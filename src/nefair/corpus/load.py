"""Loader *metadata-only* do corpus CORAA-MUPE.

A coluna `audio` (struct<bytes, path>) concentra ~99,9% do peso dos parquets
(~469 MB/shard vs. ~0,5 MB de todos os metadados juntos). Portanto NUNCA a lemos.

Estratégia de leitura (medida empiricamente como a única rápida o suficiente):
a latência de um range request à HF é ~0,6 s e cada shard tem ~38 row-groups —
ler as colunas de metadados via `HfFileSystem`/pyarrow emite dezenas de requisições
*sequenciais* por shard (~45 s/shard). Em vez disso:

1. lemos apenas o footer do parquet (metadados do arquivo) via *suffix range*,
   que já devolve o tamanho total do arquivo;
2. calculamos os intervalos de bytes das colunas de metadados (tudo menos `audio`);
3. coalescemos intervalos adjacentes (sem cruzar o "buraco" de áudio de ~12 MB);
4. buscamos esses intervalos EM PARALELO e servimos a pyarrow por um arquivo
   esparso em memória (só os bytes buscados; nenhum áudio).

Nenhum byte da coluna `audio` é transferido. O footprint de leitura é medido
exatamente pelo `total_compressed_size` das colunas no footer.
"""

from __future__ import annotations

import bisect
import io
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, get_session, hf_hub_url
from huggingface_hub.utils import build_hf_headers

from nefair.config import CorpusAuditConfig

# Layout confirmado empiricamente em nilc-nlp/CORAA-MUPE-ASR: parquet nativo em
# `main`, arquivos `data/<split>-NNNNN-of-MMMMM.parquet`.
_DATA_PREFIX = "data"

# Concorrência: o cliente httpx da huggingface_hub tem pool de 100 conexões.
# shard_workers * fetch_workers deve ficar confortavelmente abaixo disso.
_FETCH_WORKERS = 10  # range requests simultâneos por shard
_SHARD_WORKERS = 6  # shards processados em paralelo (memória desprezível cada)
# Intervalos separados por menos que isto são unidos numa única requisição.
# 64 KiB é bem menor que o "buraco" de áudio (~12 MB): nunca baixamos áudio.
_COALESCE_GAP = 64 * 1024
# Tamanho do bloco de footer a sondar (o metadado do arquivo cabe aqui).
_FOOTER_PROBE = 256 * 1024
_REQUEST_TIMEOUT = 60
_MAX_RETRIES = 4
_RETRY_BACKOFF = 0.5


@dataclass(frozen=True)
class LoadResult:
    """Resultado do carregamento de metadados, com proveniência e footprint."""

    frame: pd.DataFrame
    dataset_id: str
    requested_revision: str | None
    resolved_revision: str
    parquet_files: tuple[str, ...]
    columns_read: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    read_bytes: int
    avoided_bytes: int
    load_method: str
    split_row_counts: dict[str, int]


def resolve_revision(api: HfApi, dataset_id: str, revision: str | None) -> str:
    """Resolve a revisão pedida (ou default) para um commit SHA imutável."""
    info = api.dataset_info(dataset_id, revision=revision)
    return info.sha


def list_split_files(
    api: HfApi, dataset_id: str, revision: str, splits: tuple[str, ...]
) -> dict[str, list[str]]:
    """Mapeia cada split para seus shards parquet, em ordem estável.

    Casamos pelo prefixo `data/<split>-` para evitar que substrings colidam.
    """
    all_files = api.list_repo_files(dataset_id, repo_type="dataset", revision=revision)
    parquet = sorted(f for f in all_files if f.endswith(".parquet"))
    result: dict[str, list[str]] = {}
    for split in splits:
        prefix = f"{_DATA_PREFIX}/{split}-"
        shards = sorted(f for f in parquet if f.startswith(prefix))
        if not shards:
            raise ValueError(
                f"Nenhum shard parquet encontrado para split '{split}' "
                f"(prefixo '{prefix}') em {dataset_id}@{revision}."
            )
        result[split] = shards
    return result


class _RangeReader:
    """Busca intervalos de bytes de um arquivo remoto via range requests."""

    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self._url = url
        self._headers = headers
        self._session = get_session()

    def _request(self, range_header: str):
        """GET com o header Range dado, com retries em falhas transitórias."""
        headers = dict(self._headers, Range=range_header)
        last: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.get(self._url, headers=headers, timeout=_REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp
            except Exception as exc:  # noqa: BLE001 - timeout/reset/5xx transitórios
                last = exc
                time.sleep(_RETRY_BACKOFF * (attempt + 1))
        raise RuntimeError(f"falha ao ler {self._url} ({range_header})") from last

    def get(self, lo: int, hi: int) -> tuple[int, bytes]:
        """Lê [lo, hi) e devolve (lo, bytes). `hi` é exclusivo."""
        resp = self._request(f"bytes={lo}-{hi - 1}")
        return lo, resp.content

    def get_suffix(self, n: int) -> tuple[int, int, bytes]:
        """Lê os últimos `n` bytes; devolve (tamanho_total, offset_inicial, bytes)."""
        resp = self._request(f"bytes=-{n}")
        # Content-Range: bytes {start}-{end}/{total}
        total = int(resp.headers["content-range"].split("/")[-1])
        data = resp.content
        return total, total - len(data), data

    def get_many(self, ranges: list[tuple[int, int]]) -> list[tuple[int, bytes]]:
        with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
            return list(pool.map(lambda r: self.get(*r), ranges))


class _SparseFile(io.RawIOBase):
    """Arquivo seekável em memória contendo apenas segmentos pré-buscados.

    pyarrow lê footer + chunks das colunas projetadas, todos dentro de segmentos
    já buscados. Leituras fora dos segmentos (não esperadas) caem num fallback
    que busca sob demanda, garantindo robustez.
    """

    def __init__(self, reader: _RangeReader, size: int) -> None:
        super().__init__()
        self._reader = reader
        self._size = size
        self._pos = 0
        self._starts: list[int] = []
        self._segments: dict[int, tuple[int, bytes]] = {}

    def add(self, start: int, data: bytes) -> None:
        i = bisect.bisect_left(self._starts, start)
        if i < len(self._starts) and self._starts[i] == start:
            return
        self._starts.insert(i, start)
        self._segments[start] = (start + len(data), data)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = pos
        elif whence == io.SEEK_CUR:
            self._pos += pos
        elif whence == io.SEEK_END:
            self._pos = self._size + pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._pos
        data = self._read_at(self._pos, size)
        self._pos += len(data)
        return data

    def readinto(self, b) -> int:  # noqa: ANN001
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def _read_at(self, lo: int, n: int) -> bytes:
        hi = min(lo + n, self._size)
        i = bisect.bisect_right(self._starts, lo) - 1
        if i >= 0:
            start = self._starts[i]
            end, data = self._segments[start]
            if lo >= start and hi <= end:
                return bytes(data[lo - start : hi - start])
        _, fetched = self._reader.get(lo, hi)  # fallback sob demanda
        self.add(lo, fetched)
        return fetched


def _column_ranges(
    metadata: pq.FileMetaData, exclude: frozenset[str]
) -> tuple[list[tuple[int, int]], int, int]:
    """Intervalos de bytes das colunas mantidas + (bytes lidos, bytes evitados)."""
    ranges: list[tuple[int, int]] = []
    read_bytes = 0
    avoided_bytes = 0
    for rg in range(metadata.num_row_groups):
        rgm = metadata.row_group(rg)
        for c in range(rgm.num_columns):
            chunk = rgm.column(c)
            top = chunk.path_in_schema.split(".")[0]
            if top in exclude:
                avoided_bytes += chunk.total_compressed_size
                continue
            read_bytes += chunk.total_compressed_size
            start = (
                chunk.dictionary_page_offset
                if chunk.has_dictionary_page
                else chunk.data_page_offset
            )
            ranges.append((start, start + chunk.total_compressed_size))
    return ranges, read_bytes, avoided_bytes


def _coalesce(ranges: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Une intervalos ordenados separados por menos que `gap` bytes."""
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(ranges):
        if merged and lo - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _load_shard(
    url: str, headers: dict[str, str], exclude: frozenset[str]
) -> tuple[pd.DataFrame, list[str], int, int]:
    """Carrega um shard inteiro (colunas de metadados) para um DataFrame."""
    reader = _RangeReader(url, headers)

    size, footer_start, footer = reader.get_suffix(_FOOTER_PROBE)
    sparse = _SparseFile(reader, size)
    sparse.add(footer_start, footer)
    _, magic = reader.get(0, 8)
    sparse.add(0, magic)

    declared = int.from_bytes(footer[-8:-4], "little")
    if declared + 8 > len(footer):  # footer maior que a sondagem: busca exato
        lo = size - declared - 8
        _, data = reader.get(lo, size)
        sparse.add(lo, data)

    probe = pq.ParquetFile(sparse)
    columns = [name for name in probe.schema_arrow.names if name not in exclude]
    ranges, read_bytes, avoided_bytes = _column_ranges(probe.metadata, exclude)

    for lo, data in reader.get_many(_coalesce(ranges, _COALESCE_GAP)):
        sparse.add(lo, data)

    table = pq.ParquetFile(sparse).read(columns=columns)
    return table.to_pandas(), columns, read_bytes, avoided_bytes


def load_metadata(config: CorpusAuditConfig, token: str | None = None) -> LoadResult:
    """Carrega SOMENTE metadados dos splits configurados em um único DataFrame.

    Adiciona a coluna `split` (origem) e nunca toca na(s) coluna(s) de áudio.
    Os shards são lidos em paralelo, mas remontados em ordem estável (o resultado
    independe da ordem de conclusão), preservando o determinismo.
    """
    api = HfApi(token=token)
    headers = build_hf_headers(token=token)
    exclude = frozenset(config.audio_columns)

    resolved = resolve_revision(api, config.dataset_id, config.dataset.revision)
    split_files = list_split_files(api, config.dataset_id, resolved, config.dataset.splits)

    # Lista ordenada e estável de (split, caminho_relativo).
    tasks: list[tuple[str, str]] = [
        (split, rel_path) for split in config.dataset.splits for rel_path in split_files[split]
    ]

    def work(task: tuple[str, str]):
        split, rel_path = task
        url = hf_hub_url(config.dataset_id, rel_path, repo_type="dataset", revision=resolved)
        frame, cols, r_bytes, a_bytes = _load_shard(url, headers, exclude)
        frame["split"] = split
        return split, rel_path, frame, cols, r_bytes, a_bytes

    with ThreadPoolExecutor(max_workers=_SHARD_WORKERS) as pool:
        results = list(pool.map(work, tasks))  # preserva a ordem de `tasks`

    frames: list[pd.DataFrame] = []
    parquet_files: list[str] = []
    columns_read: list[str] = []
    read_bytes = 0
    avoided_bytes = 0
    split_row_counts: dict[str, int] = dict.fromkeys(config.dataset.splits, 0)

    for split, rel_path, frame, cols, r_bytes, a_bytes in results:
        frames.append(frame)
        parquet_files.append(rel_path)
        columns_read = cols
        read_bytes += r_bytes
        avoided_bytes += a_bytes
        split_row_counts[split] += len(frame)

    combined = pd.concat(frames, ignore_index=True)

    return LoadResult(
        frame=combined,
        dataset_id=config.dataset_id,
        requested_revision=config.dataset.revision,
        resolved_revision=resolved,
        parquet_files=tuple(parquet_files),
        columns_read=tuple(columns_read),
        excluded_columns=tuple(sorted(exclude)),
        read_bytes=read_bytes,
        avoided_bytes=avoided_bytes,
        load_method=(
            "prefetch paralelo de intervalos de bytes das colunas de metadados "
            "(range requests coalescidos sobre a URL de resolve da HF) + pyarrow "
            "em arquivo esparso de memória; coluna(s) de áudio nunca transferida(s)"
        ),
        split_row_counts=split_row_counts,
    )
