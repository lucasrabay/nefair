"""Download seletivo do áudio das janelas selecionadas (Etapa 2).

A coluna `audio` do CORAA-MUPE pesa 41,8 GB; as janelas da Etapa 1 somam ~29 h,
uma fração pequena disso. Baixar o dataset inteiro para depois jogar fora 90% é
caro e desnecessário — e a Etapa 0 já construiu a máquina certa para evitá-lo:
`RangeReader` / `SparseFile` / `coalesce_ranges`, reusados aqui sem
reimplementação de range request, retry ou arquivo esparso.

O que muda em relação à Etapa 0 é só a coluna e o escopo: lá se buscavam TODAS as
colunas de metadados de TODOS os row groups; aqui se busca a coluna `audio`
apenas dos row groups que contêm segmentos de janelas selecionadas. O grão do
parquet é o row group: não existe "baixar uma linha". Por isso o plano de
download é, literalmente, um conjunto de pares `(shard, row_group)`.

O módulo é dividido em duas metades explícitas:

- **lógica pura** — decidir quais row groups baixar, quantos bytes isso custa,
  decodificar/reamostrar/concatenar e montar o manifesto. Tudo testável offline
  contra um parquet sintético escrito em `tmp_path`;
- **I/O de rede** — abrir o footer remoto, buscar os intervalos de bytes. Isolada
  no fim do arquivo, atrás de funções que recebem a URL.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

from nefair.corpus.load import COALESCE_GAP, FOOTER_PROBE, RangeReader, SparseFile, coalesce_ranges
from nefair.schema import ExclusionLedger, Window

# Taxa alvo. 16 kHz mono é a entrada canônica de todo ASR relevante (Whisper e
# derivados reamostram para 16 kHz internamente): normalizar aqui uma vez evita
# que cada modelo aplique um reamostrador diferente ao mesmo áudio, o que faria
# a condição `asr` variar por um detalhe de pré-processamento.
TARGET_SAMPLE_RATE = 16_000

# Coluna de áudio no parquet do CORAA-MUPE. É `struct<bytes, path>` no dataset
# real e `binary` puro na fixture sintética; `segment_bytes` trata os dois.
AUDIO_COLUMN = "audio"

# Concorrência da fase de indexação (footer + coluna `file_path` por shard).
# Mesma ordem de grandeza da Etapa 0, pelas mesmas razões (pool httpx de 100
# conexões e risco de 429 na HF).
INDEX_WORKERS = 6

# Motivos de exclusão de janela (vocabulário fechado, aparece no manifesto).
STATUS_OK = "ok"
STATUS_MISSING_SEGMENT = "segmento_ausente_no_dataset"
STATUS_DECODE_FAILED = "falha_na_decodificacao"


# =========================================================================== #
# LÓGICA PURA — plano de download
# =========================================================================== #
def row_group_bounds(metadata: pq.FileMetaData) -> np.ndarray:
    """Fronteiras de linha dos row groups: `bounds[g]` é a 1ª linha do grupo `g`.

    Tem `num_row_groups + 1` elementos, de modo que `bounds[-1] == num_rows`.
    """
    sizes = [metadata.row_group(g).num_rows for g in range(metadata.num_row_groups)]
    return np.concatenate(([0], np.cumsum(sizes))).astype("int64")


def rows_to_row_groups(bounds: np.ndarray, rows: Sequence[int]) -> dict[int, tuple[int, ...]]:
    """Agrupa linhas (índices globais no shard) pelo row group que as contém."""
    grouped: dict[int, list[int]] = {}
    for row in rows:
        group = int(np.searchsorted(bounds, row, side="right") - 1)
        grouped.setdefault(group, []).append(int(row))
    return {group: tuple(sorted(rows_in)) for group, rows_in in sorted(grouped.items())}


def column_chunk_ranges(
    metadata: pq.FileMetaData,
    column: str,
    row_groups: Iterable[int] | None = None,
) -> tuple[list[tuple[int, int]], int]:
    """Intervalos de bytes dos chunks de `column`, e quantos bytes somam.

    O casamento é pelo PRIMEIRO componente de `path_in_schema`, porque uma coluna
    `struct<bytes, path>` aparece no footer como dois chunks (`audio.bytes` e
    `audio.path`): ambos pertencem à coluna `audio` e ambos precisam ser buscados.
    """
    selected = (
        range(metadata.num_row_groups)
        if row_groups is None
        else sorted(set(int(g) for g in row_groups))
    )
    ranges: list[tuple[int, int]] = []
    total = 0
    for group in selected:
        row_group = metadata.row_group(group)
        for index in range(row_group.num_columns):
            chunk = row_group.column(index)
            if chunk.path_in_schema.split(".")[0] != column:
                continue
            start = (
                chunk.dictionary_page_offset
                if chunk.has_dictionary_page
                else chunk.data_page_offset
            )
            ranges.append((start, start + chunk.total_compressed_size))
            total += chunk.total_compressed_size
    return ranges, total


@dataclass(frozen=True)
class ShardPlan:
    """O que baixar de um shard: row groups, linhas-alvo e o custo em bytes."""

    shard: str
    split: str
    n_row_groups_total: int
    row_groups: tuple[int, ...]
    rows_by_group: Mapping[int, tuple[int, ...]]
    file_paths_by_group: Mapping[int, tuple[str, ...]]
    audio_bytes_to_download: int
    audio_bytes_total: int

    @property
    def audio_bytes_avoided(self) -> int:
        return self.audio_bytes_total - self.audio_bytes_to_download

    @property
    def n_rows_targeted(self) -> int:
        return sum(len(rows) for rows in self.rows_by_group.values())

    @property
    def is_empty(self) -> bool:
        return not self.row_groups


def plan_shard(
    metadata: pq.FileMetaData,
    file_paths: Sequence[str],
    wanted: frozenset[str],
    *,
    shard: str,
    split: str,
    audio_column: str = AUDIO_COLUMN,
) -> ShardPlan:
    """Decide quais row groups deste shard precisam ser baixados.

    `file_paths` é a coluna `file_path` do shard NA ORDEM DAS LINHAS — barata de ler
    (alguns KB) e a única forma robusta de localizar um segmento: as estatísticas
    de min/max do footer só serviriam se `file_path` fosse monotônico por shard, o
    que o dataset não promete.
    """
    ids = np.asarray(file_paths, dtype=object)
    hits = np.flatnonzero(np.isin(ids, np.asarray(sorted(wanted), dtype=object)))
    bounds = row_group_bounds(metadata)
    rows_by_group = rows_to_row_groups(bounds, hits.tolist())
    file_paths_by_group = {
        group: tuple(str(ids[row]) for row in rows) for group, rows in rows_by_group.items()
    }
    groups = tuple(sorted(rows_by_group))
    _, to_download = column_chunk_ranges(metadata, audio_column, groups)
    _, total = column_chunk_ranges(metadata, audio_column, None)
    return ShardPlan(
        shard=shard,
        split=split,
        n_row_groups_total=metadata.num_row_groups,
        row_groups=groups,
        rows_by_group=rows_by_group,
        file_paths_by_group=file_paths_by_group,
        audio_bytes_to_download=to_download,
        audio_bytes_total=total,
    )


@dataclass(frozen=True)
class DownloadPlan:
    """Plano completo: um `ShardPlan` por shard tocado, mais os totais."""

    shards: tuple[ShardPlan, ...]
    missing_file_paths: tuple[str, ...]

    @property
    def bytes_to_download(self) -> int:
        return sum(plan.audio_bytes_to_download for plan in self.shards)

    @property
    def bytes_avoided(self) -> int:
        return sum(plan.audio_bytes_avoided for plan in self.shards)

    @property
    def n_row_groups(self) -> int:
        return sum(len(plan.row_groups) for plan in self.shards)

    @property
    def n_row_groups_total(self) -> int:
        return sum(plan.n_row_groups_total for plan in self.shards)

    @property
    def non_empty(self) -> tuple[ShardPlan, ...]:
        return tuple(plan for plan in self.shards if not plan.is_empty)

    def to_frame(self) -> pd.DataFrame:
        """Uma linha por shard tocado — os bytes VERDADEIROS, sem atribuição."""
        return pd.DataFrame(
            [
                {
                    "shard": plan.shard,
                    "split": plan.split,
                    "n_row_groups_total": plan.n_row_groups_total,
                    "n_row_groups_selected": len(plan.row_groups),
                    "row_groups": ";".join(str(g) for g in plan.row_groups),
                    "n_rows_targeted": plan.n_rows_targeted,
                    "audio_bytes_to_download": plan.audio_bytes_to_download,
                    "audio_bytes_avoided": plan.audio_bytes_avoided,
                }
                for plan in self.non_empty
            ],
            columns=[
                "shard",
                "split",
                "n_row_groups_total",
                "n_row_groups_selected",
                "row_groups",
                "n_rows_targeted",
                "audio_bytes_to_download",
                "audio_bytes_avoided",
            ],
        )


def build_plan(shard_plans: Sequence[ShardPlan], wanted: frozenset[str]) -> DownloadPlan:
    """Junta os planos por shard e apura o que NÃO foi encontrado em lugar nenhum.

    Um segmento de janela que não aparece em nenhum shard é um erro de
    contrato entre as Etapas 1 e 2 (revisão do dataset diferente, por exemplo).
    Ele é DEVOLVIDO, nunca ignorado: quem chama decide falhar ou contabilizar.
    """
    found: set[int] = set()
    for plan in shard_plans:
        for ids in plan.file_paths_by_group.values():
            found.update(ids)
    return DownloadPlan(shards=tuple(shard_plans), missing_file_paths=tuple(sorted(wanted - found)))


def required_file_paths(windows: Sequence[Window]) -> frozenset[str]:
    """Conjunto de `file_path` (chave do SEGMENTO) que a Etapa 2 materializa."""
    return frozenset(path for window in windows for path in window.segment_file_paths)


# =========================================================================== #
# LÓGICA PURA — decodificação, reamostragem, concatenação
# =========================================================================== #
def segment_bytes(value: object) -> bytes:
    """Extrai os bytes codificados de uma célula da coluna `audio`.

    O dataset real usa `struct<bytes, path>` (formato `Audio` do `datasets`); a
    fixture sintética usa `binary` puro. Aceitar os dois é o que permite testar o
    caminho de decodificação sem rede.
    """
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is None:
            raise ValueError("célula de áudio sem campo `bytes` (só `path`): áudio não embutido.")
        return bytes(raw)
    if isinstance(value, bytes | bytearray | memoryview):
        return bytes(value)
    raise TypeError(f"célula de áudio de tipo inesperado: {type(value)!r}")


def decode_segment(raw: bytes) -> tuple[np.ndarray, int]:
    """Decodifica um segmento para mono `float32` e devolve (amostras, taxa).

    A mistura para mono é a MÉDIA dos canais, não o canal esquerdo: descartar um
    canal perderia energia de fala se a gravação tiver o informante mais forte de
    um lado.
    """
    with io.BytesIO(raw) as buffer:
        samples, sample_rate = sf.read(buffer, dtype="float32", always_2d=True)
    return samples.mean(axis=1).astype("float32"), int(sample_rate)


def resample(samples: np.ndarray, sample_rate: int, target: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Reamostra para `target` Hz com filtro polifásico anti-aliasing.

    Interpolação linear pura seria mais simples e NÃO serve: reduzir 48 kHz para
    16 kHz sem filtro anti-aliasing dobra o conteúdo acima de 8 kHz de volta para
    dentro da banda, exatamente onde vivem as fricativas — e um artefato desses
    entraria no WER como se fosse sotaque.

    `scipy` chega garantidamente junto com `statsmodels` (dependência declarada),
    mas NÃO está declarado por si em `pyproject.toml`; o import é preguiçoso e o
    erro diz o que falta, em vez de degradar a qualidade em silêncio.
    """
    if sample_rate == target:
        return samples.astype("float32")
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover - ambiente sem scipy
        raise RuntimeError(
            "reamostragem exige `scipy` (chega via `statsmodels`, mas não está "
            "declarado em pyproject.toml). Declare-o antes de rodar a Etapa 2."
        ) from exc
    from math import gcd

    divisor = gcd(int(sample_rate), int(target))
    return resample_poly(samples, int(target) // divisor, int(sample_rate) // divisor).astype(
        "float32"
    )


def concatenate_segments(segments: Sequence[np.ndarray]) -> np.ndarray:
    """Concatena os segmentos já reamostrados, na ordem temporal recebida.

    Sem crossfade nem silêncio inserido: a janela é fala contígua por construção
    (Etapa 1), e inserir qualquer coisa entre os segmentos mudaria a duração em
    relação à soma dos metadados — que é a duração declarada no manifesto.
    """
    if not segments:
        return np.zeros(0, dtype="float32")
    return np.concatenate(segments).astype("float32")


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> Path:
    """Escreve o WAV 16 bits e devolve o caminho. PCM_16 por determinismo."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, samples, sample_rate, subtype="PCM_16", format="WAV")
    return out


def sha256_of_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_wav_path(outputs_dir: str | Path, window: Window) -> Path:
    """`outputs/audio/<speaker_code>/<window_id>.wav`."""
    return Path(outputs_dir) / "audio" / window.speaker_code / f"{window.window_id}.wav"


# =========================================================================== #
# LÓGICA PURA — manifesto
# =========================================================================== #
MANIFEST_COLUMNS: tuple[str, ...] = (
    "window_id",
    "speaker_code",
    "region",
    "split",
    "audio_name",
    "n_segments",
    "segment_file_paths",
    "duration_requested_s",
    "duration_obtained_s",
    "duration_delta_s",
    "sample_rate",
    "n_samples",
    "wav_path",
    "sha256",
    "bytes_downloaded",
    "bytes_avoided",
    "status",
)


def attribute_bytes(weights: Sequence[int], total: int) -> list[int]:
    """Reparte `total` entre as janelas em proporção a `weights`, em inteiros.

    Bytes baixados não são atribuíveis a uma janela: o grão do download é o row
    group, e um mesmo row group costuma servir a várias janelas. A repartição
    proporcional ao nº de segmentos é uma CONVENÇÃO — declarada aqui e no
    manifesto — e usa o método do maior resto para que a coluna some exatamente
    o total verdadeiro, sem perder nem inventar bytes no arredondamento.
    """
    total_weight = sum(weights)
    if total_weight <= 0 or not weights:
        return [0] * len(weights)
    exact = [total * weight / total_weight for weight in weights]
    floors = [int(value) for value in exact]
    remainder = total - sum(floors)
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - floors[i]), i))
    for index in order[:remainder]:
        floors[index] += 1
    return floors


@dataclass(frozen=True)
class WindowAudio:
    """Resultado da materialização de uma janela (ou o motivo de não ter havido)."""

    window: Window
    status: str
    n_samples: int = 0
    sha256: str = ""
    wav_path: str = ""


def build_manifest(
    results: Sequence[WindowAudio], bytes_downloaded: int, bytes_avoided: int
) -> pd.DataFrame:
    """Manifesto janela → segmentos, com as durações pedida e obtida lado a lado.

    As duas durações ficam em colunas separadas (e a diferença numa terceira) de
    propósito: divergência sistemática entre a soma dos metadados e o áudio
    decodificado é sinal de que os cortes do corpus não batem com os arquivos, e
    isso precisa ser visível sem recalcular nada.
    """
    weights = [result.window.n_segments for result in results]
    downloaded = attribute_bytes(weights, bytes_downloaded)
    avoided = attribute_bytes(weights, bytes_avoided)
    rows = []
    for result, down, avoid in zip(results, downloaded, avoided, strict=True):
        window = result.window
        obtained = result.n_samples / TARGET_SAMPLE_RATE
        rows.append(
            {
                "window_id": window.window_id,
                "speaker_code": window.speaker_code,
                "region": window.region,
                "split": window.split,
                "audio_name": window.audio_name,
                "n_segments": window.n_segments,
                "segment_file_paths": ";".join(window.segment_file_paths),
                "duration_requested_s": round(window.duration_s, 4),
                "duration_obtained_s": round(obtained, 4),
                "duration_delta_s": round(obtained - window.duration_s, 4),
                "sample_rate": TARGET_SAMPLE_RATE,
                "n_samples": result.n_samples,
                "wav_path": result.wav_path,
                "sha256": result.sha256,
                "bytes_downloaded": down,
                "bytes_avoided": avoid,
                "status": result.status,
            }
        )
    frame = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    return frame.sort_values("window_id", kind="stable").reset_index(drop=True)


def manifest_ledger(results: Sequence[WindowAudio]) -> ExclusionLedger:
    """Trilha de exclusão das janelas: nenhuma some sem motivo registrado."""
    ledger = ExclusionLedger(unit="janelas", total_in=len(results))
    for result in results:
        if result.status == STATUS_OK:
            ledger.kept += 1
        else:
            ledger.drop(result.status)
    return ledger


# =========================================================================== #
# I/O DE REDE — a partir daqui tudo toca a HF
# =========================================================================== #
@dataclass
class ShardHandle:
    """Footer de um shard remoto, já buscado, pronto para servir leituras.

    Guardamos o footer (e não um `SparseFile` vivo) porque o `SparseFile`
    acumula em memória TUDO o que já foi buscado: reusar um só entre todos os
    row groups de um shard faria a memória crescer com o download inteiro. Com o
    footer em mãos, cada row group é lido sobre um arquivo esparso novo, e a
    memória fica limitada a um row group de cada vez.
    """

    rel_path: str
    split: str
    reader: RangeReader
    size: int
    footer_start: int
    footer: bytes
    magic: bytes
    metadata: pq.FileMetaData

    def fresh_sparse(self) -> SparseFile:
        sparse = SparseFile(self.reader, self.size)
        sparse.add(0, self.magic)
        sparse.add(self.footer_start, self.footer)
        return sparse


def open_shard(url: str, headers: dict[str, str], rel_path: str, split: str) -> ShardHandle:
    """Busca só o footer do shard (suffix range) e devolve o handle."""
    reader = RangeReader(url, headers)
    size, footer_start, footer = reader.get_suffix(FOOTER_PROBE)
    _, magic = reader.get(0, 8)

    declared = int.from_bytes(footer[-8:-4], "little")
    if declared + 8 > len(footer):  # footer maior que a sondagem: busca exato
        footer_start = size - declared - 8
        _, footer = reader.get(footer_start, size)

    sparse = SparseFile(reader, size)
    sparse.add(0, magic)
    sparse.add(footer_start, footer)
    return ShardHandle(
        rel_path=rel_path,
        split=split,
        reader=reader,
        size=size,
        footer_start=footer_start,
        footer=footer,
        magic=magic,
        metadata=pq.ParquetFile(sparse).metadata,
    )


def read_shard_file_paths(handle: ShardHandle) -> np.ndarray:
    """Lê a coluna `file_path` inteira do shard (alguns KB) — o índice do plano.

    `file_path` e não `audio_id`: só ele identifica o SEGMENTO. Um `audio_id`
    cobre a entrevista inteira, e casar por ele baixaria a gravação toda.
    """
    ranges, _ = column_chunk_ranges(handle.metadata, "file_path", None)
    sparse = handle.fresh_sparse()
    for offset, data in handle.reader.get_many(coalesce_ranges(ranges, COALESCE_GAP)):
        sparse.add(offset, data)
    table = pq.ParquetFile(sparse).read(columns=["file_path"])
    return np.asarray(table.column("file_path").to_pylist(), dtype=object)


def fetch_row_group_cells(
    handle: ShardHandle, row_group: int, wanted: frozenset[str]
) -> dict[str, bytes]:
    """Baixa os chunks de `audio` de UM row group e devolve {file_path: bytes}.

    `file_path` é lido junto (custa alguns bytes) para casar célula com segmento
    sem aritmética de deslocamento entre row groups — que é onde erros desse tipo
    de código costumam se esconder.
    """
    audio_ranges, _ = column_chunk_ranges(handle.metadata, AUDIO_COLUMN, [row_group])
    id_ranges, _ = column_chunk_ranges(handle.metadata, "file_path", [row_group])
    sparse = handle.fresh_sparse()
    for offset, data in handle.reader.get_many(
        coalesce_ranges(audio_ranges + id_ranges, COALESCE_GAP)
    ):
        sparse.add(offset, data)
    table = pq.ParquetFile(sparse).read_row_groups(
        [row_group], columns=[AUDIO_COLUMN, "file_path"]
    )
    ids = table.column("file_path").to_pylist()
    cells = table.column(AUDIO_COLUMN).to_pylist()
    return {
        str(path): segment_bytes(cell)
        for path, cell in zip(ids, cells, strict=True)
        if str(path) in wanted
    }


def index_shards(
    dataset_id: str,
    revision: str,
    split_files: Mapping[str, Sequence[str]],
    headers: dict[str, str],
    wanted: frozenset[str],
) -> tuple[list[ShardHandle], list[ShardPlan]]:
    """Abre todos os shards, lê `file_path` e monta um `ShardPlan` por shard.

    Os shards são lidos em paralelo mas remontados na ordem estável de `tasks`:
    o plano não depende da ordem de conclusão das requisições.
    """
    from huggingface_hub import hf_hub_url

    tasks = [(split, rel_path) for split in sorted(split_files) for rel_path in split_files[split]]

    def work(task: tuple[str, str]) -> tuple[ShardHandle, ShardPlan]:
        split, rel_path = task
        url = hf_hub_url(dataset_id, rel_path, repo_type="dataset", revision=revision)
        handle = open_shard(url, headers, rel_path, split)
        file_paths = read_shard_file_paths(handle)
        plan = plan_shard(handle.metadata, file_paths, wanted, shard=rel_path, split=split)
        return handle, plan

    with ThreadPoolExecutor(max_workers=INDEX_WORKERS) as pool:
        results = list(pool.map(work, tasks))  # preserva a ordem de `tasks`
    return [handle for handle, _ in results], [plan for _, plan in results]


def materialize_windows(
    windows: Sequence[Window],
    cells: Mapping[int, bytes],
    outputs_dir: str | Path,
) -> list[WindowAudio]:
    """Decodifica, reamostra, concatena e grava um WAV por janela.

    Recebe as células já baixadas: a função não toca a rede, o que a mantém
    testável com bytes de WAV construídos em memória.
    """
    results: list[WindowAudio] = []
    for window in windows:
        missing = [p for p in window.segment_file_paths if p not in cells]
        if missing:
            results.append(WindowAudio(window=window, status=STATUS_MISSING_SEGMENT))
            continue
        try:
            pieces = []
            for path in window.segment_file_paths:
                samples, sample_rate = decode_segment(cells[path])
                pieces.append(resample(samples, sample_rate))
            concatenated = concatenate_segments(pieces)
        except Exception as exc:  # noqa: BLE001 - qualquer falha de codec vira status
            print(f"  ! {window.window_id}: falha na decodificação ({exc})")
            results.append(WindowAudio(window=window, status=STATUS_DECODE_FAILED))
            continue
        path = window_wav_path(outputs_dir, window)
        write_wav(path, concatenated)
        results.append(
            WindowAudio(
                window=window,
                status=STATUS_OK,
                n_samples=int(len(concatenated)),
                sha256=sha256_of_file(path),
                wav_path=str(path.relative_to(Path(outputs_dir))),
            )
        )
    return results


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #
def run_fetch_audio(
    config_base_id: str,
    revision: str | None,
    windows: Sequence[Window],
    outputs_dir: str | Path,
    *,
    token: str | None = None,
    dry_run: bool = False,
) -> DownloadPlan:
    """Executa a Etapa 2 sobre as janelas dadas e escreve WAVs + manifesto.

    Com `dry_run`, só a fase de indexação acontece (footers + coluna `file_path`,
    poucos MB no total): o plano é reportado e nenhum byte de áudio é buscado.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import build_hf_headers

    from nefair.corpus.load import list_split_files, resolve_revision

    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=token)
    headers = build_hf_headers(token=token)
    resolved = resolve_revision(api, config_base_id, revision)
    splits = tuple(sorted({window.split for window in windows}))
    split_files = list_split_files(api, config_base_id, resolved, splits)

    wanted = required_file_paths(windows)
    handles, shard_plans = index_shards(config_base_id, resolved, split_files, headers, wanted)
    plan = build_plan(shard_plans, wanted)

    print(f"Revisão resolvida: {resolved}")
    print(f"Janelas: {len(windows)} | segmentos pedidos: {len(wanted)}")
    print(
        f"Row groups a baixar: {plan.n_row_groups} de {plan.n_row_groups_total} "
        f"({len(plan.non_empty)} shards de {len(plan.shards)})"
    )
    print(
        f"Bytes de áudio a baixar: {plan.bytes_to_download / 1e9:.3f} GB | "
        f"evitados: {plan.bytes_avoided / 1e9:.3f} GB"
    )
    if plan.missing_file_paths:
        print(
            f"! {len(plan.missing_file_paths)} segmento(s) de janela NÃO encontrados nesta "
            f"revisão (ex.: {list(plan.missing_file_paths[:5])})"
        )

    plan.to_frame().to_csv(out / "audio_download_plan.csv", index=False)
    if dry_run:
        print("\n(dry-run) nenhum byte de áudio foi baixado.")
        print("# Artefatos escritos em", out)
        print("  - audio_download_plan.csv")
        return plan

    by_path = {handle.rel_path: handle for handle in handles}
    cells: dict[int, bytes] = {}
    for shard_plan in plan.non_empty:
        handle = by_path[shard_plan.shard]
        for row_group in shard_plan.row_groups:
            cells.update(fetch_row_group_cells(handle, row_group, wanted))
        print(f"  {shard_plan.shard}: {len(shard_plan.row_groups)} row group(s)")

    results = materialize_windows(windows, cells, out)
    ledger = manifest_ledger(results)
    ledger.assert_balanced()

    manifest = build_manifest(results, plan.bytes_to_download, plan.bytes_avoided)
    manifest.to_csv(out / "audio_manifest.csv", index=False)

    print("\n" + "\n".join(ledger.to_markdown()))
    print("\n# Artefatos escritos em", out)
    for name in ("audio_download_plan.csv", "audio_manifest.csv", "audio/<falante>/*.wav"):
        print(f"  - {name}")
    return plan
