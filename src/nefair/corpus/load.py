"""Loader *metadata-only* do corpus CORAA-MUPE.

A coluna `audio` (struct<bytes, path>) concentra ~99,9% do peso dos parquets
(~469 MB/shard vs. ~0,5 MB de todos os metadados juntos). Portanto NUNCA a lemos:
usamos projeção de colunas do parquet via `HfFileSystem` (leitura por range
requests sobre `hf://`), de modo que apenas os chunks das colunas de metadados
trafegam pela rede. Nenhum áudio é baixado nem decodificado.

O footprint de leitura é medido exatamente a partir do `total_compressed_size`
de cada coluna no footer dos parquets (bytes lidos vs. bytes evitados).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from nefair.config import CorpusAuditConfig

# Layout confirmado empiricamente em nilc-nlp/CORAA-MUPE-ASR: parquet nativo em
# `main`, arquivos `data/<split>-NNNNN-of-MMMMM.parquet`.
_DATA_PREFIX = "data"


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


def _hf_path(dataset_id: str, rel_path: str) -> str:
    return f"datasets/{dataset_id}/{rel_path}"


def resolve_revision(api: HfApi, dataset_id: str, revision: str | None) -> str:
    """Resolve a revisão pedida (ou default) para um commit SHA imutável."""
    info = api.dataset_info(dataset_id, revision=revision)
    return info.sha


def list_split_files(
    api: HfApi, dataset_id: str, revision: str, splits: tuple[str, ...]
) -> dict[str, list[str]]:
    """Mapeia cada split para seus shards parquet, em ordem estável.

    Casamos pelo prefixo `data/<split>-` para evitar que substrings colidam
    (ex.: um split "test" não captura arquivos de outro nome).
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


def _read_shard(
    fs: HfFileSystem,
    dataset_id: str,
    rel_path: str,
    revision: str,
    exclude: frozenset[str],
) -> tuple[pd.DataFrame, list[str], int, int]:
    """Lê um shard projetando fora as colunas `exclude`.

    Retorna (frame, colunas_lidas, bytes_lidos, bytes_evitados), onde os bytes
    vêm do tamanho comprimido das colunas no footer do parquet.
    """
    path = _hf_path(dataset_id, rel_path)
    with fs.open(path, "rb", revision=revision) as fh:
        pf = pq.ParquetFile(fh)
        schema_names = list(pf.schema_arrow.names)
        columns = [name for name in schema_names if name not in exclude]

        read_bytes = 0
        avoided_bytes = 0
        md = pf.metadata
        for rg in range(md.num_row_groups):
            rgm = md.row_group(rg)
            for c in range(rgm.num_columns):
                col = rgm.column(c)
                top = col.path_in_schema.split(".")[0]
                if top in exclude:
                    avoided_bytes += col.total_compressed_size
                else:
                    read_bytes += col.total_compressed_size

        table = pf.read(columns=columns)
    return table.to_pandas(), columns, read_bytes, avoided_bytes


def load_metadata(config: CorpusAuditConfig, token: str | None = None) -> LoadResult:
    """Carrega SOMENTE metadados dos splits configurados em um único DataFrame.

    Adiciona a coluna `split` (origem) e nunca toca na(s) coluna(s) de áudio.
    """
    api = HfApi(token=token)
    fs = HfFileSystem(token=token)
    exclude = frozenset(config.audio_columns)

    resolved = resolve_revision(api, config.dataset_id, config.dataset.revision)
    split_files = list_split_files(api, config.dataset_id, resolved, config.dataset.splits)

    frames: list[pd.DataFrame] = []
    parquet_files: list[str] = []
    columns_read: list[str] = []
    read_bytes = 0
    avoided_bytes = 0
    split_row_counts: dict[str, int] = {}

    for split in config.dataset.splits:
        split_rows = 0
        for rel_path in split_files[split]:
            frame, cols, r_bytes, a_bytes = _read_shard(
                fs, config.dataset_id, rel_path, resolved, exclude
            )
            frame = frame.copy()
            frame["split"] = split
            frames.append(frame)
            parquet_files.append(rel_path)
            columns_read = cols  # idêntico entre shards (mesmo schema)
            read_bytes += r_bytes
            avoided_bytes += a_bytes
            split_rows += len(frame)
        split_row_counts[split] = split_rows

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
            "projeção de colunas via HfFileSystem (hf://) + pyarrow; "
            "coluna(s) de áudio nunca lida(s)"
        ),
        split_row_counts=split_row_counts,
    )
