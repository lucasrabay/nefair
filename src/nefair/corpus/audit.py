"""Lógica da auditoria de metadados do corpus (Etapa 0).

Nesta fase, o módulo cobre o mapa de região (função pura, testável) e a inspeção
de schema. As etapas de filtragem, escolha de chave de falante, cross-tabs e
relatório são adicionadas após a confirmação empírica do schema.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nefair.config import CorpusAuditConfig
from nefair.corpus.load import LoadResult, load_metadata

# Sentinela para birth_state que não mapeia nem para NE nem para SE. Estados
# nesse estado são LISTADOS no relatório (nunca dropados em silêncio).
OUT_OF_SCOPE = "fora_de_escopo"

# Colunas categóricas de baixa cardinalidade cujos valores distintos são
# inspecionados e reportados (Princípio 1 do briefing).
CATEGORICAL_LOW_CARD: tuple[str, ...] = (
    "speaker_type",
    "birth_state",
    "birth_country",
    "education",
    "speaker_gender",
    "audio_quality",
    "racial_category",
)


def map_state_to_region(state: str, region_map: dict[str, str]) -> str:
    """Mapeia um `birth_state` para 'NE'/'SE' ou OUT_OF_SCOPE.

    A região deriva EXCLUSIVAMENTE de `birth_state`. Ela NUNCA deve ser inferida
    do prefixo de `speaker_code` (o "MA" em `MA_HV273` é código de entrevista,
    não estado — a falante correspondente tem `birth_state = São Paulo`).
    """
    return region_map.get(state, OUT_OF_SCOPE)


def add_region_column(
    frame: pd.DataFrame, region_map: dict[str, str], *, column: str = "region"
) -> pd.DataFrame:
    """Adiciona a coluna de região derivada de `birth_state` (nunca de outra)."""
    result = frame.copy()
    result[column] = frame["birth_state"].map(lambda s: map_state_to_region(s, region_map))
    return result


def inspect_schema(frame: pd.DataFrame) -> str:
    """Descreve o schema empírico: colunas, dtypes, nulos e categóricas.

    Retorna um bloco de texto determinístico (ordenação estável), pronto para ir
    ao stdout e ao relatório.
    """
    lines: list[str] = []
    n_rows = len(frame)
    lines.append(f"Linhas totais (splits concatenados): {n_rows}")
    lines.append(f"Colunas ({len(frame.columns)}):")

    lines.append("")
    lines.append("## Colunas, dtype e nulos")
    lines.append(f"{'coluna':22s} {'dtype':16s} {'n_nulos':>10s} {'% nulos':>9s}")
    for col in frame.columns:
        n_null = int(frame[col].isna().sum())
        pct = (100.0 * n_null / n_rows) if n_rows else 0.0
        lines.append(f"{col:22s} {str(frame[col].dtype):16s} {n_null:>10d} {pct:>8.3f}%")

    lines.append("")
    lines.append("## Valores distintos das categóricas de baixa cardinalidade")
    for col in CATEGORICAL_LOW_CARD:
        lines.append("")
        if col not in frame.columns:
            lines.append(f"### {col}: (coluna ausente)")
            continue
        counts = frame[col].value_counts(dropna=False)
        # Ordenação estável: por valor (string) para bytes idênticos entre runs.
        items = sorted(counts.items(), key=lambda kv: "" if pd.isna(kv[0]) else str(kv[0]))
        lines.append(f"### {col}: {len(items)} distinto(s)")
        for value, count in items:
            shown = "<NA>" if pd.isna(value) else repr(value)
            lines.append(f"   {shown:40s} {int(count):>10d}")

    return "\n".join(lines)


def run_audit(
    config: CorpusAuditConfig, outputs_dir: str | Path, token: str | None = None
) -> LoadResult:
    """Executa a auditoria.

    ETAPA ATUAL (pré-confirmação de schema): carrega os metadados (sem áudio) e
    imprime/salva a inspeção de schema. As demais etapas (filtragem, chave de
    falante, cross-tabs, relatório) são adicionadas após a confirmação dos
    valores reais de `speaker_type` e `birth_state`.
    """
    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)

    load_result = load_metadata(config, token=token)

    schema_text = inspect_schema(load_result.frame)
    print(schema_text)
    print("\n# Proveniência de carregamento")
    print(f"dataset: {load_result.dataset_id}")
    print(f"revisão pedida: {load_result.requested_revision}")
    print(f"revisão resolvida (SHA): {load_result.resolved_revision}")
    print(f"método: {load_result.load_method}")
    print(f"shards lidos: {len(load_result.parquet_files)}")
    print(f"colunas lidas: {list(load_result.columns_read)}")
    print(f"colunas excluídas (áudio): {list(load_result.excluded_columns)}")
    print(f"footprint lido (metadados): {load_result.read_bytes / 1e6:.3f} MB")
    print(f"footprint evitado (áudio): {load_result.avoided_bytes / 1e9:.3f} GB")
    print(f"linhas por split: {load_result.split_row_counts}")

    (out / "schema_inspection.txt").write_text(schema_text + "\n", encoding="utf-8")

    return load_result
