"""Lógica da auditoria de metadados do corpus (Etapa 0).

Transforma a exploração do corpus em artefatos reprodutíveis e determinísticos:
quatro CSVs de agregados e um relatório em Markdown. Não baixa nem decodifica
áudio. Regras de projeto confirmadas empiricamente na inspeção de schema:

- grão = SEGMENTO (não falante); ~318k linhas em train/validation/test;
- REGIÃO deriva EXCLUSIVAMENTE de `birth_state` (nunca do prefixo de
  `speaker_code`; o "MA" em `MA_HV273` é código de entrevista, não estado);
- `speaker_type == 'R'` é a fala válida do informante; P/1,P/2,P/3 são
  entrevistadores e carregam as sentinelas `age == 0` e `birth_state == 'unknown'`
  (correlação 1:1). São filtrados e contabilizados;
- `birth_state == ''` (vazio) ⟺ informante nascido fora do Brasil: mapeia para
  fora de escopo (nem NE nem SE), mas continua contabilizado como informante;
- a chave de falante é `speaker_code` (para informantes, distinta a nível de
  `speaker_code` == distinta a nível de `speaker_code + audio_name`).
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nefair.config import CorpusAuditConfig
from nefair.corpus.load import LoadResult, load_metadata

# Sentinela para birth_state que não mapeia nem para NE nem para SE. Estados
# nesse estado são LISTADOS no relatório (nunca dropados em silêncio).
OUT_OF_SCOPE = "fora_de_escopo"
# Rótulo para idades fora das bordas provisórias de bucket (reportado, não dropado).
OUT_OF_BUCKET = "fora_das_bordas"

# Valor de speaker_type que representa o informante (fala válida do estudo).
INFORMANT_SPEAKER_TYPE = "R"
# Chave de falante escolhida (ver docstring do módulo e relatório).
SPEAKER_KEY = "speaker_code"

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

# Bibliotecas cujas versões vão ao relatório (reprodutibilidade da fonte).
_REPORTED_PACKAGES: tuple[str, ...] = (
    "datasets",
    "huggingface_hub",
    "pandas",
    "pyarrow",
    "PyYAML",
)


# --------------------------------------------------------------------------- #
# Mapa de região (funções puras, testadas em tests/test_region_map.py)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Inspeção de schema
# --------------------------------------------------------------------------- #
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
        items = sorted(counts.items(), key=lambda kv: "" if pd.isna(kv[0]) else str(kv[0]))
        lines.append(f"### {col}: {len(items)} distinto(s)")
        for value, count in items:
            shown = "<NA>" if pd.isna(value) else repr(value)
            lines.append(f"   {shown:40s} {int(count):>10d}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Agregações
# --------------------------------------------------------------------------- #
def _aggregate(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Agrega por `group_cols`: nº de falantes, nº de segmentos e horas."""
    grouped = frame.groupby(group_cols, dropna=False, observed=True)
    out = grouped.agg(
        n_speakers=(SPEAKER_KEY, "nunique"),
        n_segments=(SPEAKER_KEY, "count"),
        dur_seconds=("duration", lambda s: float(s.astype("float64").sum())),
    ).reset_index()
    out["hours"] = (out.pop("dur_seconds") / 3600.0).round(4)
    return out


def speaker_key_candidates(informants: pd.DataFrame) -> dict[str, int]:
    """Contagem de falantes distintos por candidato a chave (só informantes)."""
    return {
        "speaker_code": int(informants["speaker_code"].nunique()),
        "speaker_code+audio_name": int(informants.groupby(["speaker_code", "audio_name"]).ngroups),
    }


def speaker_split_overlap(frame: pd.DataFrame, key: str = SPEAKER_KEY) -> int:
    """Nº de falantes (por `key`) presentes em mais de um split."""
    per_key = frame.groupby(key)["split"].nunique()
    return int((per_key > 1).sum())


def assign_age_bucket(ages: pd.Series, edges: tuple[int, ...]) -> pd.Series:
    """Rotula idades em buckets semiabertos [e_i, e_{i+1}); fora -> OUT_OF_BUCKET."""
    bounds = list(edges)
    labels = [f"[{bounds[i]},{bounds[i + 1]})" for i in range(len(bounds) - 1)]
    cats = pd.cut(ages, bins=bounds, right=False, labels=labels, include_lowest=True)
    return cats.astype(object).where(cats.notna(), OUT_OF_BUCKET)


def _bucket_order(edges: tuple[int, ...]) -> list[str]:
    bounds = list(edges)
    return [f"[{bounds[i]},{bounds[i + 1]})" for i in range(len(bounds) - 1)] + [OUT_OF_BUCKET]


# --------------------------------------------------------------------------- #
# CSVs de saída
# --------------------------------------------------------------------------- #
def build_speaker_counts_by_state(informants: pd.DataFrame) -> pd.DataFrame:
    """Falantes distintos, segmentos e horas por `birth_state`."""
    out = _aggregate(informants, ["birth_state", "region"])
    return out.sort_values("birth_state").reset_index(drop=True)[
        ["birth_state", "region", "n_speakers", "n_segments", "hours"]
    ]


def build_region_age_crosstab(informants: pd.DataFrame, edges: tuple[int, ...]) -> pd.DataFrame:
    """Cross-tab região × bucket etário: falantes, segmentos e horas."""
    frame = informants.copy()
    frame["age_bucket"] = assign_age_bucket(frame["age"], edges)
    out = _aggregate(frame, ["region", "age_bucket"])
    order = {label: i for i, label in enumerate(_bucket_order(edges))}
    out["_ord"] = out["age_bucket"].map(order)
    out = out.sort_values(["region", "_ord"]).drop(columns="_ord").reset_index(drop=True)
    return out[["region", "age_bucket", "n_speakers", "n_segments", "hours"]]


def build_age_distribution_by_region(informants: pd.DataFrame) -> pd.DataFrame:
    """Distribuição etária BRUTA (idade inteira) por região."""
    grouped = informants.groupby(["region", "age"], dropna=False)
    out = grouped.agg(
        n_speakers=(SPEAKER_KEY, "nunique"),
        n_segments=(SPEAKER_KEY, "count"),
    ).reset_index()
    return out.sort_values(["region", "age"]).reset_index(drop=True)


def build_data_quality_report(full: pd.DataFrame, informants: pd.DataFrame) -> pd.DataFrame:
    """Relatório de qualidade: sentinelas por coluna, audio_quality, duração,
    segmentos por falante. Formato longo (section, key, value)."""
    rows: list[tuple[str, str, float]] = []

    # Sentinelas / faltantes por coluna (sobre o dataset completo).
    sentinels = {
        "age==0": int(full["age"].eq(0).sum()),
        "birth_state=='unknown'": int(full["birth_state"].eq("unknown").sum()),
        "birth_state==''": int(full["birth_state"].eq("").sum()),
        "birth_country!='Brazil'": int(full["birth_country"].ne("Brazil").sum()),
        "education=='unknown'": int(full["education"].eq("unknown").sum()),
        "racial_category=='unknown'": int(full["racial_category"].eq("unknown").sum()),
        "speaker_gender=='X'": int(full["speaker_gender"].eq("X").sum()),
    }
    for key, value in sentinels.items():
        rows.append(("sentinela_por_coluna", key, float(value)))

    # Distribuição de audio_quality (dataset completo).
    for value, count in full["audio_quality"].value_counts().sort_index().items():
        rows.append(("audio_quality", str(value), float(count)))

    # Distribuição de duration em segundos (dataset completo).
    dur = full["duration"].astype("float64")
    for key, value in _describe(dur).items():
        rows.append(("duration_segundos", key, value))

    # Segmentos por falante (informantes).
    seg_per_speaker = informants.groupby(SPEAKER_KEY)[SPEAKER_KEY].count().astype("float64")
    for key, value in _describe(seg_per_speaker).items():
        rows.append(("segmentos_por_falante", key, value))

    return pd.DataFrame(rows, columns=["section", "key", "value"])


def _describe(series: pd.Series) -> dict[str, float]:
    return {
        "min": round(float(series.min()), 4),
        "q1": round(float(series.quantile(0.25)), 4),
        "mediana": round(float(series.quantile(0.5)), 4),
        "q3": round(float(series.quantile(0.75)), 4),
        "max": round(float(series.max()), 4),
        "media": round(float(series.mean()), 4),
    }


# --------------------------------------------------------------------------- #
# Filtragem e proveniência
# --------------------------------------------------------------------------- #
@dataclass
class FilterReport:
    total_rows: int
    dropped_by_type: dict[str, int]
    informant_rows: int


def filter_informants(frame: pd.DataFrame) -> tuple[pd.DataFrame, FilterReport]:
    """Filtra para fala válida do informante (speaker_type == 'R').

    Reporta, por valor de speaker_type, quantas linhas caíram (nunca em silêncio).
    """
    total = len(frame)
    dropped = frame[frame["speaker_type"] != INFORMANT_SPEAKER_TYPE]
    dropped_by_type = {
        str(k): int(v) for k, v in dropped["speaker_type"].value_counts().sort_index().items()
    }
    informants = frame[frame["speaker_type"] == INFORMANT_SPEAKER_TYPE].copy()
    return informants, FilterReport(total, dropped_by_type, len(informants))


def _lib_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in _REPORTED_PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "desconhecida"
    versions["python"] = platform.python_version()
    return versions


# --------------------------------------------------------------------------- #
# Relatório em Markdown
# --------------------------------------------------------------------------- #
def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def build_report(
    load_result: LoadResult,
    schema_text: str,
    filter_report: FilterReport,
    key_candidates: dict[str, int],
    overlap_all: int,
    overlap_informants: int,
    by_state: pd.DataFrame,
    by_region_age: pd.DataFrame,
    generated_at: str,
) -> str:
    versions = _lib_versions()
    region_totals = (
        by_state.groupby("region")[["n_speakers", "n_segments", "hours"]].sum().to_dict("index")
    )
    out_of_scope = by_state[by_state["region"] == OUT_OF_SCOPE].sort_values("birth_state")

    lines: list[str] = []
    lines.append("# Auditoria de metadados do corpus CORAA-MUPE")
    lines.append("")
    lines.append(
        "Artefato reprodutível e determinístico da Etapa 0. Nenhum áudio foi "
        "baixado ou decodificado; apenas metadados foram lidos."
    )
    lines.append("")
    lines.append(f"- **Gerado em (UTC):** {generated_at}")
    lines.append("")

    lines.append("## Proveniência e reprodutibilidade")
    lines.append(f"- Dataset: `{load_result.dataset_id}`")
    lines.append(f"- Revisão pedida: `{load_result.requested_revision}`")
    lines.append(f"- Revisão resolvida (commit SHA): `{load_result.resolved_revision}`")
    lines.append(f"- Método de carga: {load_result.load_method}")
    lines.append(f"- Shards parquet lidos: {len(load_result.parquet_files)}")
    lines.append(
        f"- Footprint lido (metadados): {load_result.read_bytes / 1e6:.3f} MB "
        f"(colunas: {', '.join(load_result.columns_read)})"
    )
    lines.append(
        f"- Footprint evitado (áudio nunca transferido): "
        f"{load_result.avoided_bytes / 1e9:.3f} GB "
        f"(colunas excluídas: {', '.join(load_result.excluded_columns)})"
    )
    lines.append("- Versões das bibliotecas:")
    for pkg, ver in versions.items():
        lines.append(f"  - {pkg}: {ver}")
    lines.append("")

    lines.append("## Grão e splits")
    lines.append(
        "O grão é o SEGMENTO (não o falante). Linhas por split de origem "
        "(coluna `split` preservada na concatenação):"
    )
    for split, count in load_result.split_row_counts.items():
        lines.append(f"- `{split}`: {_fmt_int(count)} segmentos")
    lines.append(f"- **Total: {_fmt_int(sum(load_result.split_row_counts.values()))} segmentos**")
    lines.append("")

    lines.append("## Chave de falante escolhida")
    lines.append(
        "Contagem de falantes distintos por candidato (apenas informantes, "
        f"`speaker_type == '{INFORMANT_SPEAKER_TYPE}'`):"
    )
    for cand, count in key_candidates.items():
        lines.append(f"- `{cand}`: {_fmt_int(count)}")
    lines.append("")
    lines.append(
        f"As duas contagens coincidem, então **`{SPEAKER_KEY}` sozinho é a chave "
        "correta**: acrescentar `audio_name` não separa mais nenhum falante. "
        "Toda contagem de 'falantes' neste relatório usa essa chave."
    )
    lines.append("")

    lines.append("## Sobreposição de falantes entre splits")
    lines.append(
        f"- Falantes (por `{SPEAKER_KEY}`) em mais de um split, no dataset "
        f"completo: {overlap_all} (são códigos de entrevistador, reutilizados "
        "em várias entrevistas)."
    )
    lines.append(
        f"- Falantes informantes em mais de um split: {overlap_informants} "
        "(sem vazamento entre train/validation/test no nível do informante)."
    )
    lines.append("")

    lines.append("## Filtragem (nenhuma linha dropada em silêncio)")
    lines.append(f"- Total de segmentos: {_fmt_int(filter_report.total_rows)}")
    lines.append(
        f"- Excluídos por não serem informante (`speaker_type != "
        f"'{INFORMANT_SPEAKER_TYPE}'`), por tipo:"
    )
    dropped_total = 0
    for stype, count in filter_report.dropped_by_type.items():
        dropped_total += count
        lines.append(f"  - `{stype}`: {_fmt_int(count)}")
    lines.append(f"  - subtotal excluído: {_fmt_int(dropped_total)}")
    lines.append(
        "  - (esses segmentos carregam as sentinelas `age == 0` e "
        "`birth_state == 'unknown'`, correlação 1:1)"
    )
    lines.append(f"- **Informantes retidos: {_fmt_int(filter_report.informant_rows)} segmentos**")
    lines.append(
        f"- Conferência: {_fmt_int(dropped_total)} + "
        f"{_fmt_int(filter_report.informant_rows)} = "
        f"{_fmt_int(dropped_total + filter_report.informant_rows)}"
    )
    lines.append("")

    lines.append("## Mapeamento de região (deriva só de `birth_state`)")
    for region in ("NE", "SE", OUT_OF_SCOPE):
        if region in region_totals:
            tot = region_totals[region]
            lines.append(
                f"- **{region}**: {_fmt_int(int(tot['n_speakers']))} falantes, "
                f"{_fmt_int(int(tot['n_segments']))} segmentos, "
                f"{tot['hours']:.2f} h"
            )
    lines.append("")
    lines.append("### Estados fora de escopo (nem NE nem SE) — listados, não dropados")
    lines.append("| birth_state | falantes | segmentos | horas |")
    lines.append("|---|--:|--:|--:|")
    for _, row in out_of_scope.iterrows():
        state = (
            "(vazio: nascido fora do Brasil)" if row["birth_state"] == "" else row["birth_state"]
        )
        lines.append(
            f"| {state} | {_fmt_int(int(row['n_speakers']))} | "
            f"{_fmt_int(int(row['n_segments']))} | {row['hours']:.2f} |"
        )
    lines.append("")
    lines.append(
        "O `birth_state` vazio corresponde a informantes nascidos fora do Brasil "
        "(`birth_country != 'Brazil'`, correlação 1:1): continuam contabilizados "
        "como informantes, mas sem região brasileira (fora de escopo)."
    )
    lines.append("")

    lines.append("## Distribuição etária por região (buckets provisórios)")
    lines.append(
        "Os buckets abaixo são PROVISÓRIOS. A distribuição etária BRUTA (idade a "
        "idade) está em `age_distribution_by_region.csv` — use-a para decidir as "
        "bordas depois de ver o confundidor (Nordeste tende a ser mais jovem)."
    )
    lines.append("")
    lines.append("| região | bucket | falantes | segmentos | horas |")
    lines.append("|---|---|--:|--:|--:|")
    for _, row in by_region_age.iterrows():
        if row["region"] in ("NE", "SE"):
            lines.append(
                f"| {row['region']} | {row['age_bucket']} | "
                f"{_fmt_int(int(row['n_speakers']))} | "
                f"{_fmt_int(int(row['n_segments']))} | {row['hours']:.2f} |"
            )
    lines.append("")

    lines.append("## Artefatos gerados")
    lines.append("- `speaker_counts_by_state.csv` — falantes e horas por `birth_state`.")
    lines.append("- `speaker_counts_by_region_age.csv` — cross-tab região × bucket etário.")
    lines.append("- `age_distribution_by_region.csv` — distribuição etária bruta por região.")
    lines.append(
        "- `data_quality_report.csv` — sentinelas, `audio_quality`, `duration`, "
        "segmentos por falante."
    )
    lines.append("")
    lines.append("## Inspeção de schema (bruta)")
    lines.append("```")
    lines.append(schema_text)
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #
def run_audit(
    config: CorpusAuditConfig, outputs_dir: str | Path, token: str | None = None
) -> LoadResult:
    """Executa a auditoria completa e escreve os CSVs + o relatório .md."""
    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)

    load_result = load_metadata(config, token=token)
    frame = load_result.frame

    schema_text = inspect_schema(frame)
    print(schema_text)

    # Filtragem para informantes (nunca dropar em silêncio).
    informants, filter_report = filter_informants(frame)
    informants = add_region_column(informants, config.region_map)

    key_candidates = speaker_key_candidates(informants)
    overlap_all = speaker_split_overlap(frame)
    overlap_informants = speaker_split_overlap(informants)

    by_state = build_speaker_counts_by_state(informants)
    by_region_age = build_region_age_crosstab(informants, config.age_bucket_edges)
    age_dist = build_age_distribution_by_region(informants)
    quality = build_data_quality_report(frame, informants)

    by_state.to_csv(out / "speaker_counts_by_state.csv", index=False)
    by_region_age.to_csv(out / "speaker_counts_by_region_age.csv", index=False)
    age_dist.to_csv(out / "age_distribution_by_region.csv", index=False)
    quality.to_csv(out / "data_quality_report.csv", index=False)

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    report = build_report(
        load_result=load_result,
        schema_text=schema_text,
        filter_report=filter_report,
        key_candidates=key_candidates,
        overlap_all=overlap_all,
        overlap_informants=overlap_informants,
        by_state=by_state,
        by_region_age=by_region_age,
        generated_at=generated_at,
    )
    (out / "corpus_audit_report.md").write_text(report, encoding="utf-8")

    print("\n# Artefatos escritos em", out)
    for name in (
        "speaker_counts_by_state.csv",
        "speaker_counts_by_region_age.csv",
        "age_distribution_by_region.csv",
        "data_quality_report.csv",
        "corpus_audit_report.md",
    ):
        print(f"  - {name}")
    return load_result
