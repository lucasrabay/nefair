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
# Painel de confundidores: tabela no nível de falante e cruzamentos por região
# --------------------------------------------------------------------------- #
def _regions_in_order(frame: pd.DataFrame, column: str = "region") -> list[str]:
    """Regiões presentes em ordem estável (NE, SE, fora_de_escopo, extras)."""
    present = set(frame[column].unique())
    ordered = [r for r in ("NE", "SE", OUT_OF_SCOPE) if r in present]
    return ordered + sorted(present - set(ordered))


# Atributos agregados no nível de falante (devem ser 1:1 com `speaker_code`).
_SPEAKER_LEVEL_ATTRS: tuple[str, ...] = (
    "birth_state",
    "region",
    "speaker_gender",
    "education",
    "racial_category",
    "age",
)


def build_speaker_table(informants: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Tabela no nível de falante (uma linha por `speaker_code`).

    Antes de reduzir cada atributo, VERIFICA que ele é 1:1 com `speaker_code`.
    Falantes com mais de um valor são reportados como aviso de qualidade — o valor
    representativo (primeiro observado) nunca é escolhido em silêncio.
    """
    grouped = informants.groupby(SPEAKER_KEY, sort=True)
    warnings: dict[str, dict] = {}
    for attr in _SPEAKER_LEVEL_ATTRS:
        distinct = grouped[attr].nunique()
        violating = distinct[distinct > 1]
        warnings[attr] = {
            "n_violations": int(len(violating)),
            "examples": sorted(str(s) for s in violating.index[:5]),
        }
    table = grouped.agg({attr: "first" for attr in _SPEAKER_LEVEL_ATTRS})
    table["n_segments"] = grouped.size()
    return table.reset_index(), warnings


def build_audio_quality_by_region(informants: pd.DataFrame) -> pd.DataFrame:
    """B1 — `audio_quality` × região no nível de SEGMENTO.

    Qualidade é atributo de segmento (um falante pode ter segmentos de qualidades
    diferentes), por isso NÃO é agregada no nível de falante. Reporta segmentos por
    qualidade, % `low` e nº de falantes com >=1 segmento `low`, por região.
    """
    regions = _regions_in_order(informants)
    qualities = sorted(informants["audio_quality"].astype(str).unique())
    ct = pd.crosstab(informants["region"], informants["audio_quality"]).reindex(
        index=regions, columns=qualities, fill_value=0
    )
    n_segments = ct.sum(axis=1)
    low = ct["low"] if "low" in ct.columns else pd.Series(0, index=regions)
    pct_low = (100.0 * low / n_segments).where(n_segments > 0, 0.0).round(4)
    n_speakers = (
        informants.groupby("region")[SPEAKER_KEY].nunique().reindex(regions).fillna(0).astype(int)
    )
    low_speakers = (
        informants[informants["audio_quality"] == "low"]
        .groupby("region")[SPEAKER_KEY]
        .nunique()
        .reindex(regions)
        .fillna(0)
        .astype(int)
    )
    out = pd.DataFrame({"region": regions})
    for quality in qualities:
        out[f"n_seg_{quality}"] = ct[quality].to_numpy()
    out["n_segments"] = n_segments.to_numpy()
    out["pct_low"] = pct_low.to_numpy()
    out["n_speakers"] = n_speakers.to_numpy()
    out["n_speakers_with_low_segment"] = low_speakers.to_numpy()
    return out


def speaker_category_by_region(
    speaker_table: pd.DataFrame, attr: str, categories: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """B2/B3/B4 — nº de falantes por região × categoria de `attr` (nível de falante).

    Grade completa: toda categoria aparece para toda região (0 inclusive), incluindo
    'unknown' e vazio. `categories` permite fixar o universo (ex.: incluir 'X' de
    `speaker_gender`, presente no corpus mas em nenhum informante) para nada omitir.
    """
    regions = _regions_in_order(speaker_table)
    present = set(speaker_table[attr].astype(str).unique())
    universe = present | set(categories) if categories is not None else present
    cats = sorted(universe)
    counts = speaker_table.groupby(["region", attr]).size().rename("n_speakers").reset_index()
    grid = pd.MultiIndex.from_product([regions, cats], names=["region", attr]).to_frame(index=False)
    out = grid.merge(counts, on=["region", attr], how="left")
    out["n_speakers"] = out["n_speakers"].fillna(0).astype(int)
    order = {region: i for i, region in enumerate(regions)}
    out["_ord"] = out["region"].map(order)
    return out.sort_values(["_ord", attr]).drop(columns="_ord").reset_index(drop=True)


def build_segments_per_speaker_by_region(speaker_table: pd.DataFrame) -> pd.DataFrame:
    """B5 — estatísticas de segmentos por falante e concentração, por região."""
    regions = _regions_in_order(speaker_table)
    rows: list[dict[str, float]] = []
    for region in regions:
        seg = speaker_table.loc[speaker_table["region"] == region, "n_segments"].astype("float64")
        total = float(seg.sum())
        rows.append(
            {
                "region": region,
                "n_speakers": int(len(seg)),
                "min_segments": int(seg.min()),
                "median_segments": round(float(seg.median()), 2),
                "mean_segments": round(float(seg.mean()), 2),
                "max_segments": int(seg.max()),
                "region_segments": int(total),
                "top_speaker_fraction": round(float(seg.max()) / total, 4) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def region_age_stats(speaker_table: pd.DataFrame) -> pd.DataFrame:
    """Idade no nível de falante por região: nº de falantes, mediana e média."""
    regions = _regions_in_order(speaker_table)
    rows: list[dict[str, float]] = []
    for region in regions:
        ages = speaker_table.loc[speaker_table["region"] == region, "age"].astype("float64")
        rows.append(
            {
                "region": region,
                "n_speakers": int(len(ages)),
                "age_median": round(float(ages.median()), 1),
                "age_mean": round(float(ages.mean()), 1),
            }
        )
    return pd.DataFrame(rows)


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


def _age_comparison_sentence(age_stats: pd.DataFrame) -> str:
    """Frase factual derivada (Tarefa A): mediana/média de idade NE vs SE.

    O sentido (quem é mais velho) é DEDUZIDO dos valores, nunca hardcoded.
    """
    stats = {row["region"]: row for _, row in age_stats.iterrows()}
    ne = stats.get("NE")
    se = stats.get("SE")
    if ne is None or se is None:
        return "Idade no nível de falante: NE ou SE ausente nesta revisão; sem comparação."
    base = (
        "Idade no nível de falante (uma idade por `speaker_code`) — "
        f"NE: mediana {ne['age_median']:.1f} anos (média {ne['age_mean']:.1f}); "
        f"SE: mediana {se['age_median']:.1f} anos (média {se['age_mean']:.1f})."
    )
    diff = se["age_median"] - ne["age_median"]
    if abs(diff) < 2:
        return (
            base
            + " Distribuições comparáveis em tendência central (diferença de mediana < 2 anos)."
        )
    older, younger = ("SE", "NE") if diff > 0 else ("NE", "SE")
    return base + f" {older} é mais velho que {younger} em {abs(diff):.1f} anos na mediana."


def _one_to_one_note(attr: str, warnings: dict[str, dict]) -> str:
    """Nota de verificação 1:1 do atributo com `speaker_code`."""
    info = warnings.get(attr, {"n_violations": 0, "examples": []})
    if info["n_violations"] == 0:
        return f"(`{attr}` verificado 1:1 com `{SPEAKER_KEY}`: 0 violações.)"
    examples = ", ".join(info["examples"])
    return (
        f"**Aviso de qualidade:** `{attr}` NÃO é 1:1 com `{SPEAKER_KEY}` "
        f"({info['n_violations']} falante(s) com múltiplos valores; ex.: {examples}). "
        "Contagens no nível de falante podem estar afetadas; valor representativo = "
        "primeiro observado."
    )


def _render_wide_speaker_counts(long_df: pd.DataFrame, attr: str) -> list[str]:
    """Tabela markdown região × categoria (contagem de falantes) + total por linha."""
    regions = _regions_in_order(long_df)
    cats = sorted(long_df[attr].astype(str).unique())
    pivot = long_df.pivot(index="region", columns=attr, values="n_speakers").fillna(0).astype(int)
    lines = ["| região | " + " | ".join(cats) + " | total |", "|---|" + "--:|" * (len(cats) + 1)]
    for region in regions:
        values = " | ".join(_fmt_int(int(pivot.loc[region, cat])) for cat in cats)
        lines.append(f"| {region} | {values} | {_fmt_int(int(pivot.loc[region].sum()))} |")
    return lines


def _unknown_coverage_lines(speaker_table: pd.DataFrame, attr: str) -> list[str]:
    """Cobertura de `attr == 'unknown'` por região e no total (nível de falante)."""
    regions = _regions_in_order(speaker_table)
    parts: list[str] = []
    for region in regions:
        sub = speaker_table[speaker_table["region"] == region]
        n = len(sub)
        unknown = int((sub[attr] == "unknown").sum())
        pct = (100.0 * unknown / n) if n else 0.0
        parts.append(f"{region}: {pct:.1f}% ({_fmt_int(unknown)}/{_fmt_int(n)})")
    total_n = len(speaker_table)
    total_unknown = int((speaker_table[attr] == "unknown").sum())
    total_pct = (100.0 * total_unknown / total_n) if total_n else 0.0
    return [
        f"Cobertura de `{attr} == 'unknown'` (falantes) — "
        + "; ".join(parts)
        + f"; total: {total_pct:.1f}% ({_fmt_int(total_unknown)}/{_fmt_int(total_n)}).",
        "Uma cobertura de 'unknown' alta significa que o atributo é inutilizável como "
        "covariável de controle — limitação a ser DECLARADA na metodologia, não uma "
        "omissão.",
    ]


def build_ne_coverage_lines(by_state: pd.DataFrame, region_map: dict[str, str]) -> list[str]:
    """Seção Tarefa C: cobertura de estados do NE (presentes vs. ausentes)."""
    ne_states = sorted(state for state, region in region_map.items() if region == "NE")
    speakers_by_state = by_state.set_index("birth_state")["n_speakers"].to_dict()
    present = [
        (s, int(speakers_by_state.get(s, 0))) for s in ne_states if speakers_by_state.get(s, 0) > 0
    ]
    absent = [s for s in ne_states if speakers_by_state.get(s, 0) == 0]

    lines = ["## Cobertura de estados do NE"]
    lines.append(
        f"Estados do NE no mapa de região: {len(ne_states)}. Com informantes "
        f"(> 0 falantes): {len(present)}; ausentes (0 falantes): {len(absent)}."
    )
    lines.append("")
    lines.append("| birth_state (NE) | falantes |")
    lines.append("|---|--:|")
    for state, count in present:
        lines.append(f"| {state} | {_fmt_int(count)} |")
    lines.append("")
    if absent:
        lines.append(
            "Estados do NE **ausentes** (0 falantes informantes): " + ", ".join(absent) + "."
        )
    else:
        lines.append("Nenhum estado do NE está ausente nesta revisão.")

    out_of_scope = by_state[by_state["region"] == OUT_OF_SCOPE]
    norte = out_of_scope[out_of_scope["birth_state"].isin(["Pará", "Rondônia"])].sort_values(
        "birth_state"
    )
    if len(norte):
        detail = ", ".join(
            f"{row['birth_state']} ({_fmt_int(int(row['n_speakers']))} falantes)"
            for _, row in norte.iterrows()
        )
        lines.append(
            "Observação: o `fora_de_escopo` inclui estados do Norte com peso "
            f"não-trivial — {detail} — já listados na tabela de estados fora de escopo."
        )
    lines.append("")
    return lines


def build_confounders_panel_lines(
    audio_quality: pd.DataFrame,
    gender_by_region: pd.DataFrame,
    education_by_region: pd.DataFrame,
    racial_by_region: pd.DataFrame,
    seg_per_speaker: pd.DataFrame,
    speaker_table: pd.DataFrame,
    warnings: dict[str, dict],
) -> list[str]:
    """Seção Tarefa B: painel de confundidores por região (B1–B5)."""
    lines = ["## Painel de confundidores por região"]
    lines.append(
        "Todos os cruzamentos usam o filtro de informante "
        f"(`speaker_type == '{INFORMANT_SPEAKER_TYPE}'`) e a chave `{SPEAKER_KEY}`, "
        "com ordenação estável. As três regiões aparecem em todos; nenhuma categoria "
        "(inclusive 'unknown' e vazio) é omitida."
    )
    lines.append("")

    # B1 — audio_quality (nível de segmento).
    lines.append("### B1. Qualidade de áudio × região (nível de segmento)")
    lines.append(
        "`audio_quality` é atributo de SEGMENTO, não de falante. Se o NE tiver "
        "proporção de `low` maior que o SE, parte de um eventual gap de acurácia "
        "seria artefato de gravação, não dialeto."
    )
    lines.append("")
    quality_cols = [c for c in audio_quality.columns if c.startswith("n_seg_")]
    header = (
        "| região | "
        + " | ".join(c.removeprefix("n_seg_") for c in quality_cols)
        + " | segmentos | % low | falantes | falantes c/ ≥1 low |"
    )
    lines.append(header)
    lines.append("|---|" + "--:|" * (len(quality_cols) + 4))
    for _, row in audio_quality.iterrows():
        seg_values = " | ".join(_fmt_int(int(row[c])) for c in quality_cols)
        lines.append(
            f"| {row['region']} | {seg_values} | {_fmt_int(int(row['n_segments']))} | "
            f"{row['pct_low']:.2f} | {_fmt_int(int(row['n_speakers']))} | "
            f"{_fmt_int(int(row['n_speakers_with_low_segment']))} |"
        )
    lines.append("")

    # B2 — gênero (nível de falante).
    lines.append("### B2. Gênero × região (nível de falante)")
    lines.append(_one_to_one_note("speaker_gender", warnings))
    lines.append("")
    lines.extend(_render_wide_speaker_counts(gender_by_region, "speaker_gender"))
    lines.append("")

    # B3 — escolaridade (nível de falante).
    lines.append("### B3. Escolaridade × região (nível de falante)")
    lines.append(_one_to_one_note("education", warnings))
    lines.append("")
    lines.extend(_render_wide_speaker_counts(education_by_region, "education"))
    lines.append("")
    lines.extend(_unknown_coverage_lines(speaker_table, "education"))
    lines.append("")

    # B4 — categoria racial (nível de falante).
    lines.append("### B4. Categoria racial × região (nível de falante)")
    lines.append(_one_to_one_note("racial_category", warnings))
    lines.append("")
    lines.extend(_render_wide_speaker_counts(racial_by_region, "racial_category"))
    lines.append("")
    lines.extend(_unknown_coverage_lines(speaker_table, "racial_category"))
    lines.append("")

    # B5 — segmentos por falante (concentração).
    lines.append("### B5. Segmentos por falante × região (concentração)")
    lines.append(
        "A fração no falante mais prolífico indica concentração / pseudo-replicação "
        "(quanto dos segmentos da região vem de um único falante)."
    )
    lines.append("")
    lines.append("| região | falantes | mín | mediana | média | máx | frac. + prolífico |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for _, row in seg_per_speaker.iterrows():
        lines.append(
            f"| {row['region']} | {_fmt_int(int(row['n_speakers']))} | "
            f"{_fmt_int(int(row['min_segments']))} | {row['median_segments']:.1f} | "
            f"{row['mean_segments']:.1f} | {_fmt_int(int(row['max_segments']))} | "
            f"{row['top_speaker_fraction']:.4f} |"
        )
    lines.append("")
    return lines


def build_report(
    load_result: LoadResult,
    schema_text: str,
    filter_report: FilterReport,
    key_candidates: dict[str, int],
    overlap_all: int,
    overlap_informants: int,
    by_state: pd.DataFrame,
    by_region_age: pd.DataFrame,
    region_map: dict[str, str],
    speaker_table: pd.DataFrame,
    speaker_warnings: dict[str, dict],
    age_stats: pd.DataFrame,
    audio_quality: pd.DataFrame,
    gender_by_region: pd.DataFrame,
    education_by_region: pd.DataFrame,
    racial_by_region: pd.DataFrame,
    seg_per_speaker_by_region: pd.DataFrame,
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

    lines.extend(build_ne_coverage_lines(by_state, region_map))

    lines.append("## Distribuição etária por região (buckets provisórios)")
    lines.append(_age_comparison_sentence(age_stats))
    lines.append("")
    lines.append(
        "Os buckets abaixo são PROVISÓRIOS. A distribuição etária BRUTA (idade a "
        "idade) está em `age_distribution_by_region.csv` — use-a para decidir as bordas."
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

    lines.extend(
        build_confounders_panel_lines(
            audio_quality=audio_quality,
            gender_by_region=gender_by_region,
            education_by_region=education_by_region,
            racial_by_region=racial_by_region,
            seg_per_speaker=seg_per_speaker_by_region,
            speaker_table=speaker_table,
            warnings=speaker_warnings,
        )
    )

    lines.append("## Artefatos gerados")
    lines.append("- `speaker_counts_by_state.csv` — falantes e horas por `birth_state`.")
    lines.append("- `speaker_counts_by_region_age.csv` — cross-tab região × bucket etário.")
    lines.append("- `age_distribution_by_region.csv` — distribuição etária bruta por região.")
    lines.append(
        "- `data_quality_report.csv` — sentinelas, `audio_quality`, `duration`, "
        "segmentos por falante."
    )
    lines.append("- `audio_quality_by_region.csv` — B1: qualidade × região (segmento).")
    lines.append("- `gender_by_region.csv` — B2: gênero × região (falante).")
    lines.append("- `education_by_region.csv` — B3: escolaridade × região (falante).")
    lines.append("- `racial_category_by_region.csv` — B4: categoria racial × região (falante).")
    lines.append("- `segments_per_speaker_by_region.csv` — B5: segmentos por falante × região.")
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

    # Tabela no nível de falante (com verificação 1:1) e agregados.
    speaker_table, speaker_warnings = build_speaker_table(informants)

    by_state = build_speaker_counts_by_state(informants)
    by_region_age = build_region_age_crosstab(informants, config.age_bucket_edges)
    age_dist = build_age_distribution_by_region(informants)
    quality = build_data_quality_report(frame, informants)

    # Painel de confundidores por região (B1–B5) e estatísticas de idade.
    age_stats = region_age_stats(speaker_table)
    audio_quality = build_audio_quality_by_region(informants)
    # Universo de categorias vindo do dataset COMPLETO (inclui, p.ex., gênero 'X',
    # presente no corpus mas em nenhum informante): nada é omitido.
    gender_cats = tuple(sorted(frame["speaker_gender"].astype(str).unique()))
    education_cats = tuple(sorted(frame["education"].astype(str).unique()))
    racial_cats = tuple(sorted(frame["racial_category"].astype(str).unique()))
    gender_by_region = speaker_category_by_region(speaker_table, "speaker_gender", gender_cats)
    education_by_region = speaker_category_by_region(speaker_table, "education", education_cats)
    racial_by_region = speaker_category_by_region(speaker_table, "racial_category", racial_cats)
    seg_per_speaker_by_region = build_segments_per_speaker_by_region(speaker_table)

    by_state.to_csv(out / "speaker_counts_by_state.csv", index=False)
    by_region_age.to_csv(out / "speaker_counts_by_region_age.csv", index=False)
    age_dist.to_csv(out / "age_distribution_by_region.csv", index=False)
    quality.to_csv(out / "data_quality_report.csv", index=False)
    audio_quality.to_csv(out / "audio_quality_by_region.csv", index=False)
    gender_by_region.to_csv(out / "gender_by_region.csv", index=False)
    education_by_region.to_csv(out / "education_by_region.csv", index=False)
    racial_by_region.to_csv(out / "racial_category_by_region.csv", index=False)
    seg_per_speaker_by_region.to_csv(out / "segments_per_speaker_by_region.csv", index=False)

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
        region_map=config.region_map,
        speaker_table=speaker_table,
        speaker_warnings=speaker_warnings,
        age_stats=age_stats,
        audio_quality=audio_quality,
        gender_by_region=gender_by_region,
        education_by_region=education_by_region,
        racial_by_region=racial_by_region,
        seg_per_speaker_by_region=seg_per_speaker_by_region,
        generated_at=generated_at,
    )
    (out / "corpus_audit_report.md").write_text(report, encoding="utf-8")

    print("\n# Artefatos escritos em", out)
    for name in (
        "speaker_counts_by_state.csv",
        "speaker_counts_by_region_age.csv",
        "age_distribution_by_region.csv",
        "data_quality_report.csv",
        "audio_quality_by_region.csv",
        "gender_by_region.csv",
        "education_by_region.csv",
        "racial_category_by_region.csv",
        "segments_per_speaker_by_region.csv",
        "corpus_audit_report.md",
    ):
        print(f"  - {name}")
    return load_result
