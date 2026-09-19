"""Análises de sensibilidade da Etapa 6.

A pergunta que estas análises respondem é sempre a mesma: **a conclusão depende
de uma escolha de desenho que poderia ter sido outra?** Um gap NE−SE que aparece
com 10 itens por falante e some com 5 não é um achado — é um artefato do tamanho
da amostra. Um gap que só existe nas janelas mais longas diz algo sobre duração,
não sobre região.

Três eixos, com custos muito diferentes (é o que separa o que está ligado do que
está desligado por padrão):

- **nº de itens por falante** — barato: subamostra os itens JÁ coletados. Nenhuma
  chamada de modelo nova, nenhuma revisão humana nova;
- **duração da janela** — barato: reestima em subconjuntos por faixa de duração.
  Também não gera dado novo;
- **nº de alternativas (D10)** — CARO: exigiria regerar os itens com 3 ou 5
  alternativas e revisá-los de novo, duplicando a revisão humana, que é o caminho
  crítico do cronograma. Desligado por padrão; quando desligado, o motivo vai ao
  relatório em vez de a análise simplesmente não aparecer.

Toda variante reusa as MESMAS reamostras de falantes da análise principal. Os
intervalos das variantes ficam assim comparáveis entre si: a diferença entre eles
vem do subconjunto de dados, não de um sorteio diferente.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nefair.analysis.regression import SpeakerResamples, estimate_gap
from nefair.schema import stable_hash

# Colunas de saída comuns a todas as variantes, para que possam ser empilhadas
# num único CSV e numa única tabela LaTeX.
SENSITIVITY_COLUMNS: tuple[str, ...] = (
    "axis",
    "variant",
    "model_key",
    "condition",
    "gap_ne_se",
    "ci_low",
    "ci_high",
    "n_trials",
    "n_items",
    "n_speakers_ne",
    "n_speakers_se",
    "n_failed_resamples",
)

# Nome esperado da coluna de duração da janela no frame de ensaios. Não existe em
# `RunRecord` (a duração é atributo de `Window`): o script da Etapa 6 junta essa
# coluna a partir de `windows.parquet` antes de chamar a sensibilidade.
DURATION_COLUMN = "window_duration_s"


@dataclass(frozen=True)
class SensitivityOutcome:
    """Resultado de um eixo: tabela de estimativas + notas para o relatório.

    `notes` nunca é decorativo. É onde fica registrado o que NÃO foi feito e por
    quê (eixo desligado, faixa sem falantes suficientes, falante com menos itens
    do que a grade pede). Uma análise de sensibilidade que omite suas próprias
    limitações não serve para o fim a que se destina.
    """

    axis: str
    enabled: bool
    table: pd.DataFrame
    notes: list[str]


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=list(SENSITIVITY_COLUMNS))


def _row(axis: str, variant: str, estimate_row: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {"axis": axis, "variant": variant}
    for column in SENSITIVITY_COLUMNS[2:]:
        row[column] = estimate_row[column]
    return row


def _estimate_subset(
    subset: pd.DataFrame,
    resamples: SpeakerResamples,
    *,
    axis: str,
    variant: str,
    covariates: Sequence[str],
    ci_level: float,
    age_min: int,
    age_max: int,
    max_failure_fraction: float,
    notes: list[str],
) -> list[dict[str, object]]:
    """Reestima o gap para cada modelo × condição de um subconjunto.

    Falhas não somem: viram nota no relatório com o modelo, a condição e a causa.
    """
    rows: list[dict[str, object]] = []
    pairs = subset[["model_key", "condition"]].drop_duplicates()
    keys = sorted((str(m), str(c)) for m, c in pairs.itertuples(index=False, name=None))
    for model_key, condition in keys:
        cell = subset[(subset["model_key"] == model_key) & (subset["condition"] == condition)]
        try:
            estimate = estimate_gap(
                cell.reset_index(drop=True),
                resamples,
                covariates=covariates,
                ci_level=ci_level,
                age_min=age_min,
                age_max=age_max,
                model_key=model_key,
                condition=condition,
                max_failure_fraction=max_failure_fraction,
            )
        except ValueError as exc:
            notes.append(f"{axis} / {variant} / {model_key} × {condition}: não estimável — {exc}")
            continue
        rows.append(_row(axis, variant, estimate.to_row()))
    return rows


# --------------------------------------------------------------------------- #
# Eixo 1 — nº de itens por falante
# --------------------------------------------------------------------------- #
def subsample_items_per_speaker(
    frame: pd.DataFrame, n_items: int, *, seed: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Escolhe até `n_items` itens por falante, de forma determinística.

    Duas decisões de projeto:

    1. **Sorteio semeado, não "os primeiros `k`".** Os itens de um falante estão
       ordenados pela posição na entrevista, e o começo de uma entrevista tem
       conteúdo sistematicamente diferente do meio (apresentação, aquecimento).
       Pegar os `k` primeiros mediria a sensibilidade ao TRECHO da entrevista, e
       não ao número de itens. O sorteio é semeado por `(seed, speaker_code)` —
       reprodutível e independente da ordem em que os falantes são processados.
    2. **Seleção no nível do ITEM, não da linha.** O item selecionado entra com
       todas as suas linhas (todos os modelos, todas as condições), preservando o
       pareamento de que a decomposição depende.

    Devolve também quantos falantes tinham menos itens do que o pedido — não é
    erro, mas precisa aparecer no relatório: numa grade de 10, um falante com 8
    itens contribui com 8, e a variante "10 itens" não é exatamente 10.
    """
    if n_items < 1:
        raise ValueError(f"n_items deve ser ≥ 1; recebido {n_items}.")
    selected: list[str] = []
    n_short = 0
    for speaker_code in sorted(frame["speaker_code"].astype(str).unique()):
        item_ids = sorted(
            frame.loc[frame["speaker_code"] == speaker_code, "item_id"].astype(str).unique()
        )
        if len(item_ids) <= n_items:
            selected.extend(item_ids)
            if len(item_ids) < n_items:
                n_short += 1
            continue
        # Semente derivada do par (semente global, falante): estável entre
        # execuções e insensível à ordem de iteração.
        digest = stable_hash([seed, speaker_code])[:16]
        rng = np.random.default_rng(int(digest, 16))
        picks = rng.permutation(len(item_ids))[:n_items]
        selected.extend(sorted(item_ids[i] for i in picks))
    subset = frame[frame["item_id"].astype(str).isin(set(selected))]
    stats = {
        "n_items_selecionados": len(selected),
        "n_falantes_com_menos_itens_que_a_grade": n_short,
    }
    return subset.reset_index(drop=True), stats


def sensitivity_n_items(
    frame: pd.DataFrame,
    resamples: SpeakerResamples,
    *,
    enabled: bool,
    grid: Sequence[int],
    covariates: Sequence[str],
    seed: int,
    ci_level: float = 0.95,
    age_min: int = 20,
    age_max: int = 69,
    max_failure_fraction: float = 0.05,
) -> SensitivityOutcome:
    """Reestima o gap com `k` itens por falante, para cada `k` da grade."""
    notes: list[str] = []
    if not enabled:
        notes.append(
            "Eixo desligado em `analysis.yaml` (`sensitivity.n_items.enabled: false`)."
        )
        return SensitivityOutcome("n_itens_por_falante", False, _empty_table(), notes)

    rows: list[dict[str, object]] = []
    for n_items in sorted(set(int(k) for k in grid)):
        subset, stats = subsample_items_per_speaker(frame, n_items, seed=seed)
        notes.append(
            f"k={n_items}: {stats['n_items_selecionados']} itens; "
            f"{stats['n_falantes_com_menos_itens_que_a_grade']} falante(s) tinham "
            "menos itens do que a grade pede e entraram com todos os que têm."
        )
        rows.extend(
            _estimate_subset(
                subset,
                resamples,
                axis="n_itens_por_falante",
                variant=f"{n_items} itens/falante",
                covariates=covariates,
                ci_level=ci_level,
                age_min=age_min,
                age_max=age_max,
                max_failure_fraction=max_failure_fraction,
                notes=notes,
            )
        )
    table = pd.DataFrame(rows, columns=list(SENSITIVITY_COLUMNS))
    return SensitivityOutcome("n_itens_por_falante", True, table, notes)


# --------------------------------------------------------------------------- #
# Eixo 2 — duração da janela
# --------------------------------------------------------------------------- #
def duration_bin_labels(bins: Sequence[float]) -> list[tuple[str, float, float]]:
    """Faixas `[b_i, b_{i+1})`, com a última fechada à direita.

    A última faixa é fechada porque 60 s é o limite superior EXATO da janela
    (Etapa 1): deixá-la aberta descartaria justamente as janelas de 60 s.
    """
    edges = [float(b) for b in bins]
    if len(edges) < 2:
        raise ValueError(f"`bins` precisa de pelo menos duas bordas; recebido {edges}.")
    if edges != sorted(edges):
        raise ValueError(f"`bins` precisa estar em ordem crescente; recebido {edges}.")
    labels: list[tuple[str, float, float]] = []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        closing = "]" if index == len(edges) - 2 else ")"
        labels.append((f"[{low:g}, {high:g}{closing} s", low, high))
    return labels


def sensitivity_window_duration(
    frame: pd.DataFrame,
    resamples: SpeakerResamples,
    *,
    enabled: bool,
    bins: Sequence[float],
    covariates: Sequence[str],
    ci_level: float = 0.95,
    age_min: int = 20,
    age_max: int = 69,
    max_failure_fraction: float = 0.05,
) -> SensitivityOutcome:
    """Reestima o gap dentro de cada faixa de duração de janela."""
    axis = "duracao_da_janela"
    notes: list[str] = []
    if not enabled:
        notes.append(
            "Eixo desligado em `analysis.yaml` (`sensitivity.window_duration.enabled: false`)."
        )
        return SensitivityOutcome(axis, False, _empty_table(), notes)
    if DURATION_COLUMN not in frame.columns:
        notes.append(
            f"Coluna `{DURATION_COLUMN}` ausente do frame de ensaios: a duração é "
            "atributo de `Window` e precisa ser juntada a partir de "
            "`outputs/windows.parquet`. Eixo não executado."
        )
        return SensitivityOutcome(axis, False, _empty_table(), notes)

    rows: list[dict[str, object]] = []
    durations = frame[DURATION_COLUMN].astype(float)
    labels = duration_bin_labels(bins)
    covered = pd.Series(False, index=frame.index)
    for label, low, high in labels:
        last = label.endswith("] s")
        mask = (durations >= low) & (durations <= high if last else durations < high)
        covered = covered | mask
        subset = frame.loc[mask]
        if subset.empty:
            notes.append(f"{label}: nenhuma janela nesta faixa; variante não estimada.")
            continue
        notes.append(
            f"{label}: {len(subset)} ensaios, "
            f"{subset['speaker_code'].nunique()} falante(s)."
        )
        rows.extend(
            _estimate_subset(
                subset,
                resamples,
                axis=axis,
                variant=label,
                covariates=covariates,
                ci_level=ci_level,
                age_min=age_min,
                age_max=age_max,
                max_failure_fraction=max_failure_fraction,
                notes=notes,
            )
        )
    n_outside = int((~covered).sum())
    if n_outside:
        # Nada dropado em silêncio: janelas fora de todas as faixas são contadas.
        notes.append(
            f"{n_outside} ensaio(s) com duração fora de todas as faixas declaradas "
            "em `sensitivity.window_duration.bins` — não entram em nenhuma variante."
        )
    table = pd.DataFrame(rows, columns=list(SENSITIVITY_COLUMNS))
    return SensitivityOutcome(axis, True, table, notes)


# --------------------------------------------------------------------------- #
# Eixo 3 — nº de alternativas (D10)
# --------------------------------------------------------------------------- #
D10_RATIONALE = (
    "Variar o nº de alternativas exigiria REGERAR os itens com 3 e com 5 "
    "alternativas e submeter cada versão nova à revisão humana completa. A revisão "
    "humana é o caminho crítico do cronograma (≈2.300 itens), e cada variante a "
    "duplicaria. Por isso o eixo fica desligado por padrão em `analysis.yaml` "
    "(`sensitivity.n_alternatives.enabled: false`); se for ligado, o escopo "
    "declarado (`scope`) limita a reexecução ao subconjunto do piloto."
)


def sensitivity_n_alternatives(
    frame: pd.DataFrame,
    resamples: SpeakerResamples,
    *,
    enabled: bool,
    scope: str,
    covariates: Sequence[str],
    ci_level: float = 0.95,
    age_min: int = 20,
    age_max: int = 69,
    max_failure_fraction: float = 0.05,
) -> SensitivityOutcome:
    """Sensibilidade ao nº de alternativas (D10) — só roda se habilitada.

    Quando desligada, devolve o MOTIVO em vez de nada: o relatório precisa
    registrar que a análise foi considerada e por que ficou de fora, e não deixar
    a impressão de que ninguém pensou nela.

    Quando ligada, este módulo **não regera itens** (isso é da Etapa 3): ele
    reestima o gap sobre o que já existir no frame com uma coluna
    `n_alternatives`, uma variante por valor distinto.
    """
    axis = "n_alternativas"
    notes: list[str] = []
    if not enabled:
        notes.append(f"Eixo desligado (D10). {D10_RATIONALE}")
        return SensitivityOutcome(axis, False, _empty_table(), notes)
    if "n_alternatives" not in frame.columns:
        notes.append(
            "Eixo habilitado, mas o frame de ensaios não tem a coluna "
            "`n_alternatives`: os itens com 3/5 alternativas precisam ser gerados e "
            "revisados na Etapa 3 antes de a Etapa 6 poder reestimar. "
            f"Escopo declarado: `{scope}`."
        )
        return SensitivityOutcome(axis, True, _empty_table(), notes)

    rows: list[dict[str, object]] = []
    notes.append(f"Eixo habilitado (D10), escopo `{scope}`.")
    for value in sorted(frame["n_alternatives"].astype(int).unique()):
        subset = frame[frame["n_alternatives"].astype(int) == value]
        rows.extend(
            _estimate_subset(
                subset,
                resamples,
                axis=axis,
                variant=f"{value} alternativas",
                covariates=covariates,
                ci_level=ci_level,
                age_min=age_min,
                age_max=age_max,
                max_failure_fraction=max_failure_fraction,
                notes=notes,
            )
        )
    table = pd.DataFrame(rows, columns=list(SENSITIVITY_COLUMNS))
    return SensitivityOutcome(axis, True, table, notes)


def combine(outcomes: Sequence[SensitivityOutcome]) -> pd.DataFrame:
    """Empilha as tabelas dos eixos numa só, em ordem canônica."""
    frames = [o.table for o in outcomes if not o.table.empty]
    if not frames:
        return _empty_table()
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["axis", "variant", "model_key", "condition"]).reset_index(
        drop=True
    )
