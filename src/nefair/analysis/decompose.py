"""Decomposição do erro da cascata: quanto é ASR e quanto é raciocínio (Etapa 6).

A cascata erra por duas razões distintas, e a política que cada uma sugere é
oposta. Se o erro vem do ASR, melhorar o reconhecimento resolve; se vem do
raciocínio sobre um texto já correto, melhorar o ASR não muda nada. A
decomposição separa as duas:

    Δ_ASR        = acc(reference) − acc(asr)
    Δ_raciocínio = teto − acc(reference)

e vale a identidade `acc(asr) + Δ_ASR + Δ_raciocínio = teto`, verificada em teste.

**Dois cuidados que este módulo carrega:**

1. **O teto é uma decisão, não um dado (D4).** Com `mode="constructed"` o teto é
   1,0 por construção: os itens foram revisados por humanos sobre a transcrição
   de referência, então "um leitor competente da referência acerta tudo" é uma
   suposição do desenho. Com `mode="human_sample"` o teto é uma acurácia humana
   medida numa amostra, e aí Δ_raciocínio passa a ser uma quantidade empírica.
   Os dois números NÃO são comparáveis entre si, e por isso o modo é **rotulado
   em toda linha de saída** — uma tabela sem esse rótulo convidaria a comparar
   decomposições de desenhos diferentes.

2. **O IC usa as MESMAS reamostras nas duas condições.** Δ_ASR é uma diferença
   entre duas medidas feitas nos MESMOS itens e nos MESMOS falantes. Reamostrar
   cada condição de forma independente destruiria esse pareamento e inflaria a
   variância da diferença (a covariância positiva entre as duas condições
   deixaria de ser descontada). Por isso a reamostragem acontece sobre um frame
   que contém as duas condições ao mesmo tempo: sorteado o falante, vêm juntas
   todas as respostas dele, nas duas condições.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nefair.analysis.regression import (
    REGION_BASELINE,
    REGION_EXPOSED,
    SpeakerResamples,
    SpeakerRowIndex,
    bootstrap_over_speakers,
)

CONDITION_ASR = "asr"
CONDITION_REFERENCE = "reference"

# Rótulo usado quando a decomposição não é restrita a uma região.
SCOPE_OVERALL = "geral"


@dataclass(frozen=True)
class CeilingSpec:
    """Teto da tarefa, com o modo que o produziu (D4).

    Espelha `TaskCeilingConfig`, mas como valor simples para que este módulo não
    dependa do carregador de YAML e possa ser exercitado com tetos arbitrários
    nos testes.
    """

    mode: str
    value: float

    def __post_init__(self) -> None:
        if self.mode not in ("constructed", "human_sample"):
            raise ValueError(
                f"Modo de teto (D4) inválido: '{self.mode}'. Use 'constructed' ou 'human_sample'."
            )
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Teto fora de [0, 1]: {self.value}.")


@dataclass(frozen=True)
class Decomposition:
    """Uma linha da decomposição, sempre rotulada com o modo do teto."""

    model_key: str
    scope: str
    ceiling_mode: str
    ceiling: float
    acc_asr: float
    acc_reference: float
    delta_asr: float
    delta_reasoning: float
    delta_asr_ci_low: float
    delta_asr_ci_high: float
    delta_reasoning_ci_low: float
    delta_reasoning_ci_high: float
    ci_level: float
    n_items_paired: int
    n_speakers: int
    n_resamples: int
    n_failed_resamples: int

    def to_row(self) -> dict[str, object]:
        return {
            "model_key": self.model_key,
            "scope": self.scope,
            "ceiling_mode": self.ceiling_mode,
            "ceiling": self.ceiling,
            "acc_asr": self.acc_asr,
            "acc_reference": self.acc_reference,
            "delta_asr": self.delta_asr,
            "delta_asr_ci_low": self.delta_asr_ci_low,
            "delta_asr_ci_high": self.delta_asr_ci_high,
            "delta_reasoning": self.delta_reasoning,
            "delta_reasoning_ci_low": self.delta_reasoning_ci_low,
            "delta_reasoning_ci_high": self.delta_reasoning_ci_high,
            "ci_level": self.ci_level,
            "n_items_paired": self.n_items_paired,
            "n_speakers": self.n_speakers,
            "n_resamples": self.n_resamples,
            "n_failed_resamples": self.n_failed_resamples,
        }


def paired_conditions_frame(frame: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Restringe ao par de condições `asr` × `reference` do mesmo modelo, pareado.

    Só sobrevivem os itens que têm resposta interpretável nas DUAS condições. Um
    item presente só em `reference` inflaria `acc(reference)` sem contrapartida em
    `acc(asr)`, e a diferença deixaria de ser Δ_ASR para virar uma mistura de
    efeito de condição com efeito de composição da amostra.
    """
    subset = frame[
        (frame["model_key"] == model_key)
        & (frame["condition"].isin([CONDITION_ASR, CONDITION_REFERENCE]))
    ]
    if subset.empty:
        raise ValueError(
            f"Modelo '{model_key}' não tem linhas nas condições "
            f"'{CONDITION_ASR}'/'{CONDITION_REFERENCE}'."
        )
    counts = subset.groupby("item_id")["condition"].nunique()
    paired_items = set(counts[counts == 2].index)
    if not paired_items:
        raise ValueError(
            f"Modelo '{model_key}': nenhum item com resposta nas duas condições. "
            "Sem pareamento não há decomposição."
        )
    out = subset[subset["item_id"].isin(paired_items)]
    return out.sort_values(["speaker_code", "item_id", "condition"]).reset_index(drop=True)


def _accuracy_by_condition(
    is_correct: np.ndarray, is_asr: np.ndarray, rows: np.ndarray
) -> tuple[float, float] | None:
    """Acurácia bruta em cada condição, sobre o mesmo conjunto de linhas."""
    selected_asr = is_asr[rows]
    n_asr = int(selected_asr.sum())
    n_ref = int(rows.size - n_asr)
    if n_asr == 0 or n_ref == 0:
        return None
    values = is_correct[rows]
    acc_asr = float(values[selected_asr].mean())
    acc_ref = float(values[~selected_asr].mean())
    return acc_asr, acc_ref


def decompose_model(
    frame: pd.DataFrame,
    resamples: SpeakerResamples,
    ceiling: CeilingSpec,
    *,
    model_key: str,
    scope: str = SCOPE_OVERALL,
    ci_level: float = 0.95,
    max_failure_fraction: float = 0.05,
) -> Decomposition:
    """Decompõe o erro de uma cascata, com IC pelas mesmas reamostras.

    A acurácia usada aqui é a **bruta** (proporção simples de acertos), não a
    ajustada pela regressão. A decomposição é uma contabilidade do erro dentro de
    uma mesma população de itens — as duas condições veem exatamente os mesmos
    itens e os mesmos falantes, então não há diferença de composição para
    ajustar. Ajustar por idade aqui responderia a outra pergunta.
    """
    paired = paired_conditions_frame(frame, model_key)
    if scope != SCOPE_OVERALL:
        paired = paired[paired["region"].astype(str) == scope].reset_index(drop=True)
        if paired.empty:
            raise ValueError(f"Modelo '{model_key}': nenhum item pareado na região '{scope}'.")

    is_correct = paired["is_correct"].to_numpy(dtype=float)
    is_asr = (paired["condition"].astype(str) == CONDITION_ASR).to_numpy()
    row_index = SpeakerRowIndex(resamples.speaker_codes, paired["speaker_code"])
    all_rows = np.arange(len(paired), dtype=np.int64)

    observed = _accuracy_by_condition(is_correct, is_asr, all_rows)
    if observed is None:
        raise ValueError(
            f"Modelo '{model_key}' ({scope}): falta uma das condições após o pareamento."
        )
    acc_asr, acc_reference = observed

    def _point_delta_asr() -> float | None:
        return acc_reference - acc_asr

    def _replicate_delta_asr(rows: np.ndarray) -> float | None:
        result = _accuracy_by_condition(is_correct, is_asr, rows)
        return None if result is None else result[1] - result[0]

    boot_asr = bootstrap_over_speakers(
        _replicate_delta_asr,
        _point_delta_asr,
        resamples,
        row_index,
        ci_level=ci_level,
        max_failure_fraction=max_failure_fraction,
    )

    def _point_delta_reasoning() -> float | None:
        return ceiling.value - acc_reference

    def _replicate_delta_reasoning(rows: np.ndarray) -> float | None:
        result = _accuracy_by_condition(is_correct, is_asr, rows)
        return None if result is None else ceiling.value - result[1]

    # MESMAS reamostras do Δ_ASR: as duas quantidades vêm da mesma reamostragem,
    # então somá-las continua devolvendo `teto − acc(asr)` réplica a réplica.
    boot_reasoning = bootstrap_over_speakers(
        _replicate_delta_reasoning,
        _point_delta_reasoning,
        resamples,
        row_index,
        ci_level=ci_level,
        max_failure_fraction=max_failure_fraction,
    )

    return Decomposition(
        model_key=model_key,
        scope=scope,
        ceiling_mode=ceiling.mode,
        ceiling=ceiling.value,
        acc_asr=acc_asr,
        acc_reference=acc_reference,
        delta_asr=boot_asr.point,
        delta_reasoning=boot_reasoning.point,
        delta_asr_ci_low=boot_asr.ci_low,
        delta_asr_ci_high=boot_asr.ci_high,
        delta_reasoning_ci_low=boot_reasoning.ci_low,
        delta_reasoning_ci_high=boot_reasoning.ci_high,
        ci_level=ci_level,
        n_items_paired=int(paired["item_id"].nunique()),
        n_speakers=int(paired["speaker_code"].nunique()),
        n_resamples=boot_asr.n_resamples,
        n_failed_resamples=boot_asr.n_failed + boot_reasoning.n_failed,
    )


def decomposition_table(
    frame: pd.DataFrame,
    resamples: SpeakerResamples,
    ceiling: CeilingSpec,
    *,
    ci_level: float = 0.95,
    scopes: Sequence[str] = (SCOPE_OVERALL, REGION_EXPOSED, REGION_BASELINE),
    max_failure_fraction: float = 0.05,
) -> tuple[pd.DataFrame, list[str]]:
    """Decomposição por modelo e por escopo (geral, NE, SE), em ordem canônica.

    O recorte por região é o que responde à pergunta central do TCC: se Δ_ASR for
    maior no NE do que no SE, parte do gap de acurácia é do reconhecimento, não
    da compreensão. Escopos não estimáveis viram avisos no relatório.
    """
    rows: list[dict[str, object]] = []
    warnings_out: list[str] = []
    cascade_models = sorted(
        frame.loc[frame["condition"].isin([CONDITION_ASR, CONDITION_REFERENCE]), "model_key"]
        .astype(str)
        .unique()
    )
    for model_key in cascade_models:
        for scope in scopes:
            try:
                decomposition = decompose_model(
                    frame,
                    resamples,
                    ceiling,
                    model_key=model_key,
                    scope=scope,
                    ci_level=ci_level,
                    max_failure_fraction=max_failure_fraction,
                )
            except ValueError as exc:
                warnings_out.append(f"{model_key} ({scope}): não decomponível — {exc}")
                continue
            rows.append(decomposition.to_row())
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["model_key", "scope"]).reset_index(drop=True)
    return table, warnings_out


def check_identity(decomposition: Decomposition, tolerance: float = 1e-9) -> None:
    """Verifica `acc(asr) + Δ_ASR + Δ_raciocínio == teto`.

    A identidade é o que garante que a decomposição é uma partição do erro e não
    duas quantidades soltas. Chamada no script da etapa: se ela quebrar, algum
    número da tabela está errado, e é melhor o script falhar do que o TCC
    publicar uma soma que não fecha.
    """
    total = decomposition.acc_asr + decomposition.delta_asr + decomposition.delta_reasoning
    if abs(total - decomposition.ceiling) > tolerance:
        raise ValueError(
            f"Decomposição não fecha para {decomposition.model_key} "
            f"({decomposition.scope}): acc_asr + Δ_ASR + Δ_raciocínio = {total}, "
            f"teto = {decomposition.ceiling}."
        )
