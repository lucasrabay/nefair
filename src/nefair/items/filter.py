"""Controle textual dos itens (Etapa 3, decisão D3).

A pergunta que este módulo responde é: *o item depende mesmo da fala?* O modelo
recebe **só** a pergunta e as alternativas — sem áudio, sem transcrição de
referência, sem hipótese de ASR. Item que continua sendo acertado nessas
condições não mede compreensão de fala; mede conhecimento geral, plausibilidade
ou pista de formato. Ele é descartado.

Três detalhes que fazem a diferença entre um filtro que funciona e um teatro:

1. **Temperatura > 0 e ordem embaralhada.** "Maioria de três execuções" com
   temperatura 0 e ordem fixa devolveria três respostas idênticas, e o critério
   de maioria seria vazio. A permutação muda a cada execução e o modelo é
   amostrado, então as três execuções são de fato três amostras.
2. **A permutação é derivada, nunca sorteada de um RNG global.** Ela deriva de
   `(item_id, model_key, run_index, seed)`. Reexecutar o filtro amanhã produz as
   mesmas permutações; rodar os testes em outra ordem não muda nada.
3. **A linha de base de acaso é calculada e reportada.** Um item que de fato
   exige áudio, respondido ao acaso com `p = 1/n_alternativas`, passa no critério
   "≥ k de n" com probabilidade `sum(C(n,i) p^i (1-p)^(n-i))` para `i` de `k` a
   `n`. Com n=3, k=2, p=0,25 isso dá ≈ 0,156: o filtro descarta ~16% dos itens
   **válidos** por puro acaso. Sem esse número ao lado, a taxa de descarte
   observada é ininterpretável — não dá para saber se 20% de descarte significa
   "quase nada vazou" ou "o filtro não está fazendo nada".

Resposta não interpretável nunca é imputada: ela conta como "não acertou" para
efeito do critério (o filtro é conservador: na dúvida, mantém o item), mas é
contada à parte e reportada.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from nefair.config import TextFilterConfig
from nefair.items.generate import derive_seed, with_filter_status
from nefair.models.base import TextQA
from nefair.models.prompts import labels_for, render_answer_prompt
from nefair.schema import ExclusionLedger, Item

# Único motivo de descarte desta etapa: o item foi respondido sem a fala.
DROP_ANSWERABLE_WITHOUT_SPEECH = "respondivel_sem_audio_nem_transcricao"

LEDGER_UNIT = "itens (controle textual)"

FILTER_STATUS_KEPT = "kept"
FILTER_STATUS_DISCARDED = "discarded"

# Estratégias de extração da letra, em cascata. É um parser MÍNIMO e local: a
# cascata completa (com o corpus de strings patológicas) é de `run/parse.py`,
# do Bloco R, que ainda não existe. Aqui basta o caso simples, porque o prompt
# pede explicitamente "apenas a letra"; o que não casar vira `None` e é CONTADO.
_RE_BARE_LABEL = re.compile(r"^\(?([A-Ea-e])\)?[).:\-\s]*$")
_RE_LEADING_LABEL = re.compile(r"^\(?([A-Ea-e])\)?[).:\-]\s")
_RE_ALTERNATIVA = re.compile(r"\balternativa\s+\(?([A-Ea-e])\)?\b", re.IGNORECASE)


def extract_label(raw_text: str, labels: Sequence[str]) -> str | None:
    """Letra escolhida, ou `None` se a resposta não for interpretável.

    `None` é uma categoria própria, jamais convertida em acerto ou erro. Ver a
    nota acima sobre a divisão de responsabilidade com `run/parse.py`.
    """
    text = (raw_text or "").strip()
    if not text:
        return None
    valid = {label.upper() for label in labels}
    for pattern in (_RE_BARE_LABEL, _RE_LEADING_LABEL, _RE_ALTERNATIVA):
        match = pattern.match(text) if pattern is not _RE_ALTERNATIVA else pattern.search(text)
        if match:
            candidate = match.group(1).upper()
            if candidate in valid:
                return candidate
    return None


# --------------------------------------------------------------------------- #
# Linha de base de acaso (CALCULADA, nunca hardcoded)
# --------------------------------------------------------------------------- #
def chance_pass_probability(*, n_runs: int, n_correct_at_least: int, n_alternatives: int) -> float:
    """P(acertar ≥ `n_correct_at_least` de `n_runs`) sob resposta aleatória.

    Cauda superior da binomial com `p = 1/n_alternatives`:
    `sum_{i=k}^{n} C(n,i) p^i (1-p)^(n-i)`.

    É a fração de itens **válidos** (que realmente exigem a fala) que o filtro
    descarta por azar. Fica ao lado da taxa de descarte observada em todo
    relatório — a taxa sozinha não é interpretável.
    """
    if n_alternatives < 2:
        raise ValueError(f"n_alternatives={n_alternatives} inválido para a linha de base.")
    if not 0 <= n_correct_at_least <= n_runs:
        raise ValueError(f"n_correct_at_least={n_correct_at_least} fora de [0, n_runs={n_runs}].")
    p = 1.0 / n_alternatives
    return sum(
        math.comb(n_runs, i) * (p**i) * ((1.0 - p) ** (n_runs - i))
        for i in range(n_correct_at_least, n_runs + 1)
    )


# --------------------------------------------------------------------------- #
# Permutação determinística das alternativas
# --------------------------------------------------------------------------- #
def derive_permutation(
    *, item_id: str, model_key: str, run_index: int, seed: int, n_alternatives: int
) -> tuple[int, ...]:
    """Permutação das alternativas para uma execução, derivada de forma estável.

    `permutation[i]` é o índice CANÔNICO da alternativa exibida na posição `i`.
    Depende de `(item_id, model_key, run_index, seed)` e de nada mais: dois
    modelos veem ordens diferentes do mesmo item (para que "qualquer modelo
    acertar" não seja um efeito de posição compartilhado), e a mesma execução
    reproduz a mesma ordem em qualquer máquina.
    """
    rng = np.random.default_rng(derive_seed(seed, item_id, model_key, run_index))
    return tuple(int(i) for i in rng.permutation(n_alternatives))


def apply_permutation(item: Item, permutation: Sequence[int]) -> tuple[list[str], str]:
    """Textos na ordem exibida + o rótulo da alternativa correta nessa ordem."""
    labels = labels_for(item.n_alternatives)
    texts = [item.alternatives[index].text for index in permutation]
    correct_index = next(i for i, alt in enumerate(item.alternatives) if alt.is_correct)
    shown_position = list(permutation).index(correct_index)
    return texts, labels[shown_position]


# --------------------------------------------------------------------------- #
# Execuções
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FilterRun:
    """Uma execução do controle textual sobre um item, por um modelo."""

    item_id: str
    speaker_code: str
    model_key: str
    run_index: int
    permutation: tuple[int, ...]
    shown_correct_label: str
    raw_text: str
    parsed_label: str | None
    is_correct: bool | None
    error: str = ""


@dataclass(frozen=True)
class ItemFilterDecision:
    """Veredito do filtro para um item, com o detalhamento por modelo."""

    item_id: str
    speaker_code: str
    region: str
    discarded: bool
    reason: str
    n_correct_by_model: dict[str, int]
    n_unparsed_by_model: dict[str, int]
    triggering_models: tuple[str, ...]


def run_filter_for_item(
    item: Item,
    models: Mapping[str, TextQA],
    config: TextFilterConfig,
    *,
    region: str = "desconhecida",
) -> tuple[ItemFilterDecision, tuple[FilterRun, ...]]:
    """Roda `n_runs` execuções por modelo e decide se o item cai.

    Os modelos são percorridos em ordem alfabética de chave para que a trilha de
    execuções seja byte a byte reprodutível.
    """
    runs: list[FilterRun] = []
    n_correct: dict[str, int] = {}
    n_unparsed: dict[str, int] = {}

    for model_key in sorted(models):
        model = models[model_key]
        correct = 0
        unparsed = 0
        for run_index in range(config.n_runs):
            permutation = (
                derive_permutation(
                    item_id=item.item_id,
                    model_key=model_key,
                    run_index=run_index,
                    seed=config.seed,
                    n_alternatives=item.n_alternatives,
                )
                if config.shuffle_alternatives
                else tuple(range(item.n_alternatives))
            )
            texts, shown_correct = apply_permutation(item, permutation)
            prompt = render_answer_prompt(
                question=item.question,
                alternative_texts=texts,
                transcript=None,
                audio_provided=False,
            )
            result = model.answer(prompt)
            parsed = extract_label(result.raw_text, labels_for(item.n_alternatives))
            is_correct = None if parsed is None else parsed == shown_correct
            if parsed is None:
                unparsed += 1
            elif is_correct:
                correct += 1
            runs.append(
                FilterRun(
                    item_id=item.item_id,
                    speaker_code=item.speaker_code,
                    model_key=model_key,
                    run_index=run_index,
                    permutation=permutation,
                    shown_correct_label=shown_correct,
                    raw_text=result.raw_text,
                    parsed_label=parsed,
                    is_correct=is_correct,
                    error=result.error,
                )
            )
        n_correct[model_key] = correct
        n_unparsed[model_key] = unparsed

    triggering = tuple(
        key for key in sorted(n_correct) if n_correct[key] >= config.discard_if_correct_at_least
    )
    # `any_model_discards` (D3): basta UM modelo responder sem a fala para que o
    # item seja suspeito. A alternativa (exigir que todos acertem) deixaria
    # passar itens vazáveis para uma família só — e a comparação entre famílias
    # é justamente o experimento.
    discarded = bool(triggering) if config.any_model_discards else len(triggering) == len(models)

    reason = ""
    if discarded:
        reason = (
            f"{DROP_ANSWERABLE_WITHOUT_SPEECH}: "
            f"{', '.join(f'{k}={n_correct[k]}/{config.n_runs}' for k in triggering)}"
        )

    decision = ItemFilterDecision(
        item_id=item.item_id,
        speaker_code=item.speaker_code,
        region=region,
        discarded=discarded,
        reason=reason,
        n_correct_by_model=n_correct,
        n_unparsed_by_model=n_unparsed,
        triggering_models=triggering,
    )
    return decision, tuple(runs)


# --------------------------------------------------------------------------- #
# Agregados do relatório
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegionDiscardRate:
    """Taxa de descarte de uma região — o número que o gate do piloto olha."""

    region: str
    n_items: int
    n_discarded: int

    @property
    def discard_rate(self) -> float:
        return self.n_discarded / self.n_items if self.n_items else 0.0


@dataclass(frozen=True)
class ModelDiscardRate:
    """Quantos itens cada modelo sozinho derrubaria (diagnóstico do D3)."""

    model_key: str
    n_items: int
    n_triggering: int

    @property
    def trigger_rate(self) -> float:
        return self.n_triggering / self.n_items if self.n_items else 0.0


@dataclass(frozen=True)
class FilterSummary:
    """Números do controle textual, prontos para o relatório."""

    n_items_in: int
    n_kept: int
    n_discarded: int
    n_runs: int
    discard_if_correct_at_least: int
    any_model_discards: bool
    shuffle_alternatives: bool
    temperature: float
    n_alternatives: int
    seed: int
    models: tuple[str, ...]
    chance_discard_rate: float
    n_total_runs: int
    n_unparsed_runs: int
    by_region: tuple[RegionDiscardRate, ...]
    by_model: tuple[ModelDiscardRate, ...]

    @property
    def discard_rate(self) -> float:
        return self.n_discarded / self.n_items_in if self.n_items_in else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items_in": self.n_items_in,
            "n_kept": self.n_kept,
            "n_discarded": self.n_discarded,
            "n_runs": self.n_runs,
            "discard_if_correct_at_least": self.discard_if_correct_at_least,
            "any_model_discards": self.any_model_discards,
            "shuffle_alternatives": self.shuffle_alternatives,
            "temperature": self.temperature,
            "n_alternatives": self.n_alternatives,
            "seed": self.seed,
            "models": list(self.models),
            "chance_discard_rate": self.chance_discard_rate,
            "n_total_runs": self.n_total_runs,
            "n_unparsed_runs": self.n_unparsed_runs,
            "by_region": [
                {"region": r.region, "n_items": r.n_items, "n_discarded": r.n_discarded}
                for r in self.by_region
            ],
            "by_model": [
                {"model_key": m.model_key, "n_items": m.n_items, "n_triggering": m.n_triggering}
                for m in self.by_model
            ],
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> FilterSummary:
        return FilterSummary(
            n_items_in=int(payload["n_items_in"]),
            n_kept=int(payload["n_kept"]),
            n_discarded=int(payload["n_discarded"]),
            n_runs=int(payload["n_runs"]),
            discard_if_correct_at_least=int(payload["discard_if_correct_at_least"]),
            any_model_discards=bool(payload["any_model_discards"]),
            shuffle_alternatives=bool(payload["shuffle_alternatives"]),
            temperature=float(payload["temperature"]),
            n_alternatives=int(payload["n_alternatives"]),
            seed=int(payload["seed"]),
            models=tuple(str(m) for m in payload["models"]),
            chance_discard_rate=float(payload["chance_discard_rate"]),
            n_total_runs=int(payload["n_total_runs"]),
            n_unparsed_runs=int(payload["n_unparsed_runs"]),
            by_region=tuple(
                RegionDiscardRate(
                    region=str(r["region"]),
                    n_items=int(r["n_items"]),
                    n_discarded=int(r["n_discarded"]),
                )
                for r in payload["by_region"]
            ),
            by_model=tuple(
                ModelDiscardRate(
                    model_key=str(m["model_key"]),
                    n_items=int(m["n_items"]),
                    n_triggering=int(m["n_triggering"]),
                )
                for m in payload["by_model"]
            ),
        )


@dataclass(frozen=True)
class FilterOutcome:
    """Saída do controle textual: itens marcados, trilha e agregados."""

    items: tuple[Item, ...]
    decisions: tuple[ItemFilterDecision, ...]
    runs: tuple[FilterRun, ...]
    ledger: ExclusionLedger
    summary: FilterSummary

    @property
    def kept_items(self) -> tuple[Item, ...]:
        return tuple(it for it in self.items if it.filter_status == FILTER_STATUS_KEPT)


def apply_text_filter(
    items: Sequence[Item],
    models: Mapping[str, TextQA],
    config: TextFilterConfig,
    *,
    region_by_speaker: Mapping[str, str] | None = None,
) -> FilterOutcome:
    """Aplica o controle textual a todos os itens e devolve a trilha completa.

    Os itens voltam TODOS, com `filter_status` preenchido — os descartados
    inclusive. Guardar o descartado (em vez de sumir com ele) é o que permite
    auditar depois se o filtro derrubou NE e SE em proporções diferentes, que é
    exatamente uma das condições do gate do piloto.
    """
    regions = dict(region_by_speaker or {})
    ordered = sorted(items, key=lambda it: it.item_id)
    ledger = ExclusionLedger(unit=LEDGER_UNIT, total_in=len(ordered))

    marked: list[Item] = []
    decisions: list[ItemFilterDecision] = []
    all_runs: list[FilterRun] = []
    region_totals: dict[str, list[int]] = {}
    model_triggers: dict[str, int] = {key: 0 for key in models}

    for item in ordered:
        region = regions.get(item.speaker_code, "desconhecida")
        decision, runs = run_filter_for_item(item, models, config, region=region)
        decisions.append(decision)
        all_runs.extend(runs)
        for key in decision.triggering_models:
            model_triggers[key] = model_triggers.get(key, 0) + 1

        bucket = region_totals.setdefault(region, [0, 0])
        bucket[0] += 1
        if decision.discarded:
            bucket[1] += 1
            ledger.drop(DROP_ANSWERABLE_WITHOUT_SPEECH)
            marked.append(
                with_filter_status(item, status=FILTER_STATUS_DISCARDED, reason=decision.reason)
            )
        else:
            marked.append(with_filter_status(item, status=FILTER_STATUS_KEPT))

    ledger.kept = sum(1 for it in marked if it.filter_status == FILTER_STATUS_KEPT)
    ledger.assert_balanced()

    n_alternatives = ordered[0].n_alternatives if ordered else 0
    chance = (
        chance_pass_probability(
            n_runs=config.n_runs,
            n_correct_at_least=config.discard_if_correct_at_least,
            n_alternatives=n_alternatives,
        )
        if n_alternatives >= 2
        else 0.0
    )

    summary = FilterSummary(
        n_items_in=len(ordered),
        n_kept=ledger.kept,
        n_discarded=ledger.total_dropped,
        n_runs=config.n_runs,
        discard_if_correct_at_least=config.discard_if_correct_at_least,
        any_model_discards=config.any_model_discards,
        shuffle_alternatives=config.shuffle_alternatives,
        temperature=config.temperature,
        n_alternatives=n_alternatives,
        seed=config.seed,
        models=tuple(sorted(models)),
        chance_discard_rate=chance,
        n_total_runs=len(all_runs),
        n_unparsed_runs=sum(1 for r in all_runs if r.parsed_label is None),
        by_region=tuple(
            RegionDiscardRate(region=region, n_items=counts[0], n_discarded=counts[1])
            for region, counts in sorted(region_totals.items())
        ),
        by_model=tuple(
            ModelDiscardRate(
                model_key=key, n_items=len(ordered), n_triggering=model_triggers.get(key, 0)
            )
            for key in sorted(models)
        ),
    )
    return FilterOutcome(
        items=tuple(marked),
        decisions=tuple(decisions),
        runs=tuple(all_runs),
        ledger=ledger,
        summary=summary,
    )
