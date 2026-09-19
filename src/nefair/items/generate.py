"""Geração dos itens de múltipla escolha (Etapa 3, decisões D2 e D11).

O que este módulo faz, em uma frase: pega uma janela-alvo, entrega ao LLM gerador
a transcrição dela **mais** trechos de outras janelas da mesma entrevista, e
transforma a resposta num `Item` — ou numa linha contada na trilha de exclusão.

Três compromissos de projeto governam o código:

1. **Proveniência do distrator é verificada, não prometida** (D2). O texto do TCC
   afirma que os distratores são "extraídos de outros trechos da mesma
   entrevista". Sem `Alternative.source_window_id` apontando para uma janela que
   de fato entrou no prompt, essa afirmação é não verificável. Com
   `require_provenance: true`, distrator sem proveniência — ou com proveniência
   inventada — torna o item malformado.
2. **Nada é reparado em silêncio.** Não há tentativa de "consertar" uma resposta
   do gerador (completar uma alternativa que faltou, desempatar duas corretas).
   Cada motivo de rejeição é uma categoria própria da `ExclusionLedger`, e o
   relatório mostra a distribuição dos motivos — que é, ela mesma, um dado sobre
   a qualidade do gerador.
3. **O gerador chega por injeção de dependência** (D11). Este módulo conhece
   apenas o `Protocol` `TextQA`; quem constrói o cliente é o script. Isso é o que
   permite rodar a etapa inteira hoje, sem chave de API, com `FakeTextQA`, e é o
   que impede que um SDK de provedor seja importado aqui.

Chave da entrevista: usamos `speaker_code`. A Etapa 0 mostrou que, para
informantes, `speaker_code` distingue exatamente os mesmos falantes que
`speaker_code + audio_name` — ou seja, um informante corresponde a uma entrevista,
mesmo quando ela foi gravada em mais de um arquivo de áudio.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from nefair.config import ItemsConfig, PilotConfig
from nefair.models.base import TextQA
from nefair.models.prompts import (
    GENERATION_RESPONSE_ROOT_KEY,
    PROMPT_VERSION,
    labels_for,
    render_generation_prompt,
)
from nefair.schema import Alternative, ExclusionLedger, Item, Window, stable_hash

# --------------------------------------------------------------------------- #
# Categorias da trilha de exclusão.
#
# Cada motivo é uma constante, e não uma string solta, por dois motivos: o teste
# afirma a categoria exata (e não "caiu em alguma"), e o relatório lista os
# motivos em ordem estável. Um motivo novo exige uma constante nova — o que
# obriga a pensar se ele é mesmo distinto dos que já existem.
# --------------------------------------------------------------------------- #
DROP_MODEL_ERROR = "erro_do_modelo"
DROP_EMPTY_RESPONSE = "resposta_vazia"
DROP_NOT_JSON = "resposta_nao_e_json"
DROP_BAD_ENVELOPE = "envelope_json_invalido"
DROP_MISSING_ITEM = "item_faltando_na_resposta"
DROP_EMPTY_QUESTION = "pergunta_vazia"
DROP_WRONG_N_ALTERNATIVES = "n_alternativas_incorreto"
DROP_NOT_EXACTLY_ONE_CORRECT = "corretas_diferente_de_uma"
DROP_EMPTY_ALTERNATIVE = "alternativa_vazia"
DROP_DUPLICATE_ALTERNATIVE = "alternativas_duplicadas"
DROP_MISSING_PROVENANCE = "distrator_sem_proveniencia"
DROP_UNKNOWN_PROVENANCE = "proveniencia_desconhecida"
DROP_NO_CONTEXT_WINDOWS = "sem_janelas_de_contexto"

# Ordem canônica dos motivos no relatório (estável, independente de dicionário).
DROP_REASONS: tuple[str, ...] = (
    DROP_MODEL_ERROR,
    DROP_EMPTY_RESPONSE,
    DROP_NOT_JSON,
    DROP_BAD_ENVELOPE,
    DROP_MISSING_ITEM,
    DROP_EMPTY_QUESTION,
    DROP_WRONG_N_ALTERNATIVES,
    DROP_NOT_EXACTLY_ONE_CORRECT,
    DROP_EMPTY_ALTERNATIVE,
    DROP_DUPLICATE_ALTERNATIVE,
    DROP_MISSING_PROVENANCE,
    DROP_UNKNOWN_PROVENANCE,
    DROP_NO_CONTEXT_WINDOWS,
)

LEDGER_UNIT = "itens (tentativas de geração)"


# --------------------------------------------------------------------------- #
# Semente derivada (usada aqui e no filtro textual)
# --------------------------------------------------------------------------- #
def derive_seed(*parts: object) -> int:
    """Semente inteira derivada deterministicamente de `parts`.

    Toda aleatoriedade do bloco de itens passa por aqui. Nunca usamos o RNG
    global de `random` nem `np.random.seed`: a permutação de um item precisa
    depender só da identidade do item, do modelo e da execução — e não da ordem
    em que os itens foram processados, nem de quantos testes rodaram antes.
    """
    return int(stable_hash([str(part) for part in parts])[:16], 16)


# --------------------------------------------------------------------------- #
# Seleção dos trechos de contexto (D2)
# --------------------------------------------------------------------------- #
def select_context_windows(
    target: Window,
    interview_windows: Sequence[Window],
    *,
    n_context_windows: int,
    seed: int,
) -> tuple[Window, ...]:
    """Escolhe até `n_context_windows` OUTRAS janelas da mesma entrevista.

    A escolha é aleatória (com semente derivada da janela-alvo), e não "as `n`
    primeiras", porque pegar sempre as primeiras ancoraria todos os distratores
    no começo da entrevista: o conteúdo dos distratores passaria a correlacionar
    com a posição na entrevista, e a dificuldade do item deixaria de ser
    comparável entre janelas.

    O retorno vem ordenado por `window_id` para que o prompt seja byte a byte
    reprodutível independentemente da ordem sorteada.
    """
    pool = [w for w in interview_windows if w.window_id != target.window_id]
    pool.sort(key=lambda w: w.window_id)
    if not pool or n_context_windows <= 0:
        return ()
    take = min(n_context_windows, len(pool))
    rng = np.random.default_rng(derive_seed(seed, "context", target.window_id))
    chosen_idx = rng.choice(len(pool), size=take, replace=False)
    chosen = [pool[int(i)] for i in chosen_idx]
    chosen.sort(key=lambda w: w.window_id)
    return tuple(chosen)


def group_windows_by_interview(windows: Iterable[Window]) -> dict[str, list[Window]]:
    """Agrupa janelas por entrevista (`speaker_code`), em ordem estável."""
    grouped: dict[str, list[Window]] = {}
    for window in windows:
        grouped.setdefault(window.speaker_code, []).append(window)
    for bucket in grouped.values():
        bucket.sort(key=lambda w: w.window_id)
    return grouped


# --------------------------------------------------------------------------- #
# Piloto (§Etapa 3: "piloto primeiro")
# --------------------------------------------------------------------------- #
def select_pilot_speakers(
    windows: Sequence[Window], pilot: PilotConfig
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Sorteia `n_speakers_per_region` falantes por região, dentro da faixa etária.

    Devolve `(falantes, disponíveis_por_região)`. O segundo item existe porque um
    piloto com menos falantes do que o pedido numa das regiões não é um detalhe:
    o gate compara taxas NE × SE, e comparar 6 contra 3 é outra coisa.

    O sorteio é semeado por `(pilot.seed, região)` e feito sobre a lista ordenada
    de falantes elegíveis, então é reprodutível e independente da ordem em que as
    janelas chegaram.
    """
    by_region: dict[str, set[str]] = {}
    for window in windows:
        if window.region not in pilot.regions:
            continue
        if not pilot.age_min <= window.age <= pilot.age_max:
            continue
        by_region.setdefault(window.region, set()).add(window.speaker_code)

    available = {region: len(codes) for region, codes in sorted(by_region.items())}
    selected: list[str] = []
    for region in pilot.regions:
        codes = sorted(by_region.get(region, set()))
        if not codes:
            continue
        take = min(pilot.n_speakers_per_region, len(codes))
        rng = np.random.default_rng(derive_seed(pilot.seed, "pilot", region))
        chosen = rng.choice(len(codes), size=take, replace=False)
        selected.extend(codes[int(i)] for i in chosen)
    return tuple(sorted(selected)), available


# --------------------------------------------------------------------------- #
# Validação da resposta do gerador
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParsedAlternative:
    """Alternativa como o gerador a devolveu, antes de virar `Alternative`."""

    text: str
    is_correct: bool
    source_window_id: str | None


@dataclass(frozen=True)
class ParsedItem:
    """Item como o gerador o devolveu, já validado contra o schema estrito."""

    question: str
    alternatives: tuple[ParsedAlternative, ...]


def parse_generation_envelope(raw_text: str) -> tuple[list[Any] | None, str | None]:
    """Extrai a lista de itens da resposta bruta do gerador.

    Devolve `(itens, None)` ou `(None, motivo)`. Nenhuma tentativa de reparo:
    um JSON quebrado é um JSON quebrado, e o número de vezes que isso acontece é
    um dado sobre o gerador que o relatório precisa mostrar.

    A única tolerância é a cerca de código (```json ... ```), que vários modelos
    adicionam mesmo quando instruídos a não adicionar. Removê-la não muda o
    conteúdo; não removê-la transformaria um item válido numa exclusão.
    """
    text = (raw_text or "").strip()
    if not text:
        return None, DROP_EMPTY_RESPONSE
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return None, DROP_NOT_JSON
    if not isinstance(payload, dict):
        return None, DROP_BAD_ENVELOPE
    items = payload.get(GENERATION_RESPONSE_ROOT_KEY)
    if not isinstance(items, list):
        return None, DROP_BAD_ENVELOPE
    return items, None


def validate_item_payload(
    payload: Any,
    *,
    n_alternatives: int,
    target_window_id: str,
    allowed_source_ids: frozenset[str],
    require_provenance: bool,
) -> tuple[ParsedItem | None, str | None]:
    """Valida UM item contra o schema estrito. Devolve `(item, None)` ou `(None, motivo)`.

    A ordem das checagens é deliberada: o motivo reportado é o primeiro problema
    encontrado, e queremos que problemas estruturais (nº de alternativas, nº de
    corretas) apareçam antes de problemas de conteúdo. Assim a distribuição de
    motivos no relatório diz "o gerador erra a estrutura" ou "o gerador erra a
    proveniência", e não uma mistura dos dois.
    """
    if not isinstance(payload, dict):
        return None, DROP_BAD_ENVELOPE

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return None, DROP_EMPTY_QUESTION

    raw_alternatives = payload.get("alternatives")
    if not isinstance(raw_alternatives, list) or len(raw_alternatives) != n_alternatives:
        return None, DROP_WRONG_N_ALTERNATIVES

    parsed: list[ParsedAlternative] = []
    for raw in raw_alternatives:
        if not isinstance(raw, dict):
            return None, DROP_BAD_ENVELOPE
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            return None, DROP_EMPTY_ALTERNATIVE
        source = raw.get("source_window_id")
        if source is not None and not isinstance(source, str):
            return None, DROP_BAD_ENVELOPE
        parsed.append(
            ParsedAlternative(
                text=text.strip(),
                is_correct=bool(raw.get("is_correct", False)),
                source_window_id=(source.strip() or None) if isinstance(source, str) else None,
            )
        )

    n_correct = sum(1 for alt in parsed if alt.is_correct)
    if n_correct != 1:
        return None, DROP_NOT_EXACTLY_ONE_CORRECT

    # Duplicata é comparada sem caixa e sem espaço nas bordas: "São Paulo" e
    # "são paulo " são a mesma alternativa para quem responde.
    normalized = [alt.text.strip().casefold() for alt in parsed]
    if len(set(normalized)) != len(normalized):
        return None, DROP_DUPLICATE_ALTERNATIVE

    if require_provenance:
        for alt in parsed:
            if alt.source_window_id is None:
                return None, DROP_MISSING_PROVENANCE
            expected = {target_window_id} if alt.is_correct else allowed_source_ids
            if alt.source_window_id not in expected:
                # Proveniência inventada é pior do que proveniência ausente: o
                # item pareceria auditável e não seria. Categoria própria.
                return None, DROP_UNKNOWN_PROVENANCE

    return ParsedItem(question=question.strip(), alternatives=tuple(parsed)), None


def build_item(
    parsed: ParsedItem,
    *,
    window: Window,
    index: int,
    generator_model: str,
) -> Item:
    """Converte um `ParsedItem` validado em `Item`, na ordem canônica.

    Os rótulos A, B, C… são atribuídos na ordem em que o gerador devolveu as
    alternativas. Essa é a ordem CANÔNICA do item: toda reordenação posterior
    (filtro textual, execução) é uma permutação registrada à parte, e nunca
    sobrescreve esta.
    """
    labels = labels_for(len(parsed.alternatives))
    alternatives = tuple(
        Alternative(
            label=label,
            text=alt.text,
            is_correct=alt.is_correct,
            source_window_id=alt.source_window_id,
        )
        for label, alt in zip(labels, parsed.alternatives, strict=True)
    )
    return Item(
        item_id=Item.make_id(window.window_id, index),
        window_id=window.window_id,
        speaker_code=window.speaker_code,
        question=parsed.question,
        alternatives=alternatives,
        generator_model=generator_model,
        prompt_version=PROMPT_VERSION,
    )


def item_from_dict(payload: Mapping[str, Any]) -> Item:
    """Reconstrói um `Item` a partir do dicionário lido de um JSONL.

    Contraparte de `schema.write_jsonl`: as etapas 3→4→5 trocam itens por
    arquivo, e cada script precisa recuperar a dataclass sem adivinhar campos.
    """
    alternatives = tuple(
        Alternative(
            label=str(alt["label"]),
            text=str(alt["text"]),
            is_correct=bool(alt["is_correct"]),
            source_window_id=(
                None if alt.get("source_window_id") is None else str(alt["source_window_id"])
            ),
        )
        for alt in payload["alternatives"]
    )
    minutes = payload.get("review_minutes")
    return Item(
        item_id=str(payload["item_id"]),
        window_id=str(payload["window_id"]),
        speaker_code=str(payload["speaker_code"]),
        question=str(payload["question"]),
        alternatives=alternatives,
        generator_model=str(payload["generator_model"]),
        prompt_version=str(payload["prompt_version"]),
        filter_status=str(payload.get("filter_status", "pending")),
        filter_reason=str(payload.get("filter_reason", "")),
        review_status=str(payload.get("review_status", "pending")),
        review_reason=str(payload.get("review_reason", "")),
        review_minutes=None if minutes is None else float(minutes),
    )


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GenerationAttempt:
    """Trilha de uma chamada ao gerador, para auditoria da Etapa 3.

    Guardamos o prompt e a resposta bruta porque, quando o gerador começa a
    produzir itens malformados, a única forma de saber se o problema é o prompt
    ou o modelo é olhar as duas coisas lado a lado.
    """

    window_id: str
    speaker_code: str
    prompt: str
    raw_text: str
    n_requested: int
    n_accepted: int
    drop_reasons: tuple[str, ...]
    context_window_ids: tuple[str, ...]
    error: str = ""


@dataclass(frozen=True)
class RegionCount:
    """Contagem por região, em ordem estável (o relatório reporta por região)."""

    region: str
    n_items: int


@dataclass(frozen=True)
class GenerationSummary:
    """Números da geração, prontos para o relatório e para o JSON de estágio."""

    generator_model_id: str
    generator_version: str
    prompt_version: str
    n_speakers: int
    n_windows: int
    n_requested: int
    n_items: int
    n_windows_without_context: int
    n_windows_with_partial_context: int
    dropped: dict[str, int]
    by_region: tuple[RegionCount, ...]
    distractors_source: str
    require_provenance: bool
    n_context_windows: int
    n_alternatives: int
    n_items_per_window: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_model_id": self.generator_model_id,
            "generator_version": self.generator_version,
            "prompt_version": self.prompt_version,
            "n_speakers": self.n_speakers,
            "n_windows": self.n_windows,
            "n_requested": self.n_requested,
            "n_items": self.n_items,
            "n_windows_without_context": self.n_windows_without_context,
            "n_windows_with_partial_context": self.n_windows_with_partial_context,
            "dropped": dict(sorted(self.dropped.items())),
            "by_region": [{"region": r.region, "n_items": r.n_items} for r in self.by_region],
            "distractors_source": self.distractors_source,
            "require_provenance": self.require_provenance,
            "n_context_windows": self.n_context_windows,
            "n_alternatives": self.n_alternatives,
            "n_items_per_window": self.n_items_per_window,
            "seed": self.seed,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> GenerationSummary:
        return GenerationSummary(
            generator_model_id=str(payload["generator_model_id"]),
            generator_version=str(payload["generator_version"]),
            prompt_version=str(payload["prompt_version"]),
            n_speakers=int(payload["n_speakers"]),
            n_windows=int(payload["n_windows"]),
            n_requested=int(payload["n_requested"]),
            n_items=int(payload["n_items"]),
            n_windows_without_context=int(payload["n_windows_without_context"]),
            n_windows_with_partial_context=int(payload["n_windows_with_partial_context"]),
            dropped={str(k): int(v) for k, v in payload["dropped"].items()},
            by_region=tuple(
                RegionCount(region=str(r["region"]), n_items=int(r["n_items"]))
                for r in payload["by_region"]
            ),
            distractors_source=str(payload["distractors_source"]),
            require_provenance=bool(payload["require_provenance"]),
            n_context_windows=int(payload["n_context_windows"]),
            n_alternatives=int(payload["n_alternatives"]),
            n_items_per_window=int(payload["n_items_per_window"]),
            seed=int(payload["seed"]),
        )


@dataclass(frozen=True)
class GenerationOutcome:
    """Saída completa da geração: itens, trilha de exclusão e trilha de chamadas."""

    items: tuple[Item, ...]
    ledger: ExclusionLedger
    attempts: tuple[GenerationAttempt, ...]
    summary: GenerationSummary


def generate_items(
    windows: Sequence[Window],
    *,
    model: TextQA,
    config: ItemsConfig,
    region_by_speaker: Mapping[str, str] | None = None,
) -> GenerationOutcome:
    """Gera `n_items_per_window` itens por janela, com contexto da mesma entrevista.

    `model` é qualquer coisa que satisfaça `TextQA` — o `FakeTextQA` nos testes,
    um adaptador de provedor quando houver credencial. Este módulo nunca
    instancia um cliente: é isso que mantém a Etapa 3 executável offline.

    A trilha de exclusão fecha por construção: `total_in` é o nº de itens
    PEDIDOS (janelas × `n_items_per_window`), e cada pedido termina em item
    retido ou em uma categoria de descarte.
    """
    ordered = sorted(windows, key=lambda w: w.window_id)
    by_interview = group_windows_by_interview(ordered)
    regions = dict(region_by_speaker or {})

    n_per_window = config.n_items_per_window
    ledger = ExclusionLedger(unit=LEDGER_UNIT, total_in=len(ordered) * n_per_window)

    items: list[Item] = []
    attempts: list[GenerationAttempt] = []
    n_without_context = 0
    n_partial_context = 0

    for window in ordered:
        interview = by_interview.get(window.speaker_code, [window])
        context = select_context_windows(
            window,
            interview,
            n_context_windows=config.distractors.n_context_windows,
            seed=config.seed,
        )
        if not context:
            n_without_context += 1
        elif len(context) < config.distractors.n_context_windows:
            n_partial_context += 1

        # Sem nenhuma outra janela da entrevista não há de onde extrair distrator
        # ancorado (D2). Descartar aqui é mais honesto do que pedir ao gerador um
        # distrator que ele inventaria e declararia como proveniente da própria
        # janela-alvo.
        if not context and config.distractors.require_provenance:
            ledger.drop(DROP_NO_CONTEXT_WINDOWS, n_per_window)
            attempts.append(
                GenerationAttempt(
                    window_id=window.window_id,
                    speaker_code=window.speaker_code,
                    prompt="",
                    raw_text="",
                    n_requested=n_per_window,
                    n_accepted=0,
                    drop_reasons=(DROP_NO_CONTEXT_WINDOWS,) * n_per_window,
                    context_window_ids=(),
                )
            )
            continue

        prompt = render_generation_prompt(
            target_window_id=window.window_id,
            target_transcript=window.reference_text,
            context_excerpts=tuple((w.window_id, w.reference_text) for w in context),
            n_alternatives=config.n_alternatives,
            n_items=n_per_window,
            require_provenance=config.distractors.require_provenance,
        )
        result = model.answer(prompt)

        reasons: list[str] = []
        accepted = 0
        if result.error:
            reasons = [DROP_MODEL_ERROR] * n_per_window
        else:
            payloads, envelope_reason = parse_generation_envelope(result.raw_text)
            if envelope_reason is not None:
                reasons = [envelope_reason] * n_per_window
            else:
                allowed = frozenset(w.window_id for w in context)
                assert payloads is not None  # garantido por envelope_reason is None
                for slot in range(n_per_window):
                    if slot >= len(payloads):
                        reasons.append(DROP_MISSING_ITEM)
                        continue
                    parsed, reason = validate_item_payload(
                        payloads[slot],
                        n_alternatives=config.n_alternatives,
                        target_window_id=window.window_id,
                        allowed_source_ids=allowed,
                        require_provenance=config.distractors.require_provenance,
                    )
                    if reason is not None or parsed is None:
                        reasons.append(reason or DROP_BAD_ENVELOPE)
                        continue
                    items.append(
                        build_item(
                            parsed,
                            window=window,
                            index=slot + 1,
                            generator_model=model.spec.model_id,
                        )
                    )
                    accepted += 1

        for reason in reasons:
            ledger.drop(reason)
        attempts.append(
            GenerationAttempt(
                window_id=window.window_id,
                speaker_code=window.speaker_code,
                prompt=prompt,
                raw_text=result.raw_text,
                n_requested=n_per_window,
                n_accepted=accepted,
                drop_reasons=tuple(reasons),
                context_window_ids=tuple(w.window_id for w in context),
                error=result.error,
            )
        )

    items.sort(key=lambda it: it.item_id)
    ledger.kept = len(items)
    ledger.assert_balanced()

    by_region: dict[str, int] = {}
    for item in items:
        region = regions.get(item.speaker_code, "desconhecida")
        by_region[region] = by_region.get(region, 0) + 1

    summary = GenerationSummary(
        generator_model_id=model.spec.model_id,
        generator_version=model.spec.version,
        prompt_version=PROMPT_VERSION,
        n_speakers=len(by_interview),
        n_windows=len(ordered),
        n_requested=ledger.total_in,
        n_items=len(items),
        n_windows_without_context=n_without_context,
        n_windows_with_partial_context=n_partial_context,
        dropped=dict(ledger.dropped),
        by_region=tuple(RegionCount(region=r, n_items=n) for r, n in sorted(by_region.items())),
        distractors_source=config.distractors.source,
        require_provenance=config.distractors.require_provenance,
        n_context_windows=config.distractors.n_context_windows,
        n_alternatives=config.n_alternatives,
        n_items_per_window=n_per_window,
        seed=config.seed,
    )
    return GenerationOutcome(
        items=tuple(items),
        ledger=ledger,
        attempts=tuple(attempts),
        summary=summary,
    )


def with_filter_status(item: Item, *, status: str, reason: str = "") -> Item:
    """Devolve uma cópia do item com a trilha do filtro preenchida.

    `Item` é frozen de propósito: a trilha de vida do item (`filter_status`,
    `review_status`) só muda por substituição explícita, nunca por mutação em
    algum ponto distante do pipeline.
    """
    return replace(item, filter_status=status, filter_reason=reason)
