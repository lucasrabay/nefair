"""Testes do bloco de itens (Etapa 3): geração, controle textual e revisão.

Três coisas precisam ser verdade para que a Etapa 3 signifique o que o texto do
TCC diz que ela significa, e cada uma delas tem seu grupo de testes aqui:

1. **Nenhum item malformado vira item, e nenhum descarte é silencioso** (D2).
   Cada resposta ruim do gerador cai numa categoria PRÓPRIA da trilha de
   exclusão — não numa categoria genérica. A distribuição dos motivos é, ela
   mesma, um resultado sobre a qualidade do gerador: "o modelo erra a estrutura"
   e "o modelo inventa proveniência" são diagnósticos diferentes e exigem
   respostas diferentes. Por isso os testes afirmam a chave literal, e não
   apenas "algum descarte aconteceu".
2. **O controle textual mede o que promete** (D3). O número que torna a taxa de
   descarte interpretável é a linha de base de acaso: com 3 execuções, critério
   2-de-3 e 4 alternativas, ~15,6% dos itens VÁLIDOS caem por puro azar. Sem
   esse número, "descartamos 20%" não distingue "quase nada vazou" de "o filtro
   não fez nada".
3. **A planilha de revisão é reprodutível e a importação é intolerante.** Item
   não revisado que passe em silêncio entra na Etapa 5 sem validação humana, e
   aí o texto afirma algo que não aconteceu.

Tudo offline: o gerador e os modelos do filtro são `FakeTextQA` com `responder`
encenando a resposta exata que o teste quer. A config é a do repositório
(`configs/items.yaml`) — o que se testa aqui é o que vai rodar.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import re
from pathlib import Path

import pytest

from nefair.config import load_items_config
from nefair.items.filter import (
    DROP_ANSWERABLE_WITHOUT_SPEECH,
    FILTER_STATUS_DISCARDED,
    FILTER_STATUS_KEPT,
    apply_text_filter,
    chance_pass_probability,
    derive_permutation,
    extract_label,
    run_filter_for_item,
)
from nefair.items.generate import (
    DROP_BAD_ENVELOPE,
    DROP_DUPLICATE_ALTERNATIVE,
    DROP_EMPTY_ALTERNATIVE,
    DROP_EMPTY_RESPONSE,
    DROP_MISSING_PROVENANCE,
    DROP_NO_CONTEXT_WINDOWS,
    DROP_NOT_EXACTLY_ONE_CORRECT,
    DROP_NOT_JSON,
    DROP_REASONS,
    DROP_UNKNOWN_PROVENANCE,
    DROP_WRONG_N_ALTERNATIVES,
    derive_seed,
    generate_items,
    parse_generation_envelope,
    select_context_windows,
)
from nefair.items.review import (
    DECISION_COLUMN,
    MINUTES_COLUMN,
    REASON_COLUMN,
    REVIEW_STATUS_ACCEPTED,
    REVIEW_STATUS_REJECTED,
    export_review_sheet,
    import_review_sheet,
    sheet_columns,
)
from nefair.models.base import FakeTextQA, fake_spec
from nefair.models.prompts import PROMPT_VERSION
from nefair.schema import ALTERNATIVE_LABELS, Alternative, Item, Window

CONFIG_PATH = "configs/items.yaml"

# Duas entrevistas, uma por região: é o mínimo para que os agregados por região
# do relatório (e o gate do piloto) tenham as duas colunas que comparam.
REGION_BY_SPEAKER = {"NE_A": "NE", "SE_B": "SE", "NE_SOLO": "NE"}


@pytest.fixture(scope="module")
def config():
    """Config real do repositório: 4 alternativas, 1 item por janela, D2 ligado."""
    return load_items_config(CONFIG_PATH)


# --------------------------------------------------------------------------- #
# Construtores de janelas e itens (dataclasses congeladas, montadas à mão)
# --------------------------------------------------------------------------- #
def make_window(speaker_code: str, index: int, *, region: str = "NE") -> Window:
    return Window(
        window_id=Window.make_id(speaker_code, index),
        speaker_code=speaker_code,
        audio_name=f"{speaker_code}_audio",
        split="train",
        region=region,
        age=34,
        segment_file_paths=tuple(
            f"train/{speaker_code}/{speaker_code}_{index:02d}_{k}.wav" for k in range(8)
        ),
        start_time=float(index * 40),
        end_time=float(index * 40 + 40),
        duration_s=40.0,
        n_segments=8,
        reference_text=f"trecho {index} da entrevista de {speaker_code}",
        position_index=index - 1,
        n_candidates_for_speaker=4,
    )


# Uma entrevista com 4 janelas: cada alvo tem exatamente 3 outras janelas, que é
# `distractors.n_context_windows` do config — assim o prompt de geração sai com o
# contexto cheio e os descartes testados não são efeito de contexto faltando.
INTERVIEW = tuple(make_window("NE_A", index) for index in range(1, 5))

DEFAULT_TEXTS = (
    "ele foi a feira de manha",
    "ele trabalhava na roca",
    "ele mudou de cidade",
    "ele ficou em casa",
)


def make_item(
    item_id: str,
    *,
    speaker_code: str = "NE_A",
    texts: tuple[str, ...] = DEFAULT_TEXTS,
    correct_index: int = 0,
    filter_status: str = "pending",
) -> Item:
    window_id = item_id.rsplit("__", 1)[0]
    alternatives = tuple(
        Alternative(
            label=label,
            text=text,
            is_correct=(position == correct_index),
            source_window_id=(
                window_id if position == correct_index else f"{speaker_code}__w{position + 5:02d}"
            ),
        )
        for position, (label, text) in enumerate(
            zip(ALTERNATIVE_LABELS[: len(texts)], texts, strict=True)
        )
    )
    return Item(
        item_id=item_id,
        window_id=window_id,
        speaker_code=speaker_code,
        question=f"O que o falante diz em {window_id}?",
        alternatives=alternatives,
        generator_model="fake-gerador",
        prompt_version=PROMPT_VERSION,
        filter_status=filter_status,
    )


# --------------------------------------------------------------------------- #
# Encenação do gerador: o `responder` lê o prompt como o LLM leria
# --------------------------------------------------------------------------- #
_TARGET_RE = re.compile(r"TRANSCRIÇÃO-ALVO \(janela (.+?)\)")
_CONTEXT_RE = re.compile(r"^- \(janela (.+?)\) ", re.MULTILINE)


def _ids_from_prompt(prompt: str) -> tuple[str, tuple[str, ...]]:
    """Ids da janela-alvo e dos trechos de contexto, lidos do próprio prompt.

    Ler do prompt (em vez de fixar ids no teste) é o que faz a encenação valer:
    o falso gerador só pode declarar proveniência válida se o prompt de fato
    ofereceu os trechos — que é exatamente a exigência do D2.
    """
    target = _TARGET_RE.search(prompt)
    assert target is not None, prompt
    return target.group(1), tuple(_CONTEXT_RE.findall(prompt))


def _valid_payload(target: str, context: tuple[str, ...]) -> dict:
    """Item bem formado: correta ancorada no alvo, distratores ancorados no contexto."""
    alternatives: list[dict] = [
        {
            "text": f"o falante fala do assunto de {target}",
            "is_correct": True,
            "source_window_id": target,
        }
    ]
    for position, window_id in enumerate(context, start=1):
        alternatives.append(
            {
                "text": f"distrator {position} vindo de {window_id}",
                "is_correct": False,
                "source_window_id": window_id,
            }
        )
    return {"question": f"O que o falante diz em {target}?", "alternatives": alternatives}


def _generator(mutate=None, *, raw: str | None = None) -> FakeTextQA:
    """Gerador falso: devolve `raw` fixo, ou um envelope válido já mutilado por `mutate`."""

    def responder(prompt: str) -> str:
        if raw is not None:
            return raw
        target, context = _ids_from_prompt(prompt)
        payload = _valid_payload(target, context)
        if mutate is not None:
            mutate(payload)
        return json.dumps({"items": [payload]}, ensure_ascii=False)

    return FakeTextQA(spec=fake_spec(key="gerador", model_id="fake-gerador"), responder=responder)


def _generate(config, mutate=None, *, raw: str | None = None, windows=INTERVIEW):
    model = _generator(mutate, raw=raw)
    outcome = generate_items(
        windows, model=model, config=config, region_by_speaker=REGION_BY_SPEAKER
    )
    outcome.ledger.assert_balanced()
    return outcome, model


def _two_correct(payload: dict) -> None:
    payload["alternatives"][1]["is_correct"] = True


def _empty_alternative(payload: dict) -> None:
    payload["alternatives"][2]["text"] = "   "


def _duplicate_alternative(payload: dict) -> None:
    payload["alternatives"][2]["text"] = payload["alternatives"][1]["text"].upper()


def _distractor_without_provenance(payload: dict) -> None:
    payload["alternatives"][3].pop("source_window_id")


def _invented_provenance(payload: dict) -> None:
    payload["alternatives"][1]["source_window_id"] = "NE_A__w99"


# =========================================================================== #
# Geração (D2)
# =========================================================================== #
def test_valid_generation_produces_one_item_per_window(config):
    """Caso-base: sem ele, um teste de descarte não prova nada (tudo cairia)."""
    outcome, model = _generate(config)
    assert model.n_calls == len(INTERVIEW)
    assert outcome.ledger.dropped == {}
    assert outcome.ledger.kept == len(INTERVIEW)
    assert [item.item_id for item in outcome.items] == [f"NE_A__w{i:02d}__i01" for i in range(1, 5)]
    first = outcome.items[0]
    assert [alt.label for alt in first.alternatives] == ["A", "B", "C", "D"]
    assert first.prompt_version == PROMPT_VERSION
    assert first.filter_status == "pending"  # a trilha de vida começa vazia
    # A proveniência viaja junto: a correta aponta para o alvo, os distratores
    # para janelas que estiveram de fato no prompt.
    assert first.alternatives[0].source_window_id == first.window_id
    context_ids = {w.window_id for w in INTERVIEW} - {first.window_id}
    assert {alt.source_window_id for alt in first.alternatives[1:]} <= context_ids
    assert outcome.summary.by_region[0].region == "NE"
    assert outcome.summary.by_region[0].n_items == 4


def test_drop_reasons_are_thirteen_distinct_snake_case_keys():
    """As chaves viram nomes de coluna e de seção no relatório: sem acento, sem maiúscula.

    Uma chave renomeada em silêncio quebra a comparação entre execuções da mesma
    auditoria — o motivo some de um relatório e aparece com outro nome no outro.
    """
    assert len(DROP_REASONS) == len(set(DROP_REASONS)) == 13
    assert set(DROP_REASONS) == {
        "erro_do_modelo",
        "resposta_vazia",
        "resposta_nao_e_json",
        "envelope_json_invalido",
        "item_faltando_na_resposta",
        "pergunta_vazia",
        "n_alternativas_incorreto",
        "corretas_diferente_de_uma",
        "alternativa_vazia",
        "alternativas_duplicadas",
        "distrator_sem_proveniencia",
        "proveniencia_desconhecida",
        "sem_janelas_de_contexto",
    }
    for reason in DROP_REASONS:
        assert re.fullmatch(r"[a-z_]+", reason), reason


def test_non_json_answer_is_counted_as_such(config):
    """Recusa educada do gerador é resposta não-JSON — não é item, nem erro de rede."""
    outcome, _ = _generate(config, raw="Desculpe, não posso ajudar com isso.")
    assert outcome.items == ()
    assert outcome.ledger.dropped == {DROP_NOT_JSON: 4}


def test_two_correct_alternatives_are_counted_as_such(config):
    """Duas corretas não são desempatadas: o item é ambíguo e a contagem diz isso."""
    outcome, _ = _generate(config, _two_correct)
    assert outcome.items == ()
    assert outcome.ledger.dropped == {DROP_NOT_EXACTLY_ONE_CORRECT: 4}


def test_blank_alternative_is_counted_as_such(config):
    """Alternativa só com espaços é vazia: nada é 'consertado' preenchendo texto."""
    outcome, _ = _generate(config, _empty_alternative)
    assert outcome.items == ()
    assert outcome.ledger.dropped == {DROP_EMPTY_ALTERNATIVE: 4}


def test_duplicate_alternatives_are_counted_ignoring_case(config):
    """Duas alternativas iguais só na caixa são a MESMA alternativa para quem responde."""
    outcome, _ = _generate(config, _duplicate_alternative)
    assert outcome.items == ()
    assert outcome.ledger.dropped == {DROP_DUPLICATE_ALTERNATIVE: 4}


def test_distractor_without_provenance_is_counted_as_such(config):
    """D2: distrator sem `source_window_id` torna não verificável a frase do TCC."""
    outcome, _ = _generate(config, _distractor_without_provenance)
    assert outcome.items == ()
    assert outcome.ledger.dropped == {DROP_MISSING_PROVENANCE: 4}


def test_invented_provenance_has_its_own_category(config):
    """Proveniência inventada é pior que ausente: o item PARECERIA auditável.

    Por isso a categoria é própria, e não a mesma de 'sem proveniência': a
    primeira é falha de formato do gerador, a segunda é alucinação.
    """
    outcome, _ = _generate(config, _invented_provenance)
    assert outcome.items == ()
    assert outcome.ledger.dropped == {DROP_UNKNOWN_PROVENANCE: 4}


def test_window_without_context_is_dropped_before_calling_the_model(config):
    """Entrevista de uma janela só não tem de onde extrair distrator ancorado (D2).

    Descartar aqui é mais honesto — e mais barato — do que pedir ao gerador um
    distrator que ele inventaria e declararia como vindo da própria janela-alvo.
    """
    solo = (make_window("NE_SOLO", 1),)
    outcome, model = _generate(config, windows=solo)
    assert outcome.items == ()
    assert outcome.ledger.dropped == {DROP_NO_CONTEXT_WINDOWS: 1}
    assert model.n_calls == 0, "sem contexto, nem se gasta chamada de API"
    assert outcome.attempts[0].prompt == ""
    assert outcome.attempts[0].context_window_ids == ()
    assert outcome.summary.n_windows_without_context == 1


def test_without_require_provenance_the_lonely_window_still_reaches_the_model(config):
    """Contraste que isola o D2: com a exigência desligada, a janela é tentada.

    O relatório então precisa dizer que a proveniência não foi checada — é essa
    a troca, e ela tem de estar visível em algum número.
    """
    loose = dataclasses.replace(
        config,
        distractors=dataclasses.replace(config.distractors, require_provenance=False),
    )
    outcome, model = _generate(loose, windows=(make_window("NE_SOLO", 1),))
    assert model.n_calls == 1
    assert DROP_NO_CONTEXT_WINDOWS not in outcome.ledger.dropped
    # Sem contexto, o gerador falso só consegue montar a alternativa correta.
    assert outcome.ledger.dropped == {DROP_WRONG_N_ALTERNATIVES: 1}


# --------------------------------------------------------------------------- #
# Envelope da resposta
# --------------------------------------------------------------------------- #
def test_envelope_tolerates_a_code_fence(config):
    """Vários modelos cercam o JSON mesmo instruídos a não cercar.

    Não tolerar a cerca transformaria item válido em exclusão — e a taxa de
    descarte passaria a medir formatação, não qualidade do item.
    """
    payload = '{"items": [{"question": "q", "alternatives": []}]}'
    fenced = f"```json\n{payload}\n```"
    items, reason = parse_generation_envelope(fenced)
    assert reason is None
    assert items == [{"question": "q", "alternatives": []}]
    assert parse_generation_envelope(f"```\n{payload}\n```") == (items, None)
    assert parse_generation_envelope(payload) == (items, None)


def test_envelope_distinguishes_empty_broken_and_wrong_shape():
    """Três falhas diferentes do gerador, três categorias — não uma genérica."""
    assert parse_generation_envelope("") == (None, DROP_EMPTY_RESPONSE)
    assert parse_generation_envelope("   \n ") == (None, DROP_EMPTY_RESPONSE)
    assert parse_generation_envelope('{"items": [') == (None, DROP_NOT_JSON)
    assert parse_generation_envelope("[1, 2, 3]") == (None, DROP_BAD_ENVELOPE)
    assert parse_generation_envelope('"só um texto"') == (None, DROP_BAD_ENVELOPE)
    assert parse_generation_envelope('{"itens": []}') == (None, DROP_BAD_ENVELOPE)
    assert parse_generation_envelope('{"items": {"a": 1}}') == (None, DROP_BAD_ENVELOPE)


def test_envelope_never_raises_for_any_garbage():
    """A resposta do gerador é entrada não confiável: exceção aqui derrubaria a etapa.

    Uma exclusão contada é um dado; um traceback no meio de 2.300 janelas é um
    dia de execução perdido.
    """
    garbage = [
        "",
        "   ",
        "```",
        "```json\n```",
        "{",
        "}",
        "null",
        "true",
        "3.14",
        "[]",
        '{"items": null}',
        "\x00\x01",
        "{'items': []}",  # aspas simples não são JSON
        "```json\n{\n```",
    ]
    for raw in garbage:
        items, reason = parse_generation_envelope(raw)
        assert items is None, raw
        assert reason in DROP_REASONS, (raw, reason)


# --------------------------------------------------------------------------- #
# Seleção dos trechos de contexto e sementes
# --------------------------------------------------------------------------- #
def test_context_excludes_the_target_and_is_deterministic(config):
    target = INTERVIEW[0]
    chosen = select_context_windows(target, INTERVIEW, n_context_windows=2, seed=config.seed)
    assert len(chosen) == 2
    assert target.window_id not in {w.window_id for w in chosen}
    assert [w.window_id for w in chosen] == sorted(w.window_id for w in chosen)
    again = select_context_windows(target, INTERVIEW, n_context_windows=2, seed=config.seed)
    assert [w.window_id for w in again] == [w.window_id for w in chosen]
    # Entrevista de uma janela só: nada a oferecer, e sem erro.
    solo = make_window("NE_SOLO", 1)
    assert select_context_windows(solo, [solo], n_context_windows=3, seed=config.seed) == ()


def test_derive_seed_depends_on_the_parts_and_not_on_the_process():
    """Semente derivada por hash estável: a mesma entrada dá o mesmo inteiro sempre.

    Se alguém trocar por `hash()` (randomizado por processo), a reprodutibilidade
    entre máquinas morre em silêncio.
    """
    assert derive_seed(1, "a") == derive_seed(1, "a")
    assert derive_seed(1, "a") != derive_seed("a", 1)
    assert derive_seed(1, "a") != derive_seed(2, "a")


# =========================================================================== #
# Controle textual (D3)
# =========================================================================== #
_SHOWN_RE = re.compile(r"^([A-E])\) (.+)$", re.MULTILINE)


def _shown(prompt: str) -> list[tuple[str, str]]:
    """Pares (rótulo, texto) na ordem EXIBIDA pelo prompt de resposta."""
    return [(label, text) for label, text in _SHOWN_RE.findall(prompt)]


def _wrong_label(shown: list[tuple[str, str]], correct_text: str) -> str:
    return next(label for label, text in shown if text != correct_text)


def _oracle_first_n(correct_text: str, n_correct: int):
    """Modelo falso que acerta nas `n_correct` primeiras chamadas e erra depois.

    É assim que se controla "2 de 3" e "1 de 3" sem depender de sorte: o oráculo
    conhece a alternativa correta e DECIDE vazar ou não, o que separa o critério
    de maioria do acaso da amostragem.
    """
    state = {"n": 0}

    def responder(prompt: str) -> str:
        state["n"] += 1
        shown = _shown(prompt)
        if state["n"] <= n_correct:
            return next(label for label, text in shown if text == correct_text)
        return _wrong_label(shown, correct_text)

    return responder


def _oracle_leaking(leaky_texts: frozenset[str], correct_texts: frozenset[str]):
    """Acerta sempre os itens cuja correta está em `leaky_texts`; erra nos demais.

    O oráculo recebe TODAS as corretas para poder errar de propósito no item
    limpo. Um falso modelo que respondesse sempre "A" acertaria o item limpo
    sempre que a permutação pusesse a correta na frente, e o teste passaria a
    depender de sorte em vez de medir o critério.
    """

    def responder(prompt: str) -> str:
        shown = _shown(prompt)
        label, text = next((lab, txt) for lab, txt in shown if txt in correct_texts)
        return label if text in leaky_texts else _wrong_label(shown, text)

    return responder


def _model(key: str, responder) -> FakeTextQA:
    return FakeTextQA(spec=fake_spec(key=key, model_id=f"fake-{key}"), responder=responder)


CORRECT_TEXT = DEFAULT_TEXTS[0]


def test_chance_baseline_is_the_number_that_makes_the_discard_rate_readable(config):
    """2-de-3 com 4 alternativas ⇒ 0,15625: ~15,6% dos itens VÁLIDOS caem por azar.

    O valor é calculado (cauda da binomial), nunca escrito à mão no relatório: se
    o config mudar para 3-de-3 ou 5 alternativas, o número acompanha.
    """
    assert chance_pass_probability(n_runs=3, n_correct_at_least=2, n_alternatives=4) == 0.15625
    # E é o que sai da config real do repositório.
    assert (
        chance_pass_probability(
            n_runs=config.text_filter.n_runs,
            n_correct_at_least=config.text_filter.discard_if_correct_at_least,
            n_alternatives=config.n_alternatives,
        )
        == 0.15625
    )
    # Critério mais exigente ⇒ menos itens válidos perdidos por azar.
    assert chance_pass_probability(n_runs=3, n_correct_at_least=3, n_alternatives=4) < 0.15625


def test_chance_baseline_rejects_a_degenerate_number_of_alternatives():
    """Com 1 alternativa, p = 1 e a 'linha de base' seria 100% — número sem sentido."""
    with pytest.raises(ValueError, match="n_alternatives=1"):
        chance_pass_probability(n_runs=3, n_correct_at_least=2, n_alternatives=1)


def test_chance_baseline_rejects_a_criterion_above_the_number_of_runs():
    """Exigir 4 acertos em 3 execuções é config incoerente: falha, não devolve 0."""
    with pytest.raises(ValueError, match="n_correct_at_least=4"):
        chance_pass_probability(n_runs=3, n_correct_at_least=4, n_alternatives=4)


def test_permutation_is_seeded_by_item_model_run_and_seed(config):
    """A ordem muda a cada execução — é isso que dá sentido à 'maioria de 3'.

    E é derivada, não sorteada de um RNG global: rodar os testes em outra ordem,
    ou reexecutar o filtro amanhã, produz exatamente as mesmas permutações.
    """
    kwargs = {"item_id": "NE_A__w01__i01", "model_key": "m1", "seed": config.text_filter.seed}
    first = derive_permutation(run_index=0, n_alternatives=4, **kwargs)
    assert first == derive_permutation(run_index=0, n_alternatives=4, **kwargs)
    assert sorted(first) == [0, 1, 2, 3]
    assert first != derive_permutation(run_index=1, n_alternatives=4, **kwargs)
    assert first != derive_permutation(
        item_id="NE_A__w01__i02",
        model_key="m1",
        seed=config.text_filter.seed,
        run_index=0,
        n_alternatives=4,
    )
    assert first != derive_permutation(
        item_id="NE_A__w01__i01",
        model_key="m2",
        seed=config.text_filter.seed,
        run_index=0,
        n_alternatives=4,
    )
    assert first != derive_permutation(
        item_id="NE_A__w01__i01", model_key="m1", seed=1, run_index=0, n_alternatives=4
    )


def test_shuffle_disabled_uses_the_identity_permutation(config):
    """Com `shuffle_alternatives: false`, a ordem exibida é a canônica — e só ela.

    O contraste importa: sem embaralhar, as três execuções veem o mesmo prompt, e
    a 'maioria de três' vira uma execução repetida três vezes.
    """
    item = make_item("NE_A__w01__i01")
    fixed = dataclasses.replace(config.text_filter, shuffle_alternatives=False)
    models = {"m1": _model("m1", _oracle_first_n(CORRECT_TEXT, 0))}
    _, runs = run_filter_for_item(item, models, fixed)
    assert [run.permutation for run in runs] == [(0, 1, 2, 3)] * fixed.n_runs
    assert {run.shown_correct_label for run in runs} == {item.correct_label}

    _, shuffled = run_filter_for_item(item, models, config.text_filter)
    assert any(run.permutation != (0, 1, 2, 3) for run in shuffled)


def test_item_answered_twice_in_three_runs_is_discarded(config):
    """O critério do D3 em ação, com a frase literal que vai para `filter_reason`."""
    item = make_item("NE_A__w01__i01")
    models = {"m1": _model("m1", _oracle_first_n(CORRECT_TEXT, 2))}
    decision, runs = run_filter_for_item(item, models, config.text_filter, region="NE")
    assert decision.discarded
    assert decision.reason == f"{DROP_ANSWERABLE_WITHOUT_SPEECH}: m1=2/3"
    assert decision.n_correct_by_model == {"m1": 2}
    assert decision.triggering_models == ("m1",)
    assert decision.region == "NE"
    assert len(runs) == 3


def test_item_answered_once_in_three_runs_is_kept(config):
    """Abaixo do critério o item fica: o filtro é conservador por construção."""
    item = make_item("NE_A__w01__i01")
    models = {"m1": _model("m1", _oracle_first_n(CORRECT_TEXT, 1))}
    decision, _ = run_filter_for_item(item, models, config.text_filter)
    assert not decision.discarded
    assert decision.reason == ""
    assert decision.triggering_models == ()


def test_any_model_discards_lets_a_single_model_drop_the_item(config):
    """D3: basta UM modelo responder sem a fala para o item ser suspeito.

    Exigir que todos acertem deixaria passar item vazável para uma família só — e
    a comparação entre famílias é justamente o experimento.
    """
    item = make_item("NE_A__w01__i01")
    models = {
        "vaza": _model("vaza", _oracle_first_n(CORRECT_TEXT, 3)),
        "nao_vaza": _model("nao_vaza", _oracle_first_n(CORRECT_TEXT, 0)),
    }
    decision, _ = run_filter_for_item(item, models, config.text_filter)
    assert config.text_filter.any_model_discards is True
    assert decision.discarded
    assert decision.triggering_models == ("vaza",)
    assert decision.reason == f"{DROP_ANSWERABLE_WITHOUT_SPEECH}: vaza=3/3"


def test_requiring_every_model_keeps_the_item_when_only_one_leaks(config):
    """Contraste: com `any_model_discards: false`, um modelo sozinho não derruba."""
    item = make_item("NE_A__w01__i01")
    models = {
        "vaza": _model("vaza", _oracle_first_n(CORRECT_TEXT, 3)),
        "nao_vaza": _model("nao_vaza", _oracle_first_n(CORRECT_TEXT, 0)),
    }
    strict = dataclasses.replace(config.text_filter, any_model_discards=False)
    decision, _ = run_filter_for_item(item, models, strict)
    assert not decision.discarded


def test_filter_without_any_model_refuses_to_decide(config):
    """Sem modelo não há filtro: a etapa não rodou, e isso não pode virar veredito.

    Antes deste teste, `len(triggering) == len(models)` com `0 == 0` fazia o
    conjunto vazio de modelos DESCARTAR todo item sob `any_model_discards=False`
    — um vazamento declarado que ninguém mediu, com `reason` truncado. Agora
    falha alto, como a planilha vazia da revisão.
    """
    item = make_item("NE_A__w01__i01")
    strict = dataclasses.replace(config.text_filter, any_model_discards=False)
    with pytest.raises(ValueError, match="sem nenhum modelo"):
        run_filter_for_item(item, {}, strict)
    with pytest.raises(ValueError, match="sem nenhum modelo"):
        run_filter_for_item(item, {}, config.text_filter)


def test_unparsable_answers_are_counted_and_never_imputed(config):
    """Resposta ininterpretável não é acerto nem erro: é categoria própria.

    Imputar como erro inflaria a taxa de retenção do filtro; imputar como acerto
    derrubaria itens bons. Contar à parte é a única leitura honesta.
    """
    item = make_item("NE_A__w01__i01")
    models = {"m1": _model("m1", lambda prompt: "não tenho como saber")}
    decision, runs = run_filter_for_item(item, models, config.text_filter)
    assert decision.n_unparsed_by_model == {"m1": 3}
    assert decision.n_correct_by_model == {"m1": 0}
    assert not decision.discarded
    assert all(run.parsed_label is None and run.is_correct is None for run in runs)


def test_extract_label_reads_the_easy_cases_and_refuses_the_rest():
    """Parser MÍNIMO e local: o que não casar vira `None` e é CONTADO."""
    labels = ("A", "B", "C", "D")
    assert extract_label("B", labels) == "B"
    assert extract_label("(C)", labels) == "C"
    assert extract_label("d)", labels) == "D"
    assert extract_label("B) ele foi a feira", labels) == "B"
    assert extract_label("Alternativa C", labels) == "C"
    assert extract_label("", labels) is None
    assert extract_label("não sei dizer", labels) is None
    # Letra fora do vocabulário do item (item de 2 alternativas) não vira acerto.
    assert extract_label("C", ("A", "B")) is None


def test_apply_text_filter_marks_every_item_and_balances_the_ledger(config):
    """O descartado VOLTA marcado, não some: é assim que se audita NE × SE.

    Sumir com o item descartado tornaria impossível checar se o filtro derrubou
    as duas regiões em proporções diferentes — uma das condições do gate.
    """
    leaky = make_item("NE_A__w01__i01", speaker_code="NE_A")
    clean = make_item(
        "SE_B__w01__i01",
        speaker_code="SE_B",
        texts=("resposta que exige a fala", "outra coisa", "mais outra", "nenhuma"),
    )
    corretas = frozenset({leaky.alternatives[0].text, clean.alternatives[0].text})
    vazadas = frozenset({leaky.alternatives[0].text})
    models = {"m1": _model("m1", _oracle_leaking(vazadas, corretas))}
    outcome = apply_text_filter(
        [clean, leaky], models, config.text_filter, region_by_speaker=REGION_BY_SPEAKER
    )
    outcome.ledger.assert_balanced()
    assert outcome.ledger.total_in == 2
    assert outcome.ledger.dropped == {DROP_ANSWERABLE_WITHOUT_SPEECH: 1}
    statuses = {item.item_id: item.filter_status for item in outcome.items}
    assert statuses == {
        "NE_A__w01__i01": FILTER_STATUS_DISCARDED,
        "SE_B__w01__i01": FILTER_STATUS_KEPT,
    }
    assert [item.item_id for item in outcome.kept_items] == ["SE_B__w01__i01"]
    discarded = next(i for i in outcome.items if i.filter_status == FILTER_STATUS_DISCARDED)
    assert discarded.filter_reason.startswith(DROP_ANSWERABLE_WITHOUT_SPEECH)
    # A linha de base de acaso acompanha a taxa observada em todo relatório.
    assert outcome.summary.chance_discard_rate == 0.15625
    assert outcome.summary.discard_rate == 0.5
    assert outcome.summary.n_total_runs == 6
    assert outcome.summary.n_unparsed_runs == 0
    by_region = {r.region: (r.n_items, r.n_discarded) for r in outcome.summary.by_region}
    assert by_region == {"NE": (1, 1), "SE": (1, 0)}


# =========================================================================== #
# Planilha de revisão
# =========================================================================== #
REVIEW_ITEMS = (
    make_item("NE_A__w01__i01", speaker_code="NE_A", filter_status=FILTER_STATUS_KEPT),
    make_item("SE_B__w01__i01", speaker_code="SE_B", filter_status=FILTER_STATUS_KEPT),
    make_item("NE_A__w02__i01", speaker_code="NE_A", filter_status=FILTER_STATUS_DISCARDED),
)
REVIEWABLE_IDS = ["NE_A__w01__i01", "SE_B__w01__i01"]


def _read_sheet(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _write_sheet(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _filled_sheet(
    tmp_path: Path,
    *,
    decision: str = REVIEW_STATUS_ACCEPTED,
    reason: str = "",
    minutes: str = "2,5",
    name: str = "revisao.csv",
) -> tuple[Path, list[str], list[dict[str, str]]]:
    """Exporta a planilha e a devolve preenchida como um revisor a devolveria."""
    path = tmp_path / name
    export_review_sheet(REVIEW_ITEMS, path, region_by_speaker=REGION_BY_SPEAKER)
    columns, rows = _read_sheet(path)
    for row in rows:
        row[DECISION_COLUMN] = decision
        row[REASON_COLUMN] = reason
        row[MINUTES_COLUMN] = minutes
    _write_sheet(path, columns, rows)
    return path, columns, rows


def _import(path: Path, config):
    return import_review_sheet(
        path,
        [i for i in REVIEW_ITEMS if i.filter_status == FILTER_STATUS_KEPT],
        review_decisions=config.review_decisions,
        region_by_speaker=REGION_BY_SPEAKER,
    )


def test_sheet_columns_are_the_twenty_four_canonical_columns():
    """Ordem fixa e derivada do nº de alternativas (D10), não escrita à mão.

    Se a ordem mudasse entre exportações, reexportar no meio da revisão
    embaralharia o trabalho já feito pelo revisor.
    """
    assert sheet_columns(4) == (
        "item_id",
        "window_id",
        "speaker_code",
        "region",
        "question",
        "alt_A_text",
        "alt_A_is_correct",
        "alt_A_source_window_id",
        "alt_B_text",
        "alt_B_is_correct",
        "alt_B_source_window_id",
        "alt_C_text",
        "alt_C_is_correct",
        "alt_C_source_window_id",
        "alt_D_text",
        "alt_D_is_correct",
        "alt_D_source_window_id",
        "correct_label",
        "generator_model",
        "prompt_version",
        "filter_status",
        "review_decision",
        "review_reason",
        "review_minutes",
    )
    assert len(sheet_columns(4)) == 24
    assert len(sheet_columns(3)) == 21  # três colunas a menos por alternativa


def test_export_takes_only_the_items_the_filter_kept(tmp_path):
    """Revisar depois do filtro é o que segura o custo humano no cronograma."""
    path = tmp_path / "revisao.csv"
    written = export_review_sheet(REVIEW_ITEMS, path, region_by_speaker=REGION_BY_SPEAKER)
    columns, rows = _read_sheet(path)
    assert written == 2
    assert columns == list(sheet_columns(4))
    assert [row["item_id"] for row in rows] == REVIEWABLE_IDS
    assert [row["region"] for row in rows] == ["NE", "SE"]
    assert rows[0]["correct_label"] == "A"
    assert rows[0]["alt_A_is_correct"] == "true"
    assert all(row[column] == "" for row in rows for column in (DECISION_COLUMN, MINUTES_COLUMN))

    everything = tmp_path / "tudo.csv"
    assert export_review_sheet(REVIEW_ITEMS, everything, only_filter_kept=False) == 3


def test_export_is_byte_identical_between_writes(tmp_path):
    """Reexportar no meio da revisão tem de ser idempotente, byte a byte."""
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    export_review_sheet(REVIEW_ITEMS, first, region_by_speaker=REGION_BY_SPEAKER)
    export_review_sheet(REVIEW_ITEMS, second, region_by_speaker=REGION_BY_SPEAKER)
    assert first.read_bytes() == second.read_bytes()


def test_export_refuses_an_empty_sheet(tmp_path):
    """Planilha vazia esconderia 'o filtro derrubou tudo' atrás de 'não rodou'."""
    todos_descartados = [
        dataclasses.replace(item, filter_status=FILTER_STATUS_DISCARDED) for item in REVIEW_ITEMS
    ]
    with pytest.raises(ValueError, match="Nenhum item a revisar"):
        export_review_sheet(todos_descartados, tmp_path / "vazia.csv")


def test_review_round_trip_applies_the_decisions(tmp_path, config):
    """Exportar → preencher → importar, com os números que o gate do piloto lê."""
    path, columns, rows = _filled_sheet(tmp_path)
    rows[1][DECISION_COLUMN] = REVIEW_STATUS_REJECTED
    rows[1][REASON_COLUMN] = "pergunta respondível pelo formato"
    rows[1][MINUTES_COLUMN] = "3.5"
    _write_sheet(path, columns, rows)

    outcome = _import(path, config)
    outcome.ledger.assert_balanced()
    by_id = {item.item_id: item for item in outcome.items}
    assert by_id["NE_A__w01__i01"].review_status == REVIEW_STATUS_ACCEPTED
    # Vírgula decimal: a planilha volta de uma ferramenta em pt-BR.
    assert by_id["NE_A__w01__i01"].review_minutes == 2.5
    assert by_id["SE_B__w01__i01"].review_status == REVIEW_STATUS_REJECTED
    assert by_id["SE_B__w01__i01"].review_reason == "pergunta respondível pelo formato"
    assert by_id["SE_B__w01__i01"].review_minutes == 3.5
    assert [item.item_id for item in outcome.accepted_items] == ["NE_A__w01__i01"]
    # O item aceito e retido pelo filtro é o único elegível para a Etapa 5.
    assert by_id["NE_A__w01__i01"].is_eligible_for_eval
    assert not by_id["SE_B__w01__i01"].is_eligible_for_eval

    assert outcome.ledger.dropped == {"revisao_rejeitada:pergunta respondível pelo formato": 1}
    assert outcome.summary.n_accepted == 1
    assert outcome.summary.n_rejected == 1
    assert outcome.summary.total_minutes == 6.0
    assert outcome.summary.minutes_per_item == 3.0
    by_region = {r.region: (r.n_items, r.n_rejected) for r in outcome.summary.by_region}
    assert by_region == {"NE": (1, 0), "SE": (1, 1)}


def test_import_rejects_a_decision_outside_the_vocabulary(tmp_path, config):
    """Vocabulário fechado: 'talvez' não é uma decisão de revisão."""
    path, columns, rows = _filled_sheet(tmp_path, decision="talvez")
    with pytest.raises(ValueError, match="fora do vocabulário"):
        _import(path, config)


def test_import_rejects_blank_minutes(tmp_path, config):
    """Minutos em branco enviesariam a projeção de custo da revisão completa."""
    path, columns, rows = _filled_sheet(tmp_path, minutes="")
    with pytest.raises(ValueError, match="sem 'review_minutes'"):
        _import(path, config)


def test_import_rejects_non_numeric_minutes(tmp_path, config):
    """'uns 3' não é um número: a projeção do cronograma precisa de aritmética."""
    path, columns, rows = _filled_sheet(tmp_path, minutes="uns 3")
    with pytest.raises(ValueError, match="não numérico"):
        _import(path, config)


def test_import_rejects_negative_minutes(tmp_path, config):
    """Tempo negativo é erro de digitação — e encolheria o custo projetado."""
    path, columns, rows = _filled_sheet(tmp_path, minutes="-1")
    with pytest.raises(ValueError, match="minutos negativos"):
        _import(path, config)


def test_import_rejects_a_repeated_item_id(tmp_path, config):
    """Linha duplicada: qual das duas decisões vale? Parar é mais barato."""
    path, columns, rows = _filled_sheet(tmp_path)
    _write_sheet(path, columns, [*rows, dict(rows[0])])
    with pytest.raises(ValueError, match="repetido"):
        _import(path, config)


def test_import_rejects_an_unknown_item(tmp_path, config):
    """Item que não existe nesta execução: a planilha é de outra rodada."""
    path, columns, rows = _filled_sheet(tmp_path)
    rows[0]["item_id"] = "NE_A__w99__i01"
    _write_sheet(path, columns, rows)
    with pytest.raises(ValueError, match="item desconhecido"):
        _import(path, config)


def test_import_rejects_a_missing_reviewable_item(tmp_path, config):
    """Item ausente da planilha é item não revisado — e o texto exige revisar TODO item."""
    path, columns, rows = _filled_sheet(tmp_path)
    _write_sheet(path, columns, rows[:1])
    with pytest.raises(ValueError, match="não revisado"):
        _import(path, config)


def test_import_rejects_a_rejection_without_a_reason(tmp_path, config):
    """A distribuição dos motivos de rejeição é resultado da Etapa 3, não burocracia."""
    path, columns, rows = _filled_sheet(tmp_path, decision=REVIEW_STATUS_REJECTED, reason="")
    with pytest.raises(ValueError, match="rejeitado sem"):
        _import(path, config)
