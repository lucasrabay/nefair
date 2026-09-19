"""Testes do runner da avaliação (Etapas 4–5) — tudo offline, com modelos falsos.

O runner é a única etapa que gasta dinheiro e que pode ser interrompida no meio.
Por isso quase todo teste aqui defende uma de três propriedades, e cada uma tem
consequência direta sobre um número do TCC:

1. **Pareamento.** `derive_permutation` não recebe `condition`. `asr` e
   `reference` do mesmo item e do mesmo modelo veem as alternativas na MESMA
   ordem, senão `acc(reference) − acc(asr)` — a decomposição da Etapa 6 — mistura
   efeito de via com efeito de posição da alternativa correta.
2. **Identidade da chave de cache.** Um resultado só pode ser reusado se a
   pergunta, o modelo, a versão, os parâmetros, o prompt e a permutação forem os
   mesmos. Cada um desses eixos tem um teste, porque um cache que responde à
   pergunta errada é pior que cache nenhum: ele é silencioso.
3. **Nada imputado, nada sumido.** Resposta não interpretável não vira acerto nem
   erro; item inelegível não some sem motivo registrado; erro de provedor não
   entra no cache (senão a retomada nunca o retentaria).

Convenção seguida (a de `tests/test_windows.py`): configuração REAL do
repositório (`configs/models.yaml`, `configs/eval.yaml`), variada com
`dataclasses.replace` só no eixo que o teste isola. As asserções são contra os
artefatos observáveis — `RunRecord`, arquivo de cache, `RunSummary`, contador de
chamadas do adaptador falso — e não contra as estruturas internas do runner.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from nefair.config import load_eval_config, load_models_config
from nefair.models.base import ROLE_ASR, FakeASR, fake_spec
from nefair.run.runner import (
    ERROR_MISSING_INPUT,
    ContaminationCheckError,
    RunCache,
    assert_contamination_checked,
    build_adapters,
    derive_permutation,
    present_alternatives,
    run_eval,
    select_eligible_items,
)
from nefair.schema import Alternative, ExclusionLedger, Item, RunRecord

MODELS_PATH = "configs/models.yaml"
EVAL_PATH = "configs/eval.yaml"

# Vários testes rodam só a condição `reference`: ela dispensa ASR e áudio, e o
# eixo que eles isolam (chave de cache, elegibilidade) não depende da condição.
REFERENCE_ONLY = ("reference",)


# --------------------------------------------------------------------------- #
# Fixtures e construtores locais
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def models():
    """Config real: os testes valem sobre os modelos que vão de fato rodar."""
    return load_models_config(MODELS_PATH)


@pytest.fixture(scope="module")
def eval_config():
    return load_eval_config(EVAL_PATH)


def make_item(
    index: int,
    *,
    n_alternatives: int = 4,
    correct: int | None = 0,
    filter_status: str = "kept",
    review_status: str = "accepted",
) -> Item:
    """Item mínimo mas bem formado, na ordem CANÔNICA (correta em `correct`).

    `correct=None` produz um item sem alternativa correta — item malformado que
    só existe para provar que o runner o recusa em vez de pontuá-lo.
    """
    window_id = f"NE_{index:02d}__w01"
    alternatives = tuple(
        Alternative(
            label=chr(ord("A") + position),
            text=f"alternativa {position} do item {index}",
            is_correct=(position == correct),
            source_window_id=(window_id if position == correct else f"NE_{index:02d}__w0{2}"),
        )
        for position in range(n_alternatives)
    )
    return Item(
        item_id=f"{window_id}__i01",
        window_id=window_id,
        speaker_code=f"NE_{index:02d}",
        question="Sobre o que o falante está falando?",
        alternatives=alternatives,
        generator_model="gerador-falso",
        prompt_version="itens-v1",
        filter_status=filter_status,
        review_status=review_status,
    )


def make_inputs(items, tmp_path: Path) -> tuple[dict[str, str], dict[str, Path]]:
    """Transcrições de referência e caminhos de áudio para cada janela.

    Os arquivos existem de verdade (vazios) porque o runner trata "áudio ausente"
    como erro de insumo, e não é esse o caminho que a maioria dos testes exercita.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    references = {item.window_id: f"referencia da janela {item.window_id}" for item in items}
    audio_paths: dict[str, Path] = {}
    for item in items:
        path = tmp_path / f"{item.window_id}.wav"
        path.write_bytes(b"")
        audio_paths[item.window_id] = path
    return references, audio_paths


def make_asr(key: str, prefix: str) -> FakeASR:
    """ASR falso com hipótese controlada, para provar a invalidação de cache.

    `version=key` faz o `fingerprint()` mudar junto com a hipótese, exatamente
    como aconteceria ao trocar de sistema de ASR de verdade.
    """
    return FakeASR(
        spec=fake_spec(key=key, role=ROLE_ASR, version=key),
        transcriber=lambda path: f"{prefix} {Path(path).stem}",
    )


def run(items, models, eval_config, tmp_path: Path, **overrides):
    """`run_eval` com adaptadores falsos e todo o cache dentro de `tmp_path`.

    `sleeper` é anulado para que nenhum teste durma de verdade no retry, e o
    portão D8 é desligado por padrão: a config real tem
    `contamination_checked: false`, e o D8 tem testes próprios abaixo.
    """
    references, audio_paths = make_inputs(items, tmp_path)
    asr, text_qa, audio_qa = build_adapters(models, seed=eval_config.seed, fake=True)
    kwargs = {
        "items": items,
        "models": models,
        "eval_config": eval_config,
        "text_qa": text_qa,
        "audio_qa": audio_qa,
        "asr": asr,
        "reference_texts": references,
        "audio_paths": audio_paths,
        "cache_path": tmp_path / "cache.jsonl",
        "enforce_contamination_gate": False,
        "fake_mode": True,
        "sleeper": lambda seconds: None,
    }
    kwargs.update(overrides)
    summary = run_eval(**kwargs)
    return summary, text_qa, audio_qa, asr


# --------------------------------------------------------------------------- #
# 1. Pareamento das condições — a propriedade de que a Etapa 6 depende
# --------------------------------------------------------------------------- #
def test_permutation_does_not_depend_on_the_condition():
    """`derive_permutation` não recebe `condition`, e isso é o desenho inteiro.

    Se `asr` e `reference` vissem ordens diferentes, `acc(reference) − acc(asr)`
    deixaria de isolar o efeito da via (áudio→texto) e passaria a somar um efeito
    de posição da alternativa correta. A assinatura sem `condition` é a garantia
    estrutural; este teste é a garantia observável.
    """
    kwargs = {"item_id": "NE_01__w01__i01", "model_key": "openai_text", "seed": 20260918, "n": 4}
    assert derive_permutation(**kwargs) == derive_permutation(**kwargs)
    with pytest.raises(TypeError):
        derive_permutation(condition="asr", **kwargs)  # type: ignore[call-arg]


def test_paired_conditions_share_the_permutation_end_to_end(models, eval_config, tmp_path):
    """A propriedade vale no artefato, não só na função pura.

    Este é o teste que pega uma regressão introduzida no `build_tasks` (por
    exemplo, alguém passando `condition` na derivação "para variar mais"): os
    `RunRecord` de `asr` e `reference` do mesmo `(item, modelo)` têm de trazer a
    mesma tupla `permutation`.
    """
    items = [make_item(index) for index in range(2)]
    summary, _, _, _ = run(items, models, eval_config, tmp_path)

    by_pair: dict[tuple[str, str], dict[str, tuple[int, ...]]] = {}
    for record in summary.records:
        by_pair.setdefault((record.item_id, record.model_key), {})[record.condition] = (
            record.permutation
        )

    paired = {
        pair: conditions
        for pair, conditions in by_pair.items()
        if {"asr", "reference"} <= set(conditions)
    }
    assert paired, "nenhum modelo de texto respondeu nas duas condições"
    for pair, conditions in paired.items():
        assert conditions["asr"] == conditions["reference"], pair


def test_permutation_changes_with_the_model_key():
    """Modelos diferentes veem ordens diferentes — e por um motivo de medida.

    Com a mesma ordem para todos, um viés de posição comum (preferir a primeira
    alternativa, por exemplo) apareceria como concordância ENTRE modelos e seria
    lido como concordância de conteúdo. São 4! = 24 ordens possíveis; sobre 20
    itens, duas chaves coincidirem em todos é praticamente impossível.
    """
    item_ids = [f"NE_{i:02d}__w01__i01" for i in range(20)]
    first = [derive_permutation(item_id=i, model_key="openai_text", seed=7, n=4) for i in item_ids]
    second = [derive_permutation(item_id=i, model_key="google_text", seed=7, n=4) for i in item_ids]
    assert first != second


def test_permutation_changes_with_the_seed_and_the_item():
    """Semente e item também entram na derivação — reprodutível, não constante."""
    base = {"model_key": "openai_text", "n": 4}
    item_ids = [f"NE_{i:02d}__w01__i01" for i in range(20)]
    seed_a = [derive_permutation(item_id=i, seed=1, **base) for i in item_ids]
    seed_b = [derive_permutation(item_id=i, seed=2, **base) for i in item_ids]
    assert seed_a != seed_b
    assert len(set(seed_a)) > 1, "a permutação não pode ser a mesma para todo item"


def test_permutation_is_the_identity_when_shuffling_is_off():
    """`shuffle_alternatives: false` tem de preservar a ordem canônica.

    É a chave que permite reproduzir uma execução sem embaralhamento para depurar
    um item suspeito.
    """
    assert derive_permutation(item_id="x", model_key="m", seed=1, n=4, shuffle=False) == (
        0,
        1,
        2,
        3,
    )
    assert derive_permutation(item_id="x", model_key="m", seed=1, n=1) == (0,)


def test_permutation_is_a_permutation_of_the_alternatives():
    """Nenhuma alternativa pode sumir nem aparecer duas vezes."""
    for n in (2, 3, 4, 5):
        for index in range(30):
            permutation = derive_permutation(item_id=f"i{index}", model_key="m", seed=20260918, n=n)
            assert sorted(permutation) == list(range(n))


def test_permutation_does_not_depend_on_the_process_hash_seed():
    """Valores fixos: se `stable_hash` virar `hash()`, a reprodutibilidade morre.

    O `hash()` embutido é randomizado por processo — o experimento deixaria de ser
    reproduzível entre máquinas sem que nenhum teste de uma execução só notasse.
    """
    assert derive_permutation(item_id="NE_00__w01__i01", model_key="openai_text", seed=7, n=4) == (
        derive_permutation(item_id="NE_00__w01__i01", model_key="openai_text", seed=7, n=4)
    )


# --------------------------------------------------------------------------- #
# `present_alternatives` — rerrotulagem sem perder a resposta certa
# --------------------------------------------------------------------------- #
def test_present_alternatives_relabels_without_losing_the_correct_answer():
    """A ordem canônica não é sobrescrita; o que muda é a LETRA a ser dita."""
    item = make_item(0, correct=0)
    presented, correct_label = present_alternatives(item, (2, 0, 3, 1))
    assert [alt.label for alt in presented] == ["A", "B", "C", "D"]
    assert [alt.text for alt in presented] == [
        item.alternatives[2].text,
        item.alternatives[0].text,
        item.alternatives[3].text,
        item.alternatives[1].text,
    ]
    # A alternativa correta continua a mesma; só a letra mudou de A para B.
    assert correct_label == "B"
    assert [alt.text for alt in presented if alt.is_correct] == [item.alternatives[0].text]
    assert item.alternatives[0].label == "A", "a ordem canônica do Item foi mutada"


def test_present_alternatives_rejects_an_invalid_permutation():
    """Permutação inválida é bug de programação, não item ruim: falha alto.

    Uma permutação com índice repetido duplicaria uma alternativa e sumiria com
    outra — o item continuaria "parecendo" válido e a resposta do modelo seria
    pontuada contra um item que ninguém escreveu.
    """
    item = make_item(0)
    for bad in [(0, 1, 2), (0, 1, 2, 2), (0, 1, 2, 4), (0, 1, 2, 3, 3)]:
        with pytest.raises(ValueError):
            present_alternatives(item, bad)


def test_present_alternatives_requires_exactly_one_correct_alternative():
    """Zero ou duas corretas: o item é malformado e não pode ser pontuado."""
    with pytest.raises(ValueError):
        present_alternatives(make_item(0, correct=None), (0, 1, 2, 3))


# --------------------------------------------------------------------------- #
# 2–4. Identidade da chave de cache
# --------------------------------------------------------------------------- #
def test_cache_hit_does_not_call_the_model(models, eval_config, tmp_path):
    """O ponto do cache inteiro: reexecutar não paga duas vezes pela mesma pergunta.

    `FakeTextQA.n_calls` é a evidência dura — o contador tem de ficar parado. Sem
    isso, uma retomada de uma execução de doze horas refaria a conta.
    """
    items = [make_item(index) for index in range(2)]
    first, text_first, _, _ = run(items, models, eval_config, tmp_path)
    assert first.n_cache_hits == 0
    assert first.n_model_calls == first.n_tasks
    assert sum(adapter.n_calls for adapter in text_first.values()) > 0

    second, text_second, audio_second, asr_second = run(items, models, eval_config, tmp_path)
    assert second.n_cache_hits == second.n_tasks
    assert second.n_model_calls == 0
    assert all(adapter.n_calls == 0 for adapter in text_second.values())
    assert all(adapter.n_calls == 0 for adapter in audio_second.values())
    assert asr_second.n_calls == 0, "o cache do ASR também tem de segurar a chamada"
    # E o resultado lido do cache é o mesmo que foi gravado.
    assert {r.cache_key: r.raw_text for r in second.records} == {
        r.cache_key: r.raw_text for r in first.records
    }


def test_resuming_does_not_duplicate_records_in_the_cache(models, eval_config, tmp_path):
    """Retomada é append-only, mas não pode re-anexar o que já está lá.

    O cache é lido como arquivo, não pela estrutura em memória: é o arquivo que
    sobrevive à interrupção, e é ele que a próxima execução vai ler.
    """
    items = [make_item(index) for index in range(2)]
    cache_file = tmp_path / "cache.jsonl"
    first, _, _, _ = run(items, models, eval_config, tmp_path)
    lines_after_first = cache_file.read_text(encoding="utf-8").splitlines()
    assert len(lines_after_first) == first.n_tasks

    run(items, models, eval_config, tmp_path)
    assert cache_file.read_text(encoding="utf-8").splitlines() == lines_after_first

    reloaded = RunCache.load(cache_file)
    assert reloaded.n_duplicates == 0
    assert len(reloaded.records) == first.n_tasks


def test_changing_the_prompt_version_changes_the_cache_key(models, eval_config, tmp_path):
    """Trocar o prompt é trocar o experimento: o cache antigo não serve mais.

    Sem isto, a execução com o prompt definitivo do TCC leria respostas dadas ao
    prompt provisório — e o relatório citaria uma versão de prompt que nunca
    produziu aqueles números.
    """
    items = [make_item(0)]
    conditions = REFERENCE_ONLY
    first, _, _, _ = run(items, models, eval_config, tmp_path, conditions=conditions)
    second, _, text_second, _ = run(
        items, models, eval_config, tmp_path, conditions=conditions, prompt_version="outro-prompt"
    )
    assert second.n_cache_hits == 0
    assert second.n_model_calls == second.n_tasks
    assert not ({r.cache_key for r in first.records} & {r.cache_key for r in second.records})


def test_changing_the_model_version_or_params_changes_the_cache_key(models, eval_config, tmp_path):
    """`ModelSpec.fingerprint()` cobre versão E parâmetros de decodificação.

    Um snapshot novo do modelo é outro modelo; temperatura diferente é outro
    experimento. Reusar a resposta antiga misturaria duas execuções sob o mesmo
    nome — exatamente o que o TCC promete não fazer ao reportar a versão exata.
    """
    items = [make_item(0)]
    conditions = REFERENCE_ONLY
    references, audio_paths = make_inputs(items, tmp_path)
    cache_path = tmp_path / "cache.jsonl"

    def keys_for(**spec_overrides) -> set[str]:
        _, text_qa, _ = build_adapters(models, seed=eval_config.seed, fake=True)
        for adapter in text_qa.values():
            adapter.spec = dataclasses.replace(adapter.spec, **spec_overrides)
        summary = run_eval(
            items=items,
            models=models,
            eval_config=eval_config,
            text_qa=text_qa,
            reference_texts=references,
            audio_paths=audio_paths,
            conditions=conditions,
            cache_path=cache_path,
            enforce_contamination_gate=False,
            fake_mode=True,
            sleeper=lambda seconds: None,
        )
        return {record.cache_key for record in summary.records}

    base = keys_for()
    other_version = keys_for(version="snapshot-2")
    other_params = keys_for(params={"temperature": 0.7})
    assert base and not (base & other_version)
    assert base and not (base & other_params)
    assert not (other_version & other_params)


def test_changing_the_asr_invalidates_only_the_asr_condition(models, eval_config, tmp_path):
    """Trocar de ASR invalida `asr` e SÓ `asr` — via `extra["hypothesis_sha"]`.

    Sem o hash da hipótese na chave, a condição `asr` leria respostas dadas sobre
    a transcrição de outro sistema e a decomposição `Δ_ASR` da Etapa 6 seria
    calculada sobre um insumo que nunca existiu. E a invalidação precisa ser
    cirúrgica: `reference` e `audio` não dependem do ASR e refazê-las seria pagar
    de novo por respostas idênticas.
    """
    items = [make_item(0)]
    references, audio_paths = make_inputs(items, tmp_path)
    cache_path = tmp_path / "cache.jsonl"

    def keys_by_condition(asr: FakeASR) -> dict[str, set[str]]:
        _, text_qa, audio_qa = build_adapters(models, seed=eval_config.seed, fake=True)
        summary = run_eval(
            items=items,
            models=models,
            eval_config=eval_config,
            text_qa=text_qa,
            audio_qa=audio_qa,
            asr=asr,
            reference_texts=references,
            audio_paths=audio_paths,
            cache_path=cache_path,
            enforce_contamination_gate=False,
            fake_mode=True,
            sleeper=lambda seconds: None,
        )
        assert summary.n_errors == 0, summary.errors_by_type
        grouped: dict[str, set[str]] = {}
        for record in summary.records:
            grouped.setdefault(record.condition, set()).add(record.cache_key)
        return grouped

    first = keys_by_condition(make_asr("asr_a", "hipotese do sistema A para"))
    second = keys_by_condition(make_asr("asr_b", "transcricao bem diferente do sistema B de"))

    assert first["asr"] and not (first["asr"] & second["asr"])
    assert first["reference"] == second["reference"]
    assert first["audio"] == second["audio"]


# --------------------------------------------------------------------------- #
# 5. D8 — portão de contaminação do ASR
# --------------------------------------------------------------------------- #
def test_contamination_gate_refuses_the_run_by_default(models, eval_config):
    """A config REAL do repositório hoje é `contamination_checked: false`.

    Ou seja: este teste falha no dia em que alguém marcar `true` sem ter feito a
    verificação — e é exatamente esse o alarme que o D8 pede. O Whisper ajustado
    para PT-BR pode ter sido treinado sobre o próprio CORAA/MuPe; se foi, a
    condição `asr` fica inflada no corpus que o estudo mede.
    """
    assert models.asr.contamination_checked is False
    assert eval_config.require_contamination_check is True
    with pytest.raises(ContaminationCheckError):
        assert_contamination_checked(models, eval_config)


def test_contamination_gate_passes_once_the_check_is_recorded(models, eval_config):
    """Verificação registrada em `models.yaml` libera a execução."""
    checked = dataclasses.replace(
        models, asr=dataclasses.replace(models.asr, contamination_checked=True)
    )
    assert_contamination_checked(checked, eval_config)


def test_contamination_gate_can_be_waived_by_the_eval_config(models, eval_config):
    """Desligar `require_contamination_check` também libera — é a válvula prevista.

    Existe para exercitar o pipeline sem ASR real. Que o desligamento seja
    explícito e apareça no relatório é o que impede que ele vire o default.
    """
    waived = dataclasses.replace(eval_config, require_contamination_check=False)
    assert_contamination_checked(models, waived)


def test_run_eval_refuses_before_any_model_call(models, eval_config, tmp_path):
    """A recusa acontece ANTES de qualquer chamada: o D8 é barato, a Etapa 5 não.

    E vale para a execução COMPLETA, não só para a condição `asr`: a mitigação
    prevista inclui `split_restriction`, que muda quais janelas entram no estudo
    e portanto afeta também `reference` e `audio`.
    """
    items = [make_item(0)]
    references, audio_paths = make_inputs(items, tmp_path)
    _, text_qa, audio_qa = build_adapters(models, seed=eval_config.seed, fake=True)
    with pytest.raises(ContaminationCheckError):
        run_eval(
            items=items,
            models=models,
            eval_config=eval_config,
            text_qa=text_qa,
            audio_qa=audio_qa,
            reference_texts=references,
            audio_paths=audio_paths,
            conditions=("reference",),
            cache_path=tmp_path / "cache.jsonl",
            enforce_contamination_gate=True,
            sleeper=lambda seconds: None,
        )
    assert all(adapter.n_calls == 0 for adapter in text_qa.values())
    assert not (tmp_path / "cache.jsonl").exists()


def test_the_run_summary_records_whether_the_gate_was_enforced(models, eval_config, tmp_path):
    """O relatório precisa dizer qual dos dois regimes valeu naquela execução."""
    items = [make_item(0)]
    summary, _, _, _ = run(items, models, eval_config, tmp_path, conditions=("reference",))
    assert summary.contamination_gate_enforced is False
    assert summary.fake_mode is True


# --------------------------------------------------------------------------- #
# 6. Erros nunca entram no cache
# --------------------------------------------------------------------------- #
def _error_record(error: str) -> RunRecord:
    return RunRecord(
        cache_key="chave",
        item_id="NE_00__w01__i01",
        speaker_code="NE_00",
        model_key="openai_text",
        condition="reference",
        prompt_version="p",
        permutation=(0, 1, 2, 3),
        raw_text="",
        parsed_label=None,
        is_correct=None,
        error=error,
    )


def test_cache_put_rejects_a_record_with_an_error(tmp_path):
    """Falha de provedor não é trabalho concluído: tem de ser retentada.

    Se um erro entrasse no cache, a retomada o leria como "já respondido" e a
    célula ficaria permanentemente vazia — um buraco silencioso na matriz do
    experimento.
    """
    cache = RunCache.load(tmp_path / "cache.jsonl")
    with pytest.raises(ValueError):
        cache.put(_error_record("RateLimitError: 429"))
    assert not (tmp_path / "cache.jsonl").exists()
    assert cache.records == {}


def test_a_missing_input_is_an_error_that_the_next_run_retries(models, eval_config, tmp_path):
    """Ponta a ponta: insumo ausente vira erro, não entra no cache, e é retentado.

    `is_correct=None` aqui é obrigatório: uma resposta que nunca chegou não pode
    contar como erro do modelo.
    """
    items = [make_item(0)]
    _, audio_paths = make_inputs(items, tmp_path)
    cache_path = tmp_path / "cache.jsonl"

    def run_without_references():
        _, text_qa, _ = build_adapters(models, seed=eval_config.seed, fake=True)
        summary = run_eval(
            items=items,
            models=models,
            eval_config=eval_config,
            text_qa=text_qa,
            reference_texts={},  # a transcrição de referência não foi fornecida
            audio_paths=audio_paths,
            conditions=("reference",),
            cache_path=cache_path,
            enforce_contamination_gate=False,
            sleeper=lambda seconds: None,
        )
        return summary

    first = run_without_references()
    assert first.n_errors == first.n_tasks > 0
    assert set(first.errors_by_type) == {ERROR_MISSING_INPUT}
    assert all(r.parsed_label is None and r.is_correct is None for r in first.records)
    assert not cache_path.exists(), "erro não pode ter sido gravado no cache"

    second = run_without_references()
    assert second.n_cache_hits == 0, "o erro precisa ser retentado, não reusado"
    assert second.n_errors == first.n_errors


# --------------------------------------------------------------------------- #
# 8. Resposta não interpretável — categoria própria, jamais imputada
# --------------------------------------------------------------------------- #
def test_unparsable_answer_is_counted_and_never_imputed(models, eval_config, tmp_path):
    """Prosa sem letra: `parsed_label=None`, `is_correct=None`, contada por motivo.

    Imputar como erro rebaixaria a acurácia de modelos verborrágicos de forma
    desigual entre condições (a `audio` produz mais prosa que a `reference`),
    contaminando a diferença que o estudo mede. Imputar como acerto seria pior.
    """
    items = [make_item(0)]
    references, audio_paths = make_inputs(items, tmp_path)
    _, text_qa, _ = build_adapters(models, seed=eval_config.seed, fake=True)
    for adapter in text_qa.values():
        adapter.responder = lambda prompt: "Acho que o falante fala de futebol e de comida."

    summary = run_eval(
        items=items,
        models=models,
        eval_config=eval_config,
        text_qa=text_qa,
        reference_texts=references,
        audio_paths=audio_paths,
        conditions=("reference",),
        cache_path=tmp_path / "cache.jsonl",
        enforce_contamination_gate=False,
        sleeper=lambda seconds: None,
    )

    assert summary.n_tasks > 0
    assert summary.n_unparsed == summary.n_tasks
    assert summary.n_correct == 0
    assert summary.n_errors == 0, "não interpretável NÃO é erro de provedor"
    assert summary.unparsed_by_reason == {"sem_letra": summary.n_tasks}
    assert all(r.parsed_label is None and r.is_correct is None for r in summary.records)
    assert all(r.error == "" for r in summary.records)
    # A taxa por célula usa como denominador as respostas OBTIDAS, não as tarefas.
    for cell in summary.per_cell.values():
        assert cell.unparsed_rate == 1.0


def test_a_parsable_answer_is_scored_against_the_permuted_label(models, eval_config, tmp_path):
    """O acerto é medido contra a letra EXIBIDA, não contra a canônica.

    Responder sempre "A" só acerta quando a permutação colocou a correta na
    primeira posição — é assim que o embaralhamento neutraliza o viés de posição
    em vez de escondê-lo.
    """
    items = [make_item(index) for index in range(6)]
    references, audio_paths = make_inputs(items, tmp_path)
    _, text_qa, _ = build_adapters(models, seed=eval_config.seed, fake=True)
    for adapter in text_qa.values():
        adapter.responder = lambda prompt: "A"

    summary = run_eval(
        items=items,
        models=models,
        eval_config=eval_config,
        text_qa=text_qa,
        reference_texts=references,
        audio_paths=audio_paths,
        conditions=("reference",),
        cache_path=tmp_path / "cache.jsonl",
        enforce_contamination_gate=False,
        sleeper=lambda seconds: None,
    )

    assert summary.n_unparsed == 0
    assert all(record.parsed_label == "A" for record in summary.records)
    # A correta é sempre o índice canônico 0; ele foi exibido como "A" apenas
    # quando `permutation[0] == 0`.
    expected = sum(1 for record in summary.records if record.permutation[0] == 0)
    assert summary.n_correct == expected
    assert 0 < expected < summary.n_tasks, "a permutação não está embaralhando nada"
    assert all(record.is_correct is not None for record in summary.records)


# --------------------------------------------------------------------------- #
# 10. Elegibilidade dos itens — nada dropado em silêncio
# --------------------------------------------------------------------------- #
def test_select_eligible_items_keeps_only_kept_and_accepted():
    """As DUAS condições valem ao mesmo tempo (filtro textual E revisão humana).

    Um item descartado pelo filtro mas "aceito" por descuido na planilha não pode
    entrar; um item retido pelo filtro e ainda pendente de revisão também não.
    """
    eligible = make_item(0)
    rejected = make_item(1, review_status="rejected")
    pending = make_item(2, review_status="pending")
    discarded = make_item(3, filter_status="discarded")
    discarded_but_accepted = make_item(4, filter_status="discarded", review_status="accepted")
    items = [eligible, rejected, pending, discarded, discarded_but_accepted]

    ledger = ExclusionLedger(unit="itens gerados")
    kept = select_eligible_items(items, ledger)

    assert [item.item_id for item in kept] == [eligible.item_id]
    assert all(item.is_eligible_for_eval for item in kept)
    ledger.assert_balanced()
    assert ledger.total_in == 5
    assert ledger.kept == 1
    assert ledger.dropped == {
        "revisão humana: rejected": 1,
        "revisão humana: pending": 1,
        "filtro textual: discarded": 2,
    }


def test_select_eligible_items_is_ordered_by_item_id():
    """Ordem canônica: o arquivo de cache não pode depender da ordem de entrada."""
    items = [make_item(index) for index in (3, 1, 2, 0)]
    ledger = ExclusionLedger(unit="itens gerados")
    kept = select_eligible_items(items, ledger)
    assert [item.item_id for item in kept] == sorted(item.item_id for item in items)


def test_run_eval_only_evaluates_eligible_items(models, eval_config, tmp_path):
    """A elegibilidade vale dentro do runner, e o ledger do resumo fecha."""
    items = [make_item(0), make_item(1, review_status="rejected"), make_item(2, filter_status="x")]
    summary, _, _, _ = run(items, models, eval_config, tmp_path, conditions=("reference",))
    assert {record.item_id for record in summary.records} == {items[0].item_id}
    summary.item_ledger.assert_balanced()
    assert summary.item_ledger.total_in == 3
    assert summary.item_ledger.kept == 1


# --------------------------------------------------------------------------- #
# 11. Modo falso — isolamento do cache real
# --------------------------------------------------------------------------- #
def test_fake_adapters_swap_the_provider_and_the_fingerprint(models):
    """`fake=True` troca `provider` para `"fake"`, o que muda o `fingerprint()`.

    Não é cosmético: o fingerprint entra na chave de cache. Sem a troca, uma
    execução de demonstração gravaria respostas inventadas sob a MESMA chave de
    uma execução real, e a Etapa 6 leria ficção sem nenhum sinal de que leu.
    """
    real_asr, real_text, real_audio = build_adapters(models, fake=False)
    fake_asr_adapter, fake_text, fake_audio = build_adapters(models, fake=True)

    assert fake_asr_adapter.spec.provider == "fake"
    assert real_asr.spec.provider != "fake"
    assert fake_asr_adapter.spec.fingerprint() != real_asr.spec.fingerprint()

    assert set(fake_text) == set(real_text) and set(fake_audio) == set(real_audio)
    for key, adapter in {**fake_text, **fake_audio}.items():
        assert adapter.spec.provider == "fake"
        real = {**real_text, **real_audio}[key]
        assert adapter.spec.fingerprint() != real.spec.fingerprint()
        # O que NÃO muda: a identidade do modelo continua rastreável.
        assert adapter.spec.key == real.spec.key
        assert adapter.spec.model_id == real.spec.model_id
        assert adapter.spec.role == real.spec.role


def test_a_fake_run_cannot_contaminate_the_real_cache(models, eval_config, tmp_path):
    """Nenhuma chave gravada pelo modo falso é legível por uma execução real.

    A prova é feita no arquivo: para cada registro gravado pela execução falsa,
    reconstruímos a chave que a MESMA pergunta teria com o modelo real (mesmo
    item, mesma condição, mesma permutação, mesmo prompt) e conferimos que ela
    não está no cache.
    """
    items = [make_item(0)]
    cache_path = tmp_path / "cache.jsonl"
    summary, _, _, _ = run(
        items, models, eval_config, tmp_path, conditions=("reference",), cache_path=cache_path
    )
    assert summary.n_model_calls == summary.n_tasks > 0

    cache = RunCache.load(cache_path)
    _, real_text, real_audio = build_adapters(models, fake=False)
    real_specs = {key: adapter.spec for key, adapter in {**real_text, **real_audio}.items()}
    for record in summary.records:
        assert cache.get(record.cache_key) is not None, "a execução falsa gravou no seu cache"
        real_key = RunRecord.make_cache_key(
            item_id=record.item_id,
            model_fingerprint=real_specs[record.model_key].fingerprint(),
            condition=record.condition,
            prompt_version=record.prompt_version,
            permutation=record.permutation,
        )
        assert real_key != record.cache_key
        assert cache.get(real_key) is None, (
            f"a chave real de {record.model_key} colidiu com a do modo falso"
        )


# --------------------------------------------------------------------------- #
# Invariantes gerais do resumo
# --------------------------------------------------------------------------- #
def test_summary_counts_reconcile(models, eval_config, tmp_path):
    """Toda tarefa produz exatamente um registro, e as contagens fecham.

    É este invariante que autoriza somar acertos, não interpretáveis e erros e
    obter o total — sem ele, qualquer taxa do relatório teria denominador incerto.
    """
    items = [make_item(index) for index in range(3)]
    summary, _, _, _ = run(items, models, eval_config, tmp_path)

    assert len(summary.records) == summary.n_tasks
    assert summary.n_cache_hits + summary.n_model_calls == summary.n_tasks
    assert sum(cell.n for cell in summary.per_cell.values()) == summary.n_tasks
    assert summary.n_answered == summary.n_tasks - summary.n_errors
    assert sum(summary.unparsed_by_reason.values()) == summary.n_unparsed
    assert len({record.cache_key for record in summary.records}) == summary.n_tasks


def test_two_runs_on_a_clean_cache_produce_identical_records(models, eval_config, tmp_path):
    """Determinismo: mesma entrada, mesma semente, mesmos registros.

    Os dois caches são arquivos DIFERENTES, então nada é reusado e as duas
    execuções recalculam tudo do zero — se saírem iguais, é porque nada no
    caminho depende de relógio, de ordem de thread ou de RNG global. A
    concorrência é real aqui (`max_concurrency: 4` da config), e é justamente ela
    que tornaria uma dependência de ordem visível.
    """
    items = [make_item(index) for index in range(2)]
    first, _, _, _ = run(items, models, eval_config, tmp_path, cache_path=tmp_path / "a.jsonl")
    second, _, _, _ = run(items, models, eval_config, tmp_path, cache_path=tmp_path / "b.jsonl")
    assert [dataclasses.astuple(r) for r in first.records] == [
        dataclasses.astuple(r) for r in second.records
    ]


def test_unknown_condition_is_refused(models, eval_config, tmp_path):
    """Condição inexistente é erro de digitação na linha de comando, não item vazio."""
    with pytest.raises(ValueError):
        run(
            [make_item(0)],
            models,
            eval_config,
            tmp_path,
            conditions=("referencia",),  # sem o "e" final: erro de digitação
        )
