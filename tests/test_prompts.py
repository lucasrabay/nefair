"""Testes dos templates de prompt (Etapas 3 e 5) — puros, offline, byte a byte.

`PROMPT_VERSION` entra na chave de cache do runner. Se alguém editar um byte de
um template sem incrementar a versão, o cache devolve respostas dadas a OUTRO
prompt e a comparação entre modelos passa a misturar duas perguntas diferentes
sob o mesmo nome. Os snapshots literais abaixo existem para que essa edição
quebre a suíte em vez de contaminar os resultados em silêncio: o texto esperado
está escrito aqui inteiro, à mão, e não é derivado de nenhuma estrutura de
`prompts.py` — testar o template com as listas do próprio template provaria
apenas que o código concorda consigo mesmo.

Quebrar o snapshot é, portanto, um evento com procedimento: mudar o texto E
incrementar `PROMPT_VERSION` E invalidar o cache. Atualizar só o snapshot é o
erro que este arquivo tenta tornar impossível de cometer distraidamente.

Os três modos de `render_answer_prompt` também são fixados aqui porque eles são
o contraste do §D3: o controle textual (sem áudio e sem transcrição) só mede
respondibilidade sem fala se avisar explicitamente que a fala não veio — sem o
aviso, o modelo tende a recusar, e o filtro mediria recusa em vez de item ruim.
"""

from __future__ import annotations

import pytest

from nefair.models.prompts import (
    GENERATION_RESPONSE_ROOT_KEY,
    PROMPT_VERSION,
    labels_for,
    render_answer_prompt,
    render_generation_prompt,
)
from nefair.schema import ALTERNATIVE_LABELS

# --------------------------------------------------------------------------- #
# Entradas fixas dos snapshots
# --------------------------------------------------------------------------- #
TARGET_WINDOW_ID = "NE_LONG__w03"
TARGET_TRANSCRIPT = "eu fui pra feira de manha e comprei macaxeira"
CONTEXT_EXCERPTS = (
    ("NE_LONG__w01", "ai eu trabalhava na roca com meu pai"),
    ("NE_LONG__w07", "depois a gente mudou pra joao pessoa"),
)
QUESTION = "O que o falante diz que fez de manha?"
ALTERNATIVE_TEXTS = (
    "ele foi a feira de manha",
    "ele trabalhava na roca com o pai",
    "ele mudou pra joao pessoa",
    "ele ficou em casa o dia todo",
)

# Linha literal emitida quando a entrevista não tem outras janelas (§D2): o
# prompt precisa DIZER que não há contexto, senão o gerador inventa distratores
# de conhecimento geral e a proveniência vira ficção.
NO_CONTEXT_LINE = "- (nenhum trecho de contexto disponível para esta entrevista)"

RULE_3_WITH_PROVENANCE = (
    "3. Toda alternativa tem de declarar `source_window_id`: a correta usa o id da "
    "janela-alvo; cada distrator usa o id do trecho de contexto em que se ancorou."
)
RULE_3_WITHOUT_PROVENANCE = (
    "3. O campo `source_window_id` é opcional nesta execução; quando presente, use o "
    "id da janela de onde a alternativa veio."
)

# --------------------------------------------------------------------------- #
# Snapshots literais (as quebras de linha são as do template, não as do arquivo:
# cada elemento da tupla é UMA linha do prompt; a concatenação implícita existe
# só para caber em 100 colunas)
# --------------------------------------------------------------------------- #
EXPECTED_GENERATION = "\n".join(
    (
        "Você é um especialista em elaboração de itens de compreensão de fala "
        "espontânea em português brasileiro.",
        "",
        "TAREFA",
        "Leia a TRANSCRIÇÃO-ALVO abaixo e escreva exatamente 1 item(ns) de múltipla "
        "escolha, cada um com 4 alternativas: 1 correta e 3 incorretas.",
        "",
        "REGRAS",
        "1. A alternativa correta tem de ser verificável APENAS na TRANSCRIÇÃO-ALVO, "
        "sem conhecimento geral e sem inferência sobre o resto da entrevista.",
        "2. Cada uma das 3 alternativas incorretas (distratores) tem de ser ancorada em"
        " um dos TRECHOS DE CONTEXTO, que vêm de OUTRAS janelas da MESMA entrevista.",
        "3. Toda alternativa tem de declarar `source_window_id`: a correta usa o id da "
        "janela-alvo; cada distrator usa o id do trecho de contexto em que se ancorou.",
        "4. Duas alternativas do mesmo item nunca podem ter o mesmo texto, nem texto vazio.",
        "5. A pergunta não pode ser respondível só olhando as alternativas: não use "
        "pistas de formato (alternativa mais longa, mais específica ou mais plausível) "
        "nem fatos de conhecimento geral.",
        "6. Escreva pergunta e alternativas em português brasileiro.",
        "",
        "TRANSCRIÇÃO-ALVO (janela NE_LONG__w03)",
        "eu fui pra feira de manha e comprei macaxeira",
        "",
        "TRECHOS DE CONTEXTO (outras janelas da mesma entrevista)",
        "- (janela NE_LONG__w01) ai eu trabalhava na roca com meu pai",
        "- (janela NE_LONG__w07) depois a gente mudou pra joao pessoa",
        "",
        "FORMATO DA RESPOSTA",
        "Responda com UM ÚNICO objeto JSON, sem cercas de código, sem comentários e sem"
        " texto antes ou depois:",
        "{",
        '  "items": [',
        "    {",
        '      "question": "...",',
        '      "alternatives": [',
        '        {"text": "...", "is_correct": true, "source_window_id": "<id da janela>"}',
        "      ]",
        "    }",
        "  ]",
        "}",
        "Exatamente 1 objeto(s) em `items`, exatamente 4 objeto(s) em `alternatives` e "
        "exatamente 1 com `is_correct` verdadeiro.",
        "Não escreva os rótulos (A, B, C, D) dentro de `text`: a ordem das alternativas"
        " é embaralhada depois.",
    )
)

EXPECTED_ANSWER_CONTROL = "\n".join(
    (
        "Responda à pergunta de múltipla escolha abaixo.",
        "",
        "CONTEXTO",
        "Você NÃO recebeu o áudio nem a transcrição da fala. Ainda assim, escolha a "
        "alternativa que considerar mais provável.",
        "",
        "PERGUNTA",
        "O que o falante diz que fez de manha?",
        "",
        "ALTERNATIVAS",
        "A) ele foi a feira de manha",
        "B) ele trabalhava na roca com o pai",
        "C) ele mudou pra joao pessoa",
        "D) ele ficou em casa o dia todo",
        "",
        "Responda apenas com a letra da alternativa correta (A, B, C ou D), sem "
        "explicação e sem pontuação.",
    )
)

EXPECTED_ANSWER_TRANSCRIPT = "\n".join(
    (
        "Responda à pergunta de múltipla escolha abaixo.",
        "",
        "TRANSCRIÇÃO DA FALA",
        "eu fui pra feira de manha e comprei macaxeira",
        "",
        "PERGUNTA",
        "O que o falante diz que fez de manha?",
        "",
        "ALTERNATIVAS",
        "A) ele foi a feira de manha",
        "B) ele trabalhava na roca com o pai",
        "C) ele mudou pra joao pessoa",
        "D) ele ficou em casa o dia todo",
        "",
        "Responda apenas com a letra da alternativa correta (A, B, C ou D), sem "
        "explicação e sem pontuação.",
    )
)

EXPECTED_ANSWER_AUDIO = "\n".join(
    (
        "Responda à pergunta de múltipla escolha abaixo.",
        "",
        "ÁUDIO",
        "O áudio da fala foi fornecido junto com esta mensagem.",
        "",
        "PERGUNTA",
        "O que o falante diz que fez de manha?",
        "",
        "ALTERNATIVAS",
        "A) ele foi a feira de manha",
        "B) ele trabalhava na roca com o pai",
        "C) ele mudou pra joao pessoa",
        "D) ele ficou em casa o dia todo",
        "",
        "Responda apenas com a letra da alternativa correta (A, B, C ou D), sem "
        "explicação e sem pontuação.",
    )
)


def _generation(**overrides) -> str:
    """Render de geração com as entradas fixas do snapshot, salvo o que mudar."""
    kwargs = {
        "target_window_id": TARGET_WINDOW_ID,
        "target_transcript": TARGET_TRANSCRIPT,
        "context_excerpts": CONTEXT_EXCERPTS,
        "n_alternatives": 4,
    }
    kwargs.update(overrides)
    return render_generation_prompt(**kwargs)


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #
def test_generation_prompt_matches_the_literal_snapshot():
    """Se este teste cair, a versão do prompt tem de mudar junto com o texto."""
    assert _generation() == EXPECTED_GENERATION


def test_answer_prompt_control_matches_the_literal_snapshot():
    """Controle textual do §D3: sem áudio, sem transcrição e com o aviso explícito."""
    rendered = render_answer_prompt(question=QUESTION, alternative_texts=ALTERNATIVE_TEXTS)
    assert rendered == EXPECTED_ANSWER_CONTROL


def test_answer_prompt_with_transcript_matches_the_literal_snapshot():
    """Condições `asr` e `reference` da Etapa 5: só muda o bloco de transcrição."""
    rendered = render_answer_prompt(
        question=QUESTION,
        alternative_texts=ALTERNATIVE_TEXTS,
        transcript=TARGET_TRANSCRIPT,
    )
    assert rendered == EXPECTED_ANSWER_TRANSCRIPT


def test_answer_prompt_with_audio_matches_the_literal_snapshot():
    """Condição `audio` (multimodal nativo): o áudio vai anexo, o texto avisa."""
    rendered = render_answer_prompt(
        question=QUESTION,
        alternative_texts=ALTERNATIVE_TEXTS,
        audio_provided=True,
    )
    assert rendered == EXPECTED_ANSWER_AUDIO


def test_the_three_modes_differ_only_in_the_evidence_block():
    """Pergunta, alternativas e instrução final são idênticas nos três modos.

    É isso que permite atribuir a diferença de acurácia entre condições à
    evidência recebida, e não ao enunciado: se o texto da pergunta mudasse com a
    condição, a decomposição da Etapa 6 estaria comparando dois experimentos.
    """
    control = EXPECTED_ANSWER_CONTROL.split("\n")
    transcript = EXPECTED_ANSWER_TRANSCRIPT.split("\n")
    audio = EXPECTED_ANSWER_AUDIO.split("\n")
    tail = control[control.index("PERGUNTA") :]
    assert transcript[transcript.index("PERGUNTA") :] == tail
    assert audio[audio.index("PERGUNTA") :] == tail
    # E o aviso de ausência de fala aparece SÓ no controle.
    assert "CONTEXTO" in control
    assert "CONTEXTO" not in transcript
    assert "CONTEXTO" not in audio


def test_transcript_and_audio_together_drop_the_absence_warning():
    """Com qualquer evidência de fala presente, o aviso de ausência some.

    O aviso é sobre ausência; mantê-lo com áudio anexado seria mentir para o
    modelo e induzir recusa — exatamente o que o §D3 quer evitar.
    """
    both = render_answer_prompt(
        question=QUESTION,
        alternative_texts=ALTERNATIVE_TEXTS,
        transcript=TARGET_TRANSCRIPT,
        audio_provided=True,
    )
    assert "TRANSCRIÇÃO DA FALA" in both
    assert "ÁUDIO" in both
    assert "CONTEXTO" not in both.split("\n")


# --------------------------------------------------------------------------- #
# Rótulos (D10: o nº de alternativas é configurável)
# --------------------------------------------------------------------------- #
def test_labels_for_slices_the_canonical_labels():
    assert labels_for(4) == ("A", "B", "C", "D")
    assert labels_for(2) == ("A", "B")
    assert labels_for(5) == ALTERNATIVE_LABELS


def test_labels_for_rejects_a_single_alternative():
    """Item com uma alternativa não é múltipla escolha: erro alto, não silêncio."""
    with pytest.raises(ValueError, match="n_alternatives=1"):
        labels_for(1)


def test_labels_for_rejects_more_labels_than_exist():
    """Seis alternativas não têm rótulo canônico — melhor falhar que reciclar letra."""
    with pytest.raises(ValueError, match="n_alternatives=6"):
        labels_for(6)


def test_answer_prompt_refuses_a_degenerate_alternative_list():
    """A checagem do nº de alternativas vale também na hora de exibir o item."""
    with pytest.raises(ValueError):
        render_answer_prompt(question=QUESTION, alternative_texts=["única"])


# --------------------------------------------------------------------------- #
# D2 — contexto e proveniência
# --------------------------------------------------------------------------- #
def test_empty_context_emits_the_literal_absence_line():
    """Sem trechos de contexto, o prompt diz isso — não deixa a seção vazia.

    Uma seção vazia seria lida pelo gerador como "invente", e o distrator sem
    âncora é justamente o que o D2 proíbe.
    """
    rendered = _generation(context_excerpts=())
    lines = rendered.split("\n")
    header = lines.index("TRECHOS DE CONTEXTO (outras janelas da mesma entrevista)")
    assert lines[header + 1] == NO_CONTEXT_LINE
    assert lines[header + 2] == ""  # a seção tem exatamente essa linha


def test_require_provenance_rewrites_only_rule_three():
    """`require_provenance=False` afrouxa a regra 3 — e nada mais no prompt.

    O contraste é o que torna honesto o relatório: se a proveniência não foi
    exigida, é preciso poder dizer exatamente qual instrução deixou de valer.
    """
    strict = _generation(require_provenance=True).split("\n")
    loose = _generation(require_provenance=False).split("\n")
    assert len(strict) == len(loose)
    differing = [i for i, (a, b) in enumerate(zip(strict, loose, strict=True)) if a != b]
    assert len(differing) == 1, [(strict[i], loose[i]) for i in differing]
    assert strict[differing[0]] == RULE_3_WITH_PROVENANCE
    assert loose[differing[0]] == RULE_3_WITHOUT_PROVENANCE


# --------------------------------------------------------------------------- #
# Contrato com o parser e determinismo
# --------------------------------------------------------------------------- #
def test_prompt_version_and_root_key_are_the_versioned_contract():
    """A chave raiz vive em `prompts.py` para que prompt e parser não divirjam."""
    assert PROMPT_VERSION == "mcq-v1"
    assert GENERATION_RESPONSE_ROOT_KEY == "items"
    assert f'"{GENERATION_RESPONSE_ROOT_KEY}": [' in _generation()


def test_generation_prompt_counts_follow_the_arguments():
    """n_items e n_alternatives aparecem no texto: a sensibilidade do D10 é real."""
    rendered = _generation(n_alternatives=3, n_items=2)
    assert "escreva exatamente 2 item(ns)" in rendered
    assert "cada um com 3 alternativas: 1 correta e 2 incorretas" in rendered
    assert "Exatamente 2 objeto(s) em `items`, exatamente 3 objeto(s)" in rendered
    assert "(A, B, C) dentro de `text`" in rendered


def test_renders_are_pure_functions():
    """Mesma entrada, mesma string: é o que permite comparar artefatos byte a byte."""
    assert _generation() == _generation()
    first = render_answer_prompt(question=QUESTION, alternative_texts=ALTERNATIVE_TEXTS)
    second = render_answer_prompt(question=QUESTION, alternative_texts=ALTERNATIVE_TEXTS)
    assert first == second
