"""Testes da extração da letra (Etapa 5) — função pura, tudo offline.

O que está sob teste aqui não é "um parser": é o **número de respostas não
interpretáveis** do TCC. Se `parse_label` chutar uma letra onde não havia
evidência, esse chute entra na acurácia como acerto ou erro e desloca a
diferença NE×SE que o estudo mede; se recusar uma resposta legítima, infla a
taxa de não interpretável de forma desigual entre condições (a condição `audio`
produz mais prosa que a `reference`). Os dois erros são vieses de medida, não
detalhes de formatação — por isso cada caso abaixo fixa um comportamento
declarado no módulo, com a razão ao lado.

Convenção seguida: as asserções são contra a **resposta crua** e contra o
contrato público (`label`, `reason`, `strategy`, `candidates`), nunca contra as
regex internas. Um teste que casasse com as regex provaria apenas que o módulo
concorda consigo mesmo e quebraria em qualquer refatoração honesta.
"""

from __future__ import annotations

import pytest

from nefair.run.parse import (
    DEFAULT_LABELS,
    REASON_AMBIGUOUS,
    REASON_EMPTY,
    REASON_NO_LETTER,
    REASON_OUT_OF_VOCAB,
    REASONS,
    STRATEGIES,
    STRATEGY_ENUMERATED,
    STRATEGY_MARKER,
    STRATEGY_ONLY_LETTER,
    STRATEGY_UNIQUE_ISOLATED,
    parse_label,
)


# --------------------------------------------------------------------------- #
# Respostas interpretáveis
# --------------------------------------------------------------------------- #
def test_bare_letter_is_parsed():
    """Caso majoritário quando o prompt pede só a letra: o único inequívoco."""
    result = parse_label("C")
    assert result.label == "C"
    assert result.is_parsed
    assert result.strategy == STRATEGY_ONLY_LETTER
    assert result.reason == ""


def test_lowercase_letter_is_parsed_as_uppercase():
    """Caixa é ruído de formatação, não informação: `"c"` é a alternativa C.

    Normalizar aqui evita que a taxa de não interpretável de um modelo dependa
    de ele escrever em maiúscula — o que não tem relação nenhuma com o que o
    estudo mede.
    """
    assert parse_label("c").label == "C"


def test_letter_announced_by_a_marker_is_parsed():
    """`"Alternativa C."` — pontuação de fim de frase não desfaz a escolha."""
    result = parse_label("Alternativa C.")
    assert result.label == "C"
    assert result.strategy == STRATEGY_MARKER


def test_marker_beats_the_leading_article():
    """O caso clássico do português: a primeira letra do texto é o artigo "A".

    `"A resposta correta é a letra B"` só devolve `B` porque o marcador explícito
    vem ANTES das estratégias posicionais na cascata. Se a ordem for invertida
    numa refatoração, este teste cai — e cai porque o resultado passaria a ser um
    viés sistemático a favor da alternativa A em todo modelo verborrágico.
    """
    result = parse_label("A resposta correta é a letra B")
    assert result.label == "B"
    assert result.strategy == STRATEGY_MARKER
    assert result.candidates == ("B",)


def test_accents_do_not_block_the_marker():
    """O mesmo texto com e sem acento tem de dar o mesmo rótulo.

    Os marcadores são escritos sem acento no módulo ("opcao"), o que só funciona
    porque o texto é desacentuado antes da cascata. Um modelo que responde
    "a opção correta é a letra B" não pode ser punido por escrever português
    correto.
    """
    assert parse_label("A opção correta é a letra B").label == "B"
    assert parse_label("A opcao correta e a letra B").label == "B"


def test_enumerated_answer_is_parsed():
    """`"C) porque o falante diz..."` — o modelo repete a alternativa e explica."""
    result = parse_label("C) porque o falante diz que mora no interior")
    assert result.label == "C"
    assert result.strategy == STRATEGY_ENUMERATED


# --------------------------------------------------------------------------- #
# Respostas NÃO interpretáveis — cada uma com seu motivo
# --------------------------------------------------------------------------- #
def test_empty_answer_is_its_own_reason():
    """Resposta vazia é recusa ou corte de tokens, não uma escolha errada.

    Ter o motivo separado é o que permite dizer, no relatório, se o problema foi
    do modelo (recusou) ou da execução (limite de tokens).
    """
    result = parse_label("")
    assert result.label is None
    assert not result.is_parsed
    assert result.reason == REASON_EMPTY
    assert result.candidates == ()


def test_whitespace_only_answer_is_empty_not_no_letter():
    """Espaço em branco é resposta vazia; classificar como `sem_letra` mentiria."""
    assert parse_label("   \n\t ").reason == REASON_EMPTY


def test_two_plausible_letters_are_ambiguous_not_a_guess():
    """`"C ou D"`: escolher uma das duas seria inventar um dado.

    O módulo prefere um `None` contado a um palpite silencioso, e as duas
    candidatas ficam registradas para auditoria posterior.
    """
    result = parse_label("C ou D")
    assert result.label is None
    assert result.reason == REASON_AMBIGUOUS
    assert result.strategy == STRATEGY_UNIQUE_ISOLATED
    assert set(result.candidates) == {"C", "D"}


def test_letter_outside_the_vocabulary_is_reported_as_such():
    """`"Z"` não é erro do falante nem do item: é sinal de prompt errado.

    Separar este motivo de `sem_letra` é o que faz a categoria ser diagnóstica —
    uma taxa alta de `letra_fora_do_vocabulario` aponta para `n_alternatives` ou
    para o texto do prompt, não para a competência do modelo.
    """
    result = parse_label("Z")
    assert result.label is None
    assert result.reason == REASON_OUT_OF_VOCAB
    assert result.candidates == ("Z",)


def test_prose_without_any_letter_is_no_letter():
    """Prosa pura: o modelo ignorou a instrução de formato.

    O artigo "o" NÃO pode contar como letra fora do vocabulário — se contasse,
    quase toda prosa em português cairia em `letra_fora_do_vocabulario` e a
    categoria deixaria de significar qualquer coisa.
    """
    result = parse_label("Acho que o falante fala de futebol e de comida")
    assert result.label is None
    assert result.reason == REASON_NO_LETTER
    assert result.candidates == ()


def test_sentence_opening_article_is_not_a_letter():
    """ "O falante menciona..." começa com artigo, não com a alternativa O.

    Mesma razão do teste anterior, pelo lado do maiúsculo: um artigo em início de
    oração é palavra, não rótulo.
    """
    assert parse_label("O falante menciona futebol e times.").reason == REASON_NO_LETTER
    assert parse_label("E o falante ainda diz que gosta de praia.").reason == REASON_NO_LETTER


def test_refusal_is_no_letter_not_a_wrong_answer():
    """Uma recusa é não interpretável, jamais um erro imputado ao modelo."""
    result = parse_label("Não tenho como saber com base apenas nesse trecho.")
    assert result.label is None
    assert result.reason == REASON_NO_LETTER


# --------------------------------------------------------------------------- #
# Letras dentro de palavras e de códigos — os falsos positivos caros
# --------------------------------------------------------------------------- #
def test_certamente_does_not_become_c():
    """O falso positivo mais caro de todos: uma letra colada a uma palavra.

    `"Certamente"` vira `C` em qualquer parser ingênuo baseado em "ache a
    primeira letra maiúscula". O prejuízo é silencioso: a resposta entra na
    acurácia como uma escolha que o modelo nunca fez, e entra mais vezes para
    modelos verborrágicos — exatamente o tipo de erro correlacionado com a
    condição que enviesaria a comparação NE×SE.
    """
    result = parse_label("Certamente")
    assert result.label is None
    assert result.reason == REASON_NO_LETTER


def test_letter_glued_to_digits_does_not_become_a_label():
    """`"C3PO"` é um código, não a alternativa C: letra vizinha de dígito não conta."""
    result = parse_label("C3PO")
    assert result.label is None
    assert result.reason == REASON_NO_LETTER


def test_prose_starting_with_a_word_that_begins_with_a_label_letter():
    """`"Difícil dizer..."` não é `D`. Mesmo princípio, outra letra."""
    assert parse_label("Difícil dizer com esse áudio").reason == REASON_NO_LETTER


# --------------------------------------------------------------------------- #
# `labels` do item (D10) — o vocabulário não é global
# --------------------------------------------------------------------------- #
def test_letter_valid_elsewhere_is_out_of_vocab_for_a_three_option_item():
    """Um item com 3 alternativas não pode aceitar `D`.

    O runner sempre passa `item.n_alternatives`; aceitar uma letra inexistente
    transformaria um item malformado em uma resposta aparentemente válida.
    """
    result = parse_label("D", labels=("A", "B", "C"))
    assert result.label is None
    assert result.reason == REASON_OUT_OF_VOCAB
    assert result.candidates == ("D",)
    # O MESMO texto é interpretável quando o item tem quatro alternativas.
    assert parse_label("D").label == "D"


def test_labels_are_normalized_before_use():
    """Rótulos em minúscula ou com espaço são o mesmo vocabulário."""
    assert parse_label("b", labels=(" a ", "b")).label == "B"


def test_empty_labels_raise_value_error():
    """Vocabulário vazio é erro de programação, não resposta não interpretável.

    Devolver `None` silenciosamente aqui esconderia um item quebrado dentro da
    taxa de não interpretável — o número ficaria alto e ninguém saberia por quê.
    É a única exceção que a função levanta.
    """
    with pytest.raises(ValueError):
        parse_label("C", labels=())


# --------------------------------------------------------------------------- #
# Invariantes do contrato
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw_text",
    [
        "C",
        "c",
        "Alternativa C.",
        "A resposta correta é a letra B",
        "",
        "   ",
        "C ou D",
        "Z",
        "Certamente",
        "C3PO",
        "Acho que o falante fala de futebol",
        "(B)",
        "**A**",
        "D) porque sim",
    ],
)
def test_result_is_always_internally_consistent(raw_text):
    """`label` e `reason` são mutuamente exclusivos — nunca ambos, nunca nenhum.

    É esse invariante que autoriza a análise a somar acertos, erros e não
    interpretáveis e obter o total de tentativas sem nenhuma correção ad hoc.
    """
    result = parse_label(raw_text)
    if result.label is None:
        assert not result.is_parsed
        assert result.reason in REASONS
    else:
        assert result.is_parsed
        assert result.label in DEFAULT_LABELS
        assert result.reason == ""
        assert result.strategy in STRATEGIES
        assert result.candidates == (result.label,)


def test_parse_label_is_pure_and_deterministic():
    """Duas chamadas com o mesmo texto dão o mesmo resultado.

    Pureza é o que permite recomputar o motivo de uma resposta não interpretável
    a partir do `raw_text` guardado no cache, meses depois, sem reexecutar nada.
    """
    text = "A resposta correta é a letra B"
    assert parse_label(text) == parse_label(text)


def test_strategy_names_are_the_five_reported_ones():
    """Os nomes vão para o relatório: a distribuição entre eles diz se a cascata
    está sendo honesta (90% no primeiro nível, fração mínima no último recurso).
    """
    assert len(STRATEGIES) == 5
    assert len(set(STRATEGIES)) == 5
    assert STRATEGIES[0] == STRATEGY_ONLY_LETTER
    assert STRATEGIES[-1] == STRATEGY_UNIQUE_ISOLATED
