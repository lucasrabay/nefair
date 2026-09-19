"""Testes do normalizador PT-BR do WER (D7) — offline, sem corpus.

A normalização é o que **define** o que conta como erro: cada regra que entra ou
sai muda todos os números do capítulo de resultados. Por isso a suíte aqui não
verifica "não explode" — ela fixa, regra a regra, o comportamento que o texto do
TCC declara. Se alguém trocar uma regra de lugar ou afrouxar uma delas, o número
do WER muda em silêncio e só estes testes acusam.

Três propriedades são tratadas como invariantes do módulo, não como casos:

1. **Idempotência** (`normalize(normalize(t)) == normalize(t)`). Um pipeline com
   cache pode normalizar duas vezes por acidente; se isso mudasse o texto, o WER
   mudaria junto sem nenhum sinal.
2. **Acentos preservados**. Em PT-BR o acento é fonêmico (`avô`/`avó`); removê-lo
   mediria MENOS erro do que o modelo comete de verdade — um viés na direção
   confortável, que é a pior direção.
3. **Simetria**. `normalize_pair` aplica o mesmo pipeline aos dois lados; testar
   isso é testar que o módulo não consegue inventar erros do lado da hipótese.

A versão testada vem de `configs/analysis.yaml` (`wer.normalizer_version`), e não
de uma constante escrita à mão aqui: os testes valem sobre o que vai rodar.
"""

from __future__ import annotations

import unicodedata

import pytest

from nefair.config import load_analysis_config
from nefair.metrics.normalize import (
    HESITATION_MARKS,
    NORMALIZER_VERSION,
    RULES_V1,
    apply_nfc,
    collapse_whitespace,
    expand_numbers,
    normalize,
    normalize_pair,
    remove_annotation_tags,
    remove_hesitations,
    strip_punctuation,
    to_lowercase,
    trace,
)

CONFIG_PATH = "configs/analysis.yaml"

# Amostra usada nos testes de propriedade (idempotência, acentos). Cada entrada
# exercita pelo menos uma regra, e várias exercitam duas ao mesmo tempo — é na
# interação entre regras que a idempotência costuma quebrar.
SAMPLES: tuple[str, ...] = (
    "Ah, [inint] o meu AVÔ tinha 100 reais!",
    "eh... a gente morava em 1985 (risos) na roça",
    "guarda-chuva d'água — 3,14 por cento",
    "<|pt|> hm hmm mhm nada aqui <unk>",
    "   espaços    demais\n\tem  toda   parte  ",
    "0800 007 1.500 2ª vez",
    "manhã de ahora, nenhuma hesitação de verdade",
    "",
    "!!!???...",
)


@pytest.fixture(scope="module")
def configured_version() -> str:
    """Versão declarada no YAML real — o teste acompanha a config, não a duplica."""
    return load_analysis_config(CONFIG_PATH).wer.normalizer_version


# --------------------------------------------------------------------------- #
# Contrato de versão
# --------------------------------------------------------------------------- #
def test_configured_version_is_the_one_implemented(configured_version):
    """O YAML e o módulo não podem divergir: WERs de versões distintas não comparam."""
    assert configured_version == NORMALIZER_VERSION == "v1"


def test_unknown_version_fails_loudly():
    """Versão inexistente tem de falhar, nunca cair num default silencioso.

    Um default silencioso produziria números incomparáveis com os do texto sem
    que ninguém percebesse — exatamente o erro que o D7 quer impedir.
    """
    with pytest.raises(ValueError, match="v2"):
        normalize("qualquer coisa", "v2")
    with pytest.raises(ValueError, match="v2"):
        normalize("qualquer coisa", version="v2")
    with pytest.raises(ValueError):
        trace("qualquer coisa", version="v2")
    with pytest.raises(ValueError):
        normalize_pair("a", "b", version="v2")


def test_version_is_positional_or_named():
    """As duas formas de chamada existem e concordam — o chamador escolhe."""
    assert normalize("Olá MUNDO", "v1") == normalize("Olá MUNDO", version="v1")
    assert normalize("Olá MUNDO") == normalize("Olá MUNDO", NORMALIZER_VERSION)


# --------------------------------------------------------------------------- #
# Uma regra por vez — as 7 de RULES_V1
# --------------------------------------------------------------------------- #
def test_pipeline_has_exactly_the_seven_declared_rules():
    """A ORDEM faz parte da definição da versão; fixá-la é fixar a versão.

    Em particular `numeros_por_extenso` PRECISA vir antes de `pontuacao` (senão o
    ponto de milhar some e `1.500` vira dois números) e `espacos` precisa ser a
    última (as outras deixam espaços sobrando de propósito).
    """
    assert [name for name, _ in RULES_V1] == [
        "nfc",
        "caixa_baixa",
        "marcacoes_de_anotacao",
        "numeros_por_extenso",
        "pontuacao",
        "marcas_de_hesitacao",
        "espacos",
    ]
    assert [rule for _, rule in RULES_V1] == [
        apply_nfc,
        to_lowercase,
        remove_annotation_tags,
        expand_numbers,
        strip_punctuation,
        remove_hesitations,
        collapse_whitespace,
    ]


def test_rule_nfc_composes_decomposed_accents():
    """NFD e NFC são visualmente idênticos; sem esta regra viravam substituição.

    É um erro do PIPELINE (duas fontes de texto com codificações diferentes) que
    seria atribuído ao modelo de reconhecimento.
    """
    decomposed = unicodedata.normalize("NFD", "coração ação avô")
    composed = unicodedata.normalize("NFC", "coração ação avô")
    assert decomposed != composed  # garante que a fixture é de fato NFD
    assert apply_nfc(decomposed) == composed
    assert normalize(decomposed) == normalize(composed) == "coração ação avô"


def test_rule_lowercase_removes_the_asr_capitalization():
    """O ASR capitaliza começo de frase; o `normalized_text` do CORAA não.

    Sem a regra, a primeira palavra de toda hipótese viraria uma substituição.
    """
    assert to_lowercase("O Pai De João") == "o pai de joão"
    assert normalize("O Pai De João") == "o pai de joão"


def test_rule_annotation_tags_remove_the_content_too():
    """`(risos)` sai inteiro: se sobrasse `risos`, contaria como palavra dita."""
    assert remove_annotation_tags("a [inint] gente (risos) foi <unk> embora").split() == [
        "a",
        "gente",
        "foi",
        "embora",
    ]
    # Tokens de controle do decodificador do ASR também são anotação, não fala.
    assert normalize("<|pt|><|transcribe|> a gente foi") == "a gente foi"
    assert normalize("a gente (risos) foi") == "a gente foi"


def test_rule_numbers_are_spelled_out_in_ptbr():
    """Casos escolhidos pelas irregularidades do português, não por conveniência.

    `100` → `cem` (e não `cento`), `1500` → `mil E quinhentos` (o `e` entra porque
    o resto é centena redonda) mas `1985` → `mil novecentos e oitenta e cinco`
    (sem `e` após `mil`, porque o resto passa de cem e não é redondo). Essa
    assimetria é a regra de leitura corrente do PT-BR e é fácil de quebrar numa
    refatoração ingênua.
    """
    assert normalize("100") == "cem"
    assert normalize("101") == "cento e um"
    assert normalize("1985") == "mil novecentos e oitenta e cinco"
    assert normalize("1500") == "mil e quinhentos"
    assert normalize("1.500") == "mil e quinhentos"  # separador de milhar
    assert normalize("2500") == "dois mil e quinhentos"
    assert normalize("21") == "vinte e um"
    assert normalize("1000") == "mil"  # nunca "um mil"
    assert normalize("1000000") == "um milhão"


def test_rule_numbers_decimal_uses_virgula_and_digit_by_digit():
    """A parte decimal é lida dígito a dígito — a única leitura previsível."""
    assert normalize("3,14") == "três vírgula um quatro"
    assert normalize("0,5") == "zero vírgula cinco"


def test_rule_numbers_leading_zero_means_code_not_quantity():
    """`007` é código, não quantidade: o falante diz "zero zero sete"."""
    assert normalize("007") == "zero zero sete"
    assert normalize("0800") == "zero oito zero zero"
    # Sem zero à esquerda a leitura volta a ser por quantidade.
    assert normalize("800") == "oitocentos"


def test_rule_numbers_inside_a_sentence(configured_version):
    frase = "em 1985 eu tinha 21 anos"
    assert normalize(frase, configured_version) == (
        "em mil novecentos e oitenta e cinco eu tinha vinte e um anos"
    )


def test_rule_punctuation_splits_hyphens_but_glues_apostrophes():
    """Hífen vira espaço, apóstrofo some: as duas escolhas têm motivos opostos.

    `guarda-chuva` são duas palavras nos dois lados (o ASR escreve com hífen, a
    referência sem); `d'água` é uma palavra só, e separá-la criaria uma inserção.
    """
    assert strip_punctuation("guarda-chuva").split() == ["guarda", "chuva"]
    assert strip_punctuation("d'água") == "dágua"
    assert normalize("Guarda-chuva, d'água!") == "guarda chuva dágua"
    # Vírgula da hipótese não pode virar inserção.
    assert normalize("bom, dia") == normalize("bom dia")


def test_rule_punctuation_keeps_letters_and_digits():
    """A regra remove pontuação e símbolos, não conteúdo."""
    assert normalize("100%") == "cem"
    assert normalize("...!?") == ""


def test_ordinal_marker_does_not_survive_as_a_token():
    """`2ª` → `dois`, nunca `dois ª`.

    Regressão de um bug real: `º`, `ª` (categoria Lo) e `²`, `³` (categoria No)
    passam por `str.isalnum()` e sobreviviam à regra de pontuação. Como a regra 4
    já converteu o dígito, o marcador virava um TOKEN SOLTO — uma inserção que o
    normalizador fabrica e cobra do modelo, já que a referência do corpus traz
    apenas "segunda". A limitação declarada do módulo é que o ordinal vira
    cardinal; ganhar uma palavra a mais não fazia parte dela.
    """
    assert normalize("2ª vez") == "dois vez"
    assert normalize("1º lugar") == "um lugar"
    assert normalize("10 m² de terra") == "dez m de terra"


def test_rule_hesitations_are_removed_by_whole_token_only():
    """`ah` sai; `ahora` e `manhã` ficam. Remoção por substring seria um desastre."""
    assert remove_hesitations("ah eh hmm a gente foi") == "a gente foi"
    assert remove_hesitations("manhã de ahora") == "manhã de ahora"
    assert normalize("Ah, eh... a gente foi") == "a gente foi"
    assert normalize("na manhã seguinte") == "na manhã seguinte"


def test_hesitation_list_is_short_and_excludes_real_words():
    """A lista é conservadora de propósito: 23 formas, nenhuma palavra real.

    `né`, `então`, `assim` e `é` ficaram de FORA — removê-los apagaria conteúdo
    lexical e mediria menos erro do que o modelo comete.
    """
    assert len(HESITATION_MARKS) == 23
    for palavra in ("né", "tá", "então", "assim", "é", "e", "a", "em"):
        assert palavra not in HESITATION_MARKS
    assert normalize("né então assim é") == "né então assim é"


def test_rule_whitespace_collapses_and_trims():
    assert collapse_whitespace("  a   gente \n foi\t ") == "a gente foi"
    assert normalize("  a   gente \n foi\t ") == "a gente foi"


# --------------------------------------------------------------------------- #
# Propriedades do pipeline inteiro
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", SAMPLES)
def test_normalize_is_idempotent(text):
    """Aplicar duas vezes tem de dar o mesmo resultado.

    Sem isso, um cache que normalize na escrita E na leitura mudaria o WER sem
    deixar rastro.
    """
    once = normalize(text)
    assert normalize(once) == once


@pytest.mark.parametrize(
    "text",
    ["avô e avó", "o pé e a pá", "coração", "português", "não é ação"],
)
def test_accents_are_never_removed(text):
    """Acento é fonêmico em PT-BR: `avô` ≠ `avó`, `e` ≠ `é`.

    Remover acentos colapsaria pares distintos e faria o WER parecer MENOR do que
    é. Este teste existe para impedir que alguém "melhore" o normalizador assim.
    """
    out = normalize(text)
    assert out == text.lower()
    assert any(unicodedata.combining(c) or ord(c) > 127 for c in unicodedata.normalize("NFD", out))


def test_normalize_pair_applies_the_same_pipeline_to_both_sides():
    """A simetria do D7 é estrutural: a função devolve os dois lados de uma vez.

    O caso é o clássico: referência do corpus sem pontuação, hipótese do ASR com
    pontuação e capitalização. Depois da normalização os dois lados coincidem —
    isto é, o ASR acertou e o pipeline não inventa erro nenhum.
    """
    referencia = "a gente morava em mil novecentos e oitenta e cinco"
    hipotese = "A gente morava em 1985."
    ref_norm, hyp_norm = normalize_pair(referencia, hipotese)
    assert ref_norm == hyp_norm
    assert (ref_norm, hyp_norm) == (normalize(referencia), normalize(hipotese))


def test_normalize_pair_does_not_erase_a_real_error():
    """Contraprova do teste acima: o normalizador não apaga erro de verdade."""
    ref_norm, hyp_norm = normalize_pair("meu pai trabalhava na roça", "Meu pai trabalhava na rosa.")
    assert ref_norm != hyp_norm
    assert ref_norm.split()[-1] == "roça"
    assert hyp_norm.split()[-1] == "rosa"


# --------------------------------------------------------------------------- #
# `trace` — o instrumento de depuração
# --------------------------------------------------------------------------- #
def test_trace_returns_the_seven_named_steps():
    """`trace` responde "foi o modelo ou foi a normalização?" sem `print` no meio."""
    steps = trace("Ah, [inint] o AVÔ tinha 100 reais!")
    assert len(steps) == 7
    assert [name for name, _ in steps] == [name for name, _ in RULES_V1]
    # O último passo é, por construção, o resultado de `normalize`.
    esperado = normalize("Ah, [inint] o AVÔ tinha 100 reais!")
    assert steps[-1][1] == esperado == "o avô tinha cem reais"


def test_trace_shows_where_each_change_happened():
    """Cada etapa mostra exatamente o seu efeito — é isso que a torna útil."""
    steps = dict(trace("Ah, [inint] o AVÔ tinha 100 reais!"))
    assert "AVÔ" in steps["nfc"]
    assert "avô" in steps["caixa_baixa"]
    assert "[inint]" not in steps["marcacoes_de_anotacao"]
    assert "cem" in steps["numeros_por_extenso"]
    assert "!" not in steps["pontuacao"]
    assert not steps["marcas_de_hesitacao"].startswith("ah")
    assert "  " not in steps["espacos"]


def test_trace_is_consistent_with_normalize_for_every_sample():
    for text in SAMPLES:
        assert trace(text)[-1][1] == normalize(text)
