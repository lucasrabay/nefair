"""Normalizador de texto PT-BR para o cálculo do WER (D7).

Por que este módulo existe separado de `wer.py`: a normalização é uma **decisão
metodológica**, não um detalhe de implementação. Ela determina o que conta como
erro. Trocar uma regra muda todos os números do capítulo de resultados, então a
normalização é **fixa, versionada e testada regra a regra** — e a versão usada
(`NORMALIZER_VERSION`) viaja junto com cada `WerRecord`.

Duas invariantes governam tudo abaixo:

1. **A MESMA normalização é aplicada à referência e à hipótese.** Aplicá-la só a
   um dos lados inventaria erros que não existem: se a referência do CORAA já vem
   sem pontuação e a hipótese do ASR vem com vírgulas, normalizar só a referência
   transformaria cada vírgula da hipótese numa inserção. Por isso a função pública
   preferida é `normalize_pair`, que devolve os dois lados de uma vez — a forma
   errada de usar este módulo fica desconfortável de escrever.
2. **Idempotência.** `normalize(normalize(t)) == normalize(t)` para todo `t`. Sem
   isso, aplicar a normalização duas vezes por engano (num pipeline com cache,
   por exemplo) mudaria silenciosamente o WER. A propriedade é testada.

Limitações conhecidas, declaradas de propósito em vez de escondidas:

- **Ordinais** (`1º`, `2ª`) perdem o marcador e são lidos como cardinais
  (`1º` → `um`, não `primeiro`). Ordinais são raros em fala espontânea e a regra
  vale para os dois lados; o custo é simétrico.
- **Acentos NÃO são removidos.** Em PT-BR o acento é fonêmico (`avô`/`avó`,
  `e`/`é`); removê-lo colapsaria palavras distintas e mediria menos erro do que
  o modelo comete de verdade.
- **Marcas de hesitação** incluem `ah` e `eh`, que ocasionalmente são interjeições
  legítimas. É uma escolha consciente (ver `HESITATION_MARKS`).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

# Versão do normalizador. Gravada em todo `WerRecord` e ecoada no relatório da
# Etapa 6. Qualquer mudança de regra exige um bump aqui — números de versões
# diferentes NÃO são comparáveis entre si.
NORMALIZER_VERSION = "v1"


# --------------------------------------------------------------------------- #
# Marcas de hesitação (escolha metodológica, não detalhe de implementação)
# --------------------------------------------------------------------------- #
# Pausas preenchidas ("filled pauses") do PT-BR falado. Removê-las é prática
# padrão em WER de fala espontânea: o entrevistado hesita, o ASR às vezes
# transcreve a hesitação e às vezes não, e essa variação não diz nada sobre a
# qualidade do reconhecimento do CONTEÚDO — que é o que o TCC quer medir.
#
# A lista é deliberadamente CURTA e conservadora. Entram só formas que não são
# palavras do português; ficaram DE FORA, apesar de frequentes na fala:
#   - `né`, `tá`, `então`, `assim` — marcadores discursivos, mas palavras reais;
#     removê-los apagaria conteúdo lexical de verdade;
#   - `em`, `e`, `a` — palavras gramaticais que se parecem com transcrições de
#     hesitação em alguns esquemas de anotação; o risco de apagá-las é alto
#     demais;
#   - `é` — é o verbo `ser`, não a hesitação `eh`.
#
# Risco assumido: `ah` e `eh` podem ser interjeições legítimas ("ah, entendi").
# Como a regra vale igualmente para referência e hipótese, o efeito é simétrico:
# ela não pode inflar nem desinflar o WER de um dos lados sozinha.
HESITATION_MARKS: frozenset[str] = frozenset(
    {
        "ah",
        "aham",
        "ahm",
        "ahn",
        "ahã",
        "eh",
        "ehm",
        "ehn",
        "hm",
        "hmm",
        "hmmm",
        "hum",
        "humm",
        "mhm",
        "mm",
        "mmm",
        "uh",
        "uhm",
        "uhum",
        "ã",
        "ãh",
        "ãhn",
        "ãn",
    }
)


# --------------------------------------------------------------------------- #
# Regra 1 — forma Unicode canônica
# --------------------------------------------------------------------------- #
def apply_nfc(text: str) -> str:
    """Converte para a forma canônica composta (NFC).

    Necessário porque `"ção"` pode chegar com o cedilha e o til como caracteres
    combinantes separados (NFD) de uma fonte e compostos (NFC) de outra. Sem
    esta regra, duas strings visualmente idênticas seriam contadas como
    substituição — um erro do pipeline atribuído ao modelo.
    """
    return unicodedata.normalize("NFC", text)


# --------------------------------------------------------------------------- #
# Regra 2 — caixa baixa
# --------------------------------------------------------------------------- #
def to_lowercase(text: str) -> str:
    """Caixa baixa.

    O ASR capitaliza início de frase e nomes próprios; a transcrição de
    referência do CORAA (`normalized_text`) já vem em caixa baixa. Sem esta
    regra, toda primeira palavra de cada hipótese viraria uma substituição.
    """
    return text.lower()


# --------------------------------------------------------------------------- #
# Regra 3 — marcações de anotação e tokens especiais
# --------------------------------------------------------------------------- #
# `[inint]`, `(risos)`, `<unk>`, `<|pt|>`: metadados de anotação (do corpus) ou
# tokens de controle (do decodificador do ASR). Removidos com o CONTEÚDO, não só
# com os delimitadores — senão `(risos)` viraria a palavra `risos` e contaria
# como uma palavra real dita pelo falante.
_ANNOTATION_TAG_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|<[^>]*>")


def remove_annotation_tags(text: str) -> str:
    """Remove marcações entre `[]`, `()` ou `<>`, junto com o conteúdo."""
    return _ANNOTATION_TAG_RE.sub(" ", text)


# --------------------------------------------------------------------------- #
# Regra 4 — números por extenso
# --------------------------------------------------------------------------- #
# O falante diz "mil novecentos e oitenta e cinco"; o ASR escreve "1985". Sem
# esta regra, cada número da hipótese vira 1 substituição + N deleções. A
# conversão é do lado dos DÍGITOS para o extenso (e não o contrário) porque a
# escrita por extenso é a forma canônica do `normalized_text` do corpus.
_UNITS: tuple[str, ...] = (
    "zero",
    "um",
    "dois",
    "três",
    "quatro",
    "cinco",
    "seis",
    "sete",
    "oito",
    "nove",
    "dez",
    "onze",
    "doze",
    "treze",
    "quatorze",
    "quinze",
    "dezesseis",
    "dezessete",
    "dezoito",
    "dezenove",
)
_TENS: tuple[str, ...] = (
    "",
    "",
    "vinte",
    "trinta",
    "quarenta",
    "cinquenta",
    "sessenta",
    "setenta",
    "oitenta",
    "noventa",
)
_HUNDREDS: tuple[str, ...] = (
    "",
    "cento",
    "duzentos",
    "trezentos",
    "quatrocentos",
    "quinhentos",
    "seiscentos",
    "setecentos",
    "oitocentos",
    "novecentos",
)
# Escalas em ordem decrescente. `mil` é tratado à parte no código porque não
# leva "um" na frente (1000 é "mil", nunca "um mil").
_SCALES: tuple[tuple[int, str, str], ...] = (
    (1_000_000_000, "bilhão", "bilhões"),
    (1_000_000, "milhão", "milhões"),
    (1_000, "mil", "mil"),
)
# Acima disto, a leitura por extenso deixa de ser previsível (e de ser plausível
# em fala espontânea): o número é lido dígito a dígito.
_MAX_SPELLED_INT = 1_000_000_000_000


def _below_thousand(n: int) -> str:
    """Extenso de 0–999, com o `e` interno do PT-BR (`cento e vinte e três`)."""
    if n < 20:
        return _UNITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _TENS[tens] if unit == 0 else f"{_TENS[tens]} e {_UNITS[unit]}"
    if n == 100:
        # 100 exato é "cem"; 101–199 usam "cento e ...". Esta é a irregularidade
        # clássica do português e a razão de `_HUNDREDS[1]` ser "cento".
        return "cem"
    hundreds, rest = divmod(n, 100)
    return _HUNDREDS[hundreds] if rest == 0 else f"{_HUNDREDS[hundreds]} e {_below_thousand(rest)}"


def _int_to_words(n: int) -> str:
    """Extenso de um inteiro não negativo, na leitura corrente do PT-BR.

    A regra do `e` antes do último grupo segue o uso brasileiro: entra quando o
    resto é menor que cem ou é centena redonda (`mil e quinhentos`), e não entra
    caso contrário (`mil novecentos e oitenta e cinco`).
    """
    if n < 1000:
        return _below_thousand(n)
    for value, singular, plural in _SCALES:
        if n < value:
            continue
        count, rest = divmod(n, value)
        if value == 1_000:
            head = "mil" if count == 1 else f"{_int_to_words(count)} mil"
        else:
            head = f"{_int_to_words(count)} {singular if count == 1 else plural}"
        if rest == 0:
            return head
        separator = " e " if (rest < 100 or rest % 100 == 0) else " "
        return f"{head}{separator}{_int_to_words(rest)}"
    raise AssertionError(f"número fora de todas as escalas: {n}")  # pragma: no cover


def _digits_one_by_one(digits: str) -> str:
    """Leitura dígito a dígito (`0800` → `zero oito zero zero`)."""
    return " ".join(_UNITS[int(d)] for d in digits)


def _integer_token_to_words(digits: str) -> str:
    """Converte a parte inteira de um token numérico.

    Zero à esquerda (`007`, `0800`) indica código, não quantidade: nesse caso a
    leitura é dígito a dígito, que é como o falante de fato pronuncia.
    """
    if len(digits) > 1 and digits[0] == "0":
        return _digits_one_by_one(digits)
    value = int(digits)
    if value >= _MAX_SPELLED_INT:
        return _digits_one_by_one(digits)
    return _int_to_words(value)


# Ordem das alternativas importa: o padrão com separador de milhar precisa ser
# tentado ANTES do padrão simples, senão `1.500` casaria só com `1`.
_NUMBER_RE = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?")


def _replace_number(match: re.Match[str]) -> str:
    token = match.group(0)
    # Separador de milhar do PT-BR: some antes da conversão.
    integer_part, _, decimal_part = token.replace(".", "").partition(",")
    words = _integer_token_to_words(integer_part)
    if decimal_part:
        # A parte decimal é lida dígito a dígito ("três vírgula um quatro"),
        # que é a leitura corrente e a única previsível para N casas.
        words = f"{words} vírgula {_digits_one_by_one(decimal_part)}"
    return f" {words} "


def expand_numbers(text: str) -> str:
    """Converte sequências de dígitos para extenso em PT-BR.

    Roda ANTES da remoção de pontuação porque o ponto de milhar e a vírgula
    decimal fazem parte do número: removê-los primeiro transformaria `1.500` em
    dois números soltos.
    """
    return _NUMBER_RE.sub(_replace_number, text)


# --------------------------------------------------------------------------- #
# Regra 5 — pontuação
# --------------------------------------------------------------------------- #
# Hífen, travessão e barra viram ESPAÇO (e não vazio): `guarda-chuva` deve virar
# duas palavras, do mesmo jeito nos dois lados, em vez de `guardachuva` num lado
# e `guarda chuva` no outro. Apóstrofo some sem espaço (`d'água` → `dágua`),
# porque ali ele une uma palavra só.
_SPACE_PUNCTUATION = frozenset("-–—/\\_")
_DROP_PUNCTUATION = frozenset("'’`´")


def strip_punctuation(text: str) -> str:
    """Remove pontuação, preservando letras, dígitos e espaços.

    A referência `normalized_text` do CORAA não tem pontuação; a hipótese do ASR
    tem. Sem esta regra, cada vírgula da hipótese viraria uma inserção — erro do
    formato de saída, não do reconhecimento.
    """
    out: list[str] = []
    for char in text:
        if char in _SPACE_PUNCTUATION:
            out.append(" ")
        elif char in _DROP_PUNCTUATION:
            continue
        elif char.isalnum() or char.isspace():
            out.append(char)
        else:
            # Toda a categoria Unicode de pontuação e símbolos (P*, S*) cai aqui,
            # inclusive `º`, `ª`, `%`, `$` e reticências.
            out.append(" ")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Regra 6 — marcas de hesitação
# --------------------------------------------------------------------------- #
def remove_hesitations(text: str) -> str:
    """Remove tokens que são pausas preenchidas (ver `HESITATION_MARKS`).

    A remoção é por TOKEN inteiro, nunca por substring: `ah` sai, mas `ahora` e
    `manhã` ficam intactos.
    """
    return " ".join(token for token in text.split() if token not in HESITATION_MARKS)


# --------------------------------------------------------------------------- #
# Regra 7 — espaços
# --------------------------------------------------------------------------- #
def collapse_whitespace(text: str) -> str:
    """Colapsa qualquer sequência de espaços/quebras num único espaço e apara.

    Última regra do pipeline: todas as anteriores deixam espaços sobrando de
    propósito (é mais seguro do que colar tokens que eram separados).
    """
    return " ".join(text.split())


# --------------------------------------------------------------------------- #
# Pipeline versionado
# --------------------------------------------------------------------------- #
# A ORDEM é parte da definição da versão. Documentada aqui, e não espalhada pelo
# corpo de `normalize`, para que a diferença entre `v1` e uma futura `v2` seja
# legível numa linha de diff.
RULES_V1: tuple[tuple[str, Callable[[str], str]], ...] = (
    ("nfc", apply_nfc),
    ("caixa_baixa", to_lowercase),
    ("marcacoes_de_anotacao", remove_annotation_tags),
    ("numeros_por_extenso", expand_numbers),
    ("pontuacao", strip_punctuation),
    ("marcas_de_hesitacao", remove_hesitations),
    ("espacos", collapse_whitespace),
)

_PIPELINES: dict[str, tuple[tuple[str, Callable[[str], str]], ...]] = {
    NORMALIZER_VERSION: RULES_V1,
}


def normalize(text: str, version: str = NORMALIZER_VERSION) -> str:
    """Aplica o pipeline de normalização da versão pedida.

    `version` é explícito (e validado) porque ele vem do YAML (`wer.normalizer_version`,
    D7): pedir uma versão que não existe tem de falhar alto, e não cair num
    default silencioso que produziria números incomparáveis com os do texto.
    """
    if version not in _PIPELINES:
        raise ValueError(
            f"Versão de normalizador desconhecida: '{version}'. "
            f"Disponíveis: {sorted(_PIPELINES)}."
        )
    out = text
    for _, rule in _PIPELINES[version]:
        out = rule(out)
    return out


def normalize_pair(
    reference: str, hypothesis: str, version: str = NORMALIZER_VERSION
) -> tuple[str, str]:
    """Normaliza referência e hipótese com o MESMO pipeline, de uma vez.

    Esta é a função que o resto do código deve usar. Ela existe para que a
    simetria exigida pelo D7 seja estrutural e não dependa de disciplina: quem
    chamar `normalize_pair` não consegue, por distração, normalizar só um lado.
    """
    return normalize(reference, version), normalize(hypothesis, version)


def trace(text: str, version: str = NORMALIZER_VERSION) -> list[tuple[str, str]]:
    """Resultado intermediário após cada regra — para depuração e para o relatório.

    Quando um WER sai alto demais, a primeira pergunta é "foi o modelo ou foi a
    normalização?". Esta função responde sem precisar de `print` no meio do
    pipeline.
    """
    if version not in _PIPELINES:
        raise ValueError(f"Versão de normalizador desconhecida: '{version}'.")
    steps: list[tuple[str, str]] = []
    out = text
    for name, rule in _PIPELINES[version]:
        out = rule(out)
        steps.append((name, out))
    return steps
