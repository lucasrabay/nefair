"""Extração da letra escolhida a partir da resposta em texto livre do modelo.

Este módulo existe para que **um número específico do TCC seja mensurável**: a
taxa de respostas não interpretáveis. Um modelo que responde "Acho que o falante
fala de futebol" não acertou nem errou o item — ele não respondeu. Imputar essa
resposta como erro rebaixaria artificialmente a acurácia de modelos verborrágicos
e, pior, faria isso de modo desigual entre condições (a condição `audio` tende a
produzir mais prosa que a condição `reference`), contaminando exatamente a
diferença que o estudo mede. Imputar como acerto seria pior ainda.

Por isso, o contrato é: **`label is None` significa "não interpretável", é
contado como categoria própria e nunca vira acerto nem erro.**

A função é pura: não lê arquivo, não chama modelo, não consulta relógio. Isso é o
que permite recomputar o motivo de uma resposta não interpretável a partir do
`raw_text` guardado no cache, meses depois, sem reexecutar nada.

## Cascata de estratégias (ordem importa, e o porquê de cada posição)

1. `resposta_e_so_a_letra` — a resposta inteira é uma única letra, ignorando
   espaços, pontuação e markdown: `"C"`, `"c"`, `"C."`, `"(C)"`, `"**C**"`,
   `"C)"`. É o caso majoritário quando o prompt pede só a letra, e é o único
   totalmente inequívoco — por isso vem primeiro.
2. `marcador_explicito` — há uma palavra que anuncia a escolha ("alternativa",
   "letra", "opção", "item", "resposta") seguida da letra. Vem antes das
   estratégias posicionais porque desempata o caso clássico
   `"A resposta correta é a letra B"`, em que a primeira letra isolada do texto
   é o artigo "A", não a resposta.
3. `letra_entre_delimitadores` — a letra aparece entre parênteses, colchetes,
   aspas ou asteriscos: `"acho que é (C)"`. Delimitador é intenção explícita de
   destacar a escolha.
4. `letra_enumerada` — a resposta começa repetindo a alternativa escolhida no
   formato de enumeração: `"C) porque o falante diz..."`. Vem depois dos
   delimitadores para que `"(C) ..."` não seja disputado pelas duas.
5. `unica_letra_isolada` — último recurso: varre o texto inteiro atrás de letras
   isoladas (nunca de letras dentro de palavras — `"Certamente"` **não** é `C`) e
   só decide se sobrar exatamente uma candidata. Duas ou mais → ambíguo → `None`.

Em qualquer estratégia, encontrar **duas letras plausíveis** (`"C ou D"`) é
ambiguidade, e ambiguidade é não interpretável. Preferimos um `None` contado a um
palpite silencioso.

## Palavras de uma letra do português

"a", "e" e "o" são palavras (artigo, conjunção, artigo) e aparecem o tempo todo
em prosa. Na estratégia 5 elas são descartadas como candidatas, junto com "A"/"E"
maiúsculos que abram oração (seguidos de palavra minúscula, como em
`"A resposta é C"`). O custo é conhecido e aceito: uma resposta como
`"A é a correta"` vira não interpretável em vez de virar `A`. Erramos para o lado
do `None` contado, nunca para o lado do palpite.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from nefair.schema import ALTERNATIVE_LABELS

# Default de `labels`: quatro alternativas, que é o valor de `items.n_alternatives`
# no `items.yaml`. O runner SEMPRE passa `item.n_alternatives` explicitamente —
# este default serve só para uso interativo e testes.
DEFAULT_LABELS: tuple[str, ...] = ALTERNATIVE_LABELS[:4]

# Nomes das estratégias, na ordem em que são tentadas. Vão para o relatório: saber
# que 90% dos acertos de parsing vêm de `resposta_e_so_a_letra` e 0,3% do último
# recurso é o que diz se a cascata está sendo honesta.
STRATEGY_ONLY_LETTER = "resposta_e_so_a_letra"
STRATEGY_MARKER = "marcador_explicito"
STRATEGY_DELIMITED = "letra_entre_delimitadores"
STRATEGY_ENUMERATED = "letra_enumerada"
STRATEGY_UNIQUE_ISOLATED = "unica_letra_isolada"
STRATEGIES: tuple[str, ...] = (
    STRATEGY_ONLY_LETTER,
    STRATEGY_MARKER,
    STRATEGY_DELIMITED,
    STRATEGY_ENUMERATED,
    STRATEGY_UNIQUE_ISOLATED,
)

# Motivos de não interpretabilidade. São categorias de relatório, não mensagens:
# a distribuição entre elas diz coisas diferentes sobre o modelo (resposta vazia
# = recusa ou corte de tokens; ambígua = o modelo não se decidiu; sem letra = o
# modelo ignorou a instrução de formato).
REASON_EMPTY = "vazio"
REASON_AMBIGUOUS = "ambiguo"
REASON_NO_LETTER = "sem_letra"
REASON_OUT_OF_VOCAB = "letra_fora_do_vocabulario"
REASONS: tuple[str, ...] = (REASON_EMPTY, REASON_AMBIGUOUS, REASON_NO_LETTER, REASON_OUT_OF_VOCAB)

# Uma letra que não é parte de uma palavra nem de um número. A dupla espiada
# (lookbehind + lookahead) é o que impede `"Certamente"` de virar `C` e
# `"C3PO"` de virar `C`.
_ISOLATED_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])(?![A-Za-z0-9])")

# A resposta inteira é uma letra cercada só de não-palavra (espaço, pontuação,
# markdown). `\W` em modo unicode não casa com letras acentuadas, então prosa em
# português nunca cai aqui.
_ONLY_LETTER_RE = re.compile(r"[\W_]*([A-Za-z])[\W_]*")

# Palavras que anunciam a escolha. Aplicadas ao texto SEM acento (ver
# `_strip_accents`), por isso "opcao"/"opcoes" e não "opção"/"opções".
_MARKER_RE = re.compile(
    r"\b(?:alternativas?|letras?|opcao|opcoes|itens|item|respostas?|answer|options?)\b",
    re.IGNORECASE,
)
# Quantos caracteres depois do marcador ainda contam como "perto do marcador".
_MARKER_WINDOW = 48
# Pontuação e delimitadores que podem separar o marcador da letra
# ("Alternativa: C", "alternativa -- C", "alternativa (C)").
_AFTER_MARKER_RE = re.compile(r"[\s:\-–—=\.\(\[\"'*«,]*([A-Za-z])(?![A-Za-z0-9])")

# Letra cercada por delimitadores de destaque.
_DELIMITED_RE = re.compile(r"[\(\[\{\"'*«]\s*([A-Za-z])\s*[\)\]\}\"'*»]")

# Letra abrindo uma linha no formato de enumeração ("C) ...", "C. ...", "C: ...").
_ENUMERATED_RE = re.compile(r"(?m)^[ \t\-–—•*>]*([A-Za-z])[\)\.:\-–—]+(?=\s|$)")

# Letras que também são palavras em português. Ver a nota no topo do módulo.
_ONE_LETTER_WORDS: frozenset[str] = frozenset({"a", "e", "o"})


@dataclass(frozen=True)
class ParseResult:
    """Resultado da extração, com a trilha do que decidiu (ou do que impediu).

    `label is None` é o caso não interpretável: `reason` diz qual dos quatro
    motivos e `candidates` mostra o que foi visto (vazio, ou duas letras em
    conflito). Nada disso é opcional para o TCC — a taxa de não interpretável é
    reportada por modelo e por condição, e a distribuição de motivos é o que
    permite dizer se o problema foi do prompt ou do modelo.
    """

    label: str | None
    strategy: str
    reason: str
    candidates: tuple[str, ...] = ()

    @property
    def is_parsed(self) -> bool:
        return self.label is not None


def _strip_accents(text: str) -> str:
    """Remove diacríticos preservando as letras ASCII subjacentes.

    Necessário porque os marcadores aparecem acentuados ("opção", "é") e porque
    "é" precisa virar "e" para ser reconhecido como palavra de uma letra — e não
    como um candidato à alternativa E.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _normalize_labels(labels: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(label).strip().upper() for label in labels)
    if not normalized:
        raise ValueError("parse_label: `labels` vazio; é preciso ao menos uma alternativa.")
    return normalized


def _isolated_letters(text: str) -> list[tuple[int, str]]:
    """Posições e caracteres de todas as letras isoladas do texto."""
    return [(m.start(1), m.group(1)) for m in _ISOLATED_LETTER_RE.finditer(text)]


def _is_one_letter_word(text: str, index: int, char: str) -> bool:
    """A letra nessa posição é uma palavra do português, não uma alternativa?

    Duas formas:
    - minúscula "a"/"e"/"o" — sempre palavra em texto corrido;
    - "A"/"E"/"O" maiúsculos seguidos de palavra minúscula — abertura de oração,
      como em "A resposta é C", "E o falante diz..." ou "O falante menciona...".

    "O" maiúsculo entra na regra pelo mesmo motivo que "A" e "E", e sem custo:
    `ALTERNATIVE_LABELS` vai só até "E", então "O" nunca é rótulo de alternativa
    e nada que fosse resposta válida deixa de sê-lo.
    """
    if char in _ONE_LETTER_WORDS:
        return True
    if char in ("A", "E", "O"):
        rest = text[index + 1 :].lstrip()
        return bool(rest) and rest[0].islower()
    return False


def _dedupe(candidates: Sequence[str]) -> tuple[str, ...]:
    """Candidatas distintas, na ordem em que apareceram (determinismo)."""
    seen: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
    return tuple(seen)


# --------------------------------------------------------------------------- #
# Estratégias (cada uma devolve as candidatas que ENCONTROU, sem decidir nada)
# --------------------------------------------------------------------------- #
def _strategy_only_letter(text: str, labels: tuple[str, ...]) -> list[str]:
    match = _ONLY_LETTER_RE.fullmatch(text)
    if match is None:
        return []
    letter = match.group(1).upper()
    return [letter] if letter in labels else []


def _strategy_marker(text: str, labels: tuple[str, ...]) -> list[str]:
    """Letra anunciada por um marcador ("alternativa C", "a letra B").

    Para cada marcador, duas regras, nesta ordem:

    1. o **próximo token** depois do marcador já é uma letra isolada → é ela
       (cobre "alternativa a", em que a letra é uma palavra do português mas está
       colada ao marcador, sem espaço para dúvida);
    2. senão, a primeira letra **maiúscula** isolada dentro da janela do marcador
       (cobre "a resposta correta é a letra B", em que entre o marcador e a letra
       há palavras, inclusive artigos de uma letra).

    Sem a regra 2 restrita a maiúsculas, "resposta correta e a letra B" devolveria
    "e" (o "é" sem acento) em vez de "B".
    """
    found: list[str] = []
    for marker in _MARKER_RE.finditer(text):
        tail = text[marker.end() : marker.end() + _MARKER_WINDOW]
        adjacent = _AFTER_MARKER_RE.match(tail)
        if adjacent is not None:
            letter = adjacent.group(1).upper()
            if letter in labels:
                found.append(letter)
                continue
        for index, char in _isolated_letters(tail):
            if char.isupper() and char in labels and not _is_one_letter_word(tail, index, char):
                found.append(char)
                break
    return found


def _strategy_delimited(text: str, labels: tuple[str, ...]) -> list[str]:
    found = []
    for match in _DELIMITED_RE.finditer(text):
        letter = match.group(1).upper()
        if letter in labels:
            found.append(letter)
    return found


def _strategy_enumerated(text: str, labels: tuple[str, ...]) -> list[str]:
    found = []
    for match in _ENUMERATED_RE.finditer(text):
        letter = match.group(1).upper()
        if letter in labels:
            found.append(letter)
    return found


def _strategy_unique_isolated(text: str, labels: tuple[str, ...]) -> list[str]:
    found = []
    for index, char in _isolated_letters(text):
        if _is_one_letter_word(text, index, char):
            continue
        letter = char.upper()
        if letter in labels:
            found.append(letter)
    return found


_CASCADE = (
    (STRATEGY_ONLY_LETTER, _strategy_only_letter),
    (STRATEGY_MARKER, _strategy_marker),
    (STRATEGY_DELIMITED, _strategy_delimited),
    (STRATEGY_ENUMERATED, _strategy_enumerated),
    (STRATEGY_UNIQUE_ISOLATED, _strategy_unique_isolated),
)


def parse_label(raw_text: str, labels: Sequence[str] = DEFAULT_LABELS) -> ParseResult:
    """Extrai a letra escolhida de uma resposta em texto livre.

    Função pura. Devolve `ParseResult` com `label is None` quando a resposta não
    é interpretável — o que é um resultado legítimo do experimento, contado e
    reportado, jamais convertido em acerto ou erro.

    `labels` é a lista de rótulos válidos do item (`ALTERNATIVE_LABELS[:n]`).
    Passá-la explicitamente importa por causa do D10: um item com 3 ou 5
    alternativas não pode aceitar uma letra que não existe nele.
    """
    valid = _normalize_labels(labels)
    text = _strip_accents(raw_text or "")
    if not text.strip():
        return ParseResult(label=None, strategy="", reason=REASON_EMPTY)

    for name, strategy in _CASCADE:
        candidates = _dedupe(strategy(text, valid))
        if len(candidates) == 1:
            return ParseResult(label=candidates[0], strategy=name, reason="", candidates=candidates)
        if len(candidates) > 1:
            # Duas letras plausíveis no MESMO nível de evidência ("C ou D"): não
            # há critério para escolher, e escolher seria inventar um dado.
            return ParseResult(
                label=None, strategy=name, reason=REASON_AMBIGUOUS, candidates=candidates
            )

    # Nenhuma estratégia achou letra válida. Distinguimos "o modelo escreveu uma
    # letra que não existe neste item" (útil: sinaliza prompt ou `n_alternatives`
    # errados) de "o modelo não escreveu letra nenhuma" (prosa pura).
    #
    # O filtro `_is_one_letter_word` é o MESMO da estratégia 5, e por identidade
    # de motivo: "a", "e" e "o" são palavras do português e aparecem em quase
    # toda prosa. Sem ele, "Acho que o falante fala de futebol" seria reportado
    # como `letra_fora_do_vocabulario` por causa do artigo "o", e a categoria
    # deixaria de sinalizar o que existe para sinalizar — prompt ou
    # `n_alternatives` errados.
    foreign = _dedupe(
        [
            char.upper()
            for index, char in _isolated_letters(text)
            if char.upper() not in valid and not _is_one_letter_word(text, index, char)
        ]
    )
    if foreign:
        return ParseResult(label=None, strategy="", reason=REASON_OUT_OF_VOCAB, candidates=foreign)
    return ParseResult(label=None, strategy="", reason=REASON_NO_LETTER)
