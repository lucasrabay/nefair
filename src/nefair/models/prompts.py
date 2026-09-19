"""Templates de prompt do item de múltipla escolha, versionados.

Existe **um** template de MCQ no projeto inteiro, em duas faces:

- `render_generation_prompt` — o que o LLM gerador recebe na Etapa 3 (janela-alvo
  + trechos de outras janelas da mesma entrevista, §D2);
- `render_answer_prompt` — o que qualquer modelo avaliado recebe diante de um
  item, tanto no controle textual da Etapa 3 (§D3: **sem** áudio e **sem**
  transcrição) quanto nas condições `asr`, `reference` e `audio` da Etapa 5.

`PROMPT_VERSION` entra na chave de cache do runner (Etapa 5) e é ecoado em todo
relatório. Mudar qualquer byte de um template **sem** mudar a versão é um bug
grave: o cache devolveria respostas dadas a outro prompt e a comparação entre
modelos passaria a misturar duas perguntas diferentes sob o mesmo nome. O teste
de snapshot em `tests/test_prompts.py` existe justamente para que uma alteração
acidental quebre a suíte em vez de contaminar os resultados em silêncio.

As funções são **puras**: mesma entrada, mesma string, sem ler config, sem ler
disco e sem consultar relógio. Isso é o que permite comparar artefatos byte a
byte entre execuções.
"""

from __future__ import annotations

from collections.abc import Sequence

from nefair.schema import ALTERNATIVE_LABELS

# Versão do template de MCQ. Incrementar SEMPRE que qualquer texto abaixo mudar.
PROMPT_VERSION = "mcq-v1"

# Marcador que o gerador deve devolver: um único objeto JSON, sem cercas de
# código. Fica aqui (e não em generate.py) para que prompt e parser compartilhem
# literalmente a mesma definição de contrato.
GENERATION_RESPONSE_ROOT_KEY = "items"


def labels_for(n_alternatives: int) -> tuple[str, ...]:
    """Rótulos canônicos das `n_alternatives` primeiras alternativas.

    Fatia `ALTERNATIVE_LABELS` em vez de assumir quatro letras, porque o nº de
    alternativas é uma decisão aberta (D10) e a análise de sensibilidade pode
    regerar itens com 3 ou 5.
    """
    if not 2 <= n_alternatives <= len(ALTERNATIVE_LABELS):
        raise ValueError(
            f"n_alternatives={n_alternatives} fora da faixa suportada "
            f"[2, {len(ALTERNATIVE_LABELS)}] (rótulos: {ALTERNATIVE_LABELS})."
        )
    return ALTERNATIVE_LABELS[:n_alternatives]


def _label_enumeration(labels: Sequence[str]) -> str:
    """'A, B, C ou D' — enumeração em português, para a instrução de resposta."""
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])} ou {labels[-1]}"


# --------------------------------------------------------------------------- #
# Prompt de GERAÇÃO (Etapa 3, D2)
# --------------------------------------------------------------------------- #
def render_generation_prompt(
    *,
    target_window_id: str,
    target_transcript: str,
    context_excerpts: Sequence[tuple[str, str]],
    n_alternatives: int,
    n_items: int = 1,
    require_provenance: bool = True,
) -> str:
    """Prompt do LLM gerador de itens.

    `context_excerpts` são pares `(window_id, transcrição)` de **outras** janelas
    da mesma entrevista. Eles são o mecanismo do D2: o texto do TCC diz que os
    distratores são "extraídos de outros trechos da mesma entrevista", e a única
    forma de tornar essa afirmação verificável é (a) dar os trechos ao gerador e
    (b) exigir que cada distrator declare em qual deles se ancorou.

    Com `require_provenance=False` a exigência (b) some do prompt — mas aí o
    relatório precisa dizer que a proveniência dos distratores não foi checada.
    """
    labels = labels_for(n_alternatives)
    n_distractors = n_alternatives - 1
    lines: list[str] = []

    lines.append(
        "Você é um especialista em elaboração de itens de compreensão de fala "
        "espontânea em português brasileiro."
    )
    lines.append("")
    lines.append("TAREFA")
    lines.append(
        f"Leia a TRANSCRIÇÃO-ALVO abaixo e escreva exatamente {n_items} "
        f"item(ns) de múltipla escolha, cada um com {n_alternatives} "
        f"alternativas: 1 correta e {n_distractors} incorretas."
    )
    lines.append("")
    lines.append("REGRAS")
    lines.append(
        "1. A alternativa correta tem de ser verificável APENAS na "
        "TRANSCRIÇÃO-ALVO, sem conhecimento geral e sem inferência sobre o resto "
        "da entrevista."
    )
    lines.append(
        f"2. Cada uma das {n_distractors} alternativas incorretas (distratores) "
        "tem de ser ancorada em um dos TRECHOS DE CONTEXTO, que vêm de OUTRAS "
        "janelas da MESMA entrevista."
    )
    if require_provenance:
        lines.append(
            "3. Toda alternativa tem de declarar `source_window_id`: a correta "
            "usa o id da janela-alvo; cada distrator usa o id do trecho de "
            "contexto em que se ancorou."
        )
    else:
        lines.append(
            "3. O campo `source_window_id` é opcional nesta execução; quando "
            "presente, use o id da janela de onde a alternativa veio."
        )
    lines.append(
        "4. Duas alternativas do mesmo item nunca podem ter o mesmo texto, nem texto vazio."
    )
    lines.append(
        "5. A pergunta não pode ser respondível só olhando as alternativas: não "
        "use pistas de formato (alternativa mais longa, mais específica ou mais "
        "plausível) nem fatos de conhecimento geral."
    )
    lines.append("6. Escreva pergunta e alternativas em português brasileiro.")
    lines.append("")
    lines.append(f"TRANSCRIÇÃO-ALVO (janela {target_window_id})")
    lines.append(target_transcript)
    lines.append("")
    lines.append("TRECHOS DE CONTEXTO (outras janelas da mesma entrevista)")
    if context_excerpts:
        for window_id, excerpt in context_excerpts:
            lines.append(f"- (janela {window_id}) {excerpt}")
    else:
        lines.append("- (nenhum trecho de contexto disponível para esta entrevista)")
    lines.append("")
    lines.append("FORMATO DA RESPOSTA")
    lines.append(
        "Responda com UM ÚNICO objeto JSON, sem cercas de código, sem "
        "comentários e sem texto antes ou depois:"
    )
    lines.append("{")
    lines.append(f'  "{GENERATION_RESPONSE_ROOT_KEY}": [')
    lines.append("    {")
    lines.append('      "question": "...",')
    lines.append('      "alternatives": [')
    lines.append(
        '        {"text": "...", "is_correct": true, "source_window_id": "<id da janela>"}'
    )
    lines.append("      ]")
    lines.append("    }")
    lines.append("  ]")
    lines.append("}")
    lines.append(
        f"Exatamente {n_items} objeto(s) em `{GENERATION_RESPONSE_ROOT_KEY}`, "
        f"exatamente {n_alternatives} objeto(s) em `alternatives` e exatamente "
        "1 com `is_correct` verdadeiro."
    )
    lines.append(
        "Não escreva os rótulos "
        f"({', '.join(labels)}) dentro de `text`: a ordem das alternativas é "
        "embaralhada depois."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Prompt de RESPOSTA (Etapa 3 §D3 e Etapa 5)
# --------------------------------------------------------------------------- #
def render_answer_prompt(
    *,
    question: str,
    alternative_texts: Sequence[str],
    transcript: str | None = None,
    audio_provided: bool = False,
) -> str:
    """Prompt que apresenta um item a um modelo, na ordem em que os textos vierem.

    A ordem de `alternative_texts` é a ordem EXIBIDA (já permutada por quem
    chama) — esta função não embaralha nada, para que a permutação fique
    registrada num único lugar e possa ser reproduzida.

    Três usos, distinguidos só pelos argumentos:

    - `transcript=None, audio_provided=False` → **controle textual** (D3): o
      modelo vê pergunta e alternativas e nada mais. Item respondido assim é um
      item ruim, porque não depende da fala;
    - `transcript=<texto>` → condições `asr` (hipótese do ASR) e `reference`
      (transcrição de referência) da Etapa 5;
    - `audio_provided=True` → condição `audio` do modelo multimodal nativo.

    O aviso explícito de ausência no caso do controle textual não é enfeite: sem
    ele, um modelo tende a alegar falta de contexto em vez de escolher, e o
    filtro mediria recusa em vez de respondibilidade.
    """
    labels = labels_for(len(alternative_texts))
    lines: list[str] = []

    lines.append("Responda à pergunta de múltipla escolha abaixo.")
    lines.append("")
    if transcript is not None:
        lines.append("TRANSCRIÇÃO DA FALA")
        lines.append(transcript)
        lines.append("")
    if audio_provided:
        lines.append("ÁUDIO")
        lines.append("O áudio da fala foi fornecido junto com esta mensagem.")
        lines.append("")
    if transcript is None and not audio_provided:
        lines.append("CONTEXTO")
        lines.append(
            "Você NÃO recebeu o áudio nem a transcrição da fala. Ainda assim, "
            "escolha a alternativa que considerar mais provável."
        )
        lines.append("")
    lines.append("PERGUNTA")
    lines.append(question)
    lines.append("")
    lines.append("ALTERNATIVAS")
    for label, text in zip(labels, alternative_texts, strict=True):
        lines.append(f"{label}) {text}")
    lines.append("")
    lines.append(
        f"Responda apenas com a letra da alternativa correta "
        f"({_label_enumeration(labels)}), sem explicação e sem pontuação."
    )
    return "\n".join(lines)
