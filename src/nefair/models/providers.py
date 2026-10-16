"""Utilidades compartilhadas pelos adaptadores de provedor.

Existe para que as quatro operações que todo adaptador precisa fazer — exigir
credencial, traduzir parâmetros, embutir áudio e ler a resposta — tenham **uma**
implementação, e não uma cópia por arquivo. A Etapa 0 já pagou esse preço uma vez:
a geração de relatório ficou duplicada nos quatro blocos por falta de um módulo
comum, e virou 17% das linhas de função do projeto.

Regra herdada de `models/base.py` e verificada em `tests/test_providers.py`:
**nenhum SDK de provedor é importado aqui**. Este módulo só mexe em `os.environ`,
bytes e objetos de resposta já construídos por quem chamou.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nefair.models.base import QAResult

# Nome no YAML -> nome no SDK. A tradução mora aqui, e não no `models.yaml`, para
# que o YAML siga legível como decisão metodológica ("quantos tokens de saída este
# experimento permite") em vez de payload de SDK.
#
# Vale para os três SDKs que usamos com esta forma de chamada: OpenAI, DashScope
# (compatível com OpenAI) e Anthropic, que também chamam o campo de `max_tokens`.
_PARAM_ALIASES: Mapping[str, str] = {"max_output_tokens": "max_tokens"}


def require_env(name: str, *, who: str) -> str:
    """Valor de uma variável de ambiente, ou erro que diz o que falta e para quem.

    Falhar aqui, antes da primeira requisição, é deliberado: um cliente construído
    sem credencial erraria com 401 a cada tentativa, e o runner gastaria todos os
    retries de cada item antes de alguém perceber que faltava exportar a chave.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{who} exige a variável de ambiente {name}, que está ausente ou vazia. "
            f"Preencha-a em `.env` (veja `.env.example`) ou exporte-a no ambiente. "
            "Para exercitar o pipeline sem credencial, use `--fake`."
        )
    return value


def chat_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Parâmetros do `models.yaml` traduzidos para o nome que o SDK espera.

    Hoje só `max_output_tokens` muda de nome. O resto passa como está, para que
    acrescentar um parâmetro no YAML não exija mexer em código.
    """
    return {_PARAM_ALIASES.get(key, key): value for key, value in params.items()}


def audio_b64(audio_path: Path) -> str:
    """WAV lido do disco e codificado em base64 (sem prefixo `data:`).

    O prefixo é responsabilidade de quem chama, porque os provedores divergem
    justamente aí: a OpenAI quer o base64 puro, o DashScope quer
    `data:audio/wav;base64,` na frente. Ver `tests/test_providers.py`.
    """
    return base64.b64encode(Path(audio_path).read_bytes()).decode("ascii")


def qa_from_openai_response(response: Any, model_key: str) -> QAResult:
    """`QAResult` a partir de uma resposta no formato OpenAI chat completions.

    Serve a quatro adaptadores: OpenAI (texto e áudio) e Qwen via DashScope
    (texto e áudio), que compartilham o mesmo formato de resposta.

    Três ausências são tratadas como **dado**, não como erro, porque é isso que
    elas são no experimento:

    - sem `choices`: resposta vazia do provedor;
    - `content is None`: recusa ou filtro de conteúdo;
    - sem `usage`: o provedor não reportou contagem.

    Nos três casos o `raw_text` vai vazio e o parser conta como não interpretável.
    Inventar um texto aqui destruiria justamente o número que o TCC reporta.
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        return QAResult(raw_text="", model_key=model_key, finish_reason="sem_choices")

    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    finish_reason = getattr(choice, "finish_reason", None)

    usage_obj = getattr(response, "usage", None)
    usage: dict[str, int] = {}
    if usage_obj is not None:
        prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
        completion_tokens = getattr(usage_obj, "completion_tokens", None)
        if prompt_tokens is not None:
            usage["input_tokens"] = int(prompt_tokens)
        if completion_tokens is not None:
            usage["output_tokens"] = int(completion_tokens)

    return QAResult(
        raw_text=content if content is not None else "",
        model_key=model_key,
        finish_reason=str(finish_reason) if finish_reason is not None else "",
        usage=usage,
    )


def qa_from_google_response(response: Any, model_key: str) -> QAResult:
    """`QAResult` a partir de uma resposta do `google.genai`.

    Compartilhada por `GoogleTextQA` e `GoogleAudioQA`. `response.text` vem `None`
    quando o candidato foi bloqueado por filtro de segurança: isso vira
    `raw_text=""` **com o `finish_reason` preenchido**, para que o relatório
    consiga distinguir "o modelo errou" de "o provedor bloqueou".
    """
    text = getattr(response, "text", None)

    candidates = getattr(response, "candidates", None) or []
    finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None

    meta = getattr(response, "usage_metadata", None)
    usage: dict[str, int] = {}
    if meta is not None:
        prompt_tokens = getattr(meta, "prompt_token_count", None)
        completion_tokens = getattr(meta, "candidates_token_count", None)
        if prompt_tokens is not None:
            usage["input_tokens"] = int(prompt_tokens)
        if completion_tokens is not None:
            usage["output_tokens"] = int(completion_tokens)

    return QAResult(
        raw_text=text if text is not None else "",
        model_key=model_key,
        finish_reason=str(finish_reason) if finish_reason is not None else "",
        usage=usage,
    )
