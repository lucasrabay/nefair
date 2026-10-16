"""Adaptadores multimodais (áudio → resposta).

São a via **nativa** da comparação da Etapa 5: o modelo recebe o áudio da janela
junto com o enunciado, sem nenhuma transcrição no meio. O `Protocol` é `AudioQA`
(`answer(prompt, audio_path) -> QAResult`).

Os três provedores esperam o áudio **embutido na mensagem**; nenhum aceita um
caminho de arquivo local. A Etapa 2 grava WAV 16 kHz mono, que os três aceitam.

Como o áudio entra na requisição é o que mais difere entre eles, e é a parte mais
fácil de errar:

| Provedor  | Forma |
|-----------|-------|
| OpenAI    | content part `input_audio` com o base64 **puro** |
| DashScope | o mesmo campo, mas com o prefixo `data:audio/wav;base64,` |
| Google    | `types.Part.from_bytes(data=<bytes>, mime_type="audio/wav")` |

Regras herdadas de `models/llm/clients.py`: nenhum SDK no topo do módulo, cliente
injetável para teste, resposta sem limpeza, erros subindo para o runner.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nefair.models.base import AudioQA, FakeAudioQA, ModelSpec, QAResult
from nefair.models.providers import (
    audio_b64,
    chat_params,
    qa_from_google_response,
    qa_from_openai_response,
    require_env,
)

ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_GOOGLE_API_KEY = "GOOGLE_API_KEY"
ENV_QWEN_API_KEY = "QWEN_API_KEY"
ENV_QWEN_BASE_URL = "QWEN_BASE_URL"

# Formato gravado pela Etapa 2 (`corpus/audio.py`): WAV PCM 16 kHz mono.
AUDIO_FORMAT = "wav"
AUDIO_MIME = "audio/wav"


def _audio_message(prompt: str, data: str) -> list[dict[str, Any]]:
    """Mensagem multimodal no formato compartilhado por OpenAI e DashScope.

    `data` já vem pronto porque é justamente aí que os dois divergem: a OpenAI
    espera o base64 puro, o DashScope espera o prefixo `data:`.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": data, "format": AUDIO_FORMAT}},
            ],
        }
    ]


@dataclass
class OpenAIAudioQA:
    """Modelo de áudio da OpenAI (via nativa da família `openai`).

    `modalities=["text"]` não é detalhe: sem isso o modelo pode responder **em
    áudio**, e a resposta textual vem vazia — que o parser contaria como não
    interpretável, inflando a taxa por um erro de configuração nosso, e não por
    falha do modelo. É exatamente o tipo de defeito que produziria um resultado
    plausível e errado.
    """

    spec: ModelSpec
    client: Any = None

    def _ensure_client(self) -> Any:
        if self.client is None:
            from openai import OpenAI  # import preguiçoso: extra opcional `providers`

            self.client = OpenAI(
                api_key=require_env(ENV_OPENAI_API_KEY, who=f"O modelo '{self.spec.key}'")
            )
        return self.client

    def answer(self, prompt: str, audio_path: Path) -> QAResult:
        response = self._ensure_client().chat.completions.create(
            model=self.spec.model_id,
            messages=_audio_message(prompt, audio_b64(audio_path)),
            modalities=["text"],
            **chat_params(self.spec.params),
        )
        return qa_from_openai_response(response, self.spec.key)


@dataclass
class GoogleAudioQA:
    """Gemini com entrada de áudio (via nativa da família `google`).

    Janelas de 30–60 s cabem folgadamente no limite de requisição inline, então o
    áudio vai embutido. O `client.files.upload(...)` só seria necessário para
    áudios grandes, e traria um estado remoto a limpar depois.
    """

    spec: ModelSpec
    client: Any = None
    types: Any = None

    def _ensure_client(self) -> tuple[Any, Any]:
        if self.client is None or self.types is None:
            from google import genai  # import preguiçoso: extra opcional `providers`
            from google.genai import types as genai_types

            if self.client is None:
                self.client = genai.Client(
                    api_key=require_env(ENV_GOOGLE_API_KEY, who=f"O modelo '{self.spec.key}'")
                )
            if self.types is None:
                self.types = genai_types
        return self.client, self.types

    def answer(self, prompt: str, audio_path: Path) -> QAResult:
        client, types = self._ensure_client()
        part = types.Part.from_bytes(data=Path(audio_path).read_bytes(), mime_type=AUDIO_MIME)
        response = client.models.generate_content(
            model=self.spec.model_id,
            contents=[prompt, part],
            config=types.GenerateContentConfig(
                temperature=self.spec.params.get("temperature"),
                max_output_tokens=self.spec.params.get("max_output_tokens"),
            ),
        )
        return qa_from_google_response(response, self.spec.key)


@dataclass
class QwenAudioQA:
    """Qwen3-Omni (via nativa da família open-weight).

    Fala com um endpoint compatível com OpenAI definido por `QWEN_BASE_URL`, mas
    com **uma** diferença em relação à OpenAI: o campo `data` do `input_audio`
    leva o prefixo `data:audio/wav;base64,`. Mandar o base64 puro para o DashScope
    (ou o prefixado para a OpenAI) é o erro mais fácil de cometer aqui, e
    `tests/test_providers.py` trava os dois sentidos.

    Pareamento (§Etapa 5): este backbone precisa ser o mesmo de `QwenTextQA`.
    """

    spec: ModelSpec
    client: Any = None

    def _ensure_client(self) -> Any:
        if self.client is None:
            from openai import OpenAI  # import preguiçoso: extra opcional `providers`

            who = f"O modelo '{self.spec.key}'"
            self.client = OpenAI(
                api_key=require_env(ENV_QWEN_API_KEY, who=who),
                base_url=require_env(ENV_QWEN_BASE_URL, who=who),
            )
        return self.client

    def answer(self, prompt: str, audio_path: Path) -> QAResult:
        data = f"data:{AUDIO_MIME};base64,{audio_b64(audio_path)}"
        response = self._ensure_client().chat.completions.create(
            model=self.spec.model_id,
            messages=_audio_message(prompt, data),
            **chat_params(self.spec.params),
        )
        return qa_from_openai_response(response, self.spec.key)


# Provedores que este módulo sabe construir, em ordem estável.
AUDIO_QA_PROVIDERS: tuple[str, ...] = ("fake", "openai", "google", "qwen")


def build_audio_qa(
    spec: ModelSpec,
    *,
    seed: int = 0,
    n_alternatives: int = 4,
    force_fake: bool = False,
) -> AudioQA:
    """Devolve o adaptador `AudioQA` correspondente ao `provider` do `spec`.

    `force_fake` troca o `provider` do spec para `"fake"` antes de construir, de
    modo que o `fingerprint()` — e portanto a chave de cache — de uma execução
    falsa nunca colida com o de uma real.
    """
    if force_fake or spec.provider == "fake":
        return FakeAudioQA(
            spec=replace(spec, provider="fake"), seed=seed, n_alternatives=n_alternatives
        )
    if spec.provider == "openai":
        return OpenAIAudioQA(spec=spec)
    if spec.provider == "google":
        return GoogleAudioQA(spec=spec)
    if spec.provider == "qwen":
        return QwenAudioQA(spec=spec)
    raise ValueError(
        f"Modelo multimodal '{spec.key}': provider '{spec.provider}' desconhecido; "
        f"esperado um de {AUDIO_QA_PROVIDERS}."
    )
