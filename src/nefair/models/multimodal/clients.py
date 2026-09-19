"""Adaptadores multimodais (áudio → resposta) — **stubs de contrato**.

São a via **nativa** da comparação da Etapa 5: o modelo recebe o áudio da janela
junto com o enunciado, sem nenhuma transcrição no meio. O `Protocol` é `AudioQA`
(`answer(prompt, audio_path) -> QAResult`).

Motivo dos stubs e regra do import preguiçoso: idênticos aos de
`models/llm/clients.py`. O que congelamos aqui é o contrato — como o áudio entra
na requisição de cada provedor, que é justamente a parte que difere entre eles e
a mais fácil de errar na pressa do dia da execução.

Detalhe que vale por si: todos os três provedores esperam o áudio **embutido na
mensagem** (base64 ou upload prévio), nenhum aceita um caminho de arquivo local.
A Etapa 2 grava WAV 16 kHz mono, que é o formato que os três aceitam.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from nefair.models.base import AudioQA, FakeAudioQA, ModelSpec, QAResult

ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_GOOGLE_API_KEY = "GOOGLE_API_KEY"
ENV_QWEN_API_KEY = "DASHSCOPE_API_KEY"
ENV_QWEN_BASE_URL = "QWEN_BASE_URL"


@dataclass
class OpenAIAudioQA:
    """Modelo de áudio da OpenAI (GPT-4o Audio ou sucessor).

    Contrato que a implementação real precisa cumprir:

    1. **Import preguiçoso** de `openai` dentro de `answer`; extra `[providers]`.
       `base64` também é importado aqui, junto.
    2. Exigir `os.environ[ENV_OPENAI_API_KEY]`.
    3. Ler os bytes de `audio_path`, codificar em base64 e montar o conteúdo
       multimodal: `messages=[{"role": "user", "content": [
       {"type": "text", "text": prompt},
       {"type": "input_audio", "input_audio": {"data": <b64>, "format": "wav"}}]}]`.
    4. Chamar `client.chat.completions.create(model=spec.model_id,
       messages=..., modalities=["text"], **spec.params)`. `modalities=["text"]`
       importa: sem isso o modelo pode responder em áudio, e a resposta textual
       vem vazia — que o parser contaria como não interpretável, inflando a taxa
       por um erro de configuração nosso, não do modelo.
    5. Campos da resposta: iguais aos de `OpenAITextQA`
       (`choices[0].message.content`, `choices[0].finish_reason`, `usage`).
    6. Erros de provedor **sobem**; retry e backoff são do runner.
    """

    spec: ModelSpec

    def answer(self, prompt: str, audio_path: Path) -> QAResult:
        raise NotImplementedError(
            "OpenAIAudioQA.answer não implementado (sem credencial para validar). "
            "Para implementar: extra opcional `providers` (openai); "
            "`from openai import OpenAI` e `import base64` DENTRO deste método; exigir "
            f"{ENV_OPENAI_API_KEY}; enviar o WAV como content part "
            "{'type':'input_audio','input_audio':{'data':<b64>,'format':'wav'}} com "
            f"modalities=['text'] e model='{self.spec.model_id}', "
            f"params={dict(self.spec.params)}; devolver "
            "QAResult(raw_text=response.choices[0].message.content, "
            f"model_key='{self.spec.key}'). Áudio pedido: {audio_path}."
        )


@dataclass
class GoogleAudioQA:
    """Gemini com entrada de áudio.

    Contrato que a implementação real precisa cumprir:

    1. **Import preguiçoso** de `google.genai` dentro de `answer`
       (`from google import genai` e `from google.genai import types`).
    2. Exigir `os.environ[ENV_GOOGLE_API_KEY]`.
    3. Duas formas de mandar o áudio, e a escolha depende do tamanho: janelas de
       30–60 s cabem folgadamente no limite de requisição inline, então usar
       `types.Part.from_bytes(data=audio_path.read_bytes(),
       mime_type="audio/wav")` e passar `contents=[prompt, part]`. O
       `client.files.upload(...)` só é necessário para áudios grandes, e traria
       um estado remoto a limpar depois.
    4. Configuração: `config=types.GenerateContentConfig(
       temperature=spec.params["temperature"],
       max_output_tokens=spec.params["max_output_tokens"])`.
    5. Campos da resposta: `response.text`;
       `response.candidates[0].finish_reason`; `response.usage_metadata`.
       `response.text is None` (bloqueio de segurança) vira `raw_text=""`.
    6. Erros de provedor **sobem**.
    """

    spec: ModelSpec

    def answer(self, prompt: str, audio_path: Path) -> QAResult:
        raise NotImplementedError(
            "GoogleAudioQA.answer não implementado (sem credencial para validar). "
            "Para implementar: extra opcional `providers` (google-genai); "
            "`from google import genai` e `from google.genai import types` DENTRO "
            f"deste método; exigir {ENV_GOOGLE_API_KEY}; chamar "
            f"client.models.generate_content(model='{self.spec.model_id}', "
            "contents=[prompt, types.Part.from_bytes(data=<wav>, mime_type='audio/wav')], "
            f"config=...{dict(self.spec.params)}); devolver QAResult("
            f"raw_text=response.text, model_key='{self.spec.key}'). "
            f"Áudio pedido: {audio_path}."
        )


@dataclass
class QwenAudioQA:
    """Qwen2.5-Omni / Qwen2-Audio — configuração open-weight da via nativa.

    Contrato que a implementação real precisa cumprir:

    - **Via DashScope compatível com OpenAI**: `from openai import OpenAI` com
      `base_url=os.environ[ENV_QWEN_BASE_URL]` e
      `api_key=os.environ[ENV_QWEN_API_KEY]`; o áudio vai como content part
      `{"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64,<b64>",
      "format": "wav"}}` — repare que o DashScope espera o prefixo `data:` no
      campo, diferente da OpenAI, que espera só o base64 puro.
    - **Pesos locais**: extra `[local]`;
      `transformers.Qwen2AudioForConditionalGeneration` + `AutoProcessor`, com o
      áudio lido em 16 kHz (`librosa`/`soundfile`) e passado ao processador.
      Carregar o modelo **uma única vez** em `self._model`.

    Import preguiçoso dentro de `answer` nos dois casos; `raw_text` sem limpeza;
    erros subindo para o runner.

    Pareamento (§Etapa 5): este backbone precisa ser o mesmo de `QwenTextQA`.
    """

    spec: ModelSpec

    def answer(self, prompt: str, audio_path: Path) -> QAResult:
        raise NotImplementedError(
            "QwenAudioQA.answer não implementado (sem credencial nem pesos para "
            "validar). Para implementar: via DashScope, `from openai import OpenAI` "
            f"DENTRO deste método com base_url={ENV_QWEN_BASE_URL} e "
            f"api_key={ENV_QWEN_API_KEY}, mandando o áudio como input_audio com "
            "prefixo 'data:audio/wav;base64,'; via local, "
            "`transformers.Qwen2AudioForConditionalGeneration` + `AutoProcessor` com o "
            f"extra opcional `local`. Modelo: '{self.spec.model_id}', "
            f"params={dict(self.spec.params)}. Áudio pedido: {audio_path}."
        )


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
    falsa nunca colida com o de uma execução real.
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
