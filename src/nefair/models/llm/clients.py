"""Adaptadores de LLM de texto — **stubs de contrato**, ainda não implementados.

Mesmo raciocínio de `models/asr/whisper.py`: sem credencial não há como validar
um cliente real, e código não verificável envelhece mal. O que congelamos agora é
o contrato — SDK, variável de ambiente, chamada e campo da resposta.

Estes adaptadores implementam o `Protocol` `TextQA` (`answer(prompt) ->
QAResult`) e são usados nas condições `asr` e `reference`: o LLM recebe **texto**
(hipótese do ASR ou transcrição de referência) e devolve a resposta bruta.

Ponto de projeto que vale repetir: `QAResult` carrega `raw_text`, e **não** a
letra escolhida. A extração é de `run/parse.py`, testada à parte e capaz de
contar a resposta não interpretável em vez de imputá-la. Um adaptador que
tentasse "limpar" a resposta aqui destruiria justamente o número que o TCC
precisa reportar.

Nenhum SDK é importado no topo deste módulo.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from nefair.models.base import FakeTextQA, ModelSpec, QAResult, TextQA

ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_GOOGLE_API_KEY = "GOOGLE_API_KEY"
# Qwen via serviço compatível com OpenAI (DashScope) ou endpoint próprio.
ENV_QWEN_API_KEY = "DASHSCOPE_API_KEY"
ENV_QWEN_BASE_URL = "QWEN_BASE_URL"


@dataclass
class OpenAITextQA:
    """LLM de texto da OpenAI (ponta da cascata da família `openai`).

    Contrato que a implementação real precisa cumprir:

    1. **Import preguiçoso** de `openai` dentro de `answer`
       (`from openai import OpenAI`); extra opcional `[providers]`.
    2. Exigir `os.environ[ENV_OPENAI_API_KEY]`, com erro explícito na ausência.
    3. Chamar `client.chat.completions.create(model=spec.model_id,
       messages=[{"role": "user", "content": prompt}], **spec.params)` —
       `spec.params` já traz `temperature` e `max_output_tokens`; converter o
       nome para o do SDK (`max_tokens`/`max_completion_tokens`) **no adaptador**,
       nunca no YAML, para que o YAML continue legível como decisão
       metodológica.
    4. Campos da resposta: texto em `response.choices[0].message.content`;
       `finish_reason` em `response.choices[0].finish_reason`; contagens em
       `response.usage.prompt_tokens` / `response.usage.completion_tokens`.
    5. Devolver `QAResult(raw_text=<conteúdo, sem nenhuma limpeza>,
       model_key=spec.key, finish_reason=..., usage={"input_tokens": ...,
       "output_tokens": ...})`.
    6. Conteúdo `None` (recusa, filtro de conteúdo) vira `raw_text=""` — que o
       parser conta como não interpretável. Não inventar um texto.
    7. Erros de provedor **sobem**; retry e backoff são do runner.
    """

    spec: ModelSpec

    def answer(self, prompt: str) -> QAResult:
        raise NotImplementedError(
            "OpenAITextQA.answer não implementado (sem credencial para validar). "
            "Para implementar: extra opcional `providers` (openai); "
            "`from openai import OpenAI` DENTRO deste método; exigir "
            f"{ENV_OPENAI_API_KEY}; chamar client.chat.completions.create("
            f"model='{self.spec.model_id}', messages=[{{'role':'user',"
            f"'content':prompt}}], **{dict(self.spec.params)}); devolver "
            "QAResult(raw_text=response.choices[0].message.content, "
            f"model_key='{self.spec.key}', "
            "finish_reason=response.choices[0].finish_reason). "
            f"Prompt tinha {len(prompt)} caracteres."
        )


@dataclass
class GoogleTextQA:
    """LLM de texto do Google (Gemini), ponta da cascata da família `google`.

    Contrato que a implementação real precisa cumprir:

    1. **Import preguiçoso** de `google.genai` dentro de `answer`
       (`from google import genai`); extra opcional `[providers]`
       (`google-genai`).
    2. Exigir `os.environ[ENV_GOOGLE_API_KEY]`, com erro explícito na ausência.
    3. Chamar `client.models.generate_content(model=spec.model_id,
       contents=prompt, config=types.GenerateContentConfig(
       temperature=spec.params["temperature"],
       max_output_tokens=spec.params["max_output_tokens"]))`.
    4. Campos da resposta: texto em `response.text`; motivo de parada em
       `response.candidates[0].finish_reason`; contagens em
       `response.usage_metadata.prompt_token_count` /
       `.candidates_token_count`.
    5. `response.text` pode ser `None` quando o candidato foi bloqueado por
       filtro de segurança: virar `raw_text=""` e preencher `finish_reason` com o
       motivo reportado. Resposta bloqueada é um dado do experimento (contado
       como não interpretável), não um erro a esconder.
    6. Erros de provedor **sobem**.
    """

    spec: ModelSpec

    def answer(self, prompt: str) -> QAResult:
        raise NotImplementedError(
            "GoogleTextQA.answer não implementado (sem credencial para validar). "
            "Para implementar: extra opcional `providers` (google-genai); "
            "`from google import genai` DENTRO deste método; exigir "
            f"{ENV_GOOGLE_API_KEY}; chamar client.models.generate_content("
            f"model='{self.spec.model_id}', contents=prompt, config=...{dict(self.spec.params)}); "
            f"devolver QAResult(raw_text=response.text, model_key='{self.spec.key}', "
            "finish_reason=response.candidates[0].finish_reason). "
            f"Prompt tinha {len(prompt)} caracteres."
        )


@dataclass
class QwenTextQA:
    """Backbone Qwen em modo texto (ponta da cascata da família open-weight).

    Duas hospedagens possíveis, e o adaptador precisa suportar a que for
    escolhida (a decisão entra em `models.yaml` como `provider`/`model_id`):

    - **DashScope compatível com OpenAI**: `from openai import OpenAI` com
      `base_url=os.environ[ENV_QWEN_BASE_URL]` e
      `api_key=os.environ[ENV_QWEN_API_KEY]`; daí a chamada é idêntica à de
      `OpenAITextQA` (mesmos campos de resposta).
    - **Pesos locais**: `transformers.AutoModelForCausalLM` +
      `AutoTokenizer`, extra opcional `[local]`, com o modelo carregado **uma
      única vez** em `self._model`.

    Em ambos os casos: import preguiçoso dentro de `answer`, `raw_text` sem
    nenhuma limpeza, erros subindo para o runner.

    Nota de pareamento (§Etapa 5): o Qwen de texto e o Qwen multimodal precisam
    ser o **mesmo backbone**, senão a diferença entre a via nativa e a cascata
    deixa de ser atribuível à via.
    """

    spec: ModelSpec

    def answer(self, prompt: str) -> QAResult:
        raise NotImplementedError(
            "QwenTextQA.answer não implementado (sem credencial nem pesos para "
            "validar). Para implementar: via DashScope, `from openai import OpenAI` "
            f"DENTRO deste método com base_url={ENV_QWEN_BASE_URL} e "
            f"api_key={ENV_QWEN_API_KEY}; via local, `transformers.AutoModelForCausalLM` "
            f"com o extra opcional `local`. Modelo: '{self.spec.model_id}', "
            f"params={dict(self.spec.params)}. Prompt tinha {len(prompt)} caracteres."
        )


# Provedores que este módulo sabe construir, em ordem estável.
TEXT_QA_PROVIDERS: tuple[str, ...] = ("fake", "openai", "google", "qwen")


def build_text_qa(
    spec: ModelSpec,
    *,
    seed: int = 0,
    n_alternatives: int = 4,
    force_fake: bool = False,
) -> TextQA:
    """Devolve o adaptador `TextQA` correspondente ao `provider` do `spec`.

    Ver `models/asr/whisper.build_asr` para o porquê de `force_fake` trocar o
    `provider` do spec antes de construir o falso: sem isso, uma execução falsa
    poluiria o cache de uma execução real (mesmas chaves, respostas inventadas).
    """
    if force_fake or spec.provider == "fake":
        return FakeTextQA(
            spec=replace(spec, provider="fake"), seed=seed, n_alternatives=n_alternatives
        )
    if spec.provider == "openai":
        return OpenAITextQA(spec=spec)
    if spec.provider == "google":
        return GoogleTextQA(spec=spec)
    if spec.provider == "qwen":
        return QwenTextQA(spec=spec)
    raise ValueError(
        f"Modelo de texto '{spec.key}': provider '{spec.provider}' desconhecido; "
        f"esperado um de {TEXT_QA_PROVIDERS}."
    )
