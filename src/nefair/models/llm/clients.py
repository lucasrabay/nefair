"""Adaptadores de LLM de texto.

Implementam o `Protocol` `TextQA` (`answer(prompt) -> QAResult`) e são usados nas
condições `asr` e `reference`: o LLM recebe **texto** (hipótese do ASR ou
transcrição de referência) e devolve a resposta bruta.

Ponto de projeto que vale repetir: `QAResult` carrega `raw_text`, e **não** a
letra escolhida. A extração é de `run/parse.py`, testada à parte e capaz de contar
a resposta não interpretável em vez de imputá-la. Um adaptador que tentasse
"limpar" a resposta aqui destruiria justamente o número que o TCC precisa reportar.

**Nenhum SDK é importado no topo deste módulo.** Cada adaptador importa o seu
dentro de `_ensure_client`, que só roda quando o cliente ainda não existe. Os
campos `client` (e `types`, no Google) são a costura que permite testar tudo sem
credencial e sem SDK instalado: o teste injeta um objeto com a forma do provedor.
`tests/test_providers.py` verifica mecanicamente que importar este módulo não traz
nenhum SDK para `sys.modules`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from nefair.models.base import FakeTextQA, ModelSpec, QAResult, TextQA
from nefair.models.providers import (
    chat_params,
    qa_from_google_response,
    qa_from_openai_response,
    require_env,
)

ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_GOOGLE_API_KEY = "GOOGLE_API_KEY"
# Qwen via serviço compatível com OpenAI (DashScope) ou endpoint próprio.
ENV_QWEN_API_KEY = "DASHSCOPE_API_KEY"
ENV_QWEN_BASE_URL = "QWEN_BASE_URL"
# Gerador dos itens (D11), fora do conjunto avaliado.
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"


@dataclass
class OpenAITextQA:
    """LLM de texto da OpenAI (ponta da cascata da família `openai`).

    Erros de provedor **sobem**: retry e backoff são do runner
    (`run/runner.call_with_retry`), para que a política seja uma só no pipeline.
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

    def answer(self, prompt: str) -> QAResult:
        response = self._ensure_client().chat.completions.create(
            model=self.spec.model_id,
            messages=[{"role": "user", "content": prompt}],
            **chat_params(self.spec.params),
        )
        return qa_from_openai_response(response, self.spec.key)


@dataclass
class GoogleTextQA:
    """LLM de texto do Google (Gemini), ponta da cascata da família `google`.

    `response.text` vem `None` quando o candidato cai em filtro de segurança.
    Isso vira `raw_text=""` com o `finish_reason` preenchido: resposta bloqueada é
    um dado do experimento (contado como não interpretável), não um erro a esconder.
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

    def answer(self, prompt: str) -> QAResult:
        client, types = self._ensure_client()
        response = client.models.generate_content(
            model=self.spec.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.spec.params.get("temperature"),
                max_output_tokens=self.spec.params.get("max_output_tokens"),
            ),
        )
        return qa_from_google_response(response, self.spec.key)


@dataclass
class QwenTextQA:
    """Backbone Qwen em modo texto (ponta da cascata da família open-weight).

    Fala com um endpoint **compatível com OpenAI** — DashScope, OpenRouter ou
    outro —, definido por `QWEN_BASE_URL`. É por isso que trocar de hospedagem é
    uma linha no `.env` e não uma mudança de código.

    Rodar os pesos localmente exigiria outro adaptador (provider `local`, extra
    opcional `[local]`, modelo carregado uma única vez). Não está implementado:
    a hospedagem escolhida é API, e um 30B não cabe nos 16 GB da máquina de
    desenvolvimento.

    Pareamento (§Etapa 5): este backbone precisa ser o mesmo de `QwenAudioQA`,
    senão a diferença entre a via nativa e a cascata deixa de ser atribuível à via.
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

    def answer(self, prompt: str) -> QAResult:
        response = self._ensure_client().chat.completions.create(
            model=self.spec.model_id,
            messages=[{"role": "user", "content": prompt}],
            **chat_params(self.spec.params),
        )
        return qa_from_openai_response(response, self.spec.key)


@dataclass
class AnthropicTextQA:
    """Claude como **gerador dos itens** (D11), fora do conjunto avaliado.

    Não entra na comparação: sem entrada de áudio nativa, não teria par na via
    cascata (D9), e usar um modelo avaliado como gerador faria o experimento medir
    em parte a afinidade de cada modelo com o estilo de quem escreveu as perguntas.

    Três diferenças de forma em relação aos outros adaptadores:

    - `max_tokens` é **obrigatório** no SDK da Anthropic, não opcional;
    - o texto vem em `response.content`, que é uma **lista de blocos**. Juntamos os
      blocos de tipo `text`; lista vazia (ou só blocos de outro tipo) vira
      `raw_text=""`, contado como não interpretável. Na prática a resposta vem com
      um bloco `thinking` **antes** do `text` (verificado contra o SDK real), e os
      tokens de raciocínio contam contra `max_tokens`: teto apertado devolve
      `stop_reason="max_tokens"` com nenhum bloco de texto, ou seja, item perdido
      sem erro aparente. Dimensionar `max_output_tokens` para raciocínio + item;
    - **os modelos atuais removeram o controle de amostragem**: `temperature`,
      `top_p` e `top_k` são rejeitados com 400 em Opus 5.5/5/4.8/4.7, Fable 5/5.1
      e Sonnet 5 (e em valor não-padrão em Sonnet 5.5 e Haiku 5.5). Não é
      limitação do SDK: o parâmetro saiu da API. O substituto é
      `output_config.effort`, que este adaptador aceita como `effort` em `params`.
      Ver `_sdk_params`.

    Sobre `effort`: o padrão do Opus 5.5 é `medium`, um nível abaixo do padrão do
    Opus 5. Declará-lo explicitamente no `items.yaml` não é redundância — é o que
    torna o valor reportável na Metodologia e o que o faz entrar no
    `fingerprint()`, e portanto na chave de cache.
    """

    spec: ModelSpec
    client: Any = None

    def _ensure_client(self) -> Any:
        if self.client is None:
            from anthropic import Anthropic  # import preguiçoso: extra `providers`

            self.client = Anthropic(
                api_key=require_env(ENV_ANTHROPIC_API_KEY, who=f"O gerador '{self.spec.key}'")
            )
        return self.client

    def _sdk_params(self) -> dict[str, Any]:
        """Parâmetros do YAML traduzidos, recusando os que a API removeu.

        Descartar `temperature` em silêncio seria pior do que falhar: o config é a
        descrição do experimento e entra no relatório, então o texto afirmaria uma
        temperatura que nunca foi aplicada — e o `fingerprint()` distinguiria duas
        execuções que, na prática, rodaram idênticas.

        Mesma política do portão D8 e da guarda de `PLACEHOLDER`: recusar e dizer
        o que fazer, em vez de seguir sob uma suposição errada.
        """
        params = chat_params(self.spec.params)

        rejeitados = sorted(p for p in ("temperature", "top_p", "top_k") if p in params)
        if rejeitados:
            raise ValueError(
                f"O modelo '{self.spec.key}' ({self.spec.model_id}) declara "
                f"{', '.join(rejeitados)} em `params`, mas os modelos atuais da Anthropic "
                "removeram o controle de amostragem — a API responde 400. Não é limitação "
                "do SDK, e o parâmetro não tem como ser honrado.\n"
                "  O substituto é `effort` (low|medium|high|xhigh|max), que este adaptador "
                "traduz para `output_config.effort`. Em configs/items.yaml, troque "
                f"{rejeitados[0]} por `effort` em `generator.params`."
            )

        if "max_tokens" not in params:
            raise ValueError(
                f"O modelo '{self.spec.key}' precisa de `max_output_tokens` em `params`: "
                "`max_tokens` é obrigatório no SDK da Anthropic."
            )

        # `effort` é um campo de `output_config`, não um parâmetro de topo.
        effort = params.pop("effort", None)
        if effort is not None:
            params["output_config"] = {**params.get("output_config", {}), "effort": effort}
        return params

    def answer(self, prompt: str) -> QAResult:
        response = self._ensure_client().messages.create(
            model=self.spec.model_id,
            messages=[{"role": "user", "content": prompt}],
            **self._sdk_params(),
        )

        blocks = getattr(response, "content", None) or []
        text = "".join(
            getattr(block, "text", "") for block in blocks if getattr(block, "type", None) == "text"
        )

        usage_obj = getattr(response, "usage", None)
        usage: dict[str, int] = {}
        if usage_obj is not None:
            if getattr(usage_obj, "input_tokens", None) is not None:
                usage["input_tokens"] = int(usage_obj.input_tokens)
            if getattr(usage_obj, "output_tokens", None) is not None:
                usage["output_tokens"] = int(usage_obj.output_tokens)

        stop_reason = getattr(response, "stop_reason", None)
        return QAResult(
            raw_text=text,
            model_key=self.spec.key,
            finish_reason=str(stop_reason) if stop_reason is not None else "",
            usage=usage,
        )


# Provedores que este módulo sabe construir, em ordem estável.
TEXT_QA_PROVIDERS: tuple[str, ...] = ("fake", "openai", "google", "qwen", "anthropic")


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
    if spec.provider == "anthropic":
        return AnthropicTextQA(spec=spec)
    raise ValueError(
        f"Modelo de texto '{spec.key}': provider '{spec.provider}' desconhecido; "
        f"esperado um de {TEXT_QA_PROVIDERS}."
    )
