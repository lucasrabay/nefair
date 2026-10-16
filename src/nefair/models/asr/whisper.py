"""Adaptadores de ASR (Whisper).

`WhisperAPI` está implementado: é a hospedagem escolhida para o estudo (API, sem
download de pesos e sem RAM). `WhisperLocal` segue como stub de contrato, porque
não há pesos locais neste repositório — um cliente "real" escrito hoje seria
código não verificável, e o docstring dele vale mais que uma implementação que
ninguém consegue rodar.

Regra herdada de `models/base.py` e verificada em `tests/test_providers.py`:
**nenhum SDK no topo deste módulo**. `torch`, `transformers` e `openai` só são
importados dentro do método que os usa.

Lembrete do D8: qual checkpoint responde aqui ainda é questão aberta
(`configs/models.yaml` → `asr.contamination_checked: false`), e o portão em
`run/runner.assert_contamination_checked` recusa a execução completa até que seja
resolvida. Isto é um adaptador, não uma resposta ao D8.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nefair.models.base import ASR, ASRResult, FakeASR, ModelSpec
from nefair.models.providers import require_env

# Variáveis de ambiente esperadas por cada provedor de ASR. Ficam nomeadas aqui
# (e não espalhadas pelo código) para que a mensagem de erro do stub e a
# implementação real não possam divergir.
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_HF_TOKEN = "HF_TOKEN"


@dataclass
class WhisperLocal:
    """Whisper rodando localmente via `transformers` (pesos do Hugging Face).

    Contrato que a implementação real precisa cumprir:

    1. **Import preguiçoso**, dentro de `transcribe` (ou de um `_ensure_pipeline`
       chamado por ele), de `torch` e de
       `transformers.pipeline`/`AutoModelForSpeechSeq2Seq` +
       `AutoProcessor`. Nunca no topo do módulo: `import nefair` não pode custar
       segundos de carregamento de `torch`, e a suíte de testes roda sem eles
       instalados (extra opcional `[local]` do `pyproject.toml`).
    2. **Carregar o modelo uma única vez** e guardá-lo em `self._pipeline`. O
       runner chama `transcribe` uma vez por janela; recarregar os pesos a cada
       chamada tornaria a Etapa 5 inviável.
    3. Usar `spec.model_id` como checkpoint (ex.: `openai/whisper-large-v3`) e
       `spec.params` como argumentos de geração — em especial
       `params["language"]` (`"pt"`) e `params["temperature"]` (`0.0`, para que a
       transcrição seja determinística). Passar `generate_kwargs={"language":
       ..., "task": "transcribe"}`.
    4. Ler o áudio de `audio_path` **já em 16 kHz mono** (a Etapa 2 grava assim);
       não reamostrar de novo, para não introduzir uma segunda reamostragem que
       a auditoria não registra.
    5. Devolver `ASRResult(text=<saída["text"]>, model_key=spec.key,
       raw=<json da saída completa>)`. O campo da resposta é `["text"]` no
       retorno do `pipeline("automatic-speech-recognition")`.
    6. Em falha do provedor, **levantar** a exceção: quem decide o que fazer com
       ela é o runner (retry com backoff e, esgotado, `RunRecord` com `error`).

    Não implementado por decisão: a hospedagem escolhida é API, e a máquina de
    desenvolvimento (16 GB de RAM) não comporta os pesos. `HF_TOKEN`
    (`ENV_HF_TOKEN`) só seria necessário para checkpoints privados.
    """

    spec: ModelSpec

    def transcribe(self, audio_path: Path) -> ASRResult:
        raise NotImplementedError(
            "WhisperLocal.transcribe não implementado (decisão deliberada: sem pesos "
            "locais neste repositório, o código não seria verificável). Para "
            f"implementar: instalar o extra opcional `local` (torch, transformers); "
            f"importar `torch` e `transformers.pipeline` DENTRO deste método; carregar "
            f"'{self.spec.model_id}' uma única vez em `self._pipeline`; chamar com "
            f"generate_kwargs derivados de spec.params={dict(self.spec.params)}; "
            f"devolver ASRResult(text=saida['text'], model_key='{self.spec.key}'). "
            f"Checkpoint privado exige {ENV_HF_TOKEN}. Áudio pedido: {audio_path}. "
            "Para ASR por API, use provider 'openai' (WhisperAPI)."
        )


@dataclass
class WhisperAPI:
    """Whisper servido por API (endpoint de transcrição compatível com OpenAI).

    `spec.version` é o snapshot fixado no dia da execução; ele entra no
    `fingerprint()` e, portanto, na chave de cache do runner — trocar de snapshot
    invalida o cache sozinho, em vez de misturar transcrições de dois modelos sob
    o mesmo nome.

    Erros de provedor (429, 5xx, timeout) **sobem**: o retry com backoff e o
    respeito ao `Retry-After` são do runner, não do adaptador, para que a política
    seja uma só em todo o pipeline.
    """

    spec: ModelSpec
    client: Any = None

    def _ensure_client(self) -> Any:
        if self.client is None:
            from openai import OpenAI  # import preguiçoso: extra opcional `providers`

            self.client = OpenAI(
                api_key=require_env(ENV_OPENAI_API_KEY, who=f"O ASR '{self.spec.key}'")
            )
        return self.client

    def transcribe(self, audio_path: Path) -> ASRResult:
        client = self._ensure_client()
        with Path(audio_path).open("rb") as handle:
            response = client.audio.transcriptions.create(
                model=self.spec.model_id,
                file=handle,
                response_format="json",
                **dict(self.spec.params),
            )

        text = getattr(response, "text", None)
        return ASRResult(
            text=text if text is not None else "",
            model_key=self.spec.key,
            raw=_raw_json(response),
        )


def _raw_json(response: Any) -> str:
    """Resposta serializada para a trilha de auditoria, sem derrubar a execução.

    O `raw` existe para que se possa reconstituir o que o provedor devolveu; se a
    serialização falhar, perder o `raw` é muito melhor do que perder a transcrição.
    """
    for attribute in ("model_dump_json", "to_json"):
        method = getattr(response, attribute, None)
        if callable(method):
            try:
                return str(method())
            except Exception:  # noqa: BLE001 - trilha de auditoria é best-effort
                pass
    try:
        return json.dumps(response, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001 - idem
        return repr(response)


# Provedores que este módulo sabe construir, em ordem estável (usada na mensagem
# de erro da factory).
ASR_PROVIDERS: tuple[str, ...] = ("fake", "local", "openai")


def build_asr(spec: ModelSpec, *, seed: int = 0, force_fake: bool = False) -> ASR:
    """Devolve o adaptador de ASR correspondente ao `provider` do `spec`.

    `force_fake` existe para o modo `--fake` do script 06. Ele **não** devolve
    simplesmente um falso com o spec original: troca `provider` para `"fake"`
    antes de construir. Isso muda o `fingerprint()` e, por consequência, a chave
    de cache — sem essa troca, uma execução falsa gravaria respostas inventadas
    sob a mesma chave de uma execução real, e o cache passaria a servir ficção
    para a Etapa 6.
    """
    if force_fake or spec.provider == "fake":
        return FakeASR(spec=replace(spec, provider="fake"), seed=seed)
    if spec.provider == "local":
        return WhisperLocal(spec=spec)
    if spec.provider == "openai":
        return WhisperAPI(spec=spec)
    raise ValueError(
        f"ASR '{spec.key}': provider '{spec.provider}' desconhecido; "
        f"esperado um de {ASR_PROVIDERS}."
    )
