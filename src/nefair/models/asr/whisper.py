"""Adaptadores de ASR (Whisper) — **stubs de contrato**, ainda não implementados.

Por que stubs: não há credencial de provedor nem pesos locais neste repositório,
então um cliente "real" escrito hoje seria código **não verificável** — ninguém
consegue rodá-lo, nenhum teste consegue cobri-lo, e ele apodreceria em silêncio
até o dia da execução. O que tem valor agora é o *contrato*: a forma exata do
adaptador, o SDK que ele usa, a variável de ambiente que ele exige e o campo da
resposta de onde sai o texto. É isso que está escrito aqui.

Cada método levanta `NotImplementedError` com a lista do que falta. A factory
`build_asr` devolve `FakeASR` quando `spec.provider == "fake"`, e é por isso que
`scripts/06_run_eval.py --fake` roda ponta a ponta hoje, sem rede.

Regra herdada de `models/base.py` e verificada no critério de aceite: **nenhum
import de SDK no topo deste módulo**. `torch`, `transformers` e `openai` só podem
ser importados dentro do método que os usa.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from nefair.models.base import ASR, ASRResult, FakeASR, ModelSpec

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
       Engolir o erro aqui esconderia exatamente o número que o TCC precisa
       reportar.

    `HF_TOKEN` (`ENV_HF_TOKEN`) só é necessário para checkpoints privados ou com
    licença aceita; checkpoints públicos dispensam.
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
            f"Checkpoint privado exige {ENV_HF_TOKEN}. Áudio pedido: {audio_path}."
        )


@dataclass
class WhisperAPI:
    """Whisper servido por API (endpoint de transcrição da OpenAI).

    Contrato que a implementação real precisa cumprir:

    1. **Import preguiçoso** de `openai` dentro de `transcribe`
       (`from openai import OpenAI`), nunca no topo — extra opcional
       `[providers]`.
    2. Ler a chave de `os.environ[ENV_OPENAI_API_KEY]` e **falhar com mensagem
       explícita** se ela não existir (não cair num cliente sem credencial, que
       erraria com 401 dezenas de vezes depois de dezenas de retries).
    3. Abrir `audio_path` em modo binário e chamar
       `client.audio.transcriptions.create(model=spec.model_id, file=<arquivo>,
       language=spec.params["language"], temperature=spec.params["temperature"],
       response_format="json")`.
    4. O texto sai em `response.text`. Devolver
       `ASRResult(text=response.text, model_key=spec.key, raw=<json bruto>)`.
    5. `spec.version` é o snapshot fixado no dia da execução; ele já entra no
       `fingerprint()` e, portanto, na chave de cache do runner — trocar de
       snapshot invalida o cache sozinho, em vez de misturar transcrições de dois
       modelos sob o mesmo nome.
    6. Erros de provedor (429, 5xx, timeout) devem **subir**: o retry com backoff
       e o respeito ao `Retry-After` são do runner, não do adaptador, para que a
       política seja uma só em todo o pipeline.
    """

    spec: ModelSpec

    def transcribe(self, audio_path: Path) -> ASRResult:
        raise NotImplementedError(
            "WhisperAPI.transcribe não implementado (não há credencial para validar "
            "o cliente). Para implementar: instalar o extra opcional `providers` "
            f"(openai); importar `from openai import OpenAI` DENTRO deste método; "
            f"exigir a variável de ambiente {ENV_OPENAI_API_KEY}; chamar "
            f"client.audio.transcriptions.create(model='{self.spec.model_id}', "
            f"file=open(audio_path,'rb'), **{dict(self.spec.params)}); devolver "
            f"ASRResult(text=response.text, model_key='{self.spec.key}'). "
            f"Áudio pedido: {audio_path}."
        )


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
