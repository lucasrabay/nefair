"""Testes dos adaptadores de provedor — todos offline, sem credencial e sem SDK.

Estes adaptadores são a única parte do pipeline que fala com o mundo externo, e
eram, até aqui, a única parte sem nenhum teste. Cada teste abaixo defende uma de
quatro propriedades, e cada uma tem consequência direta sobre um número do TCC:

1. **A requisição sai no formato que o provedor espera.** O par mais fácil de
   inverter é o do áudio: a OpenAI quer base64 puro, o DashScope quer o prefixo
   `data:`. E `modalities=["text"]` na OpenAI multimodal — sem ele o modelo pode
   responder em áudio, o texto vem vazio e o parser conta como não interpretável,
   inflando a taxa por erro de configuração nosso e não por falha do modelo.
2. **Nada é imputado.** Conteúdo `None` (recusa, filtro de segurança, bloqueio)
   vira `raw_text=""` e segue para o parser, que sabe contá-lo. Um adaptador que
   inventasse texto destruiria justamente o número que o TCC reporta.
3. **A ausência de credencial falha cedo e nomeando a variável.** Um cliente
   construído sem chave erraria 401 a cada retry de cada item antes de alguém
   perceber o que faltava.
4. **Nenhum SDK no topo do módulo.** Esta é verificada mecanicamente em
   `test_importar_adaptadores_nao_carrega_sdk`, em vez de depender de revisão.

A costura que torna tudo isto testável é o campo `client` de cada adaptador: o
teste injeta um objeto com a forma do provedor e inspeciona o que foi enviado.
"""

from __future__ import annotations

import base64
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from nefair.config import assert_no_placeholders, load_items_config, load_models_config
from nefair.models.asr.whisper import (
    ASR_PROVIDERS,
    WhisperAPI,
    WhisperLocal,
    build_asr,
)
from nefair.models.base import FakeASR, FakeAudioQA, FakeTextQA, ModelSpec, fake_spec
from nefair.models.llm.clients import (
    TEXT_QA_PROVIDERS,
    AnthropicTextQA,
    GoogleTextQA,
    OpenAITextQA,
    QwenTextQA,
    build_text_qa,
)
from nefair.models.multimodal.clients import (
    AUDIO_QA_PROVIDERS,
    GoogleAudioQA,
    OpenAIAudioQA,
    QwenAudioQA,
    build_audio_qa,
)
from nefair.models.providers import chat_params, qa_from_openai_response, require_env

MODELS_PATH = "configs/models.yaml"
ITEMS_PATH = "configs/items.yaml"

# Conteúdo arbitrário: o que se testa é como ele é embalado, não o que ele diz.
AUDIO_BYTES = b"RIFF....WAVEfmt mock"


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    path = tmp_path / "janela.wav"
    path.write_bytes(AUDIO_BYTES)
    return path


def spec_for(provider: str, role: str = "text_qa", **overrides: Any) -> ModelSpec:
    base = {
        "key": f"{provider}_{role}",
        "provider": provider,
        "model_id": f"{provider}-modelo-1",
        "version": "2026-10-01",
        "role": role,
        "family": provider,
        "params": {"temperature": 0.0, "max_output_tokens": 16},
    }
    base.update(overrides)
    return ModelSpec(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Clientes falsos com a forma de cada SDK
# --------------------------------------------------------------------------- #
@dataclass
class _Message:
    content: str | None


@dataclass
class _Choice:
    message: _Message
    finish_reason: str = "stop"


@dataclass
class _Usage:
    prompt_tokens: int = 11
    completion_tokens: int = 3


@dataclass
class _OpenAIResponse:
    choices: list[_Choice]
    usage: _Usage | None = field(default_factory=_Usage)


class _Completions:
    """Captura o kwargs da chamada: é sobre ele que as asserções de formato caem."""

    def __init__(self, content: str | None = "B") -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _OpenAIResponse:
        self.calls.append(kwargs)
        return _OpenAIResponse(choices=[_Choice(message=_Message(self.content))])


class FakeOpenAIClient:
    def __init__(self, content: str | None = "B") -> None:
        self.completions = _Completions(content)
        self.chat = type("Chat", (), {"completions": self.completions})()

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.completions.calls


@dataclass
class _Candidate:
    finish_reason: str = "STOP"


@dataclass
class _UsageMetadata:
    prompt_token_count: int = 11
    candidates_token_count: int = 3


@dataclass
class _GoogleResponse:
    text: str | None
    candidates: list[_Candidate] = field(default_factory=lambda: [_Candidate()])
    usage_metadata: _UsageMetadata = field(default_factory=_UsageMetadata)


class _GoogleModels:
    def __init__(self, text: str | None = "C") -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> _GoogleResponse:
        self.calls.append(kwargs)
        return _GoogleResponse(text=self.text)


class FakeGoogleClient:
    def __init__(self, text: str | None = "C") -> None:
        self.models = _GoogleModels(text)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.models.calls


class FakeGoogleTypes:
    """Dublê de `google.genai.types`, o módulo que o adaptador usa."""

    @staticmethod
    def GenerateContentConfig(**kwargs: Any) -> dict[str, Any]:  # noqa: N802 - nome do SDK
        return {"config": kwargs}

    class Part:
        @staticmethod
        def from_bytes(*, data: bytes, mime_type: str) -> dict[str, Any]:
            return {"data": data, "mime_type": mime_type}


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _AnthropicUsage:
    input_tokens: int = 21
    output_tokens: int = 5


@dataclass
class _AnthropicResponse:
    content: list[_Block]
    stop_reason: str = "end_turn"
    usage: _AnthropicUsage = field(default_factory=_AnthropicUsage)


class _Messages:
    def __init__(self, blocks: list[_Block]) -> None:
        self.blocks = blocks
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _AnthropicResponse:
        self.calls.append(kwargs)
        return _AnthropicResponse(content=self.blocks)


class FakeAnthropicClient:
    def __init__(self, blocks: list[_Block] | None = None) -> None:
        self.messages = _Messages(blocks if blocks is not None else [_Block("A")])

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls


@dataclass
class _Transcription:
    text: str | None


class _Transcriptions:
    def __init__(self, text: str | None = "a gente morava na roça") -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Transcription:
        # O arquivo precisa ser lido ANTES do `with` fechar; ler aqui prova isso.
        handle = kwargs.get("file")
        self.calls.append({**kwargs, "_bytes": handle.read() if handle else None})
        return _Transcription(self.text)


class FakeWhisperClient:
    def __init__(self, text: str | None = "a gente morava na roça") -> None:
        self.transcriptions = _Transcriptions(text)
        self.audio = type("Audio", (), {"transcriptions": self.transcriptions})()

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.transcriptions.calls


# --------------------------------------------------------------------------- #
# 1. A regra que não pode ser violada: nenhum SDK no topo do módulo
# --------------------------------------------------------------------------- #
def test_importar_adaptadores_nao_carrega_sdk():
    """Importar os adaptadores não pode trazer nenhum SDK para `sys.modules`.

    Esta é a regra que torna a suíte inteira executável sem credencial e sem
    nenhum provedor instalado, e que impede `import nefair` de custar segundos de
    carregamento de `torch`. Até agora ela dependia de revisão humana; aqui ela é
    verificada.

    Roda num subprocesso limpo porque, no processo do pytest, outro teste pode já
    ter importado um SDK por conta própria e o resultado seria inconclusivo.
    """
    code = (
        "import sys;"
        "import nefair.models.llm.clients;"
        "import nefair.models.multimodal.clients;"
        "import nefair.models.asr.whisper;"
        "import nefair.models.providers;"
        "vazados = [m for m in ('openai', 'google', 'anthropic', 'torch', 'transformers')"
        " if m in sys.modules];"
        "print(','.join(vazados))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
        env={"PYTHONPATH": "src", "PATH": ""},
    )
    assert done.stdout.strip() == "", (
        f"SDK importado no topo do módulo: {done.stdout.strip()}. "
        "O import tem de ficar dentro de `_ensure_client`."
    )


# --------------------------------------------------------------------------- #
# 2. Formato da requisição
# --------------------------------------------------------------------------- #
def test_openai_texto_mapeia_max_output_tokens():
    """`max_output_tokens` do YAML vira `max_tokens` no SDK.

    A tradução mora no adaptador para que o YAML siga legível como decisão
    metodológica. Se vazasse para o YAML, a config deixaria de ser a descrição do
    experimento e viraria payload de SDK.
    """
    client = FakeOpenAIClient()
    adapter = OpenAITextQA(spec=spec_for("openai"), client=client)
    adapter.answer("Qual alternativa?")

    sent = client.calls[0]
    assert sent["max_tokens"] == 16
    assert "max_output_tokens" not in sent
    assert sent["temperature"] == 0.0
    assert sent["model"] == "openai-modelo-1"
    assert sent["messages"] == [{"role": "user", "content": "Qual alternativa?"}]


def test_openai_audio_exige_modalities_text_e_base64_puro(wav: Path):
    """Sem `modalities=['text']` o modelo pode responder em áudio.

    O texto viria vazio, o parser contaria como não interpretável e a taxa de
    falha subiria por um defeito nosso de configuração. É o tipo de erro que
    produz um resultado plausível e falso.
    """
    client = FakeOpenAIClient()
    adapter = OpenAIAudioQA(spec=spec_for("openai", role="audio_qa"), client=client)
    adapter.answer("Qual alternativa?", wav)

    sent = client.calls[0]
    assert sent["modalities"] == ["text"]

    part = sent["messages"][0]["content"][1]["input_audio"]
    assert part["format"] == "wav"
    assert part["data"] == base64.b64encode(AUDIO_BYTES).decode("ascii")
    assert not part["data"].startswith("data:"), "a OpenAI espera base64 PURO"


def test_qwen_audio_exige_prefixo_data(wav: Path):
    """O DashScope espera `data:audio/wav;base64,` — ao contrário da OpenAI.

    Este teste e o anterior são um par: travam os dois sentidos da troca mais
    fácil de cometer entre os dois provedores compatíveis com OpenAI.
    """
    client = FakeOpenAIClient()
    adapter = QwenAudioQA(spec=spec_for("qwen", role="audio_qa"), client=client)
    adapter.answer("Qual alternativa?", wav)

    part = client.calls[0]["messages"][0]["content"][1]["input_audio"]
    esperado = base64.b64encode(AUDIO_BYTES).decode("ascii")
    assert part["data"] == f"data:audio/wav;base64,{esperado}"


def test_google_audio_manda_os_bytes_inline(wav: Path):
    """Janelas de 30–60 s cabem inline; `files.upload` traria estado remoto."""
    client = FakeGoogleClient()
    adapter = GoogleAudioQA(
        spec=spec_for("google", role="audio_qa"), client=client, types=FakeGoogleTypes
    )
    adapter.answer("Qual alternativa?", wav)

    contents = client.calls[0]["contents"]
    assert contents[0] == "Qual alternativa?"
    assert contents[1] == {"data": AUDIO_BYTES, "mime_type": "audio/wav"}


def test_anthropic_exige_max_tokens_e_junta_blocos_de_texto():
    """O SDK da Anthropic exige `max_tokens` e devolve uma LISTA de blocos."""
    client = FakeAnthropicClient(blocks=[_Block("parte 1 "), _Block("parte 2")])
    adapter = AnthropicTextQA(
        spec=spec_for("anthropic", params={"max_output_tokens": 1024}),
        client=client,
    )
    result = adapter.answer("Gere um item.")

    assert client.calls[0]["max_tokens"] == 1024
    assert result.raw_text == "parte 1 parte 2"
    assert result.finish_reason == "end_turn"
    assert result.usage == {"input_tokens": 21, "output_tokens": 5}


def test_anthropic_ignora_bloco_de_raciocinio():
    """A resposta real traz um bloco `thinking` ANTES do `text`.

    Verificado contra o SDK 1.15.0: pegar `content[0].text` cegamente devolveria
    vazio, e o item seria perdido sem erro aparente. Só os blocos `text` entram.
    """
    blocks = [_Block("raciocinio interno", type="thinking"), _Block("B")]
    client = FakeAnthropicClient(blocks=blocks)
    adapter = AnthropicTextQA(
        spec=spec_for("anthropic", params={"max_output_tokens": 1024}), client=client
    )
    assert adapter.answer("p").raw_text == "B"


def test_anthropic_recusa_temperature_em_vez_de_ignorar():
    """O SDK 1.x não aceita controle de amostragem; descartar seria mentir.

    O config é a descrição do experimento e entra no relatório. Se `temperature`
    fosse silenciosamente dropada, o texto afirmaria um parâmetro que nunca foi
    aplicado. Mesma política do portão D8: recusar e dizer o que fazer.
    """
    adapter = AnthropicTextQA(
        spec=spec_for("anthropic", params={"temperature": 0.2, "max_output_tokens": 1024}),
        client=FakeAnthropicClient(),
    )
    with pytest.raises(ValueError, match="temperature"):
        adapter.answer("p")


def test_anthropic_exige_max_output_tokens_no_config():
    """`max_tokens` é obrigatório no SDK: faltar no YAML tem de falhar cedo."""
    adapter = AnthropicTextQA(spec=spec_for("anthropic", params={}), client=FakeAnthropicClient())
    with pytest.raises(ValueError, match="max_output_tokens"):
        adapter.answer("p")


def test_whisper_api_le_o_arquivo_antes_de_fechar(wav: Path):
    """O handle é consumido dentro do `with`; fora dele já estaria fechado."""
    client = FakeWhisperClient()
    spec = spec_for("openai", role="asr", params={"language": "pt"})
    adapter = WhisperAPI(spec=spec, client=client)
    result = adapter.transcribe(wav)

    sent = client.calls[0]
    assert sent["_bytes"] == AUDIO_BYTES
    assert sent["response_format"] == "json"
    assert sent["language"] == "pt"
    assert result.text == "a gente morava na roça"
    assert result.model_key == "openai_asr"


# --------------------------------------------------------------------------- #
# 3. Nada imputado
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("adapter_cls", "client_factory"),
    [
        (OpenAITextQA, lambda: FakeOpenAIClient(content=None)),
        (QwenTextQA, lambda: FakeOpenAIClient(content=None)),
    ],
)
def test_conteudo_none_vira_texto_vazio(adapter_cls, client_factory):
    """Recusa ou filtro de conteúdo não é exceção: é resposta vazia.

    O parser conta o não interpretável; inventar texto aqui apagaria esse número.
    """
    provider = "openai" if adapter_cls is OpenAITextQA else "qwen"
    result = adapter_cls(spec=spec_for(provider), client=client_factory()).answer("p")
    assert result.raw_text == ""
    assert result.finish_reason == "stop"


def test_google_bloqueado_vira_texto_vazio_com_motivo():
    """`response.text is None` é bloqueio de segurança — dado, não erro.

    O `finish_reason` preenchido é o que permite ao relatório distinguir "o
    modelo errou" de "o provedor bloqueou".
    """
    client = FakeGoogleClient(text=None)
    adapter = GoogleTextQA(spec=spec_for("google"), client=client, types=FakeGoogleTypes)
    result = adapter.answer("p")

    assert result.raw_text == ""
    assert result.finish_reason == "STOP"


def test_anthropic_sem_bloco_de_texto_vira_vazio():
    """Lista de blocos vazia (ou só de outro tipo) não pode virar exceção.

    Acontece de verdade quando o teto de `max_tokens` é consumido pelo raciocínio
    antes de sair o primeiro bloco de texto.
    """
    client = FakeAnthropicClient(blocks=[])
    spec = spec_for("anthropic", params={"max_output_tokens": 16})
    result = AnthropicTextQA(spec=spec, client=client).answer("p")
    assert result.raw_text == ""


def test_resposta_sem_choices_nao_estoura():
    """Provedor que devolve `choices` vazio vira resposta vazia rotulada."""
    result = qa_from_openai_response(_OpenAIResponse(choices=[]), "m")
    assert result.raw_text == ""
    assert result.finish_reason == "sem_choices"


def test_resposta_sem_usage_nao_estoura():
    """Nem todo provedor reporta contagem de tokens."""
    response = _OpenAIResponse(choices=[_Choice(message=_Message("B"))], usage=None)
    result = qa_from_openai_response(response, "m")
    assert result.raw_text == "B"
    assert result.usage == {}


def test_erro_do_provedor_sobe():
    """Retry e backoff são do runner; engolir aqui duplicaria a política."""

    class Explode:
        def create(self, **kwargs: Any) -> Any:
            raise TimeoutError("502 do provedor")

    client = FakeOpenAIClient()
    client.chat = type("Chat", (), {"completions": Explode()})()

    with pytest.raises(TimeoutError):
        OpenAITextQA(spec=spec_for("openai"), client=client).answer("p")


# --------------------------------------------------------------------------- #
# 4. Credencial ausente falha cedo e nomeia a variável
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("adapter_cls", "provider", "variavel"),
    [
        (OpenAITextQA, "openai", "OPENAI_API_KEY"),
        (GoogleTextQA, "google", "GOOGLE_API_KEY"),
        (QwenTextQA, "qwen", "DASHSCOPE_API_KEY"),
        (AnthropicTextQA, "anthropic", "ANTHROPIC_API_KEY"),
    ],
)
def test_sem_credencial_erro_nomeia_a_variavel(monkeypatch, adapter_cls, provider, variavel):
    """A mensagem tem de dizer QUAL variável falta e para QUEM.

    Sem isso o cliente erraria 401 a cada retry de cada item antes de alguém
    perceber que bastava exportar uma chave.
    """
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "DASHSCOPE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    adapter = adapter_cls(spec=spec_for(provider))
    with pytest.raises((RuntimeError, ImportError, ModuleNotFoundError)) as excinfo:
        adapter.answer("p")

    # Sem o SDK instalado o import falha antes; com ele, a credencial é checada.
    # Os dois são aceitáveis — o que não pode é passar silenciosamente.
    if isinstance(excinfo.value, RuntimeError):
        assert variavel in str(excinfo.value)


def test_require_env_rejeita_valor_so_de_espacos(monkeypatch):
    """Variável definida mas vazia é tão inútil quanto ausente."""
    monkeypatch.setenv("NEFAIR_TESTE", "   ")
    with pytest.raises(RuntimeError, match="NEFAIR_TESTE"):
        require_env("NEFAIR_TESTE", who="O teste")


# --------------------------------------------------------------------------- #
# 5. Factories
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("provider", "esperado"),
    [
        ("openai", OpenAITextQA),
        ("google", GoogleTextQA),
        ("qwen", QwenTextQA),
        ("anthropic", AnthropicTextQA),
    ],
)
def test_build_text_qa_despacha(provider, esperado):
    assert isinstance(build_text_qa(spec_for(provider)), esperado)


@pytest.mark.parametrize(
    ("provider", "esperado"),
    [("openai", OpenAIAudioQA), ("google", GoogleAudioQA), ("qwen", QwenAudioQA)],
)
def test_build_audio_qa_despacha(provider, esperado):
    assert isinstance(build_audio_qa(spec_for(provider, role="audio_qa")), esperado)


@pytest.mark.parametrize(
    ("provider", "esperado"), [("openai", WhisperAPI), ("local", WhisperLocal)]
)
def test_build_asr_despacha(provider, esperado):
    assert isinstance(build_asr(spec_for(provider, role="asr")), esperado)


def test_anthropic_esta_registrado():
    """`configs/items.yaml` declara `provider: anthropic` para o gerador (D11)."""
    assert "anthropic" in TEXT_QA_PROVIDERS
    assert load_items_config(ITEMS_PATH).generator.provider in TEXT_QA_PROVIDERS


@pytest.mark.parametrize(
    ("builder", "role", "fake_cls"),
    [
        (build_text_qa, "text_qa", FakeTextQA),
        (build_audio_qa, "audio_qa", FakeAudioQA),
        (build_asr, "asr", FakeASR),
    ],
)
def test_force_fake_troca_o_provider_do_spec(builder, role, fake_cls):
    """O falso não pode herdar o `provider` real.

    O `provider` entra no `fingerprint()` e, portanto, na chave de cache. Sem a
    troca, uma execução `--fake` gravaria respostas inventadas sob a mesma chave
    de uma execução real, e a Etapa 6 leria ficção sem perceber.
    """
    real = spec_for("openai", role=role)
    adapter = builder(real, force_fake=True)

    assert isinstance(adapter, fake_cls)
    assert adapter.spec.provider == "fake"
    assert adapter.spec.fingerprint() != real.fingerprint()


@pytest.mark.parametrize(
    ("builder", "role", "conhecidos"),
    [
        (build_text_qa, "text_qa", TEXT_QA_PROVIDERS),
        (build_audio_qa, "audio_qa", AUDIO_QA_PROVIDERS),
        (build_asr, "asr", ASR_PROVIDERS),
    ],
)
def test_provider_desconhecido_lista_os_validos(builder, role, conhecidos):
    with pytest.raises(ValueError) as excinfo:
        builder(spec_for("inexistente", role=role))
    for provider in conhecidos:
        assert provider in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 6. Guarda de PLACEHOLDER
# --------------------------------------------------------------------------- #
def test_guarda_pega_os_placeholders_dos_configs_reais():
    """Os YAMLs do repositório ainda estão por preencher — a guarda tem de ver."""
    items = load_items_config(ITEMS_PATH)
    models = load_models_config(MODELS_PATH)

    with pytest.raises(ValueError) as excinfo:
        assert_no_placeholders(items, models)

    mensagem = str(excinfo.value)
    assert "items.generator.model_id" in mensagem
    assert "models.specs.openai_audio.model_id" in mensagem
    assert "models.asr.spec.version" in mensagem


def test_guarda_pega_version_mesmo_com_model_id_preenchido():
    """`version` entra no fingerprint: deixá-la como sentinela corrompe o cache."""
    items = load_items_config(ITEMS_PATH)
    quase_pronto = replace(items.generator, model_id="claude-opus-5-5")

    with pytest.raises(ValueError, match="items.generator.version"):
        assert_no_placeholders(replace(items, generator=quase_pronto), None)


def test_guarda_passa_quando_tudo_esta_preenchido():
    items = load_items_config(ITEMS_PATH)
    pronto = replace(items.generator, model_id="claude-opus-5-5", version="2026-10-01")
    assert_no_placeholders(replace(items, generator=pronto), None)


def test_guarda_aceita_none():
    """O script 03 pode rodar sem `--models`; a guarda não pode exigir os dois."""
    assert_no_placeholders(None, None)


# --------------------------------------------------------------------------- #
# 7. Utilidades compartilhadas
# --------------------------------------------------------------------------- #
def test_chat_params_preserva_o_que_nao_conhece():
    """Acrescentar um parâmetro no YAML não pode exigir mexer em código."""
    assert chat_params({"temperature": 0.7, "max_output_tokens": 16, "top_p": 0.9}) == {
        "temperature": 0.7,
        "max_tokens": 16,
        "top_p": 0.9,
    }


def test_fake_spec_segue_utilizavel():
    """Os falsos continuam construtíveis sem nenhum adaptador real."""
    adapter = build_text_qa(fake_spec())
    assert isinstance(adapter, FakeTextQA)
    assert adapter.answer("p").raw_text in {"A", "B", "C", "D"}
