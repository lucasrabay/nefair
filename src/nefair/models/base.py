"""Interfaces de modelo e implementações falsas determinísticas.

Regra de ouro deste pacote: **nenhum SDK de provedor é importado no topo de um
módulo**. Os adaptadores reais importam `openai`, `google.genai`, `torch` etc.
dentro da função que os usa, para que:

- a suíte de testes rode inteira sem nenhuma credencial e sem nenhum SDK
  instalado;
- `import nefair` não custe segundos de carregamento de `torch`;
- a ausência de um provedor não impeça o uso dos outros.

Os `Protocol`s abaixo são a única superfície que o runner conhece. Trocar de
provedor é escrever mais um adaptador; não é mexer no runner, nem na análise.

As implementações `Fake*` não são decoração de teste: elas são o que torna todo
o pipeline executável ponta a ponta hoje, sem chave de API. São determinísticas
por construção (a resposta deriva de um hash do prompt e da semente), então um
teste que passa hoje passa amanhã.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from nefair.schema import ALTERNATIVE_LABELS, stable_hash

# Papéis que um modelo pode exercer no pipeline.
ROLE_ASR = "asr"
ROLE_TEXT_QA = "text_qa"
ROLE_AUDIO_QA = "audio_qa"
ROLES: tuple[str, ...] = (ROLE_ASR, ROLE_TEXT_QA, ROLE_AUDIO_QA)


@dataclass(frozen=True)
class ModelSpec:
    """Identidade completa e versionada de um modelo avaliado.

    `version` não é decoração: o texto do TCC exige reportar a versão exata usada
    no dia da execução, e ela entra no `fingerprint` — trocar de snapshot invalida
    o cache automaticamente, em vez de misturar respostas de dois modelos
    diferentes sob o mesmo nome.

    `params` são os parâmetros de decodificação (temperatura, top_p, max_tokens).
    Entram no fingerprint pelo mesmo motivo: mudar a temperatura muda o
    experimento.
    """

    key: str  # chave curta, usada nos YAMLs e em nomes de arquivo
    provider: str  # openai | google | qwen | local | fake
    model_id: str  # id exato no provedor
    version: str  # snapshot/data fixado no dia da execução
    role: str  # asr | text_qa | audio_qa
    family: str = ""  # família para o pareamento nativo × cascata (§Etapa 5)
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(
                f"Modelo '{self.key}': papel '{self.role}' inválido; esperado um de {ROLES}."
            )
        if not self.key:
            raise ValueError("Modelo sem `key`: a chave curta é obrigatória.")

    def fingerprint(self) -> str:
        """Hash estável da identidade do modelo, para chaves de cache."""
        return stable_hash(
            {
                "provider": self.provider,
                "model_id": self.model_id,
                "version": self.version,
                "role": self.role,
                "params": dict(self.params),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "provider": self.provider,
            "model_id": self.model_id,
            "version": self.version,
            "role": self.role,
            "family": self.family,
            "params": dict(self.params),
        }


# --------------------------------------------------------------------------- #
# Resultados
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ASRResult:
    """Saída de um sistema de ASR sobre um arquivo de áudio."""

    text: str
    model_key: str
    raw: str = ""
    error: str = ""


@dataclass(frozen=True)
class QAResult:
    """Saída bruta de um modelo diante de um item de múltipla escolha.

    Deliberadamente **não** contém a letra escolhida: a extração é
    responsabilidade de `run/parse.py`, testada à parte e capaz de contar a
    resposta não interpretável em vez de imputá-la. Misturar geração e parsing
    aqui esconderia exatamente o número que precisamos reportar.
    """

    raw_text: str
    model_key: str
    finish_reason: str = ""
    usage: Mapping[str, int] = field(default_factory=dict)
    error: str = ""


# --------------------------------------------------------------------------- #
# Protocolos
# --------------------------------------------------------------------------- #
@runtime_checkable
class ASR(Protocol):
    """Áudio → transcrição."""

    spec: ModelSpec

    def transcribe(self, audio_path: Path) -> ASRResult: ...


@runtime_checkable
class TextQA(Protocol):
    """Prompt de texto → resposta bruta."""

    spec: ModelSpec

    def answer(self, prompt: str) -> QAResult: ...


@runtime_checkable
class AudioQA(Protocol):
    """Prompt de texto + áudio → resposta bruta (modelo multimodal nativo)."""

    spec: ModelSpec

    def answer(self, prompt: str, audio_path: Path) -> QAResult: ...


# --------------------------------------------------------------------------- #
# Implementações falsas determinísticas
# --------------------------------------------------------------------------- #
def _deterministic_choice(payload: str, seed: int, n_options: int) -> int:
    """Índice em [0, n_options) derivado de forma estável de `payload` + `seed`.

    Não usa `random`: o valor depende só dos argumentos, então não há estado
    global de RNG capaz de fazer um teste passar ou falhar conforme a ordem em
    que os testes rodam.
    """
    digest = hashlib.sha256(f"{seed}:{payload}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % n_options


@dataclass
class FakeASR:
    """ASR falso: devolve um texto derivado do caminho do áudio.

    Serve para exercitar o runner e a decomposição sem nenhum modelo real. Para
    testar o WER com um erro controlado, passe `transcriber`.
    """

    spec: ModelSpec
    transcriber: Callable[[Path], str] | None = None
    seed: int = 0
    n_calls: int = 0

    def transcribe(self, audio_path: Path) -> ASRResult:
        self.n_calls += 1
        if self.transcriber is not None:
            text = self.transcriber(audio_path)
        else:
            text = f"transcricao falsa de {Path(audio_path).stem}"
        return ASRResult(text=text, model_key=self.spec.key, raw=text)


@dataclass
class FakeTextQA:
    """LLM de texto falso.

    `responder` permite ao teste encenar qualquer comportamento (item JSON bem
    formado, resposta malformada, recusa). Sem ele, devolve uma letra escolhida
    deterministicamente a partir do prompt — útil para o filtro textual, onde o
    que importa é que a escolha seja reprodutível, não que seja boa.

    `n_calls` existe para o teste de cache: um acerto de cache tem de deixar este
    contador parado.
    """

    spec: ModelSpec
    responder: Callable[[str], str] | None = None
    seed: int = 0
    n_alternatives: int = 4
    n_calls: int = 0

    def answer(self, prompt: str) -> QAResult:
        self.n_calls += 1
        if self.responder is not None:
            text = self.responder(prompt)
        else:
            index = _deterministic_choice(prompt, self.seed, self.n_alternatives)
            text = ALTERNATIVE_LABELS[index]
        return QAResult(
            raw_text=text,
            model_key=self.spec.key,
            finish_reason="stop",
            usage={"input_tokens": len(prompt.split()), "output_tokens": len(text.split())},
        )


@dataclass
class FakeAudioQA:
    """Modelo multimodal falso (prompt + áudio → resposta bruta)."""

    spec: ModelSpec
    responder: Callable[[str, Path], str] | None = None
    seed: int = 0
    n_alternatives: int = 4
    n_calls: int = 0

    def answer(self, prompt: str, audio_path: Path) -> QAResult:
        self.n_calls += 1
        if self.responder is not None:
            text = self.responder(prompt, audio_path)
        else:
            index = _deterministic_choice(f"{prompt}|{audio_path}", self.seed, self.n_alternatives)
            text = ALTERNATIVE_LABELS[index]
        return QAResult(
            raw_text=text,
            model_key=self.spec.key,
            finish_reason="stop",
            usage={"input_tokens": len(prompt.split()), "output_tokens": len(text.split())},
        )


def fake_spec(key: str = "fake", role: str = ROLE_TEXT_QA, **overrides: Any) -> ModelSpec:
    """`ModelSpec` pronto para testes, com provedor `fake`."""
    base = {
        "key": key,
        "provider": "fake",
        "model_id": f"fake-{role}",
        "version": "v0",
        "role": role,
        "family": "fake",
        "params": {"temperature": 0.0},
    }
    base.update(overrides)
    return ModelSpec(**base)  # type: ignore[arg-type]
