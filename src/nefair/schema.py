"""Contratos de dados compartilhados por todas as etapas do pipeline (1–6).

Este módulo é a fronteira entre os blocos de implementação: janelas, itens,
execução e análise só se comunicam pelas estruturas definidas aqui e pelos
arquivos que elas sabem ler e escrever. Alterar uma dataclass daqui é alterar
um contrato — e quebra os quatro blocos de uma vez.

Três invariantes de projeto governam tudo abaixo:

1. **Identificadores são derivados, legíveis e estáveis.** `window_id` e
   `item_id` são construídos a partir de chaves naturais ordenadas, nunca de
   um contador global nem de `uuid4()`. O mesmo corpus na mesma revisão produz
   os mesmos ids em qualquer máquina.
2. **Conteúdo e carimbo de tempo não se misturam.** Nada que varie entre
   execuções (timestamp, latência) entra num campo que participe de hash,
   ordenação ou comparação byte a byte. Esses campos vivem no cabeçalho do
   artefato, isolados.
3. **Proveniência viaja junto com o dado.** Uma janela sabe de quais segmentos
   veio; um distrator sabe de qual janela foi extraído; um `RunRecord` sabe qual
   modelo, qual prompt e qual permutação o produziram.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Rótulos das alternativas de múltipla escolha, em ordem canônica. O nº de
# alternativas é configurável (D10), então o código deve fatiar esta tupla em vez
# de assumir quatro.
ALTERNATIVE_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E")

# Condições de avaliação (§Condições de controle e decomposição do erro).
#   asr       — o LLM recebe a hipótese do ASR
#   reference — o LLM recebe a transcrição de referência (teto da cascata)
#   audio     — o modelo multimodal recebe o áudio
CONDITIONS: tuple[str, ...] = ("asr", "reference", "audio")

# Sentinela para resposta que não pôde ser interpretada. NUNCA é convertida em
# acerto nem em erro: é contada à parte (§`run/parse.py`).
UNPARSED: None = None


def stable_hash(payload: Any) -> str:
    """SHA-256 hexadecimal de um payload serializável, estável entre execuções.

    Usa JSON com chaves ordenadas e sem espaços supérfluos para que a mesma
    estrutura lógica produza sempre os mesmos bytes, independentemente da ordem
    de inserção nos dicionários.
    """
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def short_hash(payload: Any, length: int = 12) -> str:
    """Prefixo de `stable_hash`, para ids legíveis sem perder unicidade prática."""
    return stable_hash(payload)[:length]


# --------------------------------------------------------------------------- #
# Proveniência da fonte
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Provenance:
    """De onde o dado veio, com precisão suficiente para reexecutar.

    `dataset_revision` é o commit SHA resolvido (nunca `main` nem `null`): é o
    que torna a Etapa 1 reproduzível em outra máquina, em outra data.
    """

    dataset_id: str
    dataset_revision: str
    splits: tuple[str, ...]
    config_sha: str
    code_version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "splits": list(self.splits),
            "config_sha": self.config_sha,
            "code_version": self.code_version,
        }


# --------------------------------------------------------------------------- #
# Etapa 1 — janelas
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Segment:
    """Um segmento do corpus, no grão nativo das linhas do parquet.

    Carrega o suficiente para (a) janelar sobre metadados e (b) localizar depois
    a célula de áudio correspondente sem reler o dataset inteiro.
    """

    audio_id: int
    audio_name: str
    file_path: str
    speaker_code: str
    speaker_type: str
    start_time: float
    end_time: float
    duration: float
    normalized_text: str
    original_text: str
    audio_quality: str
    split: str

    @property
    def is_informant(self) -> bool:
        """`speaker_type == 'R'` é a fala válida do informante (ver Etapa 0)."""
        return self.speaker_type == "R"


@dataclass(frozen=True)
class Window:
    """Janela de fala contínua de um informante (30–60 s, 8–15 segmentos).

    `segment_audio_ids` é a proveniência dura da janela: a Etapa 2 baixa
    exatamente essas células de áudio, e o WER da Etapa 6 compara exatamente
    contra o texto que elas geraram.

    `position_index` registra de qual bloco da entrevista a janela foi sorteada
    (amostragem estratificada por posição, §Etapa 1) — permite auditar que a
    seleção cobriu a entrevista inteira, e não só o começo.
    """

    window_id: str
    speaker_code: str
    audio_name: str
    split: str
    region: str
    age: int
    segment_audio_ids: tuple[int, ...]
    start_time: float
    end_time: float
    duration_s: float
    n_segments: int
    reference_text: str
    position_index: int
    n_candidates_for_speaker: int

    @staticmethod
    def make_id(speaker_code: str, index: int) -> str:
        """Id legível e estável: `<speaker_code>__w03`.

        Legível porque aparece na planilha de revisão humana; estável porque
        `index` vem da ordenação canônica das janelas do falante, não da ordem
        em que foram construídas.
        """
        return f"{speaker_code}__w{index:02d}"

    @property
    def content_sha(self) -> str:
        """Hash do conteúdo da janela, para detectar deriva entre etapas."""
        return short_hash(
            {
                "segments": list(self.segment_audio_ids),
                "reference_text": self.reference_text,
            }
        )


# --------------------------------------------------------------------------- #
# Etapa 3 — itens
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Alternative:
    """Uma alternativa de múltipla escolha.

    `source_window_id` é o mecanismo do D2: um distrator precisa declarar de qual
    outra janela da mesma entrevista foi extraído. A alternativa correta tem
    `source_window_id` igual à janela-alvo. Distrator sem proveniência é item
    malformado — descartado e contado, nunca aceito.
    """

    label: str
    text: str
    is_correct: bool
    source_window_id: str | None = None


@dataclass(frozen=True)
class Item:
    """Item de múltipla escolha ancorado numa janela.

    `alternatives` guarda a ordem CANÔNICA (a que o gerador produziu). Toda
    reordenação — no filtro textual (D3) e na execução — é uma permutação
    derivada deterministicamente e registrada no artefato que a usou; a ordem
    canônica nunca é sobrescrita.
    """

    item_id: str
    window_id: str
    speaker_code: str
    question: str
    alternatives: tuple[Alternative, ...]
    generator_model: str
    prompt_version: str
    # Trilha de vida do item: preenchida pelas etapas seguintes, nunca apagada.
    filter_status: str = "pending"  # pending | kept | discarded
    filter_reason: str = ""
    review_status: str = "pending"  # pending | accepted | rejected
    review_reason: str = ""
    review_minutes: float | None = None

    @staticmethod
    def make_id(window_id: str, index: int) -> str:
        """Id legível e estável: `<window_id>__i02`."""
        return f"{window_id}__i{index:02d}"

    @property
    def correct_label(self) -> str:
        """Rótulo da alternativa correta na ordem canônica."""
        correct = [alt.label for alt in self.alternatives if alt.is_correct]
        if len(correct) != 1:
            raise ValueError(
                f"Item {self.item_id} tem {len(correct)} alternativas corretas; "
                "exatamente uma é exigida."
            )
        return correct[0]

    @property
    def n_alternatives(self) -> int:
        return len(self.alternatives)

    @property
    def is_eligible_for_eval(self) -> bool:
        """Só entra na avaliação o item que passou pelo filtro E pela revisão."""
        return self.filter_status == "kept" and self.review_status == "accepted"


# --------------------------------------------------------------------------- #
# Etapas 4–5 — execução
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunRecord:
    """Uma resposta de um modelo a um item, sob uma condição.

    `cache_key` é a identidade da execução: se dois `RunRecord` têm a mesma
    chave, são a mesma pergunta feita do mesmo jeito ao mesmo modelo, e o runner
    não paga duas vezes por ela.

    `permutation` é o que preserva o pareamento exigido pela decomposição da
    Etapa 6: as condições `asr` e `reference` do mesmo item precisam ver as
    alternativas na MESMA ordem, senão a diferença de acurácia entre elas mistura
    efeito de condição com efeito de posição.

    `parsed_label is None` significa resposta não interpretável — contada como
    categoria própria, jamais imputada (`is_correct` também fica `None`).
    """

    cache_key: str
    item_id: str
    speaker_code: str
    model_key: str
    condition: str
    prompt_version: str
    permutation: tuple[int, ...]
    raw_text: str
    parsed_label: str | None
    is_correct: bool | None
    finish_reason: str = ""
    usage: Mapping[str, int] = field(default_factory=dict)
    error: str = ""

    @staticmethod
    def make_cache_key(
        *,
        item_id: str,
        model_fingerprint: str,
        condition: str,
        prompt_version: str,
        permutation: Sequence[int],
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        """Chave de cache determinística da execução.

        `model_fingerprint` já inclui provedor, id e versão do modelo e os
        parâmetros de decodificação (ver `models.base.ModelSpec.fingerprint`).
        `extra` cobre o que for específico de uma condição — por exemplo, o hash
        da hipótese do ASR na condição `asr`, sem o qual trocar de ASR não
        invalidaria o cache.
        """
        return stable_hash(
            {
                "item_id": item_id,
                "model": model_fingerprint,
                "condition": condition,
                "prompt_version": prompt_version,
                "permutation": list(permutation),
                "extra": dict(extra or {}),
            }
        )


@dataclass(frozen=True)
class WerRecord:
    """WER de uma janela, em contagens brutas.

    Guardamos S, I, D e N — e não a razão já calculada — porque o WER agregado é
    `soma(S+I+D) / soma(N)`, nunca a média das razões por janela. Janelas curtas
    teriam peso indevido na média de razões.
    """

    window_id: str
    speaker_code: str
    model_key: str
    substitutions: int
    insertions: int
    deletions: int
    n_reference_words: int
    normalizer_version: str

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def wer(self) -> float:
        if self.n_reference_words == 0:
            raise ValueError(
                f"Janela {self.window_id} tem referência vazia; WER indefinido. "
                "Janelas vazias devem ser contadas e excluídas explicitamente."
            )
        return self.errors / self.n_reference_words


# --------------------------------------------------------------------------- #
# Trilha de exclusão (Princípio 2: nada dropado em silêncio)
# --------------------------------------------------------------------------- #
@dataclass
class ExclusionLedger:
    """Contador de unidades descartadas, por motivo.

    Toda etapa que filtra alguma coisa mantém um destes e o imprime no relatório.
    `assert_balanced` transforma "nada foi dropado em silêncio" de intenção em
    asserção: entradas = retidas + descartadas, ou o relatório não é gerado.
    """

    unit: str
    total_in: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str, n: int = 1) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + n

    @property
    def total_dropped(self) -> int:
        return sum(self.dropped.values())

    def assert_balanced(self) -> None:
        if self.kept + self.total_dropped != self.total_in:
            raise ValueError(
                f"Trilha de exclusão desbalanceada para '{self.unit}': "
                f"entradas={self.total_in}, retidas={self.kept}, "
                f"descartadas={self.total_dropped} "
                f"(diferença={self.total_in - self.kept - self.total_dropped}). "
                "Alguma unidade sumiu sem motivo registrado."
            )

    def to_markdown(self) -> list[str]:
        """Linhas de markdown com a contabilidade completa, em ordem estável."""
        lines = [f"- Entradas ({self.unit}): {self.total_in}"]
        for reason in sorted(self.dropped):
            lines.append(f"  - descartadas — {reason}: {self.dropped[reason]}")
        lines.append(f"  - subtotal descartado: {self.total_dropped}")
        lines.append(f"- **Retidas: {self.kept}**")
        return lines


# --------------------------------------------------------------------------- #
# I/O JSONL (formato de intercâmbio entre etapas)
# --------------------------------------------------------------------------- #
def _to_jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    """Escreve dataclasses como JSONL, com chaves ordenadas e UTF-8 legível.

    A ordenação de chaves é o que torna o arquivo comparável byte a byte entre
    execuções. Devolve o nº de registros escritos.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            payload = _to_jsonable(
                asdict(record) if hasattr(record, "__dataclass_fields__") else record
            )
            fh.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Lê um JSONL linha a linha, ignorando linhas em branco."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str | Path, record: Any) -> None:
    """Acrescenta um registro ao fim do arquivo (cache append-only do runner)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(asdict(record) if hasattr(record, "__dataclass_fields__") else record)
    with out.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
