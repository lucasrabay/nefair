"""Execução `item × configuração × condição` com cache, pareamento e contabilidade.

Este é o motor das Etapas 4 e 5. Ele produz um `RunRecord` por célula do produto
cartesiano entre os itens elegíveis, os modelos avaliados e as condições
habilitadas. Quatro invariantes governam o desenho, e cada uma existe porque a
alternativa quebraria um número do TCC:

1. **Pareamento entre condições.** As condições `asr` e `reference` do mesmo item
   e do mesmo modelo veem as alternativas na MESMA permutação. A decomposição da
   Etapa 6 é `Δ_ASR = acc(reference) − acc(asr)`; se a ordem das alternativas
   diferisse entre as duas, essa subtração misturaria efeito de via (áudio
   degradado pelo ASR) com efeito de posição da alternativa correta. Por isso a
   permutação é derivada de `(item_id, model_key, seed)` e **nunca** da condição.

2. **Cache honesto.** A chave é `RunRecord.make_cache_key`, que já cobre item,
   fingerprint do modelo (provedor + id + versão + params de decodificação),
   condição, versão do prompt e permutação. Falta uma coisa que só o runner sabe:
   na condição `asr`, a resposta depende da **hipótese do ASR**. Sem o hash dela
   em `extra`, trocar de ASR não invalidaria o cache e a Etapa 6 leria respostas
   dadas sobre a transcrição de outro sistema. O hash entra em `extra`.

3. **Erro nunca é silêncio, e nunca é cache.** Falha de provedor vira
   `RunRecord` com `error` preenchido, `parsed_label=None` e `is_correct=None`,
   contado no relatório. Esses registros **não** são gravados no cache: se
   fossem, uma retomada trataria a falha como trabalho concluído e o item ficaria
   permanentemente sem resposta.

4. **Ordem estável.** As chamadas são concorrentes (`max_concurrency`), mas os
   resultados são remontados na ordem canônica das tarefas, independente da ordem
   de conclusão — mesmo espírito de `corpus/load.py`. Duas execuções produzem o
   mesmo arquivo de cache, byte a byte.

O prompt é **injetado** (`prompt_builder`): `models/prompts.py` pertence a outro
bloco e ainda não existe quando este módulo é escrito. `default_prompt_builder`
aqui é um substituto trivial, suficiente para exercitar o pipeline; quem for usar
o prompt oficial passa o renderizador dele e a `prompt_version` correspondente.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nefair.config import EvalConfig, ModelsConfig
from nefair.models.asr.whisper import build_asr
from nefair.models.base import ASR, AudioQA, ModelSpec, QAResult, TextQA
from nefair.models.llm.clients import build_text_qa
from nefair.models.multimodal.clients import build_audio_qa
from nefair.run.parse import ParseResult, parse_label
from nefair.schema import (
    ALTERNATIVE_LABELS,
    Alternative,
    ExclusionLedger,
    Item,
    RunRecord,
    append_jsonl,
    read_jsonl,
    stable_hash,
)

# Versão do prompt embutido neste módulo. NÃO é a versão do prompt oficial do
# estudo (`models/prompts.PROMPT_VERSION`, do Bloco I): é um rótulo próprio, para
# que uma execução feita com o prompt provisório jamais possa ser confundida —
# no cache ou no relatório — com uma execução feita com o prompt definitivo.
DEFAULT_PROMPT_VERSION = "runner-default-v1"

# Papel de modelo exigido por cada condição. `asr` e `reference` entregam TEXTO ao
# LLM (a diferença entre elas é só de qual texto); `audio` entrega o áudio ao
# modelo multimodal nativo.
CONDITION_ROLE: Mapping[str, str] = {
    "asr": "text_qa",
    "reference": "text_qa",
    "audio": "audio_qa",
}

# Política de retry, alinhada com `corpus/load.py`: backoff exponencial limitado,
# com respeito a `Retry-After` em 429 (rate limit) e 503.
_RETRY_BACKOFF = 0.5
_RETRY_MAX_DELAY = 15.0
_RETRY_AFTER_CAP = 30.0

# Prefixos dos motivos de erro gerados pelo próprio runner (e não pelo provedor).
ERROR_MISSING_INPUT = "insumo_ausente"
ERROR_ASR_FAILED = "asr_falhou"


class ContaminationCheckError(RuntimeError):
    """D8 não satisfeito: a execução completa é recusada antes de qualquer chamada."""


# --------------------------------------------------------------------------- #
# Permutação das alternativas
# --------------------------------------------------------------------------- #
def _byte_stream(seed_hex: str, n_bytes: int) -> bytes:
    """Fluxo determinístico de bytes derivado de um digest, sem RNG global.

    Deliberadamente não usa `random` nem `numpy.random`: o valor depende só dos
    argumentos, então nenhuma ordem de execução, nenhum outro módulo e nenhuma
    versão de biblioteca podem alterar a permutação de um item.
    """
    chunks: list[bytes] = []
    size = 0
    counter = 0
    while size < n_bytes:
        block = hashlib.sha256(f"{seed_hex}:{counter}".encode()).digest()
        chunks.append(block)
        size += len(block)
        counter += 1
    return b"".join(chunks)[:n_bytes]


def derive_permutation(
    *, item_id: str, model_key: str, seed: int, n: int, shuffle: bool = True
) -> tuple[int, ...]:
    """Permutação das alternativas para `(item_id, model_key, seed)`.

    `permutation[i]` é o índice CANÔNICO da alternativa exibida na posição `i`.

    A condição **não** entra na derivação, e isso é o ponto inteiro: `asr` e
    `reference` do mesmo item e do mesmo modelo recebem a mesma ordem, que é o
    que torna `acc(reference) − acc(asr)` uma diferença de via e não de posição.

    O `model_key` entra porque modelos diferentes devem ver ordens diferentes —
    caso contrário, um viés de posição comum (por exemplo, preferir a primeira
    alternativa) apareceria como concordância entre modelos e seria lido como
    concordância de conteúdo.
    """
    if not shuffle or n <= 1:
        return tuple(range(n))
    digest = stable_hash({"item_id": item_id, "model_key": model_key, "seed": seed})
    stream = _byte_stream(digest, 4 * n)
    order = list(range(n))
    for i in range(n - 1, 0, -1):  # Fisher–Yates
        draw = int.from_bytes(stream[4 * i : 4 * i + 4], "big")
        j = draw % (i + 1)
        order[i], order[j] = order[j], order[i]
    return tuple(order)


def present_alternatives(
    item: Item, permutation: Sequence[int]
) -> tuple[tuple[Alternative, ...], str]:
    """Alternativas na ordem de exibição, rerrotuladas, e o rótulo correto.

    A ordem canônica do `Item` nunca é sobrescrita (§`schema.Item`): o que muda é
    o rótulo exibido. A alternativa correta continua sendo a mesma; o que a
    permutação altera é qual letra o modelo precisa dizer para acertá-la — e é
    isso que neutraliza o viés de posição.
    """
    if sorted(permutation) != list(range(len(item.alternatives))):
        raise ValueError(
            f"Item {item.item_id}: permutação {tuple(permutation)} não é uma permutação "
            f"de 0..{len(item.alternatives) - 1}."
        )
    labels = ALTERNATIVE_LABELS[: len(permutation)]
    presented = tuple(
        Alternative(
            label=labels[position],
            text=item.alternatives[source].text,
            is_correct=item.alternatives[source].is_correct,
            source_window_id=item.alternatives[source].source_window_id,
        )
        for position, source in enumerate(permutation)
    )
    correct = [alt.label for alt in presented if alt.is_correct]
    if len(correct) != 1:
        raise ValueError(
            f"Item {item.item_id} tem {len(correct)} alternativas corretas após a "
            "permutação; exatamente uma é exigida."
        )
    return presented, correct[0]


# --------------------------------------------------------------------------- #
# Prompt injetado
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PromptRequest:
    """Tudo o que um renderizador de prompt precisa saber sobre uma célula.

    `alternatives` já vem permutada e rerrotulada; `transcript` é `None` na
    condição `audio` (o modelo ouve, não lê). Manter esta estrutura explícita é o
    que permite trocar o prompt provisório pelo oficial sem tocar no runner.
    """

    item: Item
    condition: str
    alternatives: tuple[Alternative, ...]
    transcript: str | None
    audio_path: Path | None


def default_prompt_builder(request: PromptRequest) -> str:
    """Prompt provisório, mínimo e determinístico.

    Não é o prompt do estudo — o oficial é do Bloco I (`models/prompts.py`), com
    versão própria. Este existe para que o runner e os testes sejam executáveis
    hoje, e é rotulado por `DEFAULT_PROMPT_VERSION` justamente para que nenhuma
    execução feita com ele possa se passar pela definitiva.
    """
    lines: list[str] = []
    if request.transcript is not None:
        lines.append("Transcrição:")
        lines.append(request.transcript)
        lines.append("")
    lines.append(f"Pergunta: {request.item.question}")
    for alternative in request.alternatives:
        lines.append(f"{alternative.label}) {alternative.text}")
    lines.append("")
    lines.append("Responda apenas com a letra da alternativa correta.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Retry com backoff (espírito de corpus/load.py)
# --------------------------------------------------------------------------- #
def retry_delay(exc: Exception, attempt: int) -> float:
    """Segundos até a próxima tentativa.

    Backoff exponencial limitado; respeita `Retry-After` quando o provedor
    devolve 429 (rate limit) ou 503. Ignorar o `Retry-After` num 429 é a maneira
    mais rápida de ser bloqueado no meio de uma execução de horas.
    """
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) in (429, 503):
        headers = getattr(response, "headers", {}) or {}
        retry_after = str(headers.get("Retry-After", ""))
        if retry_after.isdigit():
            return min(float(retry_after), _RETRY_AFTER_CAP)
    return min(_RETRY_BACKOFF * (2**attempt), _RETRY_MAX_DELAY)


def call_with_retry(
    call: Callable[[], Any],
    *,
    max_retries: int,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[Any, str]:
    """Executa `call` com retry; devolve `(resultado, erro)`.

    `max_retries` é o número de TENTATIVAS (mesma semântica de `_MAX_RETRIES` em
    `corpus/load.py`), não de repetições extras. Esgotadas, devolve
    `(None, "<Tipo>: <mensagem>")` — o runner transforma isso num `RunRecord` com
    `error`, que é contado no relatório e não entra no cache.

    `sleeper` é injetável para que os testes não durmam de verdade.
    """
    attempts = max(1, int(max_retries))
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return call(), ""
        except Exception as exc:  # noqa: BLE001 - transitório (timeout/reset/5xx/429) ou fatal
            last = exc
            if attempt == attempts - 1:
                break
            sleeper(retry_delay(exc, attempt))
    return None, f"{type(last).__name__}: {last}"


# --------------------------------------------------------------------------- #
# Portão D8 — contaminação do ASR
# --------------------------------------------------------------------------- #
def assert_contamination_checked(models: ModelsConfig, eval_config: EvalConfig) -> None:
    """Recusa a execução enquanto o D8 não estiver resolvido.

    O Whisper ajustado para PT-BR pode ter sido treinado sobre o próprio
    CORAA/MuPe. Se foi, a condição `asr` fica inflada justamente no corpus que o
    estudo mede, e a decomposição `Δ_ASR` deixa de significar o que o texto diz
    que significa. Checar custa uma leitura do card de treino; rodar a Etapa 5
    inteira sobre um ASR contaminado custa a Etapa 5 inteira.

    A recusa vale para a execução COMPLETA, não só para a condição `asr`: a
    mitigação prevista no D8 inclui `split_restriction`, que muda quais janelas
    entram no estudo e, portanto, afeta também `reference` e `audio`.
    """
    if not eval_config.require_contamination_check:
        return
    if models.asr.contamination_checked:
        return
    note = models.asr.contamination_note.strip() or "(sem nota em models.yaml)"
    raise ContaminationCheckError(
        "Execução recusada pelo portão D8 (contaminação do ASR).\n"
        f"  models.yaml → asr.contamination_checked = false\n"
        f"  eval.yaml   → require_contamination_check = true\n"
        f"  ASR configurado: provider={models.asr.spec.provider} "
        f"model_id={models.asr.spec.model_id} version={models.asr.spec.version}\n"
        f"  split_restriction atual: {models.asr.split_restriction!r}\n"
        f"  O que precisa ser verificado: {note}\n"
        "  Ao concluir a verificação, preencher `asr.contamination_checked: true` "
        "(e `asr.split_restriction`, se a mitigação escolhida for restringir o "
        "split) em configs/models.yaml. Para exercitar o pipeline sem ASR real, "
        "use `scripts/06_run_eval.py --fake`."
    )


# --------------------------------------------------------------------------- #
# Cache JSONL append-only
# --------------------------------------------------------------------------- #
def _record_from_dict(payload: Mapping[str, Any]) -> RunRecord:
    """Reconstrói um `RunRecord` de uma linha do cache, com tipos restaurados."""
    usage_raw = payload.get("usage") or {}
    return RunRecord(
        cache_key=str(payload["cache_key"]),
        item_id=str(payload["item_id"]),
        speaker_code=str(payload["speaker_code"]),
        model_key=str(payload["model_key"]),
        condition=str(payload["condition"]),
        prompt_version=str(payload["prompt_version"]),
        permutation=tuple(int(index) for index in payload.get("permutation", ())),
        raw_text=str(payload.get("raw_text", "")),
        parsed_label=(
            None if payload.get("parsed_label") is None else str(payload["parsed_label"])
        ),
        is_correct=(None if payload.get("is_correct") is None else bool(payload["is_correct"])),
        finish_reason=str(payload.get("finish_reason", "")),
        usage={str(k): int(v) for k, v in usage_raw.items()},
        error=str(payload.get("error", "")),
    )


@dataclass
class RunCache:
    """Cache append-only de `RunRecord`, indexado por `cache_key`.

    Append-only (e não reescrita do arquivo) porque é o que torna a execução
    retomável: uma interrupção no meio de doze horas de chamadas perde, no
    máximo, a chamada em curso. Linhas repetidas da mesma chave são possíveis
    (uma execução com `--no-resume`, por exemplo); a última vence e a repetição é
    **contada**, nunca ignorada em silêncio.
    """

    path: Path
    records: dict[str, RunRecord] = field(default_factory=dict)
    n_duplicates: int = 0
    n_loaded: int = 0

    @classmethod
    def load(cls, path: str | Path, *, enabled: bool = True) -> RunCache:
        cache = cls(path=Path(path))
        if not enabled or not cache.path.is_file():
            return cache
        for payload in read_jsonl(cache.path):
            record = _record_from_dict(payload)
            cache.n_loaded += 1
            if record.cache_key in cache.records:
                cache.n_duplicates += 1
            cache.records[record.cache_key] = record
        return cache

    def get(self, cache_key: str) -> RunRecord | None:
        return self.records.get(cache_key)

    def put(self, record: RunRecord) -> None:
        """Grava um registro no disco e no índice.

        Registros com `error` NÃO devem chegar aqui: falha de provedor precisa ser
        retentada na próxima execução, não tratada como trabalho concluído.
        """
        if record.error:
            raise ValueError(
                f"RunCache.put recebeu registro com erro ({record.error!r}); "
                "erros não entram no cache, para que a retomada os retente."
            )
        append_jsonl(self.path, record)
        self.records[record.cache_key] = record


# --------------------------------------------------------------------------- #
# Fase do ASR — hipóteses por janela
# --------------------------------------------------------------------------- #
@dataclass
class AsrPhase:
    """Hipóteses do ASR por janela, com cache próprio em JSONL.

    Por que uma fase separada, antes das chamadas aos LLMs: a chave de cache da
    condição `asr` depende do HASH da hipótese, então a hipótese precisa existir
    antes de sabermos se a resposta do LLM já está em cache. Rodar o ASR dentro
    das threads dos LLMs faria o mesmo áudio ser transcrito várias vezes (uma por
    modelo) e tornaria a ordem de escrita do cache dependente de corrida.

    As hipóteses também são o insumo do WER da Etapa 6: persistir aqui evita que
    a Etapa 6 precise reexecutar o ASR só para medir.
    """

    asr: ASR | None
    path: Path | None
    hypotheses: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    n_calls: int = 0
    n_cache_hits: int = 0
    n_loaded: int = 0

    def _key(self, window_id: str) -> str:
        assert self.asr is not None
        return stable_hash({"window_id": window_id, "asr": self.asr.spec.fingerprint()})

    def _load_cache(self) -> dict[str, str]:
        cached: dict[str, str] = {}
        if self.path is None or not self.path.is_file():
            return cached
        for payload in read_jsonl(self.path):
            cached[str(payload["key"])] = str(payload.get("text", ""))
            self.n_loaded += 1
        return cached

    def resolve(
        self,
        window_audio: Sequence[tuple[str, Path]],
        *,
        max_concurrency: int,
        max_retries: int,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        """Transcreve as janelas pedidas, em ordem estável, reusando o cache.

        `window_audio` já vem ordenado e sem repetição; a concorrência não altera
        a ordem de escrita porque os resultados são consumidos na ordem de
        `pool.map`, não na ordem de conclusão.
        """
        if self.asr is None or not window_audio:
            return
        cached = self._load_cache()
        pending: list[tuple[str, Path, str]] = []
        for window_id, audio_path in window_audio:
            key = self._key(window_id)
            if key in cached:
                self.hypotheses[window_id] = cached[key]
                self.n_cache_hits += 1
            else:
                pending.append((window_id, audio_path, key))

        if not pending:
            return

        def work(task: tuple[str, Path, str]) -> tuple[str, str, str, str]:
            window_id, audio_path, key = task
            assert self.asr is not None
            result, error = call_with_retry(
                lambda: self.asr.transcribe(audio_path),  # noqa: B023 - consumido antes do próximo
                max_retries=max_retries,
                sleeper=sleeper,
            )
            if error:
                return window_id, key, "", error
            if result.error:
                return window_id, key, "", f"{ERROR_ASR_FAILED}: {result.error}"
            return window_id, key, result.text, ""

        workers = max(1, int(max_concurrency))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for window_id, key, text, error in pool.map(work, pending):
                self.n_calls += 1
                if error:
                    self.errors[window_id] = error
                    continue
                self.hypotheses[window_id] = text
                if self.path is not None:
                    append_jsonl(
                        self.path,
                        {
                            "key": key,
                            "window_id": window_id,
                            "asr_model_key": self.asr.spec.key,
                            "asr_fingerprint": self.asr.spec.fingerprint(),
                            "text": text,
                        },
                    )


# --------------------------------------------------------------------------- #
# Tarefas e resumo
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunTask:
    """Uma célula do produto `item × modelo × condição`, já resolvida.

    Tudo o que depende de configuração (permutação, prompt, chave de cache) é
    calculado na montagem da tarefa, em thread única. O trabalho concorrente fica
    restrito à chamada ao provedor — que é a única parte lenta e a única que pode
    falhar de forma transitória.
    """

    item: Item
    model_key: str
    condition: str
    permutation: tuple[int, ...]
    correct_label: str
    labels: tuple[str, ...]
    prompt: str
    cache_key: str
    audio_path: Path | None
    input_error: str


@dataclass
class CellStats:
    """Contagens de uma célula `(modelo, condição)` do relatório."""

    n: int = 0
    n_cache_hits: int = 0
    n_calls: int = 0
    n_unparsed: int = 0
    n_errors: int = 0
    n_correct: int = 0

    @property
    def unparsed_rate(self) -> float:
        """Taxa de não interpretável sobre as respostas EFETIVAMENTE obtidas.

        O denominador exclui os erros de provedor: uma resposta que nunca chegou
        não é uma resposta ininteligível. Misturar as duas coisas esconderia uma
        queda de provedor dentro de um número que descreve o modelo.
        """
        answered = self.n - self.n_errors
        return self.n_unparsed / answered if answered else 0.0


@dataclass
class RunSummary:
    """Resultado completo de uma execução, pronto para virar relatório."""

    records: tuple[RunRecord, ...]
    prompt_version: str
    conditions: tuple[str, ...]
    cache_path: str
    asr_cache_path: str
    contamination_gate_enforced: bool
    fake_mode: bool
    n_tasks: int = 0
    n_cache_hits: int = 0
    n_model_calls: int = 0
    n_errors: int = 0
    n_unparsed: int = 0
    n_correct: int = 0
    n_cache_duplicates: int = 0
    per_cell: dict[tuple[str, str], CellStats] = field(default_factory=dict)
    errors_by_type: dict[str, int] = field(default_factory=dict)
    unparsed_by_reason: dict[str, int] = field(default_factory=dict)
    strategy_counts: dict[str, int] = field(default_factory=dict)
    item_ledger: ExclusionLedger = field(
        default_factory=lambda: ExclusionLedger(unit="itens gerados")
    )
    asr_calls: int = 0
    asr_cache_hits: int = 0
    asr_errors: int = 0
    skipped_incompatible: int = 0

    @property
    def n_answered(self) -> int:
        return self.n_tasks - self.n_errors


# --------------------------------------------------------------------------- #
# Montagem das tarefas
# --------------------------------------------------------------------------- #
def select_eligible_items(items: Sequence[Item], ledger: ExclusionLedger) -> list[Item]:
    """Itens elegíveis (`filter_status == kept` e `review_status == accepted`).

    Os inelegíveis não somem: cada um é contado por motivo no `ExclusionLedger`,
    que o relatório imprime e cujo balanço (`entradas = retidos + descartados`) é
    asserido.
    """
    ledger.total_in = len(items)
    kept: list[Item] = []
    for item in sorted(items, key=lambda it: it.item_id):
        if item.is_eligible_for_eval:
            kept.append(item)
            continue
        if item.filter_status != "kept":
            ledger.drop(f"filtro textual: {item.filter_status}")
        else:
            ledger.drop(f"revisão humana: {item.review_status}")
    ledger.kept = len(kept)
    ledger.assert_balanced()
    return kept


def _models_for_condition(
    models: ModelsConfig, condition: str, adapter_keys: Sequence[str]
) -> list[str]:
    """Chaves de modelo compatíveis com a condição, em ordem estável.

    Um modelo `text_qa` não pode responder na condição `audio` e vice-versa: o
    produto cartesiano é sobre os pares COMPATÍVEIS, não sobre todos.
    """
    role = CONDITION_ROLE[condition]
    return [
        key
        for key in sorted(models.specs)
        if models.specs[key].role == role and key in set(adapter_keys)
    ]


def build_tasks(
    *,
    items: Sequence[Item],
    models: ModelsConfig,
    conditions: Sequence[str],
    seed: int,
    shuffle_alternatives: bool,
    specs_in_use: Mapping[str, ModelSpec],
    reference_texts: Mapping[str, str],
    hypotheses: Mapping[str, str],
    hypothesis_errors: Mapping[str, str],
    audio_paths: Mapping[str, Path],
    prompt_builder: Callable[[PromptRequest], str],
    prompt_version: str,
) -> tuple[list[RunTask], int]:
    """Produto cartesiano resolvido, em ordem canônica `(item, modelo, condição)`.

    `specs_in_use` traz o `ModelSpec` do ADAPTADOR (não o do YAML): no modo falso
    o provider é trocado para `"fake"`, e usar o spec do adaptador é o que impede
    uma execução de teste de gravar respostas inventadas sob a chave de cache de
    uma execução real.
    """
    tasks: list[RunTask] = []
    skipped_incompatible = 0
    ordered_conditions = [c for c in conditions]
    for item in items:
        for condition in ordered_conditions:
            model_keys = _models_for_condition(models, condition, list(specs_in_use))
            if not model_keys:
                skipped_incompatible += 1
                continue
            for model_key in model_keys:
                permutation = derive_permutation(
                    item_id=item.item_id,
                    model_key=model_key,
                    seed=seed,
                    n=item.n_alternatives,
                    shuffle=shuffle_alternatives,
                )
                presented, correct_label = present_alternatives(item, permutation)
                labels = ALTERNATIVE_LABELS[: item.n_alternatives]

                transcript: str | None = None
                audio_path: Path | None = audio_paths.get(item.window_id)
                input_error = ""
                extra: dict[str, Any] = {}

                if condition == "reference":
                    transcript = reference_texts.get(item.window_id)
                    if transcript is None:
                        input_error = (
                            f"{ERROR_MISSING_INPUT}: transcrição de referência ausente "
                            f"para window_id={item.window_id}"
                        )
                        transcript = ""
                elif condition == "asr":
                    if item.window_id in hypothesis_errors:
                        input_error = hypothesis_errors[item.window_id]
                        transcript = ""
                    elif item.window_id in hypotheses:
                        transcript = hypotheses[item.window_id]
                    else:
                        input_error = (
                            f"{ERROR_MISSING_INPUT}: hipótese do ASR ausente para "
                            f"window_id={item.window_id}"
                        )
                        transcript = ""
                    # SEM isto, trocar de ASR não invalidaria o cache e a Etapa 6
                    # leria respostas dadas sobre a transcrição de outro sistema.
                    extra["hypothesis_sha"] = stable_hash(transcript)
                elif condition == "audio":
                    if audio_path is None:
                        input_error = (
                            f"{ERROR_MISSING_INPUT}: áudio ausente para window_id={item.window_id}"
                        )

                prompt = prompt_builder(
                    PromptRequest(
                        item=item,
                        condition=condition,
                        alternatives=presented,
                        transcript=transcript,
                        audio_path=audio_path,
                    )
                )
                cache_key = RunRecord.make_cache_key(
                    item_id=item.item_id,
                    model_fingerprint=specs_in_use[model_key].fingerprint(),
                    condition=condition,
                    prompt_version=prompt_version,
                    permutation=permutation,
                    extra=extra,
                )
                tasks.append(
                    RunTask(
                        item=item,
                        model_key=model_key,
                        condition=condition,
                        permutation=permutation,
                        correct_label=correct_label,
                        labels=labels,
                        prompt=prompt,
                        cache_key=cache_key,
                        audio_path=audio_path,
                        input_error=input_error,
                    )
                )
    return tasks, skipped_incompatible


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #
def _record_from_result(
    task: RunTask, result: QAResult | None, error: str, prompt_version: str
) -> tuple[RunRecord, ParseResult | None]:
    """Converte a saída do provedor num `RunRecord` (+ o resultado do parsing).

    Três desfechos, e nenhum deles é imputação:
    - erro (provedor ou insumo): `parsed_label=None`, `is_correct=None`, `error`
      preenchido;
    - resposta não interpretável: `parsed_label=None`, `is_correct=None`,
      `error=""` — é a categoria própria que o TCC reporta;
    - resposta interpretada: `parsed_label` e `is_correct` preenchidos.
    """
    base = {
        "cache_key": task.cache_key,
        "item_id": task.item.item_id,
        "speaker_code": task.item.speaker_code,
        "model_key": task.model_key,
        "condition": task.condition,
        "prompt_version": prompt_version,
        "permutation": task.permutation,
    }
    if error or result is None:
        return (
            RunRecord(
                **base,
                raw_text="",
                parsed_label=None,
                is_correct=None,
                error=error or "resultado ausente sem exceção registrada",
            ),
            None,
        )
    if result.error:
        return (
            RunRecord(
                **base,
                raw_text=result.raw_text,
                parsed_label=None,
                is_correct=None,
                finish_reason=result.finish_reason,
                usage=dict(result.usage),
                error=result.error,
            ),
            None,
        )
    parsed = parse_label(result.raw_text, task.labels)
    return (
        RunRecord(
            **base,
            raw_text=result.raw_text,
            parsed_label=parsed.label,
            is_correct=(None if parsed.label is None else parsed.label == task.correct_label),
            finish_reason=result.finish_reason,
            usage=dict(result.usage),
        ),
        parsed,
    )


def _tally(summary: RunSummary, record: RunRecord, parsed: ParseResult | None) -> None:
    """Contabiliza um registro no resumo (célula, erro, não interpretável)."""
    cell = summary.per_cell.setdefault((record.model_key, record.condition), CellStats())
    cell.n += 1
    if record.error:
        summary.n_errors += 1
        cell.n_errors += 1
        kind = record.error.split(":", 1)[0].strip() or "desconhecido"
        summary.errors_by_type[kind] = summary.errors_by_type.get(kind, 0) + 1
        return
    if record.parsed_label is None:
        summary.n_unparsed += 1
        cell.n_unparsed += 1
        reason = parsed.reason if parsed is not None else "nao_recomputado"
        summary.unparsed_by_reason[reason] = summary.unparsed_by_reason.get(reason, 0) + 1
        return
    if parsed is not None and parsed.strategy:
        summary.strategy_counts[parsed.strategy] = (
            summary.strategy_counts.get(parsed.strategy, 0) + 1
        )
    if record.is_correct:
        summary.n_correct += 1
        cell.n_correct += 1


def run_eval(
    *,
    items: Sequence[Item],
    models: ModelsConfig,
    eval_config: EvalConfig,
    text_qa: Mapping[str, TextQA] | None = None,
    audio_qa: Mapping[str, AudioQA] | None = None,
    asr: ASR | None = None,
    reference_texts: Mapping[str, str] | None = None,
    audio_paths: Mapping[str, Path] | None = None,
    prompt_builder: Callable[[PromptRequest], str] = default_prompt_builder,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    conditions: Sequence[str] | None = None,
    cache_path: str | Path | None = None,
    asr_cache_path: str | Path | None = None,
    resume: bool = True,
    enforce_contamination_gate: bool = True,
    fake_mode: bool = False,
    sleeper: Callable[[float], None] = time.sleep,
) -> RunSummary:
    """Executa o produto `item × modelo × condição` e devolve o resumo.

    `enforce_contamination_gate` existe para o modo `--fake`: sem ASR real não há
    risco de contaminação, e bloquear ali impediria exatamente o que o modo falso
    serve para fazer — exercitar o pipeline ponta a ponta sem credencial. Em
    execução real o portão fica ligado, e o relatório ecoa qual dos dois valeu.
    """
    if enforce_contamination_gate:
        assert_contamination_checked(models, eval_config)

    text_qa = dict(text_qa or {})
    audio_qa = dict(audio_qa or {})
    reference_texts = dict(reference_texts or {})
    audio_paths = dict(audio_paths or {})

    selected = tuple(conditions) if conditions is not None else eval_config.conditions
    unknown = [c for c in selected if c not in CONDITION_ROLE]
    if unknown:
        raise ValueError(f"Condição desconhecida: {unknown}; esperado {tuple(CONDITION_ROLE)}.")
    # Ordem canônica das condições, para que o arquivo de cache não dependa da
    # ordem em que o usuário digitou `--conditions`.
    ordered_conditions = tuple(c for c in eval_config.conditions if c in set(selected))

    cache_file = Path(cache_path or eval_config.cache_path)
    asr_cache_file = (
        Path(asr_cache_path) if asr_cache_path is not None else _asr_cache_for(cache_file)
    )

    ledger = ExclusionLedger(unit="itens gerados")
    eligible = select_eligible_items(items, ledger)

    # Fase 1 — hipóteses do ASR (só se a condição `asr` estiver habilitada).
    phase = AsrPhase(asr=asr if "asr" in ordered_conditions else None, path=asr_cache_file)
    if phase.asr is not None:
        needed: list[tuple[str, Path]] = []
        seen_windows: set[str] = set()
        for item in eligible:
            if item.window_id in seen_windows:
                continue
            seen_windows.add(item.window_id)
            path = audio_paths.get(item.window_id)
            if path is not None:
                needed.append((item.window_id, path))
        phase.resolve(
            sorted(needed),
            max_concurrency=eval_config.max_concurrency,
            max_retries=eval_config.max_retries,
            sleeper=sleeper,
        )

    # Fase 2 — montagem determinística das tarefas.
    specs_in_use: dict[str, ModelSpec] = {}
    for key, adapter in sorted(text_qa.items()):
        specs_in_use[key] = adapter.spec
    for key, adapter in sorted(audio_qa.items()):
        specs_in_use[key] = adapter.spec

    tasks, skipped_incompatible = build_tasks(
        items=eligible,
        models=models,
        conditions=ordered_conditions,
        seed=eval_config.seed,
        shuffle_alternatives=eval_config.shuffle_alternatives,
        specs_in_use=specs_in_use,
        reference_texts=reference_texts,
        hypotheses=phase.hypotheses,
        hypothesis_errors=phase.errors,
        audio_paths=audio_paths,
        prompt_builder=prompt_builder,
        prompt_version=prompt_version,
    )

    summary = RunSummary(
        records=(),
        prompt_version=prompt_version,
        conditions=ordered_conditions,
        cache_path=str(cache_file),
        asr_cache_path=str(asr_cache_file),
        contamination_gate_enforced=enforce_contamination_gate,
        fake_mode=fake_mode,
        n_tasks=len(tasks),
        item_ledger=ledger,
        asr_calls=phase.n_calls,
        asr_cache_hits=phase.n_cache_hits,
        asr_errors=len(phase.errors),
        skipped_incompatible=skipped_incompatible,
    )

    # Fase 3 — cache. Acerto de cache NÃO chama o modelo.
    cache = RunCache.load(cache_file, enabled=resume)
    summary.n_cache_duplicates = cache.n_duplicates
    records: list[RunRecord | None] = [None] * len(tasks)
    parses: list[ParseResult | None] = [None] * len(tasks)
    pending: list[int] = []
    for index, task in enumerate(tasks):
        hit = cache.get(task.cache_key)
        if hit is not None:
            records[index] = hit
            # O parsing é puro: o motivo da resposta não interpretável é
            # recomputado do `raw_text` guardado, sem reexecutar nada.
            parses[index] = parse_label(hit.raw_text, task.labels) if hit.raw_text else None
            summary.n_cache_hits += 1
            summary.per_cell.setdefault((task.model_key, task.condition), CellStats())
            summary.per_cell[(task.model_key, task.condition)].n_cache_hits += 1
        else:
            pending.append(index)

    # Fase 4 — chamadas, concorrentes mas remontadas em ordem canônica.
    def work(index: int) -> tuple[int, QAResult | None, str]:
        task = tasks[index]
        if task.input_error:
            return index, None, task.input_error
        if task.condition == "audio":
            adapter_audio = audio_qa[task.model_key]
            path = task.audio_path
            assert path is not None  # garantido por `input_error` acima
            result, error = call_with_retry(
                lambda: adapter_audio.answer(task.prompt, path),
                max_retries=eval_config.max_retries,
                sleeper=sleeper,
            )
        else:
            adapter_text = text_qa[task.model_key]
            result, error = call_with_retry(
                lambda: adapter_text.answer(task.prompt),
                max_retries=eval_config.max_retries,
                sleeper=sleeper,
            )
        return index, result, error

    if pending:
        workers = max(1, int(eval_config.max_concurrency))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # `pool.map` entrega na ordem de submissão, não de conclusão: o
            # arquivo de cache sai idêntico entre execuções.
            for index, result, error in pool.map(work, pending):
                task = tasks[index]
                if not tasks[index].input_error:
                    summary.n_model_calls += 1
                    summary.per_cell.setdefault(
                        (task.model_key, task.condition), CellStats()
                    ).n_calls += 1
                record, parsed = _record_from_result(task, result, error, prompt_version)
                records[index] = record
                parses[index] = parsed
                if not record.error:
                    cache.put(record)

    for index, record in enumerate(records):
        assert record is not None  # toda tarefa produz exatamente um registro
        _tally(summary, record, parses[index])

    summary.records = tuple(record for record in records if record is not None)
    return summary


def _asr_cache_for(cache_file: Path) -> Path:
    """Caminho do cache de hipóteses, derivado do cache de execução.

    Derivado (e não uma chave nova de YAML) porque o `configs/eval.yaml` pertence
    à camada de contratos e não é deste bloco. O valor efetivo vai ao relatório.
    """
    return cache_file.with_name(f"{cache_file.stem}_asr{cache_file.suffix or '.jsonl'}")


# --------------------------------------------------------------------------- #
# Construção dos adaptadores
# --------------------------------------------------------------------------- #
def build_adapters(
    models: ModelsConfig,
    *,
    seed: int = 0,
    fake: bool = False,
    n_alternatives: int = 4,
) -> tuple[ASR, dict[str, TextQA], dict[str, AudioQA]]:
    """Constrói ASR + adaptadores de texto e multimodais a partir do `models.yaml`.

    Com `fake=True`, todos os provedores viram falsos determinísticos — e o
    `provider` do `ModelSpec` vira `"fake"`, o que muda o `fingerprint()` e
    isola completamente o cache falso do cache real.
    """
    asr = build_asr(models.asr.spec, seed=seed, force_fake=fake)
    text: dict[str, TextQA] = {}
    audio: dict[str, AudioQA] = {}
    for key in sorted(models.specs):
        spec = models.specs[key]
        if spec.role == "text_qa":
            text[key] = build_text_qa(
                spec, seed=seed, n_alternatives=n_alternatives, force_fake=fake
            )
        elif spec.role == "audio_qa":
            audio[key] = build_audio_qa(
                spec, seed=seed, n_alternatives=n_alternatives, force_fake=fake
            )
        else:
            raise ValueError(
                f"Modelo '{key}' tem role '{spec.role}', que não é avaliável em "
                "nenhuma condição (esperado 'text_qa' ou 'audio_qa')."
            )
    return asr, text, audio


# --------------------------------------------------------------------------- #
# Relatório
# --------------------------------------------------------------------------- #
def _pct(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.1f}%" if denominator else "—"


def build_eval_report(
    summary: RunSummary,
    models: ModelsConfig,
    eval_config: EvalConfig,
    *,
    generated_at: str,
    extra_config: Mapping[str, Any] | None = None,
) -> str:
    """Relatório em Markdown da execução (Etapas 4–5).

    Contém o que o plano exige que seja auditável: contabilidade completa dos
    itens de entrada, nº de execuções e de acertos de cache, taxa de não
    interpretável **por modelo e por condição**, erros por tipo, e o valor
    EFETIVO de cada chave de config usada (nunca o valor "provável").
    """
    lines: list[str] = []
    lines.append("# Relatório de execução — Etapas 4 e 5")
    lines.append("")
    lines.append(f"- Gerado em: `{generated_at}`")
    lines.append(
        "- Modo: **falso (sem rede, sem credencial)**"
        if summary.fake_mode
        else "- Modo: **real (provedores)**"
    )
    lines.append("")

    lines.append("## Configuração efetiva")
    lines.append("")
    lines.append("| chave | valor efetivo |")
    lines.append("|---|---|")
    effective: list[tuple[str, Any]] = [
        ("eval.seed", eval_config.seed),
        ("eval.conditions (config)", list(eval_config.conditions)),
        ("condições executadas", list(summary.conditions)),
        ("eval.cache_path (efetivo)", summary.cache_path),
        ("cache de hipóteses do ASR (derivado)", summary.asr_cache_path),
        ("eval.max_concurrency", eval_config.max_concurrency),
        ("eval.max_retries", eval_config.max_retries),
        ("eval.shuffle_alternatives", eval_config.shuffle_alternatives),
        ("eval.require_contamination_check (D8)", eval_config.require_contamination_check),
        ("models.asr.contamination_checked (D8)", models.asr.contamination_checked),
        ("models.asr.split_restriction (D8)", models.asr.split_restriction),
        ("portão D8 aplicado nesta execução", summary.contamination_gate_enforced),
        ("prompt_version", summary.prompt_version),
        ("models.asr.spec", models.asr.spec.model_id + "@" + models.asr.spec.version),
        ("models.paired (D9)", [f"{p.family}: {p.native} × {p.cascade}" for p in models.paired]),
        ("models.unpaired_extra (D9)", list(models.unpaired_extra)),
    ]
    for key, value in effective:
        lines.append(f"| `{key}` | `{value}` |")
    for key, value in sorted((extra_config or {}).items()):
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Itens de entrada (nada dropado em silêncio)")
    lines.append("")
    lines.extend(summary.item_ledger.to_markdown())
    lines.append("")

    lines.append("## Execuções")
    lines.append("")
    lines.append(f"- Tarefas (item × modelo × condição): {summary.n_tasks}")
    lines.append(
        f"- Acertos de cache: {summary.n_cache_hits} "
        f"({_pct(summary.n_cache_hits, summary.n_tasks)})"
    )
    lines.append(f"- Chamadas a modelo: {summary.n_model_calls}")
    lines.append(f"- Erros: {summary.n_errors} ({_pct(summary.n_errors, summary.n_tasks)})")
    lines.append(
        f"- Respostas não interpretáveis: {summary.n_unparsed} "
        f"({_pct(summary.n_unparsed, summary.n_answered)} das respostas obtidas)"
    )
    lines.append(
        f"- Acertos: {summary.n_correct} "
        f"({_pct(summary.n_correct, summary.n_answered - summary.n_unparsed)} das "
        "respostas interpretáveis)"
    )
    lines.append(f"- Linhas repetidas no cache (última vence): {summary.n_cache_duplicates}")
    lines.append(
        f"- ASR: {summary.asr_calls} chamadas, {summary.asr_cache_hits} acertos de cache, "
        f"{summary.asr_errors} janelas com erro"
    )
    if summary.skipped_incompatible:
        lines.append(
            f"- Combinações (condição sem modelo compatível) puladas: "
            f"{summary.skipped_incompatible}"
        )
    lines.append("")

    lines.append("## Não interpretável por modelo e condição")
    lines.append("")
    lines.append(
        "Denominador = respostas efetivamente obtidas (exclui erros de provedor). "
        "Resposta não interpretável **não** é imputada como erro nem como acerto."
    )
    lines.append("")
    lines.append(
        "| modelo | condição | tarefas | cache | chamadas | erros | não interpretável | taxa |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for (model_key, condition), cell in sorted(summary.per_cell.items()):
        lines.append(
            f"| `{model_key}` | `{condition}` | {cell.n} | {cell.n_cache_hits} | "
            f"{cell.n_calls} | {cell.n_errors} | {cell.n_unparsed} | "
            f"{100.0 * cell.unparsed_rate:.1f}% |"
        )
    lines.append("")

    lines.append("## Motivos de não interpretabilidade")
    lines.append("")
    if summary.unparsed_by_reason:
        for reason in sorted(summary.unparsed_by_reason):
            lines.append(f"- `{reason}`: {summary.unparsed_by_reason[reason]}")
    else:
        lines.append("- (nenhuma)")
    lines.append("")

    lines.append("## Estratégia de parsing que decidiu")
    lines.append("")
    if summary.strategy_counts:
        for strategy in sorted(summary.strategy_counts):
            lines.append(f"- `{strategy}`: {summary.strategy_counts[strategy]}")
    else:
        lines.append("- (nenhuma resposta interpretada)")
    lines.append("")

    lines.append("## Erros por tipo")
    lines.append("")
    if summary.errors_by_type:
        for kind in sorted(summary.errors_by_type):
            lines.append(f"- `{kind}`: {summary.errors_by_type[kind]}")
    else:
        lines.append("- (nenhum)")
    lines.append("")
    lines.append(
        "Erros **não** são gravados no cache: uma retomada precisa retentá-los, "
        "em vez de tratar a falha como trabalho concluído."
    )
    lines.append("")
    return "\n".join(lines)
