"""Loader do YAML de configuração e utilitário mínimo de `.env`.

Sem dependências além de `pyyaml`: o `.env` é lido por um parser simples de
linhas `CHAVE=VALOR` para não introduzir `python-dotenv` (deps mínimas).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    """Identificação e escopo do dataset a auditar."""

    id: str
    revision: str | None
    splits: tuple[str, ...]


@dataclass(frozen=True)
class CorpusAuditConfig:
    """Configuração completa da auditoria de metadados (Etapa 0)."""

    dataset: DatasetConfig
    audio_columns: tuple[str, ...]
    region_map: dict[str, str]
    age_bucket_edges: tuple[int, ...]

    @property
    def dataset_id(self) -> str:
        return self.dataset.id


def load_dotenv(path: str | Path = ".env") -> None:
    """Carrega pares CHAVE=VALOR de um `.env` no ambiente (sem sobrescrever).

    Silencioso se o arquivo não existir. Não sobrescreve variáveis já definidas
    no ambiente, para que o ambiente do processo tenha precedência.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str | Path) -> CorpusAuditConfig:
    """Lê e valida o YAML de configuração da auditoria."""
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Config inválido em {path}: esperado mapa no topo.")

    dataset_raw = raw.get("dataset")
    if not isinstance(dataset_raw, dict):
        raise ValueError("Config inválido: seção 'dataset' ausente ou malformada.")

    splits = dataset_raw.get("splits")
    if not isinstance(splits, list) or not splits:
        raise ValueError("Config inválido: 'dataset.splits' deve ser lista não vazia.")

    dataset = DatasetConfig(
        id=str(dataset_raw["id"]),
        revision=dataset_raw.get("revision"),
        splits=tuple(str(s) for s in splits),
    )

    region_map_raw = raw.get("region_map")
    if not isinstance(region_map_raw, dict) or not region_map_raw:
        raise ValueError("Config inválido: 'region_map' deve ser mapa não vazio.")
    region_map = {str(k): str(v) for k, v in region_map_raw.items()}

    audio_columns = tuple(str(c) for c in raw.get("audio_columns", ["audio"]))

    age_buckets_raw = raw.get("age_buckets", {})
    edges = age_buckets_raw.get("edges", []) if isinstance(age_buckets_raw, dict) else []
    age_bucket_edges = tuple(int(e) for e in edges)

    return CorpusAuditConfig(
        dataset=dataset,
        audio_columns=audio_columns,
        region_map=region_map,
        age_bucket_edges=age_bucket_edges,
    )


# --------------------------------------------------------------------------- #
# Configs das Etapas 1–6
#
# Cada decisão ainda aberta do plano (D1–D11) é uma CHAVE nomeada aqui, nunca um
# default escondido no meio da lógica. A carga valida o que consegue validar
# sozinha e o relatório de cada etapa ecoa o valor efetivamente usado.
#
# `base_config` amarra toda etapa nova ao mesmo `configs/corpus_audit.yaml`, de
# modo que a revisão (SHA) do dataset e o mapa de região tenham uma única fonte
# de verdade e não possam divergir entre a Etapa 0 e as seguintes.
# --------------------------------------------------------------------------- #
import hashlib  # noqa: E402
from typing import Any  # noqa: E402

from nefair.models.base import ModelSpec  # noqa: E402
from nefair.schema import CONDITIONS  # noqa: E402


def config_sha(path: str | Path) -> str:
    """SHA-256 do arquivo de config, gravado na proveniência dos artefatos."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config inválido em {path}: esperado mapa no topo.")
    return raw


def _require(raw: dict[str, Any], key: str, path: str | Path) -> Any:
    """Lê uma chave obrigatória, falhando com o arquivo e a chave no erro.

    Preferimos falhar na carga a assumir um default: uma decisão metodológica
    ausente do YAML precisa aparecer como erro, não como silêncio.
    """
    if key not in raw:
        raise ValueError(f"Config inválido em {path}: chave obrigatória '{key}' ausente.")
    return raw[key]


def _resolve_base(raw: dict[str, Any], path: str | Path) -> CorpusAuditConfig:
    """Carrega o `corpus_audit.yaml` referenciado por `base_config`.

    O caminho é resolvido relativo ao diretório do próprio config, para que
    `configs/windows.yaml` funcione de qualquer diretório de trabalho.
    """
    base_ref = _require(raw, "base_config", path)
    base_path = Path(path).parent / str(base_ref)
    if not base_path.is_file():
        base_path = Path(str(base_ref))
    if not base_path.is_file():
        raise ValueError(f"Config inválido em {path}: `base_config` '{base_ref}' não encontrado.")
    return load_config(base_path)


# --------------------------------------------------------------------------- #
# Etapa 1 — janelas
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WindowsConfig:
    """Parâmetros do janelamento (Etapa 1).

    `interviewer_turn` é o D1: com `close_window`, um turno de entrevistador
    entre dois segmentos do informante FECHA a janela (fala contínua de verdade);
    com `ignore`, os dois segmentos podem entrar na mesma janela. A detecção só é
    possível ANTES do filtro de informante — depois, o turno do entrevistador já
    não está no frame.

    `allow_fewer_windows` é o D5: falante sem candidatas suficientes fica com
    menos janelas e é contado; a alternativa (sobrepor janelas) criaria
    pseudo-replicação dentro do próprio falante.
    """

    base: CorpusAuditConfig
    base_config_path: str
    seed: int
    min_duration_s: float
    max_duration_s: float
    min_segments: int
    max_segments: int
    gap_tolerance_s: float
    interviewer_turn: str
    n_windows_per_speaker: int
    allow_fewer_windows: bool
    reference_field: str
    age_min: int
    age_max: int
    regions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.interviewer_turn not in ("close_window", "ignore"):
            raise ValueError(
                f"windows.interviewer_turn (D1) inválido: '{self.interviewer_turn}'. "
                "Use 'close_window' ou 'ignore'."
            )
        if self.min_duration_s > self.max_duration_s:
            raise ValueError("windows: min_duration_s > max_duration_s.")
        if self.min_segments > self.max_segments:
            raise ValueError("windows: min_segments > max_segments.")
        if self.reference_field not in ("normalized_text", "original_text"):
            raise ValueError(
                f"windows.reference_field (D7) inválido: '{self.reference_field}'. "
                "Use 'normalized_text' ou 'original_text'."
            )
        if self.age_min > self.age_max:
            raise ValueError("windows: age_min > age_max.")


def load_windows_config(path: str | Path) -> WindowsConfig:
    raw = _read_yaml(path)
    base = _resolve_base(raw, path)
    win = _require(raw, "windows", path)
    age = _require(win, "age_range", path)
    return WindowsConfig(
        base=base,
        base_config_path=str(raw["base_config"]),
        seed=int(_require(win, "seed", path)),
        min_duration_s=float(_require(win, "min_duration_s", path)),
        max_duration_s=float(_require(win, "max_duration_s", path)),
        min_segments=int(_require(win, "min_segments", path)),
        max_segments=int(_require(win, "max_segments", path)),
        gap_tolerance_s=float(_require(win, "gap_tolerance_s", path)),
        interviewer_turn=str(_require(win, "interviewer_turn", path)),
        n_windows_per_speaker=int(_require(win, "n_windows_per_speaker", path)),
        allow_fewer_windows=bool(_require(win, "allow_fewer_windows", path)),
        reference_field=str(_require(win, "reference_field", path)),
        age_min=int(age[0]),
        age_max=int(age[1]),
        regions=tuple(str(r) for r in _require(win, "regions", path)),
    )


# --------------------------------------------------------------------------- #
# Modelos (Etapas 4–5)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AsrConfig:
    """ASR único compartilhado por todas as cascatas (§Etapa 5).

    `contamination_checked` é o D8: enquanto for `false`, a execução completa
    deve se recusar a rodar. O Whisper ajustado para PT-BR pode ter sido treinado
    sobre o próprio CORAA/MuPe, e um ASR contaminado inflaria a condição `asr`
    justamente no lado do corpus que queremos medir.
    """

    spec: ModelSpec
    contamination_checked: bool
    contamination_note: str
    split_restriction: str | None


@dataclass(frozen=True)
class PairedFamily:
    """Par nativo × cascata da mesma família de LLM (critério de pareamento)."""

    family: str
    native: str
    cascade: str


@dataclass(frozen=True)
class ModelsConfig:
    """Modelos avaliados, com o pareamento nativo × cascata explícito."""

    asr: AsrConfig
    specs: dict[str, ModelSpec]
    paired: tuple[PairedFamily, ...]
    unpaired_extra: tuple[str, ...]

    def __post_init__(self) -> None:
        for pair in self.paired:
            for role, key in (("native", pair.native), ("cascade", pair.cascade)):
                if key not in self.specs:
                    raise ValueError(
                        f"models.paired[{pair.family}].{role} aponta para '{key}', "
                        "que não está definido em `specs`."
                    )
        for key in self.unpaired_extra:
            if key not in self.specs:
                raise ValueError(f"models.unpaired_extra contém '{key}', ausente de `specs`.")

    @property
    def evaluated_model_ids(self) -> frozenset[str]:
        """`model_id` de todo modelo avaliado — base da checagem do D11."""
        return frozenset(spec.model_id for spec in self.specs.values())


def _parse_spec(key: str, raw: dict[str, Any], path: str | Path) -> ModelSpec:
    return ModelSpec(
        key=key,
        provider=str(_require(raw, "provider", path)),
        model_id=str(_require(raw, "model_id", path)),
        version=str(_require(raw, "version", path)),
        role=str(_require(raw, "role", path)),
        family=str(raw.get("family", "")),
        params=dict(raw.get("params", {})),
    )


def load_models_config(path: str | Path) -> ModelsConfig:
    raw = _read_yaml(path)
    asr_raw = _require(raw, "asr", path)
    specs_raw = _require(raw, "specs", path)
    specs = {key: _parse_spec(key, value, path) for key, value in sorted(specs_raw.items())}
    paired = tuple(
        PairedFamily(
            family=str(_require(p, "family", path)),
            native=str(_require(p, "native", path)),
            cascade=str(_require(p, "cascade", path)),
        )
        for p in raw.get("paired", [])
    )
    return ModelsConfig(
        asr=AsrConfig(
            spec=_parse_spec("asr", _require(asr_raw, "spec", path), path),
            contamination_checked=bool(_require(asr_raw, "contamination_checked", path)),
            contamination_note=str(asr_raw.get("contamination_note", "")),
            split_restriction=(
                None
                if asr_raw.get("split_restriction") is None
                else str(asr_raw["split_restriction"])
            ),
        ),
        specs=specs,
        paired=paired,
        unpaired_extra=tuple(str(k) for k in raw.get("unpaired_extra", [])),
    )


# --------------------------------------------------------------------------- #
# Etapa 3 — itens
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DistractorsConfig:
    """Mecanismo dos distratores (D2).

    `require_provenance` exige que cada distrator declare a janela de origem.
    Sem isso, "extraídos de outros trechos da mesma entrevista" vira uma
    afirmação não verificável no texto do TCC.
    """

    source: str
    require_provenance: bool
    n_context_windows: int


@dataclass(frozen=True)
class TextFilterConfig:
    """Controle textual sem áudio (D3).

    Temperatura > 0 e alternativas embaralhadas são o que dão sentido à "maioria
    de três execuções": a temperatura 0 com ordem fixa devolveria três respostas
    idênticas e o critério de maioria seria vazio.
    """

    models: tuple[str, ...]
    temperature: float
    n_runs: int
    discard_if_correct_at_least: int
    shuffle_alternatives: bool
    any_model_discards: bool
    seed: int

    def __post_init__(self) -> None:
        if self.discard_if_correct_at_least > self.n_runs:
            raise ValueError(
                "items.text_filter: discard_if_correct_at_least "
                f"({self.discard_if_correct_at_least}) > n_runs ({self.n_runs})."
            )
        if self.shuffle_alternatives and self.temperature == 0 and self.n_runs > 1:
            # Não é erro fatal, mas é a armadilha exata que o D3 aponta.
            pass


@dataclass(frozen=True)
class PilotConfig:
    """Piloto da Etapa 3: falantes por região, faixa etária e semente."""

    n_speakers_per_region: int
    regions: tuple[str, ...]
    age_min: int
    age_max: int
    seed: int


@dataclass(frozen=True)
class ItemsConfig:
    """Geração, filtro e revisão de itens (Etapa 3)."""

    seed: int
    n_items_per_window: int
    n_alternatives: int
    generator: ModelSpec
    distractors: DistractorsConfig
    text_filter: TextFilterConfig
    pilot: PilotConfig
    review_decisions: tuple[str, ...]


def load_items_config(path: str | Path) -> ItemsConfig:
    raw = _read_yaml(path)
    gen = _require(raw, "generator", path)
    dis = _require(raw, "distractors", path)
    flt = _require(raw, "text_filter", path)
    pil = _require(raw, "pilot", path)
    pil_age = _require(pil, "age_range", path)
    return ItemsConfig(
        seed=int(_require(raw, "seed", path)),
        n_items_per_window=int(_require(raw, "n_items_per_window", path)),
        n_alternatives=int(_require(raw, "n_alternatives", path)),
        generator=_parse_spec("generator", gen, path),
        distractors=DistractorsConfig(
            source=str(_require(dis, "source", path)),
            require_provenance=bool(_require(dis, "require_provenance", path)),
            n_context_windows=int(_require(dis, "n_context_windows", path)),
        ),
        text_filter=TextFilterConfig(
            models=tuple(str(m) for m in _require(flt, "models", path)),
            temperature=float(_require(flt, "temperature", path)),
            n_runs=int(_require(flt, "n_runs", path)),
            discard_if_correct_at_least=int(_require(flt, "discard_if_correct_at_least", path)),
            shuffle_alternatives=bool(_require(flt, "shuffle_alternatives", path)),
            any_model_discards=bool(_require(flt, "any_model_discards", path)),
            seed=int(_require(flt, "seed", path)),
        ),
        pilot=PilotConfig(
            n_speakers_per_region=int(_require(pil, "n_speakers_per_region", path)),
            regions=tuple(str(r) for r in _require(pil, "regions", path)),
            age_min=int(pil_age[0]),
            age_max=int(pil_age[1]),
            seed=int(_require(pil, "seed", path)),
        ),
        review_decisions=tuple(str(d) for d in _require(raw, "review_decisions", path)),
    )


# --------------------------------------------------------------------------- #
# Etapas 4–5 — execução
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvalConfig:
    """Execução item × configuração × condição (Etapas 4 e 5)."""

    seed: int
    conditions: tuple[str, ...]
    cache_path: str
    max_concurrency: int
    max_retries: int
    shuffle_alternatives: bool
    require_contamination_check: bool

    def __post_init__(self) -> None:
        unknown = [c for c in self.conditions if c not in CONDITIONS]
        if unknown:
            raise ValueError(f"eval.conditions desconhecida(s): {unknown}; esperado {CONDITIONS}.")


def load_eval_config(path: str | Path) -> EvalConfig:
    raw = _read_yaml(path)
    return EvalConfig(
        seed=int(_require(raw, "seed", path)),
        conditions=tuple(str(c) for c in _require(raw, "conditions", path)),
        cache_path=str(_require(raw, "cache_path", path)),
        max_concurrency=int(_require(raw, "max_concurrency", path)),
        max_retries=int(_require(raw, "max_retries", path)),
        shuffle_alternatives=bool(_require(raw, "shuffle_alternatives", path)),
        require_contamination_check=bool(_require(raw, "require_contamination_check", path)),
    )


# --------------------------------------------------------------------------- #
# Etapa 6 — análise
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WerConfig:
    """Referência e normalização do WER (D7).

    A MESMA normalização é aplicada à referência e à hipótese; aplicá-la só a um
    dos lados inventaria erros que não existem.
    """

    reference_field: str
    normalizer_version: str


@dataclass(frozen=True)
class BootstrapConfig:
    """Bootstrap no nível do falante, estratificado por região.

    Estratificar importa porque o corpus é fortemente desbalanceado (39 falantes
    NE contra 193 SE): um bootstrap não estratificado produziria reamostras com
    pouquíssimos falantes do NE e um IC largo por artefato de reamostragem.
    """

    n_resamples: int
    seed: int
    stratify_by: str
    ci_level: float


@dataclass(frozen=True)
class TaskCeilingConfig:
    """Teto da tarefa para Δ_raciocínio (D4).

    `constructed` assume 1.0 por construção (itens revisados sobre a referência);
    `human_sample` exige uma acurácia humana medida. O modo usado é ROTULADO na
    saída — a decomposição não é comparável entre modos.
    """

    mode: str
    value: float

    def __post_init__(self) -> None:
        if self.mode not in ("constructed", "human_sample"):
            raise ValueError(
                f"analysis.task_ceiling.mode (D4) inválido: '{self.mode}'. "
                "Use 'constructed' ou 'human_sample'."
            )


@dataclass(frozen=True)
class SensitivityConfig:
    """Análises de sensibilidade (D10).

    `n_alternatives.enabled` fica desligado por padrão: regerar itens com 3 ou 5
    alternativas duplica a revisão humana, que já é o caminho crítico.
    """

    n_items_enabled: bool
    n_items_grid: tuple[int, ...]
    window_duration_enabled: bool
    window_duration_bins: tuple[float, ...]
    n_alternatives_enabled: bool
    n_alternatives_scope: str


@dataclass(frozen=True)
class AnalysisConfig:
    """Regressão, decomposição, sensibilidade e export (Etapa 6)."""

    age_min: int
    age_max: int
    covariates: tuple[str, ...]
    wer: WerConfig
    bootstrap: BootstrapConfig
    task_ceiling: TaskCeilingConfig
    sensitivity: SensitivityConfig
    latex_source: str


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    raw = _read_yaml(path)
    age = _require(raw, "age_range", path)
    wer = _require(raw, "wer", path)
    boot = _require(raw, "bootstrap", path)
    ceil = _require(raw, "task_ceiling", path)
    sens = _require(raw, "sensitivity", path)
    sens_items = _require(sens, "n_items", path)
    sens_dur = _require(sens, "window_duration", path)
    sens_alts = _require(sens, "n_alternatives", path)
    return AnalysisConfig(
        age_min=int(age[0]),
        age_max=int(age[1]),
        covariates=tuple(str(c) for c in _require(raw, "covariates", path)),
        wer=WerConfig(
            reference_field=str(_require(wer, "reference_field", path)),
            normalizer_version=str(_require(wer, "normalizer_version", path)),
        ),
        bootstrap=BootstrapConfig(
            n_resamples=int(_require(boot, "n_resamples", path)),
            seed=int(_require(boot, "seed", path)),
            stratify_by=str(_require(boot, "stratify_by", path)),
            ci_level=float(_require(boot, "ci_level", path)),
        ),
        task_ceiling=TaskCeilingConfig(
            mode=str(_require(ceil, "mode", path)),
            value=float(_require(ceil, "value", path)),
        ),
        sensitivity=SensitivityConfig(
            n_items_enabled=bool(_require(sens_items, "enabled", path)),
            n_items_grid=tuple(int(n) for n in _require(sens_items, "grid", path)),
            window_duration_enabled=bool(_require(sens_dur, "enabled", path)),
            window_duration_bins=tuple(float(b) for b in _require(sens_dur, "bins", path)),
            n_alternatives_enabled=bool(_require(sens_alts, "enabled", path)),
            n_alternatives_scope=str(_require(sens_alts, "scope", path)),
        ),
        latex_source=str(_require(raw, "latex_source", path)),
    )


# --------------------------------------------------------------------------- #
# Validação cruzada entre arquivos de config
# --------------------------------------------------------------------------- #
def validate_cross_config(items: ItemsConfig, models: ModelsConfig) -> None:
    """Checagens que dependem de mais de um YAML — em especial o D11.

    D11: o LLM gerador dos itens NÃO pode ser um dos modelos avaliados. Se fosse,
    o experimento mediria em parte a afinidade de cada modelo com o estilo de
    quem escreveu as perguntas, e o gerador partiria na frente. A checagem é por
    `model_id` (não pela chave curta), porque duas chaves diferentes podem apontar
    para o mesmo modelo.
    """
    if items.generator.model_id in models.evaluated_model_ids:
        raise ValueError(
            f"D11 violado: o modelo gerador '{items.generator.model_id}' também está "
            "entre os modelos avaliados em models.yaml. Use um gerador fora do "
            "conjunto avaliado, ou remova o modelo da avaliação."
        )
    unknown = [m for m in items.text_filter.models if m not in models.specs]
    if unknown:
        raise ValueError(
            f"items.text_filter.models referencia modelo(s) ausente(s) de models.yaml: {unknown}."
        )
    if models.asr.spec.role != "asr":
        raise ValueError(f"models.asr.spec deve ter role 'asr', não '{models.asr.spec.role}'.")
