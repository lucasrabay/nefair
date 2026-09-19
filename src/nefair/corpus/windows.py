"""Janelamento de fala contínua do informante, SÓ sobre metadados (Etapa 1).

Nenhum byte de áudio é lido aqui: a janela é uma decisão inteiramente tomada
sobre `start_time`, `end_time`, `duration` e `speaker_type`. A Etapa 2 é que
materializa o áudio das janelas já escolhidas.

Três pontos de projeto valem mais que o resto do módulo:

1. **A contiguidade é avaliada no frame COMPLETO, antes do filtro de informante.**
   Um turno `P/*` entre dois segmentos `R` desaparece quando filtramos, e a fala
   pareceria contínua exatamente onde ela foi interrompida. Por isso
   `mark_interviewer_turns` recebe o frame cru, anota quanto tempo de
   entrevistador existe entre cada par de segmentos do informante, e só então
   devolve as linhas do informante. É o mecanismo do D1.

2. **As candidatas são empacotadas pelo menor tamanho válido.** O plano conta com
   isso: "com >= 8 segmentos por janela, 64 segmentos dão no máximo 8 janelas".
   Empacotar pelo maior tamanho válido reduziria o número de candidatas e, com
   ele, tanto a cobertura da entrevista quanto o número de falantes que atingem
   a meta de janelas.

3. **A seleção é estratificada por posição na entrevista.** "As 10 primeiras"
   amostraria só o começo da entrevista — justamente a parte mais protocolar
   (apresentação, dados pessoais) e menos representativa da fala espontânea.
   A semente de cada falante deriva de `(seed, speaker_code)` por SHA-256; o
   `hash()` embutido é randomizado por processo e destruiria o determinismo.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nefair.config import WindowsConfig, config_sha
from nefair.corpus.audit import (
    INFORMANT_SPEAKER_TYPE,
    SPEAKER_KEY,
    add_region_column,
    filter_informants,
)
from nefair.corpus.load import load_metadata
from nefair.schema import ExclusionLedger, Provenance, Window

# Ordenação canônica dos segmentos do informante. `audio_id` entra como último
# critério de desempate para que dois segmentos com o mesmo `start_time` (borda
# de arredondamento em float32) nunca troquem de lugar entre execuções.
SEGMENT_ORDER: tuple[str, ...] = ("speaker_code", "audio_name", "start_time", "audio_id")

# Ordenação dentro da GRAVAÇÃO, usada para detectar o turno de entrevistador.
# Aqui NÃO se ordena por `speaker_code`: informante e entrevistador têm códigos
# diferentes, e ordenar por falante desfaria justamente a intercalação que
# queremos enxergar.
RECORDING_ORDER: tuple[str, ...] = ("audio_name", "start_time", "audio_id")

# Motivos de quebra de janela, contabilizados um a um no relatório.
BREAK_AUDIO = "troca_de_gravacao"
BREAK_INTERVIEWER = "turno_de_entrevistador"
BREAK_GAP = "lacuna_temporal"
BREAK_REASONS: tuple[str, ...] = (BREAK_AUDIO, BREAK_INTERVIEWER, BREAK_GAP)

# Motivos de exclusão (vocabulário fechado; aparecem literalmente no relatório).
DROP_NON_INFORMANT = "nao_informante"
DROP_REGION = "regiao_fora_do_escopo"
DROP_AGE = "idade_fora_da_faixa"
DROP_NO_CANDIDATE = "sem_janela_candidata"
DROP_NOT_SAMPLED = "candidata_nao_sorteada"
DROP_BELOW_TARGET = "abaixo_da_meta_com_allow_fewer_windows_false"

# Versão do código gravada na proveniência dos artefatos desta etapa.
CODE_VERSION = "0.1.0"


# --------------------------------------------------------------------------- #
# D1 — detecção do turno de entrevistador (antes do filtro de informante)
# --------------------------------------------------------------------------- #
def mark_interviewer_turns(frame: pd.DataFrame) -> pd.DataFrame:
    """Devolve as linhas do informante anotadas com o que houve ANTES de cada uma.

    Acrescenta duas colunas ao frame do informante:

    - `interviewer_turns_before` — quantos turnos não-informante (`P/*`) ocorrem
      entre este segmento e o segmento anterior do informante, na MESMA gravação;
    - `interviewer_seconds_before` — quantos segundos esses turnos ocupam.

    Os segundos importam porque o modo `ignore` do D1 precisa descontá-los da
    lacuna temporal: se um turno de entrevistador de 4 s separa dois segmentos do
    informante, a lacuna bruta é 4 s e a regra de `gap_tolerance_s` quebraria a
    janela de qualquer jeito — o `ignore` viraria letra morta. O que o `ignore`
    diz é "o tempo do entrevistador não conta como silêncio do informante", e é
    exatamente isso que o desconto implementa.

    O agrupamento é por `audio_name` (a gravação), não por `speaker_code`: o
    entrevistador tem código próprio e só se liga ao informante pela gravação.
    """
    ordered = frame.sort_values(list(RECORDING_ORDER), kind="stable").reset_index(drop=True)
    is_informant = (ordered["speaker_type"] == INFORMANT_SPEAKER_TYPE).to_numpy()

    # Rank do informante dentro da gravação: um turno não-informante com rank r
    # está DEPOIS do r-ésimo informante e ANTES do (r+1)-ésimo.
    inf_rank = (
        pd.Series(is_informant.astype("int64"), index=ordered.index)
        .groupby(ordered["audio_name"], sort=False)
        .cumsum()
    )

    non_informant = ~is_informant
    accumulated = pd.DataFrame(
        {
            "audio_name": ordered["audio_name"],
            "inf_rank": inf_rank,
            "seconds": np.where(non_informant, ordered["duration"].to_numpy("float64"), 0.0),
            "turns": non_informant.astype("int64"),
        }
    )
    per_slot = accumulated.groupby(["audio_name", "inf_rank"], sort=False)[
        ["seconds", "turns"]
    ].sum()

    informants = ordered.loc[is_informant].copy()
    slot = pd.MultiIndex.from_arrays(
        [informants["audio_name"], inf_rank.loc[is_informant] - 1],
        names=["audio_name", "inf_rank"],
    )
    informants["interviewer_seconds_before"] = (
        per_slot["seconds"].reindex(slot).fillna(0.0).to_numpy("float64")
    )
    informants["interviewer_turns_before"] = (
        per_slot["turns"].reindex(slot).fillna(0).to_numpy("int64")
    )
    return informants.sort_values(list(SEGMENT_ORDER), kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Corridas contíguas e motivos de quebra
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BreakTally:
    """Contagem de quebras por motivo, mais o que o D1 deixou passar.

    `interviewer_turns_observed` é contado nos dois modos do D1: com `ignore`,
    ele registra quantas interrupções reais foram deliberadamente atravessadas —
    número que precisa aparecer no relatório, e não sumir com a decisão.
    """

    by_reason: dict[str, int]
    n_runs: int
    n_speakers: int
    interviewer_turns_observed: int

    @property
    def total(self) -> int:
        return sum(self.by_reason.values())


def assign_runs(informants: pd.DataFrame, config: WindowsConfig) -> tuple[pd.DataFrame, BreakTally]:
    """Marca cada segmento com o id da corrida contígua a que pertence.

    Uma "corrida" é a maior sequência de segmentos do informante que ainda conta
    como fala contínua. A precedência dos motivos de quebra é fixa — gravação,
    entrevistador, lacuna — para que a contagem por motivo some exatamente o
    número de quebras, sem duplo cômputo quando dois motivos coincidem.
    """
    n = len(informants)
    if n == 0:
        empty = informants.assign(run_id=pd.Series(dtype="int64"))
        return empty, BreakTally(dict.fromkeys(BREAK_REASONS, 0), 0, 0, 0)

    speaker = informants["speaker_code"].to_numpy()
    audio = informants["audio_name"].to_numpy()
    start = informants["start_time"].to_numpy("float64")
    end = informants["end_time"].to_numpy("float64")
    turns_before = informants["interviewer_turns_before"].to_numpy("int64")
    seconds_before = informants["interviewer_seconds_before"].to_numpy("float64")

    first_of_speaker = np.r_[True, speaker[1:] != speaker[:-1]]
    new_audio = np.r_[False, audio[1:] != audio[:-1]] & ~first_of_speaker

    # Lacuna LÍQUIDA: desconta o tempo de fala do entrevistador (ver docstring de
    # `mark_interviewer_turns`). Sob `close_window` o desconto é inócuo, porque a
    # quebra por entrevistador tem precedência.
    raw_gap = np.r_[0.0, start[1:] - end[:-1]]
    net_gap = raw_gap - seconds_before

    interviewer_between = (turns_before > 0) & ~first_of_speaker & ~new_audio
    interrupted = interviewer_between & (config.interviewer_turn == "close_window")
    gapped = (net_gap > config.gap_tolerance_s) & ~first_of_speaker & ~new_audio & ~interrupted

    break_at = first_of_speaker | new_audio | interrupted | gapped
    run_id = np.cumsum(break_at) - 1

    tally = BreakTally(
        by_reason={
            BREAK_AUDIO: int(new_audio.sum()),
            BREAK_INTERVIEWER: int(interrupted.sum()),
            BREAK_GAP: int(gapped.sum()),
        },
        n_runs=int(break_at.sum()),
        n_speakers=int(first_of_speaker.sum()),
        interviewer_turns_observed=int(interviewer_between.sum()),
    )
    return informants.assign(run_id=run_id), tally


# --------------------------------------------------------------------------- #
# Empacotamento das candidatas
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    """Janela candidata: um intervalo posicional fechado à esquerda, aberto à direita."""

    speaker_code: str
    audio_name: str
    run_id: int
    row_start: int
    row_end: int
    duration_s: float


def pack_run(durations: np.ndarray, config: WindowsConfig) -> list[tuple[int, int]]:
    """Empacota uma corrida em janelas DISJUNTAS pelo menor tamanho válido.

    Válido = duração somada em `[min_duration_s, max_duration_s]` E contagem de
    segmentos em `[min_segments, max_segments]` — as duas condições ao mesmo
    tempo, nunca uma só.

    O menor tamanho válido maximiza o número de candidatas disjuntas, que é o que
    dá espaço para a amostragem estratificada cobrir a entrevista inteira. Quando
    nenhum tamanho serve a partir de uma posição (segmentos longos demais, ou
    curtos demais para somar o mínimo), a posição avança de um: o segmento fica
    sem janela e é contabilizado como tal, nunca forçado para dentro.
    """
    n = len(durations)
    cumulative = np.concatenate(([0.0], np.cumsum(durations)))
    packed: list[tuple[int, int]] = []
    start = 0
    while start + config.min_segments <= n:
        chosen: int | None = None
        longest = min(config.max_segments, n - start)
        for length in range(config.min_segments, longest + 1):
            total = cumulative[start + length] - cumulative[start]
            if total > config.max_duration_s:
                break  # só cresce daqui para frente
            if total >= config.min_duration_s:
                chosen = length
                break
        if chosen is None:
            start += 1
            continue
        packed.append((start, start + chosen))
        start += chosen
    return packed


def enumerate_candidates(eligible: pd.DataFrame, config: WindowsConfig) -> list[Candidate]:
    """Todas as candidatas disjuntas, em ordem temporal, do frame já elegível.

    `eligible` precisa estar na ordem canônica, com `RangeIndex` e coluna
    `run_id`: as posições devolvidas são índices posicionais nesse frame.
    """
    if len(eligible) == 0:
        return []
    run_ids = eligible["run_id"].to_numpy("int64")
    durations = eligible["duration"].to_numpy("float64")
    speakers = eligible["speaker_code"].to_numpy()
    audios = eligible["audio_name"].to_numpy()

    run_starts = np.flatnonzero(np.r_[True, run_ids[1:] != run_ids[:-1]])
    run_ends = np.r_[run_starts[1:], len(run_ids)]

    candidates: list[Candidate] = []
    for run_start, run_end in zip(run_starts, run_ends, strict=True):
        local = durations[run_start:run_end]
        for lo, hi in pack_run(local, config):
            candidates.append(
                Candidate(
                    speaker_code=str(speakers[run_start]),
                    audio_name=str(audios[run_start]),
                    run_id=int(run_ids[run_start]),
                    row_start=int(run_start + lo),
                    row_end=int(run_start + hi),
                    duration_s=float(local[lo:hi].sum()),
                )
            )
    return candidates


# --------------------------------------------------------------------------- #
# Seleção estratificada por posição na entrevista
# --------------------------------------------------------------------------- #
def speaker_rng(seed: int, speaker_code: str) -> np.random.Generator:
    """Gerador semeado por `(seed, speaker_code)` de forma estável entre processos.

    O `hash()` embutido do Python é randomizado a cada processo (PYTHONHASHSEED):
    usá-lo aqui faria a mesma configuração sortear janelas diferentes a cada
    execução. SHA-256 do par não tem esse problema.
    """
    digest = hashlib.sha256(f"{seed}:{speaker_code}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def stratified_positions(
    n_candidates: int, n_target: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Sorteia uma candidata por bloco de posição; devolve (índice, bloco).

    Divide `range(n_candidates)` em `n_target` blocos de tamanho quase igual
    (`np.array_split`, blocos maiores primeiro — determinístico) e sorteia uma
    candidata dentro de cada bloco. Com menos candidatas do que a meta, todas
    entram e o bloco vira o próprio ordinal da candidata (D5).
    """
    if n_candidates <= n_target:
        return [(i, i) for i in range(n_candidates)]
    picked: list[tuple[int, int]] = []
    for block_index, block in enumerate(np.array_split(np.arange(n_candidates), n_target)):
        offset = int(rng.integers(0, len(block)))
        picked.append((int(block[offset]), block_index))
    return picked


# --------------------------------------------------------------------------- #
# Resultado da etapa
# --------------------------------------------------------------------------- #
@dataclass
class WindowsResult:
    """Tudo o que a Etapa 1 produz — janelas E a trilha de como se chegou nelas."""

    windows: tuple[Window, ...]
    breaks: BreakTally
    ledger_segments: ExclusionLedger
    ledger_candidates: ExclusionLedger
    ledger_speakers: ExclusionLedger
    candidates_per_speaker: dict[str, int]
    selected_per_speaker: dict[str, int]
    speakers_below_target: tuple[str, ...]
    speakers_by_region: dict[str, int]
    windows_by_region: dict[str, int]
    empty_reference_windows: int
    dropped_speakers: dict[str, list[str]] = field(default_factory=dict)

    def assert_balanced(self) -> None:
        """Fecha as três trilhas de exclusão de uma vez (Princípio 2)."""
        self.ledger_segments.assert_balanced()
        self.ledger_candidates.assert_balanced()
        self.ledger_speakers.assert_balanced()


def build_windows(frame: pd.DataFrame, config: WindowsConfig) -> WindowsResult:
    """Constrói as janelas da Etapa 1 a partir do frame de metadados completo.

    Função pura: mesmo frame + mesmo config ⇒ mesmas janelas, na mesma ordem.
    Recebe o frame COMPLETO (com entrevistadores), porque o D1 depende disso.
    """
    ledger_segments = ExclusionLedger(unit="segmentos", total_in=len(frame))
    _, filter_report = filter_informants(frame)
    for speaker_type, count in sorted(filter_report.dropped_by_type.items()):
        ledger_segments.drop(f"{DROP_NON_INFORMANT} ({speaker_type})", count)

    informants = mark_interviewer_turns(frame)
    informants = add_region_column(informants, config.base.region_map)

    ledger_speakers = ExclusionLedger(
        unit="falantes informantes", total_in=int(informants[SPEAKER_KEY].nunique())
    )
    dropped_speakers: dict[str, list[str]] = {}

    # Recorte de escopo. A precedência é fixa (região antes de idade) para que um
    # falante fora dos dois critérios seja contado uma vez só, sempre no mesmo.
    in_region = informants["region"].isin(config.regions)
    out_of_region = informants.loc[~in_region]
    ledger_segments.drop(DROP_REGION, int(len(out_of_region)))
    dropped_speakers[DROP_REGION] = sorted(out_of_region[SPEAKER_KEY].unique().tolist())
    ledger_speakers.drop(DROP_REGION, len(dropped_speakers[DROP_REGION]))
    informants = informants.loc[in_region]

    in_age = informants["age"].between(config.age_min, config.age_max)
    out_of_age = informants.loc[~in_age]
    ledger_segments.drop(DROP_AGE, int(len(out_of_age)))
    dropped_speakers[DROP_AGE] = sorted(out_of_age[SPEAKER_KEY].unique().tolist())
    ledger_speakers.drop(DROP_AGE, len(dropped_speakers[DROP_AGE]))
    eligible = informants.loc[in_age].reset_index(drop=True)

    speakers_by_region = {
        str(region): int(group[SPEAKER_KEY].nunique())
        for region, group in eligible.groupby("region", sort=True)
    }

    eligible, breaks = assign_runs(eligible, config)
    candidates = enumerate_candidates(eligible, config)

    covered = int(sum(c.row_end - c.row_start for c in candidates))
    ledger_segments.drop(DROP_NO_CANDIDATE, len(eligible) - covered)

    windows, selected_rows, below_target, refused = _select_and_build(eligible, candidates, config)

    ledger_candidates = ExclusionLedger(unit="janelas candidatas", total_in=len(candidates))
    ledger_candidates.kept = len(windows)
    ledger_candidates.drop(DROP_NOT_SAMPLED, len(candidates) - len(windows))

    ledger_segments.kept = selected_rows
    ledger_segments.drop(DROP_NOT_SAMPLED, covered - selected_rows)

    all_candidate_speakers = {c.speaker_code for c in candidates}
    eligible_speakers = set(eligible[SPEAKER_KEY].unique().tolist())
    dropped_speakers[DROP_NO_CANDIDATE] = sorted(eligible_speakers - all_candidate_speakers)
    ledger_speakers.drop(DROP_NO_CANDIDATE, len(dropped_speakers[DROP_NO_CANDIDATE]))
    dropped_speakers[DROP_BELOW_TARGET] = sorted(refused)
    ledger_speakers.drop(DROP_BELOW_TARGET, len(refused))
    ledger_speakers.kept = len({w.speaker_code for w in windows})

    candidates_per_speaker: dict[str, int] = {}
    for candidate in candidates:
        candidates_per_speaker[candidate.speaker_code] = (
            candidates_per_speaker.get(candidate.speaker_code, 0) + 1
        )
    selected_per_speaker: dict[str, int] = {}
    windows_by_region: dict[str, int] = {}
    for window in windows:
        selected_per_speaker[window.speaker_code] = (
            selected_per_speaker.get(window.speaker_code, 0) + 1
        )
        windows_by_region[window.region] = windows_by_region.get(window.region, 0) + 1

    return WindowsResult(
        windows=windows,
        breaks=breaks,
        ledger_segments=ledger_segments,
        ledger_candidates=ledger_candidates,
        ledger_speakers=ledger_speakers,
        candidates_per_speaker=dict(sorted(candidates_per_speaker.items())),
        selected_per_speaker=dict(sorted(selected_per_speaker.items())),
        speakers_below_target=tuple(sorted(below_target)),
        speakers_by_region=speakers_by_region,
        windows_by_region=dict(sorted(windows_by_region.items())),
        empty_reference_windows=sum(1 for w in windows if not w.reference_text.strip()),
        dropped_speakers={k: v for k, v in sorted(dropped_speakers.items())},
    )


def _select_and_build(
    eligible: pd.DataFrame, candidates: list[Candidate], config: WindowsConfig
) -> tuple[tuple[Window, ...], int, list[str], list[str]]:
    """Sorteia as candidatas de cada falante e materializa os `Window`.

    Devolve (janelas, nº de segmentos usados, falantes abaixo da meta, falantes
    recusados por `allow_fewer_windows: false`).
    """
    by_speaker: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_speaker.setdefault(candidate.speaker_code, []).append(candidate)

    reference = eligible[config.reference_field].to_numpy()
    audio_ids = eligible["audio_id"].to_numpy("int64")
    start_time = eligible["start_time"].to_numpy("float64")
    end_time = eligible["end_time"].to_numpy("float64")

    windows: list[Window] = []
    selected_rows = 0
    below_target: list[str] = []
    refused: list[str] = []

    for speaker_code in sorted(by_speaker):
        speaker_candidates = by_speaker[speaker_code]
        n_candidates = len(speaker_candidates)
        if n_candidates < config.n_windows_per_speaker:
            below_target.append(speaker_code)
            # D5 com `allow_fewer_windows: false` não tem saída legítima: a única
            # forma de chegar à meta seria sobrepor janelas, o que criaria
            # pseudo-replicação dentro do falante. Então o falante inteiro sai —
            # e sai CONTADO, com motivo próprio.
            if not config.allow_fewer_windows:
                refused.append(speaker_code)
                continue

        rng = speaker_rng(config.seed, speaker_code)
        picks = stratified_positions(n_candidates, config.n_windows_per_speaker, rng)
        # Numeração das janelas na ordem TEMPORAL, não na ordem do sorteio.
        picks.sort(key=lambda pair: pair[0])

        for ordinal, (candidate_index, position_index) in enumerate(picks, start=1):
            candidate = speaker_candidates[candidate_index]
            rows = slice(candidate.row_start, candidate.row_end)
            head = eligible.iloc[candidate.row_start]
            texts = [str(text).strip() for text in reference[rows]]
            windows.append(
                Window(
                    window_id=Window.make_id(speaker_code, ordinal),
                    speaker_code=speaker_code,
                    audio_name=candidate.audio_name,
                    split=str(head["split"]),
                    region=str(head["region"]),
                    age=int(head["age"]),
                    segment_audio_ids=tuple(int(a) for a in audio_ids[rows]),
                    start_time=float(start_time[candidate.row_start]),
                    end_time=float(end_time[candidate.row_end - 1]),
                    duration_s=round(candidate.duration_s, 6),
                    n_segments=candidate.row_end - candidate.row_start,
                    reference_text=" ".join(text for text in texts if text),
                    position_index=position_index,
                    n_candidates_for_speaker=n_candidates,
                )
            )
            selected_rows += candidate.row_end - candidate.row_start

    return tuple(windows), selected_rows, below_target, refused


# --------------------------------------------------------------------------- #
# Artefato em parquet
# --------------------------------------------------------------------------- #
# Ordem de colunas FIXA (Princípio 1): o parquet é comparável byte a byte entre
# execuções, e a Etapa 2 pode depender da ordem.
WINDOW_SCHEMA = pa.schema(
    [
        ("window_id", pa.string()),
        ("speaker_code", pa.string()),
        ("audio_name", pa.string()),
        ("split", pa.string()),
        ("region", pa.string()),
        ("age", pa.int64()),
        ("segment_audio_ids", pa.list_(pa.int64())),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("duration_s", pa.float64()),
        ("n_segments", pa.int64()),
        ("reference_text", pa.string()),
        ("position_index", pa.int64()),
        ("n_candidates_for_speaker", pa.int64()),
        ("content_sha", pa.string()),
    ]
)


def windows_to_table(windows: tuple[Window, ...], provenance: Provenance | None = None) -> pa.Table:
    """Tabela pyarrow com a ordem de colunas fixa e a proveniência nos metadados."""
    columns: dict[str, list] = {name: [] for name in WINDOW_SCHEMA.names}
    for window in windows:
        columns["window_id"].append(window.window_id)
        columns["speaker_code"].append(window.speaker_code)
        columns["audio_name"].append(window.audio_name)
        columns["split"].append(window.split)
        columns["region"].append(window.region)
        columns["age"].append(window.age)
        columns["segment_audio_ids"].append(list(window.segment_audio_ids))
        columns["start_time"].append(window.start_time)
        columns["end_time"].append(window.end_time)
        columns["duration_s"].append(window.duration_s)
        columns["n_segments"].append(window.n_segments)
        columns["reference_text"].append(window.reference_text)
        columns["position_index"].append(window.position_index)
        columns["n_candidates_for_speaker"].append(window.n_candidates_for_speaker)
        columns["content_sha"].append(window.content_sha)

    schema = WINDOW_SCHEMA
    if provenance is not None:
        schema = schema.with_metadata(
            {
                b"nefair_provenance": json.dumps(
                    provenance.to_dict(), sort_keys=True, ensure_ascii=False
                ).encode()
            }
        )
    return pa.table(columns, schema=schema)


def write_windows_parquet(
    windows: tuple[Window, ...], path: str | Path, provenance: Provenance | None = None
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(windows_to_table(windows, provenance), out)
    return out


def read_windows_parquet(path: str | Path) -> tuple[Window, ...]:
    """Lê o parquet de volta para `Window` — a entrada da Etapa 2."""
    table = pq.read_table(path)
    rows = table.to_pylist()
    return tuple(
        Window(
            window_id=row["window_id"],
            speaker_code=row["speaker_code"],
            audio_name=row["audio_name"],
            split=row["split"],
            region=row["region"],
            age=int(row["age"]),
            segment_audio_ids=tuple(int(a) for a in row["segment_audio_ids"]),
            start_time=float(row["start_time"]),
            end_time=float(row["end_time"]),
            duration_s=float(row["duration_s"]),
            n_segments=int(row["n_segments"]),
            reference_text=row["reference_text"],
            position_index=int(row["position_index"]),
            n_candidates_for_speaker=int(row["n_candidates_for_speaker"]),
        )
        for row in rows
    )


# --------------------------------------------------------------------------- #
# Relatório em Markdown
# --------------------------------------------------------------------------- #
# Bibliotecas cujas versões vão ao relatório (reprodutibilidade da fonte).
_REPORTED_PACKAGES: tuple[str, ...] = ("numpy", "pandas", "pyarrow", "PyYAML")


def _lib_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _REPORTED_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "desconhecida"
    versions["python"] = platform.python_version()
    return versions


def _fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def effective_config_lines(config: WindowsConfig) -> list[str]:
    """Valor EFETIVO de cada chave, com as decisões abertas marcadas.

    Princípio 4 da Etapa 0: decisão pendente nunca vira default escondido. Quem
    lê o relatório precisa saber, sem abrir o YAML, sob qual D1/D5/D7 os números
    da tabela foram produzidos.
    """
    return [
        "| chave | valor efetivo | decisão |",
        "|---|---|---|",
        f"| `base_config` | `{config.base_config_path}` | — |",
        f"| `seed` | {config.seed} | — |",
        f"| `min_duration_s` | {config.min_duration_s:g} | — |",
        f"| `max_duration_s` | {config.max_duration_s:g} | — |",
        f"| `min_segments` | {config.min_segments} | — |",
        f"| `max_segments` | {config.max_segments} | — |",
        f"| `gap_tolerance_s` | {config.gap_tolerance_s:g} | — |",
        f"| `interviewer_turn` | `{config.interviewer_turn}` | **D1** |",
        f"| `n_windows_per_speaker` | {config.n_windows_per_speaker} | — |",
        f"| `allow_fewer_windows` | {str(config.allow_fewer_windows).lower()} | **D5** |",
        f"| `reference_field` | `{config.reference_field}` | **D7** |",
        f"| `age_range` | [{config.age_min}, {config.age_max}] | — |",
        f"| `regions` | {list(config.regions)} | — |",
    ]


def _histogram_lines(selected_per_speaker: dict[str, int], target: int) -> list[str]:
    """Histograma de janelas por falante, em ordem crescente de nº de janelas."""
    counts: dict[int, int] = {}
    for n_windows in selected_per_speaker.values():
        counts[n_windows] = counts.get(n_windows, 0) + 1
    lines = ["| janelas por falante | falantes | |", "|--:|--:|---|"]
    for n_windows in sorted(counts):
        mark = "" if n_windows >= target else " (abaixo da meta)"
        bar = "█" * min(counts[n_windows], 60)
        lines.append(f"| {n_windows}{mark} | {_fmt_int(counts[n_windows])} | {bar} |")
    return lines


def build_windows_report(
    result: WindowsResult,
    config: WindowsConfig,
    provenance: Provenance,
    generated_at: str,
    *,
    load_note: str = "",
) -> str:
    """Relatório da Etapa 1 com a trilha completa, do segmento à janela."""
    windows = result.windows
    lines: list[str] = []

    lines.append("# Janelamento de fala contínua (Etapa 1)")
    lines.append("")
    lines.append(
        "Artefato reprodutível e determinístico. Nenhum byte de áudio foi lido: "
        "as janelas são decididas inteiramente sobre metadados."
    )
    lines.append("")
    lines.append(f"- **Gerado em (UTC):** {generated_at}")
    lines.append("")

    lines.append("## Proveniência e reprodutibilidade")
    lines.append(f"- Dataset: `{provenance.dataset_id}`")
    lines.append(f"- Revisão resolvida (commit SHA): `{provenance.dataset_revision}`")
    lines.append(f"- Splits: {', '.join(provenance.splits)}")
    lines.append(f"- SHA do config: `{provenance.config_sha}`")
    lines.append(f"- Versão do código: `{provenance.code_version}`")
    if load_note:
        lines.append(f"- Origem do frame: {load_note}")
    lines.append("- Versões das bibliotecas:")
    for package, version in _lib_versions().items():
        lines.append(f"  - {package}: {version}")
    lines.append("")

    lines.append("## Configuração efetiva")
    lines.append(
        "Nenhum valor abaixo é default escondido no código: todos vêm do YAML e "
        "são ecoados aqui para que a tabela de janelas seja interpretável sozinha."
    )
    lines.append("")
    lines.extend(effective_config_lines(config))
    lines.append("")

    lines.append("## Trilha de exclusão — segmentos")
    lines.append(
        "Todo segmento do frame de entrada termina em exatamente uma linha abaixo. "
        "`sem_janela_candidata` são segmentos elegíveis que sobraram das bordas de "
        "uma corrida contígua (curta demais para fechar uma janela); "
        f"`{DROP_NOT_SAMPLED}` são segmentos de candidatas que a amostragem "
        "estratificada não sorteou."
    )
    lines.append("")
    lines.extend(result.ledger_segments.to_markdown())
    lines.append("")

    lines.append("## Trilha de exclusão — falantes informantes")
    lines.append(
        "Precedência fixa dos motivos (região antes de idade), para que um falante "
        "fora dos dois critérios seja contado uma única vez e sempre no mesmo."
    )
    lines.append("")
    lines.extend(result.ledger_speakers.to_markdown())
    lines.append("")
    lines.append("Falantes elegíveis (dentro de região e faixa etária), por região:")
    for region in sorted(result.speakers_by_region):
        lines.append(f"- **{region}**: {_fmt_int(result.speakers_by_region[region])} falantes")
    lines.append("")

    lines.append("## Contiguidade: corridas e motivos de quebra")
    lines.append(
        "Uma *corrida* é a maior sequência de segmentos do informante que ainda "
        "conta como fala contínua. A detecção de turno de entrevistador acontece "
        "no frame COMPLETO, antes do filtro de informante — depois de filtrar, o "
        "turno `P/*` já não está no frame e a fala pareceria contínua."
    )
    lines.append("")
    lines.append(f"- Corridas contíguas: {_fmt_int(result.breaks.n_runs)}")
    lines.append(f"- Falantes elegíveis com ao menos um segmento: {result.breaks.n_speakers}")
    lines.append("- Quebras por motivo:")
    for reason in BREAK_REASONS:
        lines.append(f"  - `{reason}`: {_fmt_int(result.breaks.by_reason[reason])}")
    lines.append(f"  - total de quebras: {_fmt_int(result.breaks.total)}")
    lines.append(
        f"- Conferência: {result.breaks.n_speakers} falantes + "
        f"{_fmt_int(result.breaks.total)} quebras = "
        f"{_fmt_int(result.breaks.n_speakers + result.breaks.total)} corridas"
    )
    lines.append(
        "- Turnos de entrevistador observados entre segmentos consecutivos do "
        f"informante: {_fmt_int(result.breaks.interviewer_turns_observed)} "
        f"(D1 = `{config.interviewer_turn}`)"
    )
    if config.interviewer_turn == "ignore":
        lines.append(
            "  - Com `ignore`, esses turnos NÃO quebram a janela e o tempo que "
            "ocupam é descontado da lacuna temporal — sem o desconto, a regra de "
            "`gap_tolerance_s` quebraria a janela mesmo assim e o `ignore` seria "
            "letra morta. Consequência a declarar: o áudio da janela terá um "
            "corte audível onde o entrevistador foi removido."
        )
    lines.append("")

    lines.append("## Candidatas e seleção estratificada")
    lines.append(
        "As candidatas de um falante são DISJUNTAS e empacotadas pelo menor "
        "tamanho válido (maximiza o nº de candidatas, e com ele a cobertura da "
        "entrevista). A seleção divide as candidatas em "
        f"{config.n_windows_per_speaker} blocos de posição e sorteia uma por "
        "bloco, com semente derivada de `(seed, speaker_code)` por SHA-256."
    )
    lines.append("")
    lines.append(
        "Consequência a declarar: empacotar pelo menor tamanho válido concentra "
        f"as janelas perto do piso de {config.min_duration_s:g} s, não no meio da "
        f"faixa [{config.min_duration_s:g}, {config.max_duration_s:g}] s. É o "
        "preço de ter mais candidatas — e é o que a aritmética do D5 pressupõe "
        f'("com >= {config.min_segments} segmentos por janela, N segmentos dão '
        f'no máximo N/{config.min_segments} janelas"). A duração efetivamente '
        "obtida está tabulada em *Janelas selecionadas*, abaixo."
    )
    lines.append("")
    lines.extend(result.ledger_candidates.to_markdown())
    lines.append("")
    if result.candidates_per_speaker:
        per_speaker = np.array(sorted(result.candidates_per_speaker.values()), dtype="float64")
        lines.append(
            "Candidatas por falante — mín "
            f"{int(per_speaker.min())}, mediana {np.median(per_speaker):.1f}, "
            f"média {per_speaker.mean():.1f}, máx {int(per_speaker.max())}."
        )
        lines.append("")

    lines.append("## Janelas por falante (histograma)")
    lines.extend(_histogram_lines(result.selected_per_speaker, config.n_windows_per_speaker))
    lines.append("")
    below = len(result.speakers_below_target)
    lines.append(
        f"**D5 — falantes com menos de {config.n_windows_per_speaker} candidatas: "
        f"{_fmt_int(below)}.** "
        + (
            "Com `allow_fewer_windows: true`, ficam com menos janelas; nunca se "
            "sobrepõem janelas para completar a meta (isso criaria "
            "pseudo-replicação dentro do próprio falante)."
            if config.allow_fewer_windows
            else "Com `allow_fewer_windows: false`, esses falantes são EXCLUÍDOS "
            "inteiros: chegar à meta exigiria sobrepor janelas."
        )
    )
    if below:
        shown = ", ".join(f"`{s}`" for s in result.speakers_below_target[:20])
        suffix = ", …" if below > 20 else ""
        lines.append(f"Falantes afetados (até 20): {shown}{suffix}")
    lines.append("")

    lines.append("## Janelas selecionadas")
    lines.append(f"- Total de janelas: {_fmt_int(len(windows))}")
    for region in sorted(result.windows_by_region):
        lines.append(f"  - **{region}**: {_fmt_int(result.windows_by_region[region])}")
    if windows:
        durations = np.array([w.duration_s for w in windows], dtype="float64")
        segments = np.array([w.n_segments for w in windows], dtype="float64")
        total_hours = durations.sum() / 3600.0
        lines.append(
            f"- Duração somada por janela — mín {durations.min():.2f} s, "
            f"mediana {np.median(durations):.2f} s, média {durations.mean():.2f} s, "
            f"máx {durations.max():.2f} s"
        )
        lines.append(
            f"- Segmentos por janela — mín {int(segments.min())}, "
            f"mediana {np.median(segments):.1f}, máx {int(segments.max())}"
        )
        lines.append(f"- Áudio a baixar na Etapa 2: {total_hours:.2f} h")
    lines.append(f"- Janelas com transcrição de referência vazia: {result.empty_reference_windows}")
    lines.append("")

    lines.append("## Artefatos gerados")
    lines.append(
        "- `windows.parquet` — uma linha por janela (gitignorado: é derivado e "
        "regenerável a partir do config + revisão do dataset)."
    )
    lines.append("- `windows_report.md` — este relatório.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #
def run_windows(
    config: WindowsConfig,
    config_path: str | Path,
    outputs_dir: str | Path,
    *,
    token: str | None = None,
    frame: pd.DataFrame | None = None,
    dataset_revision: str | None = None,
) -> WindowsResult:
    """Executa a Etapa 1 e escreve `windows.parquet` + `windows_report.md`.

    `frame` permite rodar a etapa ponta a ponta sobre o corpus sintético de
    fixture, sem rede (critério de aceite 3 do plano de código). Quando é `None`,
    os metadados vêm da HF pelo mesmo caminho da Etapa 0.
    """
    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)

    if frame is None:
        load_result = load_metadata(config.base, token=token)
        frame = load_result.frame
        resolved_revision = load_result.resolved_revision
        load_note = load_result.load_method
    else:
        resolved_revision = dataset_revision or "frame-fornecido-sem-revisao"
        load_note = "frame fornecido pelo chamador (fixture sintética; sem rede)"

    provenance = Provenance(
        dataset_id=config.base.dataset_id,
        dataset_revision=resolved_revision,
        splits=config.base.dataset.splits,
        config_sha=config_sha(config_path),
        code_version=CODE_VERSION,
    )

    result = build_windows(frame, config)
    result.assert_balanced()

    write_windows_parquet(result.windows, out / "windows.parquet", provenance)
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    report = build_windows_report(result, config, provenance, generated_at, load_note=load_note)
    (out / "windows_report.md").write_text(report, encoding="utf-8")

    print(f"Janelas: {len(result.windows)}")
    print(f"Falantes com janela: {result.ledger_speakers.kept}")
    print(f"Falantes abaixo da meta (D5): {len(result.speakers_below_target)}")
    print("\n# Artefatos escritos em", out)
    for name in ("windows.parquet", "windows_report.md"):
        print(f"  - {name}")
    return result
