"""Testes do janelamento (Etapa 1) — todos offline, sobre o corpus sintético.

A fixture `tests/fixtures/synthetic_corpus.py` planta, de propósito, um caso por
decisão do plano: `NE_INTERRUPTED` para o D1, `NE_SHORT` para o D5, `SE_GAP` para
`gap_tolerance_s`, `SE_MULTI_AUDIO` para o limite de gravação, `SE_OUT_OF_AGE`
para o recorte etário e `XX_FOREIGN` para o recorte de região. Cada teste aqui
aponta para um desses casos — se um deles passar a valer por acidente, é porque
a fixture mudou, não porque o código melhorou.

As asserções de contiguidade NÃO reusam as estruturas internas de `windows.py`:
elas voltam ao frame cru da fixture e conferem as janelas contra ele. Testar o
janelamento com o próprio índice que o janelamento construiu provaria apenas que
o código concorda consigo mesmo.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from fixtures.synthetic_corpus import DEFAULT_PLAN, SEGMENT_DURATION, synthetic_frame
from nefair.config import load_windows_config
from nefair.corpus.audit import add_region_column
from nefair.corpus.windows import (
    BREAK_AUDIO,
    BREAK_GAP,
    BREAK_INTERVIEWER,
    DROP_AGE,
    DROP_REGION,
    assign_runs,
    build_windows,
    enumerate_candidates,
    mark_interviewer_turns,
    pack_run,
    read_windows_parquet,
    speaker_rng,
    stratified_positions,
    windows_to_table,
    write_windows_parquet,
)
from nefair.schema import Window

CONFIG_PATH = "configs/windows.yaml"


# --------------------------------------------------------------------------- #
# Fixtures locais
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def config():
    """Config real do repositório: os testes valem sobre o que vai rodar."""
    return load_windows_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_frame()


@pytest.fixture(scope="module")
def result(frame, config):
    return build_windows(frame, config)


@pytest.fixture(scope="module")
def result_ignore(frame, config):
    """Mesmo corpus, D1 = `ignore` — o contraste que isola o efeito do D1."""
    return build_windows(frame, dataclasses.replace(config, interviewer_turn="ignore"))


def _by_file_path(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Indexa por `file_path` — a chave do SEGMENTO.

    `audio_id` identifica a GRAVAÇÃO: um só valor cobre a entrevista inteira,
    então indexar por ele colapsaria todos os segmentos de um falante numa
    linha só e os testes passariam medindo nada.
    """
    return {str(row.file_path): row for row in frame.itertuples()}


def _recording_positions(frame: pd.DataFrame) -> dict[int, int]:
    """file_path -> posição na ordem da GRAVAÇÃO (incluindo o entrevistador).

    É a referência independente para "contíguo": duas linhas do informante são
    vizinhas de verdade só se suas posições diferem de 1 — se um turno `P/*`
    estiver no meio, a diferença é 2 ou mais.
    """
    ordered = frame.sort_values(
        ["audio_name", "start_time", "file_path"], kind="stable"
    ).reset_index(drop=True)
    return {str(path): position for position, path in enumerate(ordered["file_path"])}


def _windows_of(result, speaker_code: str) -> list[Window]:
    return [w for w in result.windows if w.speaker_code == speaker_code]


# --------------------------------------------------------------------------- #
# Contiguidade, não sobreposição e limites
# --------------------------------------------------------------------------- #
def test_windows_are_contiguous_in_the_recording(result, frame):
    """Sob `close_window`, os segmentos de uma janela são vizinhos na GRAVAÇÃO.

    Não basta serem vizinhos entre os segmentos do informante: é exatamente essa
    a diferença que o D1 captura.
    """
    positions = _recording_positions(frame)
    for window in result.windows:
        ordered = [positions[path] for path in window.segment_file_paths]
        assert ordered == sorted(ordered), f"{window.window_id} fora de ordem temporal"
        assert ordered == list(range(ordered[0], ordered[0] + len(ordered))), (
            f"{window.window_id} tem uma linha estranha no meio"
        )


def test_windows_never_overlap_within_speaker(result):
    seen: dict[str, set[int]] = {}
    for window in result.windows:
        used = seen.setdefault(window.speaker_code, set())
        overlap = used & set(window.segment_file_paths)
        assert not overlap, f"{window.window_id} reusa os segmentos {sorted(overlap)}"
        used.update(window.segment_file_paths)


def test_gaps_inside_a_window_are_within_tolerance(result, frame, config):
    """Nenhuma janela contém um salto temporal acima da tolerância."""
    rows = _by_file_path(frame)
    for window in result.windows:
        ids = window.segment_file_paths
        for previous, current in zip(ids[:-1], ids[1:], strict=True):
            gap = float(rows[current].start_time) - float(rows[previous].end_time)
            assert gap <= config.gap_tolerance_s + 1e-6, (
                f"{window.window_id}: lacuna de {gap:.2f} s entre {previous} e {current}"
            )


def test_duration_and_segment_count_are_both_within_bounds(result, config):
    """As DUAS condições do texto valem ao mesmo tempo, não uma só."""
    assert result.windows
    for window in result.windows:
        assert config.min_duration_s <= window.duration_s <= config.max_duration_s
        assert config.min_segments <= window.n_segments <= config.max_segments
        assert window.n_segments == len(window.segment_file_paths)


def test_window_fields_agree_with_the_source_rows(result, frame, config):
    rows = _by_file_path(frame)
    for window in result.windows:
        segments = [rows[path] for path in window.segment_file_paths]
        assert {s.speaker_type for s in segments} == {"R"}
        assert {s.speaker_code for s in segments} == {window.speaker_code}
        assert {s.audio_name for s in segments} == {window.audio_name}
        assert window.start_time == pytest.approx(float(segments[0].start_time))
        assert window.end_time == pytest.approx(float(segments[-1].end_time))
        assert window.duration_s == pytest.approx(
            sum(float(s.duration) for s in segments), abs=1e-4
        )
        expected = " ".join(
            str(getattr(s, config.reference_field)).strip()
            for s in segments
            if str(getattr(s, config.reference_field)).strip()
        )
        assert window.reference_text == expected


# --------------------------------------------------------------------------- #
# D1 — turno de entrevistador (NE_INTERRUPTED)
# --------------------------------------------------------------------------- #
def test_interviewer_turn_is_detected_before_the_informant_filter(frame):
    """A anotação do D1 sobrevive ao filtro — é esse o ponto sutil da etapa."""
    marked = mark_interviewer_turns(frame)
    assert set(marked["speaker_type"]) == {"R"}
    interrupted = marked[marked["speaker_code"] == "NE_INTERRUPTED"]
    flagged = interrupted[interrupted["interviewer_turns_before"] > 0]
    # A fixture planta 4 turnos de entrevistador nesse falante.
    assert len(flagged) == len(DEFAULT_PLAN[2].interviewer_at) == 4
    assert flagged["interviewer_seconds_before"].tolist() == [SEGMENT_DURATION] * 4


def test_close_window_breaks_on_interviewer_turn(result):
    assert result.breaks.by_reason[BREAK_INTERVIEWER] == 4
    assert result.breaks.interviewer_turns_observed == 4


def test_ignore_does_not_break_on_interviewer_turn(result_ignore, frame, config):
    """Com `ignore`, a interrupção não fecha a janela — e é contada assim mesmo.

    O contraste é a prova do D1: mesmo corpus, mesma semente, só a chave muda.
    Sob `ignore` aparecem MAIS candidatas para `NE_INTERRUPTED` (as corridas se
    fundem) e alguma janela passa por cima de um turno de entrevistador.
    """
    assert result_ignore.breaks.by_reason[BREAK_INTERVIEWER] == 0
    # O turno segue observado e reportado, ainda que não quebre nada.
    assert result_ignore.breaks.interviewer_turns_observed == 4

    marked = mark_interviewer_turns(frame)
    interrupted_ids = set(
        marked.loc[marked["interviewer_turns_before"] > 0, "file_path"].astype(str)
    )
    positions = _recording_positions(frame)
    spanned = [
        window
        for window in _windows_of(result_ignore, "NE_INTERRUPTED")
        if interrupted_ids & set(window.segment_file_paths[1:])
    ]
    assert spanned, "com `ignore`, alguma janela deve atravessar a interrupção"
    # E essa janela NÃO é contígua na gravação: o turno `P/*` ficou no meio.
    ordered = [positions[p] for p in spanned[0].segment_file_paths]
    assert ordered[-1] - ordered[0] > len(ordered) - 1


def test_ignore_yields_more_candidates_than_close_window(result, result_ignore):
    close = result.candidates_per_speaker["NE_INTERRUPTED"]
    ignore = result_ignore.candidates_per_speaker["NE_INTERRUPTED"]
    assert ignore > close, (close, ignore)
    # 120 segmentos contíguos de 4 s ⇒ 15 janelas de 8 segmentos.
    assert ignore == 15


# --------------------------------------------------------------------------- #
# Lacuna temporal (SE_GAP) e troca de gravação (SE_MULTI_AUDIO)
# --------------------------------------------------------------------------- #
def test_twelve_second_gap_breaks_the_window(result, frame):
    """A lacuna de 12 s da fixture é a única quebra por lacuna do corpus."""
    assert result.breaks.by_reason[BREAK_GAP] == 1

    speaker = frame[(frame["speaker_code"] == "SE_GAP")].sort_values("start_time")
    times = speaker[["file_path", "start_time", "end_time"]].to_numpy()
    boundary = [
        (str(times[i][0]), str(times[i + 1][0]))
        for i in range(len(times) - 1)
        if times[i + 1][1] - times[i][2] > 1.0
    ]
    assert len(boundary) == 1, boundary
    before, after = boundary[0]
    for window in _windows_of(result, "SE_GAP"):
        ids = set(window.segment_file_paths)
        assert not (before in ids and after in ids), f"{window.window_id} cruzou a lacuna"


def test_no_window_crosses_audio_name(result, frame):
    rows = _by_file_path(frame)
    for window in result.windows:
        names = {rows[path].audio_name for path in window.segment_file_paths}
        assert len(names) == 1, f"{window.window_id} cruza gravações: {names}"
    # A fixture tem exatamente uma troca de gravação (SE_MULTI_AUDIO, 2 áudios).
    assert result.breaks.by_reason[BREAK_AUDIO] == 1
    multi = _windows_of(result, "SE_MULTI_AUDIO")
    assert len({w.audio_name for w in multi}) == 2, "as duas gravações devem aparecer"


# --------------------------------------------------------------------------- #
# D5 — falante curto (NE_SHORT)
# --------------------------------------------------------------------------- #
def test_short_speaker_yields_fewer_windows_and_is_counted(result, config):
    """30 segmentos de 4 s ⇒ 3 janelas de 8; nem erro, nem sobreposição, nem 10."""
    short = _windows_of(result, "NE_SHORT")
    assert len(short) == 3 < config.n_windows_per_speaker
    assert result.candidates_per_speaker["NE_SHORT"] == 3
    assert "NE_SHORT" in result.speakers_below_target
    assert all(w.n_candidates_for_speaker == 3 for w in short)
    # Nenhum outro falante da fixture fica abaixo da meta.
    assert result.speakers_below_target == ("NE_SHORT",)


def test_allow_fewer_windows_false_excludes_the_speaker_with_a_reason(frame, config):
    """Com `allow_fewer_windows: false` não há saída legítima: o falante sai INTEIRO.

    Chegar à meta exigiria sobrepor janelas, que é justamente o que o D5 recusa.
    O falante some das janelas, mas não da contabilidade.
    """
    strict = build_windows(frame, dataclasses.replace(config, allow_fewer_windows=False))
    strict.assert_balanced()
    assert not _windows_of(strict, "NE_SHORT")
    assert "NE_SHORT" in strict.dropped_speakers["abaixo_da_meta_com_allow_fewer_windows_false"]
    assert all(
        len(_windows_of(strict, speaker)) == config.n_windows_per_speaker
        for speaker in strict.selected_per_speaker
    )


# --------------------------------------------------------------------------- #
# Recortes de escopo (SE_OUT_OF_AGE, XX_FOREIGN)
# --------------------------------------------------------------------------- #
def test_out_of_age_speaker_is_excluded_and_counted(result, config):
    assert config.age_max == 69
    assert not _windows_of(result, "SE_OUT_OF_AGE")  # 72 anos
    assert result.dropped_speakers[DROP_AGE] == ["SE_OUT_OF_AGE"]
    assert result.ledger_speakers.dropped[DROP_AGE] == 1
    # 96 segmentos do falante, todos contabilizados no motivo certo.
    assert result.ledger_segments.dropped[DROP_AGE] == 96


def test_out_of_region_speaker_is_excluded_and_counted(result):
    assert not _windows_of(result, "XX_FOREIGN")  # birth_state == '' (fora do Brasil)
    assert result.dropped_speakers[DROP_REGION] == ["XX_FOREIGN"]
    assert result.ledger_segments.dropped[DROP_REGION] == 96


def test_only_configured_regions_survive(result, config):
    assert {w.region for w in result.windows} <= set(config.regions)
    assert result.windows_by_region == {"NE": 33, "SE": 50}


# --------------------------------------------------------------------------- #
# Trilha de exclusão e determinismo
# --------------------------------------------------------------------------- #
def test_exclusion_ledgers_balance(result, frame):
    result.assert_balanced()
    assert result.ledger_segments.total_in == len(frame)
    assert result.ledger_candidates.kept == len(result.windows)
    assert result.ledger_segments.kept == sum(w.n_segments for w in result.windows)


def test_break_tally_reconciles_with_the_number_of_runs(result):
    assert result.breaks.n_runs == result.breaks.n_speakers + result.breaks.total


def test_two_calls_produce_identical_results(frame, config):
    first = build_windows(frame, config)
    second = build_windows(frame, config)
    assert first.windows == second.windows
    assert first.candidates_per_speaker == second.candidates_per_speaker
    assert first.breaks == second.breaks
    assert first.ledger_segments.dropped == second.ledger_segments.dropped


def test_speaker_rng_does_not_depend_on_the_process_hash_seed():
    """Valores fixos: se alguém trocar SHA-256 por `hash()`, este teste cai.

    O `hash()` embutido é randomizado por processo — o bug seria invisível numa
    única execução e destruiria a reprodutibilidade entre máquinas.
    """
    draws = speaker_rng(20260918, "NE_LONG").integers(0, 1000, size=5).tolist()
    assert draws == speaker_rng(20260918, "NE_LONG").integers(0, 1000, size=5).tolist()
    assert draws != speaker_rng(20260918, "NE_MID").integers(0, 1000, size=5).tolist()
    assert draws != speaker_rng(1, "NE_LONG").integers(0, 1000, size=5).tolist()


# --------------------------------------------------------------------------- #
# Empacotamento e amostragem estratificada (unidades)
# --------------------------------------------------------------------------- #
def test_pack_run_respects_the_minimum_segment_count(config):
    """Segmentos longos: 30 s cabem em 3 deles, mas 3 < min_segments ⇒ nada."""
    assert pack_run(np.repeat(10.0, 6), config) == []


def test_pack_run_respects_the_maximum_segment_count(config):
    """Segmentos curtos: 15 de 1 s somam 15 s < 30 s ⇒ nenhuma janela válida."""
    assert pack_run(np.repeat(1.0, 40), config) == []


def test_pack_run_respects_the_maximum_duration(config):
    """8 segmentos de 8 s dão 64 s > 60 s ⇒ nenhum tamanho válido a partir do 0."""
    assert pack_run(np.repeat(8.0, 32), config) == []


def test_pack_run_takes_the_smallest_valid_window(config):
    """O menor tamanho válido é o que maximiza o nº de candidatas disjuntas."""
    packed = pack_run(np.repeat(4.0, 20), config)
    assert packed == [(0, 8), (8, 16)]  # sobram 4 segmentos, sem janela


def test_pack_run_windows_are_disjoint_and_ordered(config):
    packed = pack_run(np.repeat(SEGMENT_DURATION, 160), config)
    assert len(packed) == 20
    for (_, previous_end), (next_start, _) in zip(packed[:-1], packed[1:], strict=True):
        assert next_start >= previous_end


def test_stratified_selection_covers_the_whole_interview():
    """Uma candidata por bloco: a seleção cobre a entrevista, não só o começo."""
    rng = speaker_rng(1, "S")
    picks = stratified_positions(100, 10, rng)
    indices = [index for index, _ in picks]
    blocks = [block for _, block in picks]
    assert blocks == list(range(10))
    assert len(set(indices)) == 10
    for block, index in zip(blocks, indices, strict=True):
        assert block * 10 <= index < (block + 1) * 10


def test_stratified_selection_takes_everything_when_below_target():
    picks = stratified_positions(3, 10, speaker_rng(1, "S"))
    assert picks == [(0, 0), (1, 1), (2, 2)]


def test_position_index_spans_the_interview(result, config):
    long_windows = _windows_of(result, "NE_LONG")
    assert len(long_windows) == config.n_windows_per_speaker
    assert [w.position_index for w in long_windows] == list(range(config.n_windows_per_speaker))
    assert [w.window_id for w in long_windows] == [
        f"NE_LONG__w{i:02d}" for i in range(1, config.n_windows_per_speaker + 1)
    ]


def test_enumerate_candidates_is_empty_for_an_empty_frame(config):
    empty = synthetic_frame().iloc[0:0].assign(run_id=pd.Series(dtype="int64"))
    assert enumerate_candidates(empty, config) == []


def test_candidates_are_disjoint_within_each_speaker(frame, config):
    """As candidatas (não só as sorteadas) já nascem sem sobreposição."""
    marked = mark_interviewer_turns(frame)
    marked = add_region_column(marked, config.base.region_map)
    eligible = marked[
        marked["region"].isin(config.regions)
        & marked["age"].between(config.age_min, config.age_max)
    ].reset_index(drop=True)
    eligible, _ = assign_runs(eligible, config)
    candidates = enumerate_candidates(eligible, config)
    assert len(candidates) == 125
    used: dict[str, set[int]] = {}
    for candidate in candidates:
        rows = set(range(candidate.row_start, candidate.row_end))
        speaker = used.setdefault(candidate.speaker_code, set())
        assert not (speaker & rows)
        speaker.update(rows)


# --------------------------------------------------------------------------- #
# Artefato em parquet
# --------------------------------------------------------------------------- #
def test_parquet_roundtrip_preserves_the_windows(result, tmp_path):
    path = write_windows_parquet(result.windows, tmp_path / "windows.parquet")
    assert read_windows_parquet(path) == result.windows


def test_parquet_is_byte_identical_between_writes(result, tmp_path):
    """Critério de aceite 4 do plano: duas execuções, bytes idênticos."""
    first = write_windows_parquet(result.windows, tmp_path / "a.parquet")
    second = write_windows_parquet(result.windows, tmp_path / "b.parquet")
    assert first.read_bytes() == second.read_bytes()


def test_parquet_column_order_is_fixed(result):
    table = windows_to_table(result.windows)
    assert table.column_names[:6] == [
        "window_id",
        "speaker_code",
        "audio_name",
        "split",
        "region",
        "age",
    ]
    assert table.column_names[-1] == "content_sha"
