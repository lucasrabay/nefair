"""Testes da lógica de auditoria (offline, sem rede)."""

from __future__ import annotations

import pandas as pd

from nefair.corpus.audit import (
    INFORMANT_SPEAKER_TYPE,
    OUT_OF_BUCKET,
    OUT_OF_SCOPE,
    assign_age_bucket,
    build_speaker_counts_by_state,
    filter_informants,
    speaker_key_candidates,
    speaker_split_overlap,
)

REGION_MAP = {"Bahia": "NE", "São Paulo": "SE"}


def _frame() -> pd.DataFrame:
    # 2 informantes (R) + 1 entrevistador (P/1, com sentinelas).
    return pd.DataFrame(
        {
            "speaker_type": ["R", "R", "R", "P/1"],
            "speaker_code": ["S1", "S1", "S2", "E1"],
            "audio_name": ["a", "a", "b", "a"],
            "birth_state": ["Bahia", "Bahia", "São Paulo", "unknown"],
            "age": [30, 30, 40, 0],
            "duration": [3600.0, 1800.0, 7200.0, 10.0],
            "split": ["train", "train", "test", "train"],
        }
    )


def test_filter_informants_counts_drops_by_type():
    informants, report = filter_informants(_frame())
    assert report.total_rows == 4
    assert report.informant_rows == 3
    assert report.dropped_by_type == {"P/1": 1}
    assert set(informants["speaker_type"]) == {INFORMANT_SPEAKER_TYPE}


def test_speaker_key_candidates():
    informants, _ = filter_informants(_frame())
    cands = speaker_key_candidates(informants)
    assert cands["speaker_code"] == 2
    assert cands["speaker_code+audio_name"] == 2


def test_speaker_split_overlap_zero_when_disjoint():
    informants, _ = filter_informants(_frame())
    assert speaker_split_overlap(informants) == 0


def test_assign_age_bucket_out_of_range():
    ages = pd.Series([16, 24, 25, 110, 120, 15])
    edges = (16, 25, 35, 45, 60, 120)
    buckets = list(assign_age_bucket(ages, edges))
    assert buckets[0] == "[16,25)"
    assert buckets[1] == "[16,25)"
    assert buckets[2] == "[25,35)"
    assert buckets[3] == "[60,120)"
    assert buckets[4] == OUT_OF_BUCKET  # 120 é a borda superior exclusiva
    assert buckets[5] == OUT_OF_BUCKET  # 15 < primeira borda


def test_speaker_counts_by_state_uses_region_from_state():
    from nefair.corpus.audit import add_region_column

    informants, _ = filter_informants(_frame())
    informants = add_region_column(informants, REGION_MAP)
    by_state = build_speaker_counts_by_state(informants)
    bahia = by_state[by_state["birth_state"] == "Bahia"].iloc[0]
    assert bahia["region"] == "NE"
    assert bahia["n_speakers"] == 1
    assert bahia["n_segments"] == 2
    assert bahia["hours"] == 1.5  # (3600 + 1800) / 3600
    sp = by_state[by_state["birth_state"] == "São Paulo"].iloc[0]
    assert sp["region"] == "SE"


def test_unmapped_state_out_of_scope_in_counts():
    from nefair.corpus.audit import add_region_column

    frame = _frame().assign(birth_state=["Paraná", "Paraná", "São Paulo", "unknown"])
    informants, _ = filter_informants(frame)
    informants = add_region_column(informants, REGION_MAP)
    by_state = build_speaker_counts_by_state(informants)
    parana = by_state[by_state["birth_state"] == "Paraná"].iloc[0]
    assert parana["region"] == OUT_OF_SCOPE
