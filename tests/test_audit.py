"""Testes da lógica de auditoria (offline, sem rede)."""

from __future__ import annotations

import pandas as pd

from nefair.corpus.audit import (
    INFORMANT_SPEAKER_TYPE,
    OUT_OF_BUCKET,
    OUT_OF_SCOPE,
    _age_comparison_sentence,
    add_region_column,
    assign_age_bucket,
    build_audio_quality_by_region,
    build_speaker_counts_by_state,
    build_speaker_table,
    filter_informants,
    region_age_stats,
    speaker_category_by_region,
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


def _rich_informants() -> pd.DataFrame:
    # 3 informantes 1:1: S1 (NE), S2 e S3 (SE). S1 tem 1 seg high + 1 low.
    return pd.DataFrame(
        {
            "speaker_code": ["S1", "S1", "S2", "S2", "S3"],
            "region": ["NE", "NE", "SE", "SE", "SE"],
            "birth_state": ["Bahia", "Bahia", "São Paulo", "São Paulo", "São Paulo"],
            "speaker_gender": ["F", "F", "M", "M", "F"],
            "education": ["college", "college", "unknown", "unknown", "high"],
            "racial_category": ["White", "White", "unknown", "unknown", "Black"],
            "age": [30, 30, 50, 50, 60],
            "audio_quality": ["high", "low", "high", "high", "low"],
            "duration": [3.0, 3.0, 3.0, 3.0, 3.0],
        }
    )


def test_build_speaker_table_is_one_to_one_without_warnings():
    table, warnings = build_speaker_table(_rich_informants())
    assert len(table) == 3  # uma linha por falante
    assert all(info["n_violations"] == 0 for info in warnings.values())


def test_build_speaker_table_flags_non_one_to_one():
    frame = _rich_informants()
    frame.loc[1, "speaker_gender"] = "M"  # S1 passa a ter dois gêneros
    _, warnings = build_speaker_table(frame)
    assert warnings["speaker_gender"]["n_violations"] == 1
    assert warnings["speaker_gender"]["examples"] == ["S1"]


def test_speaker_category_includes_absent_category_X():
    table, _ = build_speaker_table(_rich_informants())
    out = speaker_category_by_region(table, "speaker_gender", categories=("F", "M", "X"))
    # 'X' não existe em nenhum informante, mas deve aparecer com 0 em toda região.
    assert set(out["speaker_gender"]) == {"F", "M", "X"}
    x_rows = out[out["speaker_gender"] == "X"]
    assert (x_rows["n_speakers"] == 0).all()
    assert len(x_rows) == 2  # NE e SE


def test_audio_quality_by_region_is_segment_level():
    out = build_audio_quality_by_region(_rich_informants()).set_index("region")
    # NE: S1 tem 1 high + 1 low -> pct_low 50, 1 falante com >=1 low
    assert out.loc["NE", "n_segments"] == 2
    assert out.loc["NE", "pct_low"] == 50.0
    assert out.loc["NE", "n_speakers_with_low_segment"] == 1
    # SE: 3 seg (2 high de S2, 1 low de S3) -> só S3 tem low
    assert out.loc["SE", "n_speakers"] == 2
    assert out.loc["SE", "n_speakers_with_low_segment"] == 1


def test_age_comparison_sentence_deduces_sense():
    stats = region_age_stats(build_speaker_table(_rich_informants())[0])
    sentence = _age_comparison_sentence(stats)
    # NE mediana 30, SE mediana 55 -> SE mais velho por 25 anos (deduzido, não fixo)
    assert "SE é mais velho que NE em 25.0 anos" in sentence


def test_age_comparison_sentence_comparable_when_close():
    stats = pd.DataFrame(
        [
            {"region": "NE", "n_speakers": 1, "age_median": 40.0, "age_mean": 40.0},
            {"region": "SE", "n_speakers": 1, "age_median": 41.0, "age_mean": 41.0},
        ]
    )
    assert "comparáveis em tendência central" in _age_comparison_sentence(stats)


def test_speaker_counts_by_state_uses_region_from_state():
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
    frame = _frame().assign(birth_state=["Paraná", "Paraná", "São Paulo", "unknown"])
    informants, _ = filter_informants(frame)
    informants = add_region_column(informants, REGION_MAP)
    by_state = build_speaker_counts_by_state(informants)
    parana = by_state[by_state["birth_state"] == "Paraná"].iloc[0]
    assert parana["region"] == OUT_OF_SCOPE
