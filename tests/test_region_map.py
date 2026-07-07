"""Testes do mapa de região (birth_state -> NE/SE/fora de escopo)."""

from __future__ import annotations

import pandas as pd

from nefair.corpus.audit import OUT_OF_SCOPE, add_region_column, map_state_to_region

# region_map de teste (independente do config real), com acentuação.
REGION_MAP = {
    "Bahia": "NE",
    "Ceará": "NE",
    "Pernambuco": "NE",
    "São Paulo": "SE",
    "Minas Gerais": "SE",
    "Rio de Janeiro": "SE",
}


def test_maps_nordeste_states():
    assert map_state_to_region("Bahia", REGION_MAP) == "NE"
    assert map_state_to_region("Ceará", REGION_MAP) == "NE"


def test_maps_sudeste_states():
    assert map_state_to_region("São Paulo", REGION_MAP) == "SE"
    assert map_state_to_region("Minas Gerais", REGION_MAP) == "SE"


def test_unmapped_state_is_out_of_scope():
    # Estado real fora do escopo NE/SE não é dropado: vira OUT_OF_SCOPE.
    assert map_state_to_region("Paraná", REGION_MAP) == OUT_OF_SCOPE


def test_missing_state_sentinel_is_out_of_scope():
    assert map_state_to_region("unknown", REGION_MAP) == OUT_OF_SCOPE


def test_region_comes_from_birth_state_not_speaker_code_prefix():
    """Invariante central: um speaker_code com prefixo 'MA' (código de
    entrevista, não estado) e birth_state 'São Paulo' deve mapear para SE, nunca
    para NE. Se a região viesse do prefixo de speaker_code, cairia em NE."""
    frame = pd.DataFrame(
        {
            "speaker_code": ["MA_HV273", "SP_AB001"],
            "birth_state": ["São Paulo", "Bahia"],
        }
    )
    result = add_region_column(frame, REGION_MAP)
    assert list(result["region"]) == ["SE", "NE"]
    # A falante 'MA_HV273' é de São Paulo -> SE (e não NE por causa do "MA").
    assert result.loc[0, "region"] == "SE"


def test_add_region_column_does_not_mutate_input():
    frame = pd.DataFrame({"birth_state": ["Bahia"], "speaker_code": ["X_1"]})
    add_region_column(frame, REGION_MAP)
    assert "region" not in frame.columns
