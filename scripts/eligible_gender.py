#!/usr/bin/env python
"""Gênero dos falantes ELEGÍVEIS (após o recorte 20-69), por região.

O painel de confundidores da Etapa 0 (`outputs/gender_by_region.csv`) é do corpus
bruto (39 NE / 193 SE). Como gênero entra no modelo (D6), o texto precisa da
composição dos elegíveis (33 NE / 132 SE), que são os falantes de
`outputs/windows.parquet`.

Uso:
    uv run scripts/eligible_gender.py --config configs/corpus_audit.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Permite `python scripts/...` sem instalar o pacote (uv run já instala).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nefair.config import load_config, load_dotenv  # noqa: E402
from nefair.corpus.audit import (  # noqa: E402
    SPEAKER_KEY,
    add_region_column,
    build_speaker_table,
    filter_informants,
    speaker_category_by_region,
)
from nefair.corpus.load import load_metadata  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gênero dos falantes elegíveis por região.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--windows", type=Path, default=Path("outputs/windows.parquet"))
    parser.add_argument("--out", type=Path, default=Path("outputs/gender_eligible_by_region.csv"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env)
    config = load_config(args.config)

    frame = load_metadata(config, token=os.environ.get("HF_TOKEN") or None).frame
    informants, _ = filter_informants(frame)
    informants = add_region_column(informants, config.region_map)
    speaker_table, warnings = build_speaker_table(informants)
    if warnings["speaker_gender"]["n_violations"]:
        raise SystemExit(
            f"speaker_gender não é 1:1 com {SPEAKER_KEY}: {warnings['speaker_gender']}"
        )

    eligible = (
        pd.read_parquet(args.windows, columns=[SPEAKER_KEY, "region"])
        .drop_duplicates()
        .rename(columns={"region": "window_region"})
    )
    merged = speaker_table.merge(eligible, on=SPEAKER_KEY, how="inner")
    # Todo elegível tem que estar nos metadados, e a região tem que bater.
    if len(merged) != len(eligible):
        raise SystemExit(f"{len(eligible) - len(merged)} elegíveis ausentes dos metadados")
    mismatch = merged[merged["region"] != merged["window_region"]]
    if len(mismatch):
        raise SystemExit(f"região diverge para {len(mismatch)} falantes")

    gender_cats = tuple(sorted(frame["speaker_gender"].astype(str).unique()))
    out = speaker_category_by_region(merged, "speaker_gender", gender_cats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(merged.groupby("region").size().rename("n_elegiveis").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
