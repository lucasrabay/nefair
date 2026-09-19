#!/usr/bin/env python
"""Etapa 3.2 — controle textual dos itens (D3).

Uso:
    uv run scripts/04_text_only_filter.py --config configs/items.yaml \
        --models configs/models.yaml --fake

O modelo vê **só** pergunta e alternativas: sem áudio e sem transcrição. Item
respondido assim não mede compreensão de fala e cai. `--fake` substitui os LLMs
avaliados por `FakeTextQA` determinísticos (única opção sem chave de API).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nefair.config import (  # noqa: E402
    ItemsConfig,
    ModelsConfig,
    load_dotenv,
    load_items_config,
    load_models_config,
    validate_cross_config,
)
from nefair.items.filter import apply_text_filter, chance_pass_probability  # noqa: E402
from nefair.items.generate import item_from_dict  # noqa: E402
from nefair.items.review import (  # noqa: E402
    STAGE_SUMMARY_FILENAME,
    load_stage_summaries,
    regenerate_report,
    save_stage_summaries,
)
from nefair.models.base import FakeTextQA, TextQA  # noqa: E402
from nefair.schema import read_jsonl, write_jsonl  # noqa: E402

INPUT_FILENAME = "items_generated.jsonl"
OUTPUT_FILENAME = "items_filtered.jsonl"
RUNS_FILENAME = "items_filter_runs.jsonl"


def build_filter_models(
    items_config: ItemsConfig, models_config: ModelsConfig, *, fake: bool
) -> dict[str, TextQA]:
    """Constrói um cliente por modelo listado em `text_filter.models` (D3).

    A temperatura do config de itens SOBRESCREVE a do `models.yaml`: no controle
    textual queremos amostragem (é o que dá sentido à maioria de três execuções),
    enquanto na avaliação da Etapa 5 os mesmos modelos rodam com temperatura 0.
    São dois usos do mesmo modelo, e o fingerprint precisa refletir isso.
    """
    if not fake:
        raise SystemExit(
            "Sem `--fake` este script precisa dos adaptadores de provedor do "
            "Bloco R (`src/nefair/models/llm/`), que ainda não existem, e de "
            "credenciais. Rode com `--fake`."
        )
    clients: dict[str, TextQA] = {}
    for index, key in enumerate(sorted(items_config.text_filter.models)):
        base = models_config.specs[key]
        spec = replace(
            base,
            provider="fake",
            model_id=f"fake::{base.model_id}",
            params={**dict(base.params), "temperature": items_config.text_filter.temperature},
        )
        # Sementes distintas por modelo: modelos falsos com a mesma semente
        # responderiam de forma idêntica e `any_model_discards` viraria
        # `one_model_discards` disfarçado.
        clients[key] = FakeTextQA(
            spec=spec,
            seed=items_config.text_filter.seed + index,
            n_alternatives=items_config.n_alternatives,
        )
    return clients


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 3.2 — controle textual (D3).")
    parser.add_argument("--config", required=True, type=Path, help="YAML de itens.")
    parser.add_argument(
        "--models", required=True, type=Path, help="YAML de modelos (chaves do filtro)."
    )
    parser.add_argument(
        "--items", type=Path, default=None, help=f"Default: outputs/{INPUT_FILENAME}"
    )
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--fake", action="store_true", help="Usa LLMs falsos determinísticos.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env)
    items_config = load_items_config(args.config)
    models_config = load_models_config(args.models)
    validate_cross_config(items_config, models_config)

    out = Path(args.outputs)
    items_path = args.items or (out / INPUT_FILENAME)
    if not items_path.is_file():
        raise SystemExit(
            f"Itens não encontrados em {items_path}. Rode `scripts/03_generate_items.py`."
        )
    items = tuple(item_from_dict(row) for row in read_jsonl(items_path))
    if not items:
        raise SystemExit(f"{items_path} está vazio: não há o que filtrar.")

    stages = load_stage_summaries(out / STAGE_SUMMARY_FILENAME)
    regions = {str(k): str(v) for k, v in dict(stages.get("regions", {})).items()}

    models = build_filter_models(items_config, models_config, fake=args.fake)
    outcome = apply_text_filter(items, models, items_config.text_filter, region_by_speaker=regions)

    write_jsonl(out / OUTPUT_FILENAME, outcome.items)
    write_jsonl(out / RUNS_FILENAME, outcome.runs)

    stages["filter"] = outcome.summary.to_dict()
    stages.pop("review", None)  # a revisão anterior é de outro conjunto de itens
    save_stage_summaries(out / STAGE_SUMMARY_FILENAME, stages)

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    report = regenerate_report(out, config=items_config, generated_at=generated_at)

    chance = chance_pass_probability(
        n_runs=items_config.text_filter.n_runs,
        n_correct_at_least=items_config.text_filter.discard_if_correct_at_least,
        n_alternatives=items_config.n_alternatives,
    )
    print("\n".join(outcome.ledger.to_markdown()))
    print(
        f"[info] taxa de descarte observada: {outcome.summary.discard_rate:.4f} | "
        f"linha de base de acaso (calculada): {chance:.6f}"
    )
    print(f"[ok] {len(outcome.kept_items)} item(ns) retido(s) em {out / OUTPUT_FILENAME}")
    print(f"[ok] relatório em {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
