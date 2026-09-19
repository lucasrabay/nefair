#!/usr/bin/env python
"""Etapa 3.3 — planilha de revisão humana e relatório final da Etapa 3.

Uso:
    uv run scripts/05_review.py --config configs/items.yaml --mode export
    # ... o revisor preenche outputs/review_sheet.csv ...
    uv run scripts/05_review.py --config configs/items.yaml --mode import \
        --full-speakers 232

`--mode export --fake` preenche a planilha com decisões determinísticas, para
que o pipeline rode ponta a ponta sem revisor humano. Isso é uma DEMONSTRAÇÃO
do código: os números que saem daí não são uma revisão e o relatório diz isso.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nefair.items.filter import FILTER_STATUS_KEPT  # noqa: E402
from nefair.items.generate import derive_seed, item_from_dict  # noqa: E402

from nefair.config import load_dotenv, load_items_config  # noqa: E402  # isort: skip
from nefair.items.review import (  # noqa: E402
    DECISION_COLUMN,
    MINUTES_COLUMN,
    REASON_COLUMN,
    REVIEW_STATUS_ACCEPTED,
    REVIEW_STATUS_REJECTED,
    STAGE_SUMMARY_FILENAME,
    export_review_sheet,
    import_review_sheet,
    load_stage_summaries,
    regenerate_report,
    save_stage_summaries,
    sheet_columns,
)
from nefair.schema import read_jsonl, write_jsonl  # noqa: E402

INPUT_FILENAME = "items_filtered.jsonl"
SHEET_FILENAME = "review_sheet.csv"
OUTPUT_FILENAME = "items_reviewed.jsonl"


def prefill_sheet_for_demo(path: Path, seed: int) -> int:
    """Preenche as três colunas do revisor com decisões determinísticas.

    Só existe para demonstrar o caminho export → import sem um humano. A decisão
    deriva de um hash do `item_id`, então é reprodutível — e é justamente por ser
    reprodutível que ela não pode ser confundida com uma revisão de verdade.
    """
    import csv

    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    for row in rows:
        bucket = derive_seed(seed, "demo_review", row["item_id"]) % 100
        rejected = bucket < 20
        row[DECISION_COLUMN] = REVIEW_STATUS_REJECTED if rejected else REVIEW_STATUS_ACCEPTED
        row[REASON_COLUMN] = "demo: pergunta respondível pelo contexto" if rejected else ""
        row[MINUTES_COLUMN] = f"{1.0 + (bucket % 25) / 10.0:.1f}"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 3.3 — revisão humana e relatório.")
    parser.add_argument("--config", required=True, type=Path, help="YAML de itens.")
    parser.add_argument(
        "--mode",
        choices=("export", "import", "report"),
        default="export",
        help="export: gera a planilha; import: concilia a planilha preenchida; "
        "report: só reescreve outputs/items_report.md.",
    )
    parser.add_argument(
        "--items", type=Path, default=None, help=f"Default: outputs/{INPUT_FILENAME}"
    )
    parser.add_argument(
        "--sheet", type=Path, default=None, help=f"Default: outputs/{SHEET_FILENAME}"
    )
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument(
        "--full-speakers",
        type=int,
        default=None,
        help="Nº de falantes do conjunto completo (saída da Etapa 1), para a projeção.",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="No modo export, preenche a planilha com decisões determinísticas (demo).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env)
    config = load_items_config(args.config)

    out = Path(args.outputs)
    sheet_path = args.sheet or (out / SHEET_FILENAME)
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    if args.mode == "report":
        report = regenerate_report(
            out,
            config=config,
            generated_at=generated_at,
            projection_speakers=args.full_speakers,
        )
        print(f"[ok] relatório em {report}")
        return 0

    items_path = args.items or (out / INPUT_FILENAME)
    if not items_path.is_file():
        raise SystemExit(
            f"Itens filtrados não encontrados em {items_path}. Rode "
            "`scripts/04_text_only_filter.py`."
        )
    items = tuple(item_from_dict(row) for row in read_jsonl(items_path))
    kept = tuple(it for it in items if it.filter_status == FILTER_STATUS_KEPT)
    if not kept:
        raise SystemExit(
            "Nenhum item sobreviveu ao controle textual: não há o que revisar. "
            "Isso é, por si só, um resultado do gate do piloto."
        )

    stages = load_stage_summaries(out / STAGE_SUMMARY_FILENAME)
    regions = {str(k): str(v) for k, v in dict(stages.get("regions", {})).items()}

    if args.mode == "export":
        n = export_review_sheet(kept, sheet_path, region_by_speaker=regions)
        print(f"[ok] {n} item(ns) em {sheet_path}")
        print(f"[info] colunas: {list(sheet_columns(kept[0].n_alternatives))}")
        if args.fake:
            filled = prefill_sheet_for_demo(sheet_path, config.seed)
            print(
                f"[aviso] --fake: {filled} linha(s) preenchida(s) com decisões "
                "determinísticas de DEMONSTRAÇÃO (não são uma revisão humana)."
            )
        report = regenerate_report(
            out,
            config=config,
            generated_at=generated_at,
            projection_speakers=args.full_speakers,
        )
        print(f"[ok] relatório em {report}")
        return 0

    if not sheet_path.is_file():
        raise SystemExit(f"Planilha não encontrada em {sheet_path}. Rode `--mode export` antes.")
    outcome = import_review_sheet(
        sheet_path,
        kept,
        review_decisions=config.review_decisions,
        region_by_speaker=regions,
    )
    write_jsonl(out / OUTPUT_FILENAME, outcome.items)
    stages["review"] = outcome.summary.to_dict()
    save_stage_summaries(out / STAGE_SUMMARY_FILENAME, stages)
    report = regenerate_report(
        out,
        config=config,
        generated_at=generated_at,
        projection_speakers=args.full_speakers,
    )
    print("\n".join(outcome.ledger.to_markdown()))
    print(
        f"[info] {outcome.summary.minutes_per_item:.2f} min/item; "
        f"taxa de rejeição {outcome.summary.rejection_rate:.4f}"
    )
    print(f"[ok] {len(outcome.accepted_items)} item(ns) aceito(s) em {out / OUTPUT_FILENAME}")
    print(f"[ok] relatório em {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
