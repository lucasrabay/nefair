#!/usr/bin/env python
"""Entrypoint CLI da auditoria de metadados do corpus (Etapa 0).

Uso:
    uv run scripts/run_corpus_audit.py --config configs/corpus_audit.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Permite `python scripts/...` sem instalar o pacote (uv run já instala).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nefair.config import load_config, load_dotenv  # noqa: E402
from nefair.corpus.audit import run_audit  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditoria de metadados do corpus CORAA-MUPE.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Caminho do YAML de configuração (ex.: configs/corpus_audit.yaml).",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="Diretório de saída dos artefatos (default: outputs).",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="Arquivo .env com HF_TOKEN (default: .env; ignorado se ausente).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env)
    token = os.environ.get("HF_TOKEN") or None
    config = load_config(args.config)
    run_audit(config, args.outputs, token=token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
