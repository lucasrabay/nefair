#!/usr/bin/env python
"""Entrypoint CLI do janelamento de fala contínua (Etapa 1).

Uso:
    uv run scripts/01_build_windows.py --config configs/windows.yaml
    uv run scripts/01_build_windows.py --config configs/windows.yaml --fixture

Nenhum byte de áudio é lido: a etapa decide as janelas só sobre metadados. Com
`--fixture`, roda ponta a ponta sobre o corpus sintético de teste, sem rede e
sem `HF_TOKEN` — é assim que o pipeline se valida num ambiente sem credenciais.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Permite `python scripts/...` sem instalar o pacote (uv run já instala).
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nefair.config import load_dotenv, load_windows_config  # noqa: E402
from nefair.corpus.windows import run_windows  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Etapa 1 — janelas de fala contínua do informante (só metadados)."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Caminho do YAML de configuração (ex.: configs/windows.yaml).",
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
    parser.add_argument(
        "--fixture",
        action="store_true",
        help=(
            "Roda sobre o corpus sintético de tests/fixtures em vez da HF "
            "(offline; útil para validar o pipeline sem credenciais)."
        ),
    )
    return parser.parse_args(argv)


def _fixture_frame():
    """Importa a fixture sintética sem exigir que `tests` seja um pacote instalado."""
    sys.path.insert(0, str(_ROOT / "tests"))
    from fixtures.synthetic_corpus import synthetic_frame

    return synthetic_frame()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env)
    token = os.environ.get("HF_TOKEN") or None
    config = load_windows_config(args.config)
    frame = _fixture_frame() if args.fixture else None
    run_windows(
        config,
        args.config,
        args.outputs,
        token=token,
        frame=frame,
        dataset_revision="fixture-sintetica" if args.fixture else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
