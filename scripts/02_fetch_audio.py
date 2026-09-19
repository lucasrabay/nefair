#!/usr/bin/env python
"""Entrypoint CLI do download seletivo de áudio das janelas (Etapa 2).

Uso:
    uv run scripts/02_fetch_audio.py --config configs/windows.yaml --dry-run
    uv run scripts/02_fetch_audio.py --config configs/windows.yaml --limit 5
    uv run scripts/02_fetch_audio.py --config configs/windows.yaml

Baixa SOMENTE os chunks da coluna `audio` dos row groups que contêm segmentos
das janelas selecionadas — nunca os 41,8 GB do dataset. `--dry-run` faz só a
indexação (footers + coluna `audio_id`, poucos MB) e reporta o que baixaria;
`--limit` restringe às N primeiras janelas, para um teste barato de ponta a
ponta antes de pagar a rede inteira.
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
from nefair.corpus.audio import run_fetch_audio  # noqa: E402
from nefair.corpus.windows import read_windows_parquet  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Etapa 2 — áudio das janelas selecionadas (download seletivo)."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Caminho do YAML de configuração (ex.: configs/windows.yaml).",
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=None,
        help="Parquet de janelas da Etapa 1 (default: <outputs>/windows.parquet).",
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
        "--limit",
        type=int,
        default=None,
        help="Processa apenas as N primeiras janelas (ordem canônica do parquet).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Só indexa e reporta o que seria baixado; não busca nenhum byte de áudio.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env)
    token = os.environ.get("HF_TOKEN") or None
    config = load_windows_config(args.config)

    windows_path = args.windows or (args.outputs / "windows.parquet")
    if not Path(windows_path).is_file():
        raise SystemExit(
            f"Janelas não encontradas em '{windows_path}'. "
            "Rode a Etapa 1 antes (scripts/01_build_windows.py)."
        )
    windows = read_windows_parquet(windows_path)
    if args.limit is not None:
        windows = windows[: args.limit]
        print(f"--limit {args.limit}: processando {len(windows)} janela(s).")
    if not windows:
        raise SystemExit("Nenhuma janela a processar.")

    run_fetch_audio(
        config.base.dataset_id,
        config.base.dataset.revision,
        windows,
        args.outputs,
        token=token,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
