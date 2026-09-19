#!/usr/bin/env python
"""Etapa 3.1 — geração dos itens de múltipla escolha (D2, D11).

Uso:
    uv run scripts/03_generate_items.py --config configs/items.yaml --fake

Sem credencial de provedor não há como chamar o LLM gerador de verdade: o
`--fake` troca o gerador por um `FakeTextQA` determinístico e roda a etapa ponta
a ponta, produzindo os mesmos artefatos que a execução real produziria. É assim
que o pipeline é demonstrável hoje. Sem `--fake`, o script FALHA com uma
mensagem explícita em vez de fingir que rodou.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

# Permite `python scripts/...` sem instalar o pacote (uv run já instala).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nefair.config import (  # noqa: E402
    ItemsConfig,
    load_dotenv,
    load_items_config,
    load_models_config,
    validate_cross_config,
)
from nefair.items.generate import generate_items, select_pilot_speakers  # noqa: E402
from nefair.items.review import (  # noqa: E402
    STAGE_SUMMARY_FILENAME,
    load_stage_summaries,
    regenerate_report,
    save_stage_summaries,
)
from nefair.models.base import FakeTextQA, ModelSpec  # noqa: E402
from nefair.schema import Window, write_jsonl  # noqa: E402

ITEMS_FILENAME = "items_generated.jsonl"
ATTEMPTS_FILENAME = "items_generation_attempts.jsonl"

# Campos de `Window`, na ordem da dataclass. Usados para reconstruir as janelas
# vindas do parquet/JSONL da Etapa 1 sem adivinhar nomes de coluna.
_WINDOW_FIELDS: tuple[str, ...] = (
    "window_id",
    "speaker_code",
    "audio_name",
    "split",
    "region",
    "age",
    "segment_audio_ids",
    "start_time",
    "end_time",
    "duration_s",
    "n_segments",
    "reference_text",
    "position_index",
    "n_candidates_for_speaker",
)


# --------------------------------------------------------------------------- #
# Entrada: janelas da Etapa 1
# --------------------------------------------------------------------------- #
def _window_from_mapping(row: dict) -> Window:
    missing = [f for f in _WINDOW_FIELDS if f not in row]
    if missing:
        raise SystemExit(
            f"Janela sem os campos {missing}. O artefato de janelas não segue o "
            "contrato de `nefair.schema.Window` (Etapa 1)."
        )
    ids = row["segment_audio_ids"]
    if isinstance(ids, str):
        ids = json.loads(ids)
    return Window(
        window_id=str(row["window_id"]),
        speaker_code=str(row["speaker_code"]),
        audio_name=str(row["audio_name"]),
        split=str(row["split"]),
        region=str(row["region"]),
        age=int(row["age"]),
        segment_audio_ids=tuple(int(i) for i in ids),
        start_time=float(row["start_time"]),
        end_time=float(row["end_time"]),
        duration_s=float(row["duration_s"]),
        n_segments=int(row["n_segments"]),
        reference_text=str(row["reference_text"]),
        position_index=int(row["position_index"]),
        n_candidates_for_speaker=int(row["n_candidates_for_speaker"]),
    )


def load_windows(path: Path) -> tuple[Window, ...]:
    """Lê as janelas da Etapa 1, de `.parquet` ou `.jsonl`."""
    if path.suffix == ".jsonl":
        from nefair.schema import read_jsonl

        return tuple(_window_from_mapping(row) for row in read_jsonl(path))
    import pandas as pd

    frame = pd.read_parquet(path)
    return tuple(_window_from_mapping(row) for row in frame.to_dict("records"))


def demo_windows() -> tuple[Window, ...]:
    """Janelas SINTÉTICAS, só para demonstrar o pipeline antes da Etapa 1.

    Não são dados: são um andaime. O relatório carimba `windows_source` com o
    aviso correspondente, para que ninguém leia as taxas de um relatório de
    demonstração como se fossem do piloto.
    """
    windows: list[Window] = []
    plan = (("NE_DEMO01", "NE", 34), ("NE_DEMO02", "NE", 52), ("SE_DEMO01", "SE", 29))
    for speaker, region, age in plan:
        for index in range(1, 5):
            start = 40.0 * index
            windows.append(
                Window(
                    window_id=Window.make_id(speaker, index),
                    speaker_code=speaker,
                    audio_name=f"{speaker}_entrevista",
                    split="train",
                    region=region,
                    age=age,
                    segment_audio_ids=tuple(range(index * 10, index * 10 + 8)),
                    start_time=start,
                    end_time=start + 45.0,
                    duration_s=45.0,
                    n_segments=8,
                    reference_text=(
                        f"trecho de demonstração {index} da entrevista de {speaker}: "
                        "a gente morava na roça e depois mudou pra cidade"
                    ),
                    position_index=index - 1,
                    n_candidates_for_speaker=4,
                )
            )
    return tuple(windows)


# --------------------------------------------------------------------------- #
# Gerador falso (modo --fake)
# --------------------------------------------------------------------------- #
_RE_TARGET = re.compile(r"^TRANSCRIÇÃO-ALVO \(janela (?P<wid>[^)]+)\)$", re.MULTILINE)
_RE_CONTEXT = re.compile(r"^- \(janela (?P<wid>[^)]+)\) ", re.MULTILINE)


def fake_generator_responder(n_alternatives: int, n_items: int):
    """Responder que devolve itens BEM FORMADOS derivados do próprio prompt.

    Lê do prompt o id da janela-alvo e os ids dos trechos de contexto, e monta o
    JSON com proveniência correta. Serve para exercitar o caminho feliz ponta a
    ponta; os casos malformados são exercitados nos testes, onde cada motivo de
    rejeição tem seu próprio cenário.
    """

    def respond(prompt: str) -> str:
        target = _RE_TARGET.search(prompt)
        target_id = target.group("wid") if target else "desconhecida"
        context_ids = _RE_CONTEXT.findall(prompt)
        items = []
        for slot in range(n_items):
            alternatives = [
                {
                    "text": f"resposta correta do item {slot + 1} sobre {target_id}",
                    "is_correct": True,
                    "source_window_id": target_id,
                }
            ]
            for d in range(n_alternatives - 1):
                source = context_ids[d % len(context_ids)] if context_ids else target_id
                alternatives.append(
                    {
                        "text": f"distrator {d + 1} do item {slot + 1} ancorado em {source}",
                        "is_correct": False,
                        "source_window_id": source,
                    }
                )
            items.append(
                {
                    "question": f"O que o informante disse na janela {target_id}?",
                    "alternatives": alternatives,
                }
            )
        return json.dumps({"items": items}, ensure_ascii=False)

    return respond


def build_generator(config: ItemsConfig, *, fake: bool) -> FakeTextQA:
    """Constrói o cliente do gerador. Hoje, só o falso.

    Este bloco fala apenas com os `Protocol`s de `models/base.py`; os adaptadores
    de provedor (OpenAI/Google/Qwen) são do Bloco R. Quando existirem, é aqui que
    entram — nada dentro de `src/nefair/items/` precisa mudar.
    """
    if not fake:
        raise SystemExit(
            "Sem `--fake` este script precisa de um adaptador de provedor para o "
            f"gerador `{config.generator.model_id}`, que ainda não existe neste "
            "repositório (Bloco R) e exigiria credencial. Rode com `--fake` para "
            "exercitar a etapa ponta a ponta com um gerador determinístico."
        )
    spec = ModelSpec(
        key="generator",
        provider="fake",
        model_id=f"fake::{config.generator.model_id}",
        version=f"fake::{config.generator.version}",
        role=config.generator.role,
        family=config.generator.family,
        params=dict(config.generator.params),
    )
    return FakeTextQA(
        spec=spec,
        responder=fake_generator_responder(config.n_alternatives, config.n_items_per_window),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 3.1 — geração de itens (D2, D11).")
    parser.add_argument("--config", required=True, type=Path, help="YAML de itens.")
    parser.add_argument(
        "--models",
        type=Path,
        default=None,
        help="YAML de modelos; se informado, valida o D11 (gerador ∉ avaliados).",
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path("outputs/windows.parquet"),
        help="Janelas da Etapa 1 (.parquet ou .jsonl).",
    )
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Usa um gerador falso determinístico (única opção sem chave de API).",
    )
    parser.add_argument(
        "--all-speakers",
        action="store_true",
        help="Ignora o recorte do piloto e gera itens para todos os falantes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env)
    config = load_items_config(args.config)
    if args.models is not None:
        validate_cross_config(config, load_models_config(args.models))

    if args.windows.is_file():
        windows = load_windows(args.windows)
        windows_source = f"`{args.windows}` ({len(windows)} janelas)"
    elif args.fake:
        windows = demo_windows()
        windows_source = (
            "**JANELAS DE DEMONSTRAÇÃO** — a Etapa 1 ainda não produziu "
            f"`{args.windows}`. As taxas abaixo NÃO são do piloto."
        )
        print(f"[aviso] {args.windows} não existe; usando janelas de demonstração.")
    else:
        raise SystemExit(
            f"Janelas não encontradas em {args.windows}. Rode a Etapa 1 "
            "(`scripts/01_build_windows.py`) ou use `--fake`."
        )

    if args.all_speakers:
        selected = windows
        pilot_note = "todos os falantes (`--all-speakers`)"
    else:
        speakers, available = select_pilot_speakers(windows, config.pilot)
        selected = tuple(w for w in windows if w.speaker_code in set(speakers))
        pilot_note = (
            f"piloto: {len(speakers)} falante(s) "
            f"({config.pilot.n_speakers_per_region} por região, idade "
            f"{config.pilot.age_min}–{config.pilot.age_max}); "
            f"elegíveis por região: {available}"
        )
        if not selected:
            raise SystemExit(
                "Nenhum falante elegível para o piloto com "
                f"regions={list(config.pilot.regions)} e idade "
                f"{config.pilot.age_min}–{config.pilot.age_max}. Use `--all-speakers` "
                "ou revise `pilot` em items.yaml."
            )
    print(f"[info] {pilot_note}")

    model = build_generator(config, fake=args.fake)
    regions = {w.speaker_code: w.region for w in selected}
    outcome = generate_items(selected, model=model, config=config, region_by_speaker=regions)

    out = Path(args.outputs)
    write_jsonl(out / ITEMS_FILENAME, outcome.items)
    write_jsonl(out / ATTEMPTS_FILENAME, outcome.attempts)

    stages = load_stage_summaries(out / STAGE_SUMMARY_FILENAME)
    stages["generation"] = outcome.summary.to_dict()
    stages["regions"] = dict(sorted(regions.items()))
    stages["windows_source"] = f"{windows_source} — {pilot_note}"
    # A geração invalida o que veio depois dela: manter o filtro e a revisão de
    # uma geração anterior produziria um relatório internamente inconsistente.
    stages.pop("filter", None)
    stages.pop("review", None)
    save_stage_summaries(out / STAGE_SUMMARY_FILENAME, stages)

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    report = regenerate_report(out, config=config, generated_at=generated_at)

    print("\n".join(outcome.ledger.to_markdown()))
    print(f"[ok] {len(outcome.items)} item(ns) em {out / ITEMS_FILENAME}")
    print(f"[ok] relatório em {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
