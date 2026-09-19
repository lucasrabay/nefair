"""Planilha de revisão humana e relatório da Etapa 3.

O texto do TCC exige revisão humana de **todo** item. O código não faz a revisão:
ele entrega uma planilha estável, recebe a planilha preenchida de volta e se
recusa a aceitar qualquer coisa ambígua.

Por que a exportação é tão rígida (ordem de colunas e de linhas fixa): a planilha
é editada fora do repositório, em outra ferramenta, possivelmente em dias
diferentes. Se a ordem das linhas mudasse a cada exportação, reexportar no meio
da revisão embaralharia o trabalho já feito. Com ordem fixa, reexportar é
idempotente.

Por que a importação **falha** em vez de avisar: um item não revisado que passe
silenciosamente entra na Etapa 5 sem ter sido validado por ninguém, e o texto
passa a afirmar algo que não aconteceu. Item desconhecido na planilha é o mesmo
problema visto de trás: ou a planilha é de outra execução, ou alguém editou um
`item_id` à mão. Nos dois casos, parar é mais barato do que descobrir depois.

Aqui também mora o gerador de `outputs/items_report.md`, porque o relatório
precisa dos três estágios (geração, filtro, revisão) ao mesmo tempo — e é o
número de **minutos por item** que decide se a revisão de ~2.300 itens cabe no
cronograma (o gate do piloto).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nefair.config import ItemsConfig
from nefair.items.filter import FILTER_STATUS_KEPT, FilterSummary
from nefair.items.generate import GenerationSummary
from nefair.models.prompts import PROMPT_VERSION
from nefair.schema import ExclusionLedger, Item

REVIEW_STATUS_ACCEPTED = "accepted"
REVIEW_STATUS_REJECTED = "rejected"
LEDGER_UNIT = "itens (revisão humana)"

# Colunas que o revisor preenche. Ficam por último na planilha, nessa ordem.
DECISION_COLUMN = "review_decision"
REASON_COLUMN = "review_reason"
MINUTES_COLUMN = "review_minutes"
EDITABLE_COLUMNS: tuple[str, ...] = (DECISION_COLUMN, REASON_COLUMN, MINUTES_COLUMN)


def sheet_columns(n_alternatives: int) -> tuple[str, ...]:
    """Ordem canônica das colunas da planilha, para `n_alternatives` alternativas.

    Derivada do nº de alternativas (D10) em vez de escrita à mão: regerar itens
    com 3 ou 5 alternativas não pode exigir edição deste módulo.
    """
    from nefair.models.prompts import labels_for

    columns: list[str] = ["item_id", "window_id", "speaker_code", "region", "question"]
    for label in labels_for(n_alternatives):
        columns.extend(
            [f"alt_{label}_text", f"alt_{label}_is_correct", f"alt_{label}_source_window_id"]
        )
    columns.extend(
        ["correct_label", "generator_model", "prompt_version", "filter_status", *EDITABLE_COLUMNS]
    )
    return tuple(columns)


# --------------------------------------------------------------------------- #
# Exportação
# --------------------------------------------------------------------------- #
def export_review_sheet(
    items: Sequence[Item],
    path: str | Path,
    *,
    region_by_speaker: Mapping[str, str] | None = None,
    only_filter_kept: bool = True,
) -> int:
    """Escreve a planilha de revisão e devolve o nº de linhas escritas.

    Por padrão exporta **só** os itens que sobreviveram ao controle textual: o
    plano é explícito em revisar depois do filtro justamente para reduzir a
    carga humana, que é o caminho crítico do cronograma.

    O CSV é escrito com `\\n` e sem BOM para que duas exportações da mesma
    entrada sejam idênticas byte a byte.
    """
    regions = dict(region_by_speaker or {})
    selected = [
        item for item in items if (item.filter_status == FILTER_STATUS_KEPT or not only_filter_kept)
    ]
    selected.sort(key=lambda it: it.item_id)
    if not selected:
        raise ValueError(
            "Nenhum item a revisar: a planilha vazia esconderia a diferença entre "
            "'o filtro derrubou tudo' e 'a etapa não rodou'."
        )

    n_alternatives = selected[0].n_alternatives
    columns = sheet_columns(n_alternatives)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for item in selected:
            if item.n_alternatives != n_alternatives:
                raise ValueError(
                    f"Item {item.item_id} tem {item.n_alternatives} alternativas, mas a "
                    f"planilha foi aberta com {n_alternatives}. Planilha de largura "
                    "variável não é revisável."
                )
            row: dict[str, Any] = {
                "item_id": item.item_id,
                "window_id": item.window_id,
                "speaker_code": item.speaker_code,
                "region": regions.get(item.speaker_code, "desconhecida"),
                "question": item.question,
                "correct_label": item.correct_label,
                "generator_model": item.generator_model,
                "prompt_version": item.prompt_version,
                "filter_status": item.filter_status,
                DECISION_COLUMN: "",
                REASON_COLUMN: "",
                MINUTES_COLUMN: "",
            }
            for alt in item.alternatives:
                row[f"alt_{alt.label}_text"] = alt.text
                row[f"alt_{alt.label}_is_correct"] = "true" if alt.is_correct else "false"
                row[f"alt_{alt.label}_source_window_id"] = alt.source_window_id or ""
            writer.writerow(row)
    return len(selected)


# --------------------------------------------------------------------------- #
# Importação
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegionReviewRate:
    """Taxa de rejeição na revisão, por região."""

    region: str
    n_items: int
    n_rejected: int
    total_minutes: float

    @property
    def rejection_rate(self) -> float:
        return self.n_rejected / self.n_items if self.n_items else 0.0

    @property
    def minutes_per_item(self) -> float:
        return self.total_minutes / self.n_items if self.n_items else 0.0


@dataclass(frozen=True)
class ReviewSummary:
    """Números da revisão humana — a entrada do gate do piloto."""

    n_items_in: int
    n_accepted: int
    n_rejected: int
    total_minutes: float
    decisions_vocabulary: tuple[str, ...]
    by_region: tuple[RegionReviewRate, ...]

    @property
    def minutes_per_item(self) -> float:
        return self.total_minutes / self.n_items_in if self.n_items_in else 0.0

    @property
    def rejection_rate(self) -> float:
        return self.n_rejected / self.n_items_in if self.n_items_in else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items_in": self.n_items_in,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "total_minutes": self.total_minutes,
            "decisions_vocabulary": list(self.decisions_vocabulary),
            "by_region": [
                {
                    "region": r.region,
                    "n_items": r.n_items,
                    "n_rejected": r.n_rejected,
                    "total_minutes": r.total_minutes,
                }
                for r in self.by_region
            ],
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> ReviewSummary:
        return ReviewSummary(
            n_items_in=int(payload["n_items_in"]),
            n_accepted=int(payload["n_accepted"]),
            n_rejected=int(payload["n_rejected"]),
            total_minutes=float(payload["total_minutes"]),
            decisions_vocabulary=tuple(str(d) for d in payload["decisions_vocabulary"]),
            by_region=tuple(
                RegionReviewRate(
                    region=str(r["region"]),
                    n_items=int(r["n_items"]),
                    n_rejected=int(r["n_rejected"]),
                    total_minutes=float(r["total_minutes"]),
                )
                for r in payload["by_region"]
            ),
        )


@dataclass(frozen=True)
class ReviewOutcome:
    """Itens com a revisão aplicada + trilha de exclusão + agregados."""

    items: tuple[Item, ...]
    ledger: ExclusionLedger
    summary: ReviewSummary

    @property
    def accepted_items(self) -> tuple[Item, ...]:
        return tuple(it for it in self.items if it.review_status == REVIEW_STATUS_ACCEPTED)


def _parse_minutes(raw: str, item_id: str) -> float:
    """Minutos gastos na revisão de um item — obrigatório e numérico.

    É o número que projeta o custo da revisão completa (~2.300 itens) e decide o
    gate do piloto. Aceitar vazio faria a projeção ser calculada sobre uma
    amostra silenciosamente enviesada (só os itens cujo tempo alguém anotou).
    """
    text = (raw or "").strip().replace(",", ".")
    if not text:
        raise ValueError(
            f"Planilha de revisão: item '{item_id}' sem '{MINUTES_COLUMN}'. O tempo por "
            "item é o que projeta o custo da revisão completa; não pode ficar em branco."
        )
    try:
        minutes = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Planilha de revisão: item '{item_id}' com '{MINUTES_COLUMN}' não numérico ({raw!r})."
        ) from exc
    if minutes < 0:
        raise ValueError(
            f"Planilha de revisão: item '{item_id}' com minutos negativos ({minutes})."
        )
    return minutes


def import_review_sheet(
    path: str | Path,
    items: Sequence[Item],
    *,
    review_decisions: Sequence[str],
    region_by_speaker: Mapping[str, str] | None = None,
) -> ReviewOutcome:
    """Lê a planilha preenchida, concilia por `item_id` e aplica as decisões.

    Falha — nunca avisa e segue — quando:

    - falta alguma coluna obrigatória;
    - há `item_id` repetido;
    - há `item_id` que não existe entre os itens revisáveis;
    - há item revisável ausente da planilha (item não revisado);
    - a decisão está fora de `review_decisions` (vocabulário fechado do config);
    - `review_minutes` está vazio, não numérico ou negativo.
    """
    allowed = tuple(review_decisions)
    if REVIEW_STATUS_ACCEPTED not in allowed or REVIEW_STATUS_REJECTED not in allowed:
        raise ValueError(
            f"items.review_decisions={list(allowed)} precisa conter "
            f"'{REVIEW_STATUS_ACCEPTED}' e '{REVIEW_STATUS_REJECTED}': o pipeline "
            "distingue item aceito de item rejeitado."
        )
    regions = dict(region_by_speaker or {})
    by_id = {item.item_id: item for item in items}

    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [c for c in ("item_id", *EDITABLE_COLUMNS) if c not in fieldnames]
        if missing:
            raise ValueError(
                f"Planilha de revisão {path}: coluna(s) obrigatória(s) ausente(s): {missing}. "
                f"Colunas encontradas: {list(fieldnames)}."
            )
        rows = list(reader)

    seen: dict[str, dict[str, str]] = {}
    for row in rows:
        item_id = (row.get("item_id") or "").strip()
        if not item_id:
            raise ValueError(f"Planilha de revisão {path}: linha com 'item_id' vazio.")
        if item_id in seen:
            raise ValueError(f"Planilha de revisão {path}: 'item_id' repetido: '{item_id}'.")
        if item_id not in by_id:
            raise ValueError(
                f"Planilha de revisão {path}: item desconhecido '{item_id}'. A planilha "
                "não corresponde ao conjunto de itens desta execução."
            )
        seen[item_id] = row

    not_reviewed = sorted(set(by_id) - set(seen))
    if not_reviewed:
        raise ValueError(
            f"Planilha de revisão {path}: {len(not_reviewed)} item(ns) não revisado(s) "
            f"(ex.: {not_reviewed[:5]}). O texto exige revisão humana de TODO item."
        )

    ledger = ExclusionLedger(unit=LEDGER_UNIT, total_in=len(by_id))
    reviewed: list[Item] = []
    region_totals: dict[str, list[float]] = {}
    total_minutes = 0.0

    for item_id in sorted(by_id):
        row = seen[item_id]
        decision = (row.get(DECISION_COLUMN) or "").strip()
        if decision not in allowed:
            raise ValueError(
                f"Planilha de revisão {path}: item '{item_id}' com decisão "
                f"'{decision}' fora do vocabulário {list(allowed)}."
            )
        minutes = _parse_minutes(row.get(MINUTES_COLUMN, ""), item_id)
        reason = (row.get(REASON_COLUMN) or "").strip()
        if decision == REVIEW_STATUS_REJECTED and not reason:
            raise ValueError(
                f"Planilha de revisão {path}: item '{item_id}' rejeitado sem "
                f"'{REASON_COLUMN}'. A distribuição dos motivos de rejeição é um "
                "resultado da Etapa 3, não um detalhe administrativo."
            )
        item = by_id[item_id]
        reviewed.append(
            replace(
                item,
                review_status=decision,
                review_reason=reason,
                review_minutes=minutes,
            )
        )
        total_minutes += minutes
        region = regions.get(item.speaker_code, "desconhecida")
        bucket = region_totals.setdefault(region, [0.0, 0.0, 0.0])
        bucket[0] += 1
        if decision == REVIEW_STATUS_REJECTED:
            bucket[1] += 1
            ledger.drop(f"revisao_rejeitada:{reason or 'sem_motivo'}")
        bucket[2] += minutes

    ledger.kept = sum(1 for it in reviewed if it.review_status == REVIEW_STATUS_ACCEPTED)
    ledger.assert_balanced()

    summary = ReviewSummary(
        n_items_in=len(reviewed),
        n_accepted=ledger.kept,
        n_rejected=ledger.total_dropped,
        total_minutes=total_minutes,
        decisions_vocabulary=allowed,
        by_region=tuple(
            RegionReviewRate(
                region=region,
                n_items=int(counts[0]),
                n_rejected=int(counts[1]),
                total_minutes=counts[2],
            )
            for region, counts in sorted(region_totals.items())
        ),
    )
    return ReviewOutcome(items=tuple(reviewed), ledger=ledger, summary=summary)


# --------------------------------------------------------------------------- #
# Estado entre estágios
#
# Os scripts 03, 04 e 05 rodam em processos separados, possivelmente em dias
# diferentes. Este arquivo é a memória compartilhada entre eles — e é o que
# permite que qualquer um dos três regenere o relatório completo com as seções
# dos estágios que já rodaram.
# --------------------------------------------------------------------------- #
STAGE_SUMMARY_FILENAME = "items_stage_summaries.json"


def load_stage_summaries(path: str | Path) -> dict[str, Any]:
    """Lê o JSON de estágios; devolve `{}` se ainda não existir."""
    file = Path(path)
    if not file.is_file():
        return {}
    payload = json.loads(file.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def save_stage_summaries(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Escreve o JSON de estágios de forma determinística (chaves ordenadas)."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Relatório `outputs/items_report.md`
# --------------------------------------------------------------------------- #
def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%".replace(".", ",")


def _num(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _config_echo_lines(config: ItemsConfig) -> list[str]:
    """Valor EFETIVO de cada chave que materializa D2, D3 e D11.

    O plano é explícito: decisão pendente nunca vira default escondido. Ecoar o
    valor usado é o que permite ler o relatório de uma execução antiga e saber
    sob quais decisões aqueles números foram produzidos.
    """
    flt = config.text_filter
    return [
        "## Valores efetivos de configuração (D2, D3, D11)",
        "",
        "| decisão | chave | valor efetivo |",
        "|---|---|---|",
        f"| D11 | `generator.model_id` | `{config.generator.model_id}` |",
        f"| D11 | `generator.version` | `{config.generator.version}` |",
        f"| D11 | `generator.provider` | `{config.generator.provider}` |",
        f"| D2 | `distractors.source` | `{config.distractors.source}` |",
        f"| D2 | `distractors.require_provenance` | `{config.distractors.require_provenance}` |",
        f"| D2 | `distractors.n_context_windows` | `{config.distractors.n_context_windows}` |",
        f"| D3 | `text_filter.models` | `{list(flt.models)}` |",
        f"| D3 | `text_filter.temperature` | `{flt.temperature}` |",
        f"| D3 | `text_filter.n_runs` | `{flt.n_runs}` |",
        f"| D3 | `text_filter.discard_if_correct_at_least` | `{flt.discard_if_correct_at_least}` |",
        f"| D3 | `text_filter.shuffle_alternatives` | `{flt.shuffle_alternatives}` |",
        f"| D3 | `text_filter.any_model_discards` | `{flt.any_model_discards}` |",
        f"| D3 | `text_filter.seed` | `{flt.seed}` |",
        f"| — | `seed` | `{config.seed}` |",
        f"| — | `n_items_per_window` | `{config.n_items_per_window}` |",
        f"| D10 | `n_alternatives` | `{config.n_alternatives}` |",
        f"| — | `review_decisions` | `{list(config.review_decisions)}` |",
        f"| — | `prompt_version` | `{PROMPT_VERSION}` |",
        "",
    ]


def _generation_lines(summary: GenerationSummary) -> list[str]:
    lines = [
        "## Geração (D2, D11)",
        "",
        f"- Gerador: `{summary.generator_model_id}` (versão `{summary.generator_version}`), "
        f"prompt `{summary.prompt_version}`",
        f"- Falantes: {summary.n_speakers} — janelas: {summary.n_windows}",
        f"- Itens pedidos: {summary.n_requested} — itens bem formados: {summary.n_items}",
        f"- Taxa de itens bem formados: "
        f"{_pct(summary.n_items / summary.n_requested if summary.n_requested else 0.0)}",
        f"- Janelas sem nenhum trecho de contexto: {summary.n_windows_without_context}",
        f"- Janelas com menos de {summary.n_context_windows} trechos: "
        f"{summary.n_windows_with_partial_context}",
        "",
        "Motivos de descarte na geração (nada dropado em silêncio):",
        "",
        "| motivo | itens |",
        "|---|--:|",
    ]
    for reason in sorted(summary.dropped):
        lines.append(f"| `{reason}` | {summary.dropped[reason]} |")
    if not summary.dropped:
        lines.append("| (nenhum) | 0 |")
    lines.append(
        f"| **subtotal descartado** | **{sum(summary.dropped.values())}** |",
    )
    lines.append(f"| **retidos** | **{summary.n_items}** |")
    lines.append(f"| **conferência (pedidos)** | **{summary.n_requested}** |")
    lines.append("")
    lines.append("Itens gerados por região:")
    lines.append("")
    lines.append("| região | itens |")
    lines.append("|---|--:|")
    for row in summary.by_region:
        lines.append(f"| {row.region} | {row.n_items} |")
    lines.append("")
    return lines


def _filter_lines(summary: FilterSummary) -> list[str]:
    chance = summary.chance_discard_rate
    lines = [
        "## Controle textual (D3)",
        "",
        "O modelo viu **apenas** a pergunta e as alternativas: sem áudio e sem "
        "transcrição. Item respondido nessas condições não mede compreensão de fala.",
        "",
        f"- Itens avaliados: {summary.n_items_in}",
        f"- Descartados: {summary.n_discarded} — **taxa de descarte: "
        f"{_pct(summary.discard_rate)}**",
        f"- Retidos: {summary.n_kept}",
        f"- Execuções: {summary.n_total_runs} "
        f"({summary.n_runs} por modelo × {len(summary.models)} modelo(s) × itens)",
        f"- Respostas não interpretáveis: {summary.n_unparsed_runs} "
        f"({_pct(summary.n_unparsed_runs / summary.n_total_runs if summary.n_total_runs else 0.0)}"
        ") — contadas à parte, nunca imputadas como acerto ou erro",
        "",
        "### Linha de base de acaso (calculada, não fixada no texto)",
        "",
        f"Sob resposta aleatória com `p = 1/{summary.n_alternatives} = "
        f"{_num(1.0 / summary.n_alternatives if summary.n_alternatives else 0.0, 4)}`, a "
        f"probabilidade de acertar ≥ {summary.discard_if_correct_at_least} de "
        f"{summary.n_runs} execuções é",
        "",
        f"`sum_i=k..n C(n,i) p^i (1-p)^(n-i)` = **{_pct(chance)}** (valor exato: {chance:.6f}).",
        "",
        "Ou seja: mesmo que **todos** os itens exigissem de fato a fala, o filtro "
        f"descartaria ~{_pct(chance)} deles por puro acaso. A taxa de descarte "
        f"observada ({_pct(summary.discard_rate)}) só é interpretável ao lado desse "
        "número: descarte próximo da linha de base significa que quase nada vazou; "
        "descarte muito acima dela é vazamento real.",
        "",
        "### Taxa de descarte por região",
        "",
        "| região | itens | descartados | taxa |",
        "|---|--:|--:|--:|",
    ]
    for row in summary.by_region:
        lines.append(
            f"| {row.region} | {row.n_items} | {row.n_discarded} | {_pct(row.discard_rate)} |"
        )
    lines.append("")
    if len(summary.by_region) == 2:
        first, second = summary.by_region
        gap = abs(first.discard_rate - second.discard_rate)
        lines.append(
            f"Diferença entre {first.region} e {second.region}: "
            f"{_num(100.0 * gap, 1)} pontos percentuais. O plano pede que não haja "
            "diferença **grosseira** de descarte entre as regiões, mas não fixa o "
            "limiar — a leitura desse número é decisão do autor, e por isso ele é "
            "reportado em vez de ser transformado num veredito automático."
        )
        lines.append("")
    lines.append("### Quantos itens cada modelo sozinho derrubaria")
    lines.append("")
    lines.append(
        f"`any_model_discards = {summary.any_model_discards}`: "
        + (
            "basta UM modelo passar no critério para o item cair."
            if summary.any_model_discards
            else "o item só cai se TODOS os modelos passarem no critério."
        )
    )
    lines.append("")
    lines.append("| modelo | itens | atinge o critério | taxa |")
    lines.append("|---|--:|--:|--:|")
    for row in summary.by_model:
        lines.append(
            f"| `{row.model_key}` | {row.n_items} | {row.n_triggering} | {_pct(row.trigger_rate)} |"
        )
    lines.append("")
    return lines


def _review_lines(summary: ReviewSummary) -> list[str]:
    lines = [
        "## Revisão humana",
        "",
        f"- Itens revisados: {summary.n_items_in} (apenas os que sobreviveram ao controle textual)",
        f"- Aceitos: {summary.n_accepted} — rejeitados: {summary.n_rejected} "
        f"(**taxa de rejeição: {_pct(summary.rejection_rate)}**)",
        f"- Tempo total: {_num(summary.total_minutes, 1)} min — "
        f"**{_num(summary.minutes_per_item, 2)} min por item**",
        f"- Vocabulário de decisão aceito: `{list(summary.decisions_vocabulary)}`",
        "",
        "### Taxa de rejeição e minutos por item, por região",
        "",
        "| região | itens | rejeitados | taxa de rejeição | min/item |",
        "|---|--:|--:|--:|--:|",
    ]
    for row in summary.by_region:
        lines.append(
            f"| {row.region} | {row.n_items} | {row.n_rejected} | "
            f"{_pct(row.rejection_rate)} | {_num(row.minutes_per_item, 2)} |"
        )
    lines.append("")
    return lines


def _projection_lines(
    generation: GenerationSummary,
    filtering: FilterSummary | None,
    review: ReviewSummary | None,
    projection_speakers: int | None,
) -> list[str]:
    """Projeção do piloto para o conjunto completo — o número do gate.

    Tudo é expresso **por falante** primeiro. A projeção absoluta exige saber
    quantos falantes o conjunto completo terá, e esse número vem da Etapa 1, não
    daqui: se ele não for informado, o relatório diz isso em vez de inventar.
    """
    lines = ["## Projeção para o conjunto completo (gate do piloto)", ""]
    n_speakers = generation.n_speakers
    if n_speakers == 0:
        lines.append("Sem falantes no piloto; projeção indefinida.")
        lines.append("")
        return lines

    requested_per_speaker = generation.n_requested / n_speakers
    wellformed_rate = generation.n_items / generation.n_requested if generation.n_requested else 0.0
    keep_rate = (
        filtering.n_kept / filtering.n_items_in
        if filtering is not None and filtering.n_items_in
        else None
    )
    accept_rate = (
        review.n_accepted / review.n_items_in if review is not None and review.n_items_in else None
    )
    minutes_per_item = review.minutes_per_item if review is not None else None

    lines.append("Taxas medidas no piloto:")
    lines.append("")
    lines.append("| taxa | valor |")
    lines.append("|---|--:|")
    lines.append(f"| itens pedidos por falante | {_num(requested_per_speaker, 2)} |")
    lines.append(f"| itens bem formados / pedidos | {_pct(wellformed_rate)} |")
    lines.append(
        "| sobrevivem ao filtro textual | "
        + (f"{_pct(keep_rate)} |" if keep_rate is not None else "(filtro não executado) |")
    )
    lines.append(
        "| aceitos na revisão | "
        + (f"{_pct(accept_rate)} |" if accept_rate is not None else "(revisão não executada) |")
    )
    lines.append(
        "| minutos por item revisado | "
        + (
            f"{_num(minutes_per_item, 2)} |"
            if minutes_per_item is not None
            else "(revisão não executada) |"
        )
    )
    survival = wellformed_rate * (keep_rate or 0.0) * (accept_rate or 0.0)
    if keep_rate is not None and accept_rate is not None:
        lines.append(f"| **sobrevivência ponta a ponta** | **{_pct(survival)}** |")
    lines.append("")

    if projection_speakers is None or projection_speakers <= 0:
        lines.append(
            "**Projeção absoluta não calculada:** o nº de falantes do conjunto "
            "completo é uma saída da Etapa 1 (janelas), não desta etapa. Rode o "
            "script `05` com `--full-speakers N` para obter os totais. As taxas "
            "acima são o que o piloto mede; multiplicá-las por um N inventado aqui "
            "seria transformar uma suposição em resultado."
        )
        lines.append("")
        return lines

    projected_requested = projection_speakers * requested_per_speaker
    projected_items = projected_requested * wellformed_rate
    projected_to_review = projected_items * (keep_rate if keep_rate is not None else 1.0)
    projected_accepted = projected_to_review * (accept_rate if accept_rate is not None else 1.0)
    lines.append(f"Projeção para **{projection_speakers} falantes**:")
    lines.append("")
    lines.append("| grandeza | projeção |")
    lines.append("|---|--:|")
    lines.append(f"| itens pedidos ao gerador | {_num(projected_requested, 0)} |")
    lines.append(f"| itens bem formados | {_num(projected_items, 0)} |")
    lines.append(f"| itens que chegam à revisão humana | {_num(projected_to_review, 0)} |")
    lines.append(f"| itens aceitos (entram na Etapa 5) | {_num(projected_accepted, 0)} |")
    if minutes_per_item is not None:
        total_minutes = projected_to_review * minutes_per_item
        lines.append(
            f"| **tempo de revisão humana** | **{_num(total_minutes / 60.0, 1)} h** "
            f"({_num(total_minutes, 0)} min) |"
        )
    lines.append("")
    if minutes_per_item is None:
        lines.append(
            "O tempo de revisão projetado — a condição (c) do gate — exige a "
            "planilha de revisão preenchida (`scripts/05_review.py --mode import`)."
        )
        lines.append("")
    return lines


def build_items_report(
    *,
    config: ItemsConfig,
    generation: GenerationSummary | None,
    filtering: FilterSummary | None,
    review: ReviewSummary | None,
    generated_at: str,
    projection_speakers: int | None = None,
    windows_source: str = "",
) -> str:
    """Monta `outputs/items_report.md` a partir dos estágios que já rodaram.

    O relatório é função dos artefatos em disco: rodar só a geração produz um
    relatório com a seção de filtro marcada como não executada, em vez de um
    relatório ausente. Assim o artefato existe desde a primeira execução e vai
    ganhando seções, e nunca há um estado em que a etapa rodou mas não há trilha.
    """
    lines: list[str] = []
    lines.append("# Etapa 3 — itens de múltipla escolha (geração, filtro e revisão)")
    lines.append("")
    lines.append(
        "Artefato determinístico da Etapa 3. Nenhum áudio foi usado em nenhum "
        "estágio deste relatório: a geração parte da transcrição de referência da "
        "janela e o controle textual parte apenas da pergunta e das alternativas."
    )
    lines.append("")
    lines.append(f"- **Gerado em (UTC):** {generated_at}")
    if windows_source:
        lines.append(f"- **Fonte das janelas:** {windows_source}")
    lines.append("")
    lines.extend(_config_echo_lines(config))

    if generation is None:
        lines.append("## Geração (D2, D11)")
        lines.append("")
        lines.append("(não executada — rode `scripts/03_generate_items.py`)")
        lines.append("")
    else:
        lines.extend(_generation_lines(generation))

    if filtering is None:
        lines.append("## Controle textual (D3)")
        lines.append("")
        lines.append("(não executado — rode `scripts/04_text_only_filter.py`)")
        lines.append("")
    else:
        lines.extend(_filter_lines(filtering))

    if review is None:
        lines.append("## Revisão humana")
        lines.append("")
        lines.append(
            "(não importada — exporte com `scripts/05_review.py --mode export`, "
            "preencha a planilha e importe com `--mode import`)"
        )
        lines.append("")
    else:
        lines.extend(_review_lines(review))

    if generation is not None:
        lines.extend(_projection_lines(generation, filtering, review, projection_speakers))

    lines.append("## Condições do gate do piloto")
    lines.append("")
    lines.append(
        "(a) taxa de descarte do filtro nem ~0 nem > 60%; (b) sem diferença "
        "grosseira de descarte entre NE e SE; (c) tempo de revisão projetado "
        "compatível com o cronograma. Os três números estão acima; o veredito é "
        "do autor, não do código."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


REPORT_FILENAME = "items_report.md"


def regenerate_report(
    outputs_dir: str | Path,
    *,
    config: ItemsConfig,
    generated_at: str,
    projection_speakers: int | None = None,
) -> Path:
    """Reescreve `outputs/items_report.md` a partir do JSON de estágios.

    Chamada pelos três scripts. O relatório é sempre derivado do estado em disco,
    nunca do que este processo por acaso tem na memória: assim rodar só o filtro
    não apaga a seção de geração escrita ontem.
    """
    out = Path(outputs_dir)
    stages = load_stage_summaries(out / STAGE_SUMMARY_FILENAME)
    generation = (
        GenerationSummary.from_dict(stages["generation"]) if "generation" in stages else None
    )
    filtering = FilterSummary.from_dict(stages["filter"]) if "filter" in stages else None
    review = ReviewSummary.from_dict(stages["review"]) if "review" in stages else None
    text = build_items_report(
        config=config,
        generation=generation,
        filtering=filtering,
        review=review,
        generated_at=generated_at,
        projection_speakers=projection_speakers,
        windows_source=str(stages.get("windows_source", "")),
    )
    path = out / REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
