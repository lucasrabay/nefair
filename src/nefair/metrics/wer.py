"""WER por janela e agregado, em contagens brutas (Etapa 6).

O ponto crítico deste módulo não é calcular `(S + I + D) / N` — `jiwer` faz isso.
É **como se agrega**. Há duas formas de resumir o WER de um conjunto de janelas:

- **`pooled` (soma/soma)**: `soma(S + I + D) / soma(N)`. É o WER do conjunto
  tratado como um único texto longo. É a definição usada na literatura de ASR e
  é o que este TCC reporta.
- **`macro` (média das razões)**: `média_j(WER_j)`. Dá a cada janela o mesmo
  peso, independentemente do tamanho. Como as janelas têm de 30 a 60 s e número
  variável de palavras, uma janela curta com 3 palavras e 1 erro (WER = 0,33)
  pesaria tanto quanto uma janela longa com 150 palavras e 5 erros (WER = 0,03).

As duas são calculadas e ficam lado a lado no relatório **de propósito**: a
diferença entre elas é justamente o tipo de erro estatístico silencioso que passa
por uma suíte de testes verde. O que o texto do TCC chama de "WER" é sempre o
`pooled`.

Por isso `WerRecord` guarda S, I, D e N — e não a razão já calculada. Uma vez que
a razão é computada, a informação necessária para agregar corretamente já se
perdeu.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import jiwer
import pandas as pd

from nefair.metrics.normalize import NORMALIZER_VERSION, normalize_pair
from nefair.schema import ExclusionLedger, WerRecord

# Transformação do `jiwer`: apenas separar em palavras. TODA a normalização
# linguística já aconteceu em `metrics/normalize.py` (D7) — deixar o `jiwer`
# aplicar os defaults dele por cima significaria duas camadas de normalização,
# uma delas invisível no relatório e pensada para o inglês.
_SPLIT_ONLY = jiwer.Compose(
    [
        jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
    ]
)


class EmptyReferenceError(ValueError):
    """Referência sem nenhuma palavra após a normalização.

    Não é `ZeroDivisionError`: `N == 0` torna o WER da janela **indefinido**, não
    infinito nem zero. A janela precisa ser contada numa trilha de exclusão e
    excluída explicitamente, nunca imputada.
    """


@dataclass(frozen=True)
class EditCounts:
    """Contagens brutas de uma comparação referência × hipótese."""

    substitutions: int
    insertions: int
    deletions: int
    n_reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def wer(self) -> float:
        if self.n_reference_words == 0:
            raise EmptyReferenceError("Referência vazia: WER indefinido.")
        return self.errors / self.n_reference_words


def count_edits(
    reference: str, hypothesis: str, *, version: str = NORMALIZER_VERSION
) -> EditCounts:
    """Conta S, I, D e N de um par, aplicando o MESMO normalizador aos dois lados.

    A normalização acontece aqui dentro (via `normalize_pair`) e não no chamador:
    é a única maneira de garantir estruturalmente que ninguém compare um lado
    normalizado com um lado cru.
    """
    ref_norm, hyp_norm = normalize_pair(reference, hypothesis, version)
    if not ref_norm:
        raise EmptyReferenceError(
            "Referência vazia após a normalização "
            f"(versão '{version}'); a janela deve ser contada e excluída."
        )
    if not hyp_norm:
        # Hipótese vazia é legítima (o ASR não reconheceu nada): tudo vira
        # deleção. `jiwer` não aceita string vazia, então a contagem é direta.
        n_ref = len(ref_norm.split())
        return EditCounts(
            substitutions=0, insertions=0, deletions=n_ref, n_reference_words=n_ref
        )
    output = jiwer.process_words(
        ref_norm,
        hyp_norm,
        reference_transform=_SPLIT_ONLY,
        hypothesis_transform=_SPLIT_ONLY,
    )
    n_ref = output.substitutions + output.deletions + output.hits
    return EditCounts(
        substitutions=int(output.substitutions),
        insertions=int(output.insertions),
        deletions=int(output.deletions),
        n_reference_words=int(n_ref),
    )


@dataclass(frozen=True)
class HypothesisRow:
    """Uma hipótese de ASR para uma janela.

    Contrato mínimo de entrada do WER. Não existe dataclass equivalente em
    `schema.py` (o `RunRecord` guarda o hash da hipótese, não o texto), então
    este módulo declara a forma que espera e a recebe por parâmetro, em vez de
    importar de `nefair.run`.
    """

    window_id: str
    speaker_code: str
    model_key: str
    hypothesis: str


def compute_wer_records(
    hypotheses: Iterable[HypothesisRow],
    references: dict[str, str],
    *,
    version: str = NORMALIZER_VERSION,
) -> tuple[list[WerRecord], ExclusionLedger]:
    """Calcula um `WerRecord` por (janela, modelo), com trilha de exclusão.

    Nada é dropado em silêncio: janela sem referência conhecida e referência
    vazia após a normalização viram linhas contadas no `ExclusionLedger`, com
    motivos distintos — as duas situações têm causas diferentes (falha de junção
    vs. janela degenerada) e o relatório precisa distingui-las.

    A ordenação da saída é canônica (`model_key`, `window_id`) para que dois
    relatórios gerados da mesma entrada sejam idênticos byte a byte.
    """
    ledger = ExclusionLedger(unit="janela × modelo")
    records: list[WerRecord] = []
    for row in hypotheses:
        ledger.total_in += 1
        reference = references.get(row.window_id)
        if reference is None:
            ledger.drop("janela sem transcrição de referência")
            continue
        try:
            counts = count_edits(reference, row.hypothesis, version=version)
        except EmptyReferenceError:
            ledger.drop("referência vazia após normalização")
            continue
        records.append(
            WerRecord(
                window_id=row.window_id,
                speaker_code=row.speaker_code,
                model_key=row.model_key,
                substitutions=counts.substitutions,
                insertions=counts.insertions,
                deletions=counts.deletions,
                n_reference_words=counts.n_reference_words,
                normalizer_version=version,
            )
        )
        ledger.kept += 1
    ledger.assert_balanced()
    records.sort(key=lambda r: (r.model_key, r.window_id))
    return records, ledger


# --------------------------------------------------------------------------- #
# Agregação
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WerAggregate:
    """WER agregado de um conjunto de janelas, nas duas formas de agregação.

    `wer_pooled` é o número que vai ao texto. `wer_macro` só existe para que a
    diferença entre as duas agregações seja visível no relatório em vez de ser
    uma decisão implícita de quem escreveu o código.
    """

    n_windows: int
    n_speakers: int
    substitutions: int
    insertions: int
    deletions: int
    n_reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def wer_pooled(self) -> float:
        """`soma(S + I + D) / soma(N)` — a definição usada no TCC."""
        if self.n_reference_words == 0:
            raise EmptyReferenceError(
                "Conjunto sem nenhuma palavra de referência: WER agregado indefinido."
            )
        return self.errors / self.n_reference_words


def aggregate_wer(records: Sequence[WerRecord]) -> WerAggregate:
    """Agrega por SOMA DE CONTAGENS (a forma correta)."""
    if not records:
        raise ValueError("aggregate_wer recebeu conjunto vazio; nada a agregar.")
    return WerAggregate(
        n_windows=len(records),
        n_speakers=len({r.speaker_code for r in records}),
        substitutions=sum(r.substitutions for r in records),
        insertions=sum(r.insertions for r in records),
        deletions=sum(r.deletions for r in records),
        n_reference_words=sum(r.n_reference_words for r in records),
    )


def macro_average_wer(records: Sequence[WerRecord]) -> float:
    """Média das razões por janela — **NÃO** é o WER reportado.

    Existe para ser comparada com `aggregate_wer(...).wer_pooled` no relatório.
    Janelas curtas recebem aqui o mesmo peso de janelas longas, o que enviesa o
    resumo sempre que o comprimento da janela se correlaciona com a dificuldade
    (e, num corpus de fala espontânea, ele se correlaciona).
    """
    if not records:
        raise ValueError("macro_average_wer recebeu conjunto vazio; nada a agregar.")
    return sum(r.wer for r in records) / len(records)


def wer_table(
    records: Sequence[WerRecord],
    speaker_region: dict[str, str],
    *,
    group_by: Sequence[str] = ("model_key", "region"),
) -> pd.DataFrame:
    """Tabela de WER agregado por grupo, com as duas agregações lado a lado.

    `speaker_region` mapeia falante → região; falante ausente do mapa recebe a
    região `desconhecida` e aparece na tabela (nunca some).
    """
    if not records:
        raise ValueError("wer_table recebeu conjunto vazio.")
    frame = pd.DataFrame(
        {
            "window_id": [r.window_id for r in records],
            "speaker_code": [r.speaker_code for r in records],
            "model_key": [r.model_key for r in records],
            "region": [speaker_region.get(r.speaker_code, "desconhecida") for r in records],
            "substitutions": [r.substitutions for r in records],
            "insertions": [r.insertions for r in records],
            "deletions": [r.deletions for r in records],
            "n_reference_words": [r.n_reference_words for r in records],
            "wer_window": [r.wer for r in records],
        }
    )
    keys = list(group_by)
    grouped = frame.groupby(keys, dropna=False, sort=True)
    out = grouped.agg(
        n_windows=("window_id", "size"),
        n_speakers=("speaker_code", "nunique"),
        substitutions=("substitutions", "sum"),
        insertions=("insertions", "sum"),
        deletions=("deletions", "sum"),
        n_reference_words=("n_reference_words", "sum"),
        wer_macro=("wer_window", "mean"),
    ).reset_index()
    errors = out["substitutions"] + out["insertions"] + out["deletions"]
    out["errors"] = errors
    out["wer_pooled"] = errors / out["n_reference_words"]
    column_order = keys + [
        "n_windows",
        "n_speakers",
        "substitutions",
        "insertions",
        "deletions",
        "errors",
        "n_reference_words",
        "wer_pooled",
        "wer_macro",
    ]
    return out[column_order].sort_values(keys).reset_index(drop=True)
