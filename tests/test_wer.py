"""Testes do WER por janela e da AGREGAÇÃO (Etapa 6) — offline, sem corpus.

O cálculo de `(S + I + D) / N` numa janela é trabalho do `jiwer`. O que este
módulo pode errar em silêncio é o **resumo**: `soma(S+I+D)/soma(N)` (pooled) e
`média_j(WER_j)` (macro) são números diferentes, e trocar um pelo outro não
levanta exceção nenhuma — só muda o resultado do TCC.

Por isso o teste central daqui (`test_pooled_and_macro_diverge_...`) monta um
conjunto em que as duas agregações divergem por um fator de ~6 e traz **os dois
valores calculados à mão**. Uma suíte que só checasse "o WER agregado fica entre
0 e 1" passaria com a implementação errada.

Os demais testes cobrem as bordas que têm consequência metodológica:

- **Referência vazia** é `EmptyReferenceError`, não `ZeroDivisionError` nem 0.0:
  `N == 0` torna o WER da janela INDEFINIDO, e janela indefinida tem de ser
  contada na trilha de exclusão, nunca imputada.
- **Hipótese vazia** é legítima (o ASR não reconheceu nada) e vale `WER = 1.0`
  com tudo em deleção — confundir os dois casos esconderia falha de modelo.
- A normalização acontece DENTRO de `count_edits`, então `1985` na hipótese e
  `mil novecentos e oitenta e cinco` na referência custam zero erro.
"""

from __future__ import annotations

import pytest

from nefair.metrics.normalize import NORMALIZER_VERSION
from nefair.metrics.wer import (
    EditCounts,
    EmptyReferenceError,
    HypothesisRow,
    aggregate_wer,
    compute_wer_records,
    count_edits,
    macro_average_wer,
    wer_table,
)
from nefair.schema import WerRecord

# Motivos de descarte, literais como o módulo os grava — se um deles mudar de
# texto, o relatório da Etapa 6 muda junto e o teste tem de acusar.
DROP_SEM_REFERENCIA = "janela sem transcrição de referência"
DROP_REFERENCIA_VAZIA = "referência vazia após normalização"


def _record(
    window_id: str,
    speaker_code: str,
    *,
    s: int = 0,
    i: int = 0,
    d: int = 0,
    n: int,
    model_key: str = "asr",
) -> WerRecord:
    """Registro com contagens ESCOLHIDAS, para aritmética verificável à mão."""
    return WerRecord(
        window_id=window_id,
        speaker_code=speaker_code,
        model_key=model_key,
        substitutions=s,
        insertions=i,
        deletions=d,
        n_reference_words=n,
        normalizer_version=NORMALIZER_VERSION,
    )


# --------------------------------------------------------------------------- #
# `count_edits` — S, I, D e N conferidos alinhamento a alinhamento
# --------------------------------------------------------------------------- #
def test_count_edits_substitution():
    """`roça` → `rosa`: uma substituição, cinco palavras de referência."""
    counts = count_edits("a gente morava na roça", "a gente morava na rosa")
    assert counts == EditCounts(substitutions=1, insertions=0, deletions=0, n_reference_words=5)
    assert counts.errors == 1
    assert counts.wer == pytest.approx(1 / 5)


def test_count_edits_insertion():
    """O ASR inventou `muito`: uma inserção. N continua sendo o da REFERÊNCIA."""
    counts = count_edits("meu pai trabalhava", "meu pai muito trabalhava")
    assert counts == EditCounts(substitutions=0, insertions=1, deletions=0, n_reference_words=3)
    # WER pode passar de 1.0 justamente porque o denominador é N da referência.
    assert counts.wer == pytest.approx(1 / 3)


def test_count_edits_deletion():
    """O ASR comeu `trabalhava`: uma deleção."""
    counts = count_edits("meu pai trabalhava na roça", "meu pai na roça")
    assert counts == EditCounts(substitutions=0, insertions=0, deletions=1, n_reference_words=5)


def test_count_edits_mixes_the_three_kinds():
    """Um alinhamento com os três tipos ao mesmo tempo, conferido à mão.

    ref: o meu pai trabalhava ___   na roça todo dia   (8 palavras)
    hyp: o meu pai trabalhava muito na rosa ____ dia
                                  ↑ inserção  ↑ subst. ↑ deleção
    """
    counts = count_edits(
        "o meu pai trabalhava na roça todo dia",
        "o meu pai trabalhava muito na rosa dia",
    )
    assert counts == EditCounts(substitutions=1, insertions=1, deletions=1, n_reference_words=8)
    assert counts.wer == pytest.approx(3 / 8)


def test_count_edits_normalizes_both_sides_so_a_correct_asr_costs_nothing():
    """`1985` vs. `mil novecentos e oitenta e cinco`: acerto, não 1 S + 4 D.

    Sem a normalização simétrica dentro de `count_edits`, esta janela pagaria
    cinco erros por uma transcrição perfeita — o WER mediria o formato de saída
    do ASR, não o reconhecimento.
    """
    counts = count_edits(
        "a gente morava em mil novecentos e oitenta e cinco", "A gente morava em 1985."
    )
    assert counts.errors == 0
    assert counts.n_reference_words == 10


def test_count_edits_rejects_an_unknown_normalizer_version():
    with pytest.raises(ValueError, match="v9"):
        count_edits("a gente foi", "a gente foi", version="v9")


# --------------------------------------------------------------------------- #
# As duas bordas que NÃO podem ser confundidas
# --------------------------------------------------------------------------- #
def test_empty_reference_raises_instead_of_dividing_by_zero():
    """Referência vazia é indefinida, não zero nem infinito.

    O caso realista não é a string vazia: é a referência que fica vazia DEPOIS da
    normalização — uma janela cuja transcrição é só anotação (`[inint]`) ou só
    hesitação. Ela precisa sair contada, nunca imputada.
    """
    with pytest.raises(EmptyReferenceError):
        count_edits("", "a gente foi")
    with pytest.raises(EmptyReferenceError):
        count_edits("[inint] (risos)", "a gente foi")
    with pytest.raises(EmptyReferenceError):
        count_edits("ah eh hmm", "a gente foi")
    # E é subclasse de ValueError: quem só captura ValueError não deixa passar.
    assert issubclass(EmptyReferenceError, ValueError)


def test_empty_hypothesis_is_legitimate_and_becomes_all_deletions():
    """ASR que não reconheceu nada tem WER = 1.0, com N palavras deletadas.

    Tratar isso como erro (ou como janela a descartar) esconderia exatamente a
    falha de modelo que a auditoria existe para medir.
    """
    counts = count_edits("a gente foi", "")
    assert counts == EditCounts(substitutions=0, insertions=0, deletions=3, n_reference_words=3)
    assert counts.wer == pytest.approx(1.0)
    # Hipótese que some na normalização cai no mesmo caso, e não em erro.
    assert count_edits("a gente foi", "(risos)") == counts


def test_a_record_with_no_reference_words_refuses_to_report_a_wer():
    """`WerRecord.wer` com N = 0 levanta, em vez de devolver 0.0 ou NaN."""
    degenerate = _record("w-degenerada", "NE_01", n=0)
    with pytest.raises(ValueError, match="indefinido"):
        _ = degenerate.wer


# --------------------------------------------------------------------------- #
# O TESTE CENTRAL — pooled ≠ macro, com os dois números feitos à mão
# --------------------------------------------------------------------------- #
def test_pooled_and_macro_diverge_and_pooled_is_sum_over_sum():
    """Prova numérica de que o agregado é soma/soma, e não média de razões.

    Conjunto montado para maximizar o contraste que o corpus real produz de
    verdade (janelas de 30 a 60 s têm número de palavras bem variável):

      janela curta: S=1, I=0, D=0, N=3    → WER_j = 1/3   ≈ 0,333
      janela longa: S=2, I=0, D=0, N=100  → WER_j = 2/100 = 0,020

    pooled = (1 + 2) / (3 + 100)     = 3/103      ≈ 0,02913
    macro  = (1/3 + 2/100) / 2       = 53/300     ≈ 0,17667

    O macro é ~6× o pooled: a janela de 3 palavras, que carrega 3% das palavras
    do conjunto, pesa metade do resumo. Se alguém trocar a implementação de
    `wer_pooled` pela média das razões, o WER do TCC pula de 2,9% para 17,7% sem
    nenhuma exceção — é este teste, e só ele, que segura isso.
    """
    records = [
        _record("w-curta", "NE_01", s=1, n=3),
        _record("w-longa", "NE_01", s=2, n=100),
    ]
    aggregate = aggregate_wer(records)

    assert aggregate.errors == 3
    assert aggregate.n_reference_words == 103
    assert aggregate.wer_pooled == pytest.approx(3 / 103)
    assert aggregate.wer_pooled == pytest.approx(0.0291262, abs=1e-7)

    macro = macro_average_wer(records)
    assert macro == pytest.approx((1 / 3 + 2 / 100) / 2)
    assert macro == pytest.approx(53 / 300)
    assert macro == pytest.approx(0.1766666, abs=1e-7)

    # A divergência é o ponto: as duas NÃO podem ser tratadas como sinônimos.
    assert macro > 5 * aggregate.wer_pooled
    assert abs(macro - aggregate.wer_pooled) > 0.14


def test_the_two_aggregations_coincide_only_when_the_windows_have_equal_length():
    """Contraprova: com N igual em todas as janelas, pooled == macro.

    Isso mostra que a divergência acima não é bug de aritmética — é o peso das
    janelas. E é por isso que o desbalanço de comprimento importa.
    """
    same_length = [
        _record("w1", "NE_01", s=1, n=10),
        _record("w2", "NE_01", s=3, n=10),
        _record("w3", "SE_01", s=0, n=10),
    ]
    assert aggregate_wer(same_length).wer_pooled == pytest.approx(macro_average_wer(same_length))
    assert aggregate_wer(same_length).wer_pooled == pytest.approx(4 / 30)


def test_aggregate_counts_windows_and_distinct_speakers():
    records = [
        _record("w1", "NE_01", s=1, n=10),
        _record("w2", "NE_01", s=1, n=10),
        _record("w3", "SE_01", s=1, n=10),
    ]
    aggregate = aggregate_wer(records)
    assert (aggregate.n_windows, aggregate.n_speakers) == (3, 2)


def test_aggregations_refuse_an_empty_set():
    """Conjunto vazio não tem WER: melhor falhar que devolver 0,0 ou NaN."""
    with pytest.raises(ValueError):
        aggregate_wer([])
    with pytest.raises(ValueError):
        macro_average_wer([])


def test_aggregate_with_zero_reference_words_is_undefined_not_zero():
    with pytest.raises(EmptyReferenceError):
        _ = aggregate_wer([_record("w-degenerada", "NE_01", n=0)]).wer_pooled


# --------------------------------------------------------------------------- #
# `compute_wer_records` — trilha de exclusão com os dois motivos distintos
# --------------------------------------------------------------------------- #
def test_compute_wer_records_counts_both_drop_reasons_and_balances_the_ledger():
    """As duas causas de descarte são contadas SEPARADAS, e a soma fecha.

    Falha de junção (janela sem referência no dicionário) e janela degenerada
    (referência que some na normalização) têm diagnósticos opostos: a primeira é
    bug de pipeline, a segunda é propriedade do corpus. Somá-las num único
    contador esconderia a primeira dentro da segunda.
    """
    hypotheses = [
        HypothesisRow("w-ok", "NE_01", "asr", "a gente morava na rosa"),
        HypothesisRow("w-sem-ref", "NE_01", "asr", "qualquer coisa"),
        HypothesisRow("w-ref-vazia", "SE_01", "asr", "qualquer coisa"),
    ]
    references = {
        "w-ok": "a gente morava na roça",
        "w-ref-vazia": "[inint] ah hmm",  # some inteira na normalização
    }
    records, ledger = compute_wer_records(hypotheses, references)

    assert [r.window_id for r in records] == ["w-ok"]
    assert ledger.total_in == 3
    assert ledger.kept == 1
    assert ledger.dropped == {DROP_SEM_REFERENCIA: 1, DROP_REFERENCIA_VAZIA: 1}
    ledger.assert_balanced()  # entradas = retidas + descartadas
    assert ledger.unit == "janela × modelo"


def test_compute_wer_records_keeps_counts_and_stamps_the_normalizer_version():
    """O registro guarda S, I, D, N — e a versão que os produziu, não a razão."""
    records, ledger = compute_wer_records(
        [HypothesisRow("w1", "NE_01", "whisper", "a gente morava na rosa")],
        {"w1": "a gente morava na roça"},
    )
    (record,) = records
    assert (record.substitutions, record.insertions, record.deletions) == (1, 0, 0)
    assert record.n_reference_words == 5
    assert record.normalizer_version == NORMALIZER_VERSION
    assert record.wer == pytest.approx(0.2)
    assert (record.speaker_code, record.model_key) == ("NE_01", "whisper")
    assert ledger.kept == 1


def test_compute_wer_records_output_order_is_canonical():
    """Ordenação `(model_key, window_id)`: dois relatórios da mesma entrada são iguais.

    Sem ordenação canônica, a ordem de iteração do chamador vazaria para dentro
    do artefato e dois `outputs/` idênticos em conteúdo diferiam byte a byte.
    """
    hypotheses = [
        HypothesisRow("w03", "NE_01", "whisper", "a gente foi"),
        HypothesisRow("w01", "SE_01", "whisper", "a gente foi"),
        HypothesisRow("w02", "NE_01", "azul", "a gente foi"),
    ]
    references = dict.fromkeys(["w01", "w02", "w03"], "a gente foi embora")
    records, _ = compute_wer_records(hypotheses, references)
    assert [(r.model_key, r.window_id) for r in records] == [
        ("azul", "w02"),
        ("whisper", "w01"),
        ("whisper", "w03"),
    ]
    # Mesma entrada em outra ordem ⇒ mesma saída.
    shuffled, _ = compute_wer_records(list(reversed(hypotheses)), references)
    assert shuffled == records


def test_compute_wer_records_handles_an_empty_input():
    records, ledger = compute_wer_records([], {})
    assert records == []
    assert (ledger.total_in, ledger.kept, ledger.dropped) == (0, 0, {})


# --------------------------------------------------------------------------- #
# `wer_table` — as duas agregações lado a lado, por modelo × região
# --------------------------------------------------------------------------- #
def test_wer_table_reports_pooled_and_macro_side_by_side():
    """A tabela do relatório carrega as DUAS agregações, com os números à mão.

    NE: (1 + 2) / (3 + 100) = 3/103 ≈ 0,02913 | macro = (1/3 + 0,02)/2 ≈ 0,17667
    SE: (1 + 1) / (10 + 10) = 2/20  = 0,10    | macro = (0,10 + 0,10)/2 = 0,10

    Reparar na leitura errada que o macro habilita aqui: por ele o NE pareceria
    quase o DOBRO do erro do SE (0,177 vs. 0,100); pelo pooled o NE erra MENOS
    (0,029 vs. 0,100). A conclusão do TCC inverteria de sinal.
    """
    records = [
        _record("w-curta", "NE_01", s=1, n=3),
        _record("w-longa", "NE_02", s=2, n=100),
        _record("w-se-1", "SE_01", s=1, n=10),
        _record("w-se-2", "SE_02", s=1, n=10),
    ]
    table = wer_table(records, {"NE_01": "NE", "NE_02": "NE", "SE_01": "SE", "SE_02": "SE"})

    assert list(table.columns) == [
        "model_key",
        "region",
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
    ne = table[table["region"] == "NE"].iloc[0]
    se = table[table["region"] == "SE"].iloc[0]

    assert (int(ne["n_windows"]), int(ne["n_speakers"])) == (2, 2)
    assert int(ne["n_reference_words"]) == 103
    assert ne["wer_pooled"] == pytest.approx(3 / 103)
    assert ne["wer_macro"] == pytest.approx((1 / 3 + 0.02) / 2)
    assert se["wer_pooled"] == pytest.approx(0.10)
    assert se["wer_macro"] == pytest.approx(0.10)

    # A inversão de sinal entre as duas agregações, explicitada.
    assert ne["wer_pooled"] < se["wer_pooled"]
    assert ne["wer_macro"] > se["wer_macro"]


def test_wer_table_agrees_with_aggregate_wer_group_by_group():
    """A tabela não pode ter uma segunda implementação da agregação."""
    records = [
        _record("w1", "NE_01", s=1, n=7),
        _record("w2", "NE_01", i=2, n=13),
        _record("w3", "SE_01", d=3, n=21),
    ]
    regions = {"NE_01": "NE", "SE_01": "SE"}
    table = wer_table(records, regions)
    for region in ("NE", "SE"):
        subset = [r for r in records if regions[r.speaker_code] == region]
        row = table[table["region"] == region].iloc[0]
        assert row["wer_pooled"] == pytest.approx(aggregate_wer(subset).wer_pooled)
        assert row["wer_macro"] == pytest.approx(macro_average_wer(subset))


def test_wer_table_never_silently_drops_a_speaker_without_a_region():
    """Falante fora do mapa vira `desconhecida` e APARECE — não some da tabela."""
    records = [
        _record("w1", "NE_01", s=1, n=10),
        _record("w2", "XX_99", s=1, n=10),
    ]
    table = wer_table(records, {"NE_01": "NE"})
    assert set(table["region"]) == {"NE", "desconhecida"}
    assert int(table["n_windows"].sum()) == 2


def test_wer_table_separates_models():
    """Agrupar por modelo é o eixo de comparação da Etapa 6; não pode colapsar."""
    records = [
        _record("w1", "NE_01", s=1, n=10, model_key="whisper"),
        _record("w2", "NE_01", s=5, n=10, model_key="azul"),
    ]
    table = wer_table(records, {"NE_01": "NE"})
    assert len(table) == 2
    assert table.sort_values("model_key")["wer_pooled"].tolist() == [0.5, 0.1]


def test_wer_table_group_by_is_configurable():
    records = [
        _record("w1", "NE_01", s=1, n=10),
        _record("w2", "NE_02", s=3, n=10),
    ]
    table = wer_table(records, {"NE_01": "NE", "NE_02": "NE"}, group_by=("speaker_code",))
    assert list(table["speaker_code"]) == ["NE_01", "NE_02"]
    assert table["wer_pooled"].tolist() == [0.1, 0.3]


def test_wer_table_refuses_an_empty_set():
    with pytest.raises(ValueError):
        wer_table([], {})
