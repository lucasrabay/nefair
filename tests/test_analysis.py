"""Testes da Etapa 6 — regressão, decomposição e sensibilidade. Offline.

Este é o bloco onde um erro estatístico silencioso passa por uma suíte verde: as
funções daqui devolvem números plausíveis mesmo quando estão erradas. Um IC
estreito demais, um efeito em log-odds no lugar de pontos de acurácia ou um
bootstrap que reamostra itens em vez de falantes não levantam exceção nenhuma —
só mudam a conclusão do TCC. Por isso **nenhum teste deste arquivo se contenta em
verificar que a função retorna**: cada estimador é confrontado com um caso de
resultado conhecido, calculado à mão ou plantado na fixture.

O que cada fixture planta, e contra o quê:

| fixture                | o que tem dentro                         | prova o quê            |
|------------------------|------------------------------------------|------------------------|
| `planted_gap_frame`    | gap NE−SE = −0,12 gerado por sorteio     | recuperação do efeito  |
| `homogeneous_frame`    | taxas exatas por região, idade inerte    | AME exato, sem ruído   |
| `confounded_frame`     | idade é a ÚNICA causa; NE é mais velho   | D6: ajustar importa    |
| `clustered_frame`      | correlação forte dentro do falante       | falante × item no IC   |
| `cascade_frame`        | `asr` e `reference` pareadas por item    | decomposição e D4      |
| `imbalanced_frame`     | 39 NE contra 193 SE (o desbalanço real)  | estratificação         |

Duas armadilhas de construção de fixture ficam registradas aqui porque custaram
tempo: (1) com `stratify_by="region"`, cada `speaker_code` precisa ter UM único
valor de `region` no frame inteiro, senão `make_speaker_resamples` recusa — e com
razão, o estrato é atributo do falante, não do ensaio; (2) o logit precisa de
falantes suficientes e de variação no desfecho para convergir, então as fixtures
não são miniaturas de 4 linhas.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from nefair.analysis.decompose import (
    CONDITION_ASR,
    CONDITION_REFERENCE,
    SCOPE_OVERALL,
    CeilingSpec,
    check_identity,
    decompose_model,
    decomposition_table,
    paired_conditions_frame,
)
from nefair.analysis.regression import (
    REGION_BASELINE,
    REGION_EXPOSED,
    TRIAL_COLUMNS,
    SpeakerRowIndex,
    average_marginal_effect,
    bootstrap_over_speakers,
    build_design,
    build_trial_frame,
    estimate_gap,
    fit_logit,
    gap_table,
    make_speaker_resamples,
    restrict_age,
)
from nefair.analysis.sensitivity import (
    DURATION_COLUMN,
    duration_bin_labels,
    sensitivity_n_alternatives,
    sensitivity_n_items,
    sensitivity_window_duration,
    subsample_items_per_speaker,
)
from nefair.config import load_analysis_config
from nefair.schema import RunRecord

CONFIG_PATH = "configs/analysis.yaml"

# Poucas reamostras: a suíte precisa ser rápida. O número NÃO afeta as
# propriedades testadas (cobertura, estratificação, determinismo); afeta só a
# granularidade dos quantis, e a produção usa 2.000 (ver `analysis.yaml`).
N_RESAMPLES = 100
SEED = 20260918


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _expit(z: np.ndarray | float) -> np.ndarray | float:
    """Logística — reimplementada aqui de propósito.

    O teste do efeito marginal médio confere a g-computation contra esta fórmula;
    importar `_expit` do módulo sob teste faria o código concordar consigo mesmo.
    """
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def speaker_gender_of(speaker_code: str) -> str:
    """Gênero do falante, derivado do código de forma estável (D6).

    Precisa ser atributo do FALANTE — constante em todas as linhas dele — e
    precisa VARIAR DENTRO de cada região. Se todo falante do NE fosse `F` e todo
    do SE fosse `M`, a coluna de gênero seria colinear com a de região, a matriz
    de desenho ficaria deficiente de posto e `fit_logit` devolveria `None` em
    toda reamostra: o teste falharia por álgebra linear, não pelo estimador.

    Alternar pela paridade do índice no código garante as duas coisas.
    """
    digits = "".join(c for c in speaker_code if c.isdigit())
    return "F" if (int(digits or 0) % 2 == 0) else "M"


def _trial_rows(
    speaker_code: str,
    region: str,
    age: float,
    corrects: list[bool],
    *,
    model_key: str = "cascata",
    condition: str = CONDITION_ASR,
    **extra: object,
) -> list[dict[str, object]]:
    return [
        {
            "item_id": f"{speaker_code}__i{index:02d}",
            "speaker_code": speaker_code,
            "model_key": model_key,
            "condition": condition,
            "region": region,
            "age": age,
            "speaker_gender": speaker_gender_of(speaker_code),
            "is_correct": bool(value),
            **extra,
        }
        for index, value in enumerate(corrects)
    ]


@pytest.fixture(scope="module")
def planted_gap_frame() -> pd.DataFrame:
    """Gap NE−SE **plantado** em −0,12, com heterogeneidade entre falantes.

    A acurácia média é 0,58 no NE e 0,70 no SE; cada falante tem um desvio
    próprio (σ = 0,08) para que exista variância ENTRE falantes — sem ela o
    bootstrap de falantes teria variância zero e o IC não provaria nada.

    39/193 é o desbalanço real do corpus; aqui usamos 40/80, que preserva a
    direção do desbalanço e mantém a suíte rápida.
    """
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for region, n_speakers, base in ((REGION_EXPOSED, 40, 0.58), (REGION_BASELINE, 80, 0.70)):
        for index in range(n_speakers):
            code = f"{region}_{index:03d}"
            rate = float(np.clip(base + rng.normal(0.0, 0.08), 0.05, 0.95))
            age = float(rng.integers(20, 70))
            rows += _trial_rows(code, region, age, list(rng.random(12) < rate))
    return pd.DataFrame(rows)


def homogeneous_frame(n_correct_ne: int, n_correct_se: int, n_items: int = 20) -> pd.DataFrame:
    """Frame em que a acurácia é EXATAMENTE `k/n_items` dentro de cada região.

    Construção escolhida para que o efeito marginal médio seja exatamente a
    diferença de taxas, sem ruído amostral: como todo falante de uma região tem a
    mesma proporção de acertos, a equação de escore da idade (`Σ idade·(y − p̂)`)
    se anula falante a falante e o coeficiente de idade sai exatamente zero. Isso
    isola o estimador do ajuste e permite conferir o número à mão.
    """
    rows: list[dict[str, object]] = []
    for region, n_speakers, k in (
        (REGION_EXPOSED, 12, n_correct_ne),
        (REGION_BASELINE, 24, n_correct_se),
    ):
        for index in range(n_speakers):
            age = float(20 + (index * 3) % 50)
            corrects = [i < k for i in range(n_items)]
            rows += _trial_rows(f"{region}_{index:03d}", region, age, corrects, model_key="m")
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def confounded_frame() -> pd.DataFrame:
    """Idade é a ÚNICA causa do acerto; o NE é sistematicamente mais velho.

    `p = expit(3 − 0,05·idade)`, igual nas duas regiões. As faixas etárias se
    sobrepõem (NE 40–64, SE 25–49) para que o ajuste seja interpolação e não
    extrapolação. O efeito regional VERDADEIRO é zero — e a diferença bruta não é.
    """
    rows: list[dict[str, object]] = []
    for region, ages in ((REGION_EXPOSED, range(40, 66, 2)), (REGION_BASELINE, range(25, 51, 2))):
        for index, age in enumerate(ages):
            rate = float(_expit(3.0 - 0.05 * age))
            n_correct = int(round(rate * 20))
            corrects = [i < n_correct for i in range(20)]
            rows += _trial_rows(
                f"{region}_{index:02d}", region, float(age), corrects, model_key="m"
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def clustered_frame() -> pd.DataFrame:
    """Correlação intraclasse alta: metade dos falantes acerta ~0,25, metade ~0,85.

    É o retrato do corpus real — os 15 itens de um falante compartilham o áudio,
    o sotaque e a qualidade de gravação. Serve para mostrar o que acontece quando
    se reamostra a unidade errada.
    """
    rng = np.random.default_rng(3)
    rows: list[dict[str, object]] = []
    for region in (REGION_EXPOSED, REGION_BASELINE):
        for index in range(20):
            code = f"{region}_{index:03d}"
            rate = 0.25 if index % 2 == 0 else 0.85
            rows += _trial_rows(code, region, 40.0, list(rng.random(15) < rate), model_key="m")
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cascade_frame() -> pd.DataFrame:
    """Cascata com as duas condições pareadas item a item.

    Plantado: o ASR degrada mais o NE (0,65 contra 0,75 no SE) enquanto a
    condição `reference` é igual nas duas regiões (0,85). Ou seja, o gap desta
    fixture é de RECONHECIMENTO, não de raciocínio — e a decomposição tem de
    dizer isso, com Δ_ASR maior no NE.
    """
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for region, n_speakers, rate_asr in ((REGION_EXPOSED, 20, 0.65), (REGION_BASELINE, 30, 0.75)):
        for index in range(n_speakers):
            code = f"{region}_{index:03d}"
            age = float(20 + (index * 3) % 50)
            # Habilidade do falante: desloca as DUAS condições juntas, que é o
            # que cria a covariância positiva descontada pelo pareamento.
            skill = float(rng.normal(0.0, 0.12))
            p_reference = float(np.clip(0.85 + skill, 0.05, 0.99))
            p_asr = float(np.clip(rate_asr + skill, 0.05, 0.99))
            for item in range(10):
                item_id = f"{code}__i{item:02d}"
                for condition, rate in ((CONDITION_ASR, p_asr), (CONDITION_REFERENCE, p_reference)):
                    rows.append(
                        {
                            "item_id": item_id,
                            "speaker_code": code,
                            "model_key": "cascata",
                            "condition": condition,
                            "region": region,
                            "age": age,
                            "speaker_gender": speaker_gender_of(code),
                            "is_correct": bool(rng.random() < rate),
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def imbalanced_frame() -> pd.DataFrame:
    """O desbalanço REAL do corpus: 39 falantes do NE contra 193 do SE."""
    rows: list[dict[str, object]] = []
    for region, n_speakers in ((REGION_EXPOSED, 39), (REGION_BASELINE, 193)):
        for index in range(n_speakers):
            rows += _trial_rows(f"{region}_{index:03d}", region, 40.0, [True], model_key="m")
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def config():
    """Config real do repositório: os testes valem sobre o que vai rodar."""
    return load_analysis_config(CONFIG_PATH)


# --------------------------------------------------------------------------- #
# 1. Recuperação de um efeito conhecido
# --------------------------------------------------------------------------- #
def test_estimate_gap_recovers_a_planted_effect(planted_gap_frame, config):
    """O estimador devolve o gap que foi PLANTADO, e o IC o contém.

    Plantado: acurácia 0,58 no NE e 0,70 no SE ⇒ gap NE−SE = −0,12. A tolerância
    não é arbitrária: com 40 e 80 falantes e 12 itens cada, o erro padrão do gap
    fica na casa de 0,03, então ±0,05 é ~1,6 erro padrão. Mais apertado do que
    isso testaria o sorteio da fixture, não o estimador.

    As três asserções cobrem falhas diferentes:
    - magnitude errada (efeito em log-odds no lugar de pontos) → cai na primeira;
    - sinal invertido (SE − NE) → cai na segunda;
    - IC quebrado (largo demais, estreito demais ou deslocado) → cai na terceira.
    """
    resamples = make_speaker_resamples(
        planted_gap_frame, n_resamples=N_RESAMPLES, seed=SEED, stratify_by="region"
    )
    estimate = estimate_gap(
        planted_gap_frame,
        resamples,
        covariates=config.covariates,
        ci_level=config.bootstrap.ci_level,
        model_key="cascata",
        condition=CONDITION_ASR,
    )

    assert estimate.estimate == pytest.approx(-0.12, abs=0.05)
    assert estimate.estimate < 0  # NE pior que SE, como plantado
    assert estimate.ci_low <= -0.12 <= estimate.ci_high
    # Um efeito desta magnitude com este n tem de excluir o zero; se não
    # excluísse, o IC estaria largo demais para ser útil.
    assert estimate.ci_high < 0.0
    assert estimate.n_failed_resamples == 0


def test_estimate_gap_reports_the_sample_it_used(planted_gap_frame, config):
    """Os metadados da linha não são decoração: a tabela do TCC declara o recorte."""
    resamples = make_speaker_resamples(planted_gap_frame, n_resamples=20, seed=SEED)
    estimate = estimate_gap(
        planted_gap_frame,
        resamples,
        covariates=config.covariates,
        age_min=config.age_min,
        age_max=config.age_max,
        model_key="cascata",
        condition=CONDITION_ASR,
    )
    assert (estimate.n_speakers_ne, estimate.n_speakers_se) == (40, 80)
    assert estimate.n_trials == len(planted_gap_frame)
    assert estimate.n_items == planted_gap_frame["item_id"].nunique()
    assert (estimate.age_min, estimate.age_max) == (20, 69)
    row = estimate.to_row()
    assert row["gap_ne_se"] == estimate.estimate
    # Derivado da config, nao fixado na mao: o ponto da asercao e que `to_row`
    # serializa as covariaveis efetivamente usadas (D6 mudou a lista em
    # 2026-10-02, quando genero entrou), nao que a lista seja uma em particular.
    assert row["covariates"] == "+".join(config.covariates)


def test_average_marginal_effect_is_exact_when_the_design_is_homogeneous():
    """Caso sem ruído: 0,70 no NE e 0,80 no SE ⇒ AME exatamente −0,10.

    Aqui não há tolerância a negociar. Se o estimador estiver certo, o número é
    −0,10 até o epsilon de máquina, e o coeficiente de idade é exatamente zero
    (a idade não tem relação nenhuma com o acerto nesta fixture).
    """
    frame = homogeneous_frame(n_correct_ne=14, n_correct_se=16)
    resamples = make_speaker_resamples(frame, n_resamples=20, seed=SEED)
    design = build_design(frame, resamples, covariates=("region", "age"))
    params = fit_logit(design.endog, design.exog)

    assert params is not None
    assert params[design.names.index("age")] == pytest.approx(0.0, abs=1e-9)

    contrast = average_marginal_effect(design.endog, design.exog, design.region_column)
    assert contrast is not None
    assert contrast.acc_exposed == pytest.approx(0.70, abs=1e-12)
    assert contrast.acc_baseline == pytest.approx(0.80, abs=1e-12)
    assert contrast.effect == pytest.approx(-0.10, abs=1e-12)


def test_the_confidence_interval_is_degenerate_when_speakers_are_identical():
    """Sem variação ENTRE falantes, o bootstrap de falantes dá IC de largura zero.

    Não é bug: nesta fixture todo falante do NE acerta 14/20 e todo falante do SE
    acerta 16/20, então qualquer reamostra de falantes reproduz as mesmas taxas.
    O teste está aqui porque o resultado seria OUTRO se a reamostragem fosse de
    itens (dentro do falante ainda há variação item a item). É uma evidência
    direta de que a unidade de reamostragem é o falante.
    """
    frame = homogeneous_frame(n_correct_ne=14, n_correct_se=16)
    resamples = make_speaker_resamples(frame, n_resamples=50, seed=SEED)
    estimate = estimate_gap(frame, resamples, covariates=("region", "age"))
    assert estimate.ci_low == pytest.approx(-0.10, abs=1e-9)
    assert estimate.ci_high == pytest.approx(-0.10, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. Estratificação — o que ela faz e o que ela evita
# --------------------------------------------------------------------------- #
def test_stratified_resamples_preserve_the_stratum_sizes_exactly(imbalanced_frame):
    """Com estratificação, TODA reamostra tem 39 NE e 193 SE. Sem exceção.

    É essa constância que retira do IC a incerteza sobre o desenho amostral: o
    estudo tem 39 falantes do NE, isso é um fato, não uma variável aleatória.
    """
    resamples = make_speaker_resamples(
        imbalanced_frame, n_resamples=N_RESAMPLES, seed=SEED, stratify_by="region"
    )
    assert resamples.observed_stratum_sizes == {REGION_EXPOSED: 39, REGION_BASELINE: 193}
    for index in range(resamples.n_resamples):
        assert resamples.stratum_counts(index) == resamples.observed_stratum_sizes


def test_unstratified_resamples_let_the_ne_count_wander(imbalanced_frame):
    """Sem estratificar, o nº de falantes do NE VARIA de reamostra para reamostra.

    Com 232 sorteios de um conjunto com 39/232 de falantes do NE, a contagem
    segue uma binomial de desvio padrão ≈ 5,7 — e a amplitude observada passa de
    20 falantes. Essa variação entra no IC como se fosse incerteza sobre o
    efeito, quando é incerteza sobre um desenho que na prática é fixo.

    Este é o contraste que justifica a decisão: mesmo frame, mesma semente, só a
    chave `stratify_by` muda.
    """
    resamples = make_speaker_resamples(
        imbalanced_frame, n_resamples=N_RESAMPLES, seed=SEED, stratify_by=None
    )
    counts = [
        sum(1 for code in resamples.codes_of(index) if code.startswith(REGION_EXPOSED))
        for index in range(resamples.n_resamples)
    ]
    assert len(set(counts)) > 10, "sem estratificar, a contagem do NE tem de variar"
    assert max(counts) - min(counts) >= 10
    assert np.std(counts) > 3.0
    # E o total de falantes continua sendo o mesmo — o que varia é a composição.
    assert all(len(resamples.codes_of(i)) == 232 for i in range(resamples.n_resamples))


def test_stratum_must_be_an_attribute_of_the_speaker_not_of_the_trial(planted_gap_frame):
    """Falante com duas regiões é recusado — e a mensagem diz por quê.

    É a armadilha nº 1 de quem monta um frame de ensaios à mão: basta UMA linha
    com a região trocada para o mesmo falante passar a pertencer a dois estratos.
    Aceitar isso em silêncio o faria ser sorteado de ambos.
    """
    broken = planted_gap_frame.copy()
    broken.loc[0, "region"] = REGION_BASELINE  # o falante NE_000 fica com NE e SE
    with pytest.raises(ValueError, match="mais de um valor"):
        make_speaker_resamples(broken, n_resamples=10, seed=SEED)


def test_make_speaker_resamples_validates_its_inputs(imbalanced_frame):
    with pytest.raises(ValueError, match="n_resamples"):
        make_speaker_resamples(imbalanced_frame, n_resamples=0, seed=SEED)
    with pytest.raises(ValueError, match="speaker_code"):
        make_speaker_resamples(
            imbalanced_frame.drop(columns=["speaker_code"]), n_resamples=10, seed=SEED
        )
    with pytest.raises(ValueError, match="inexistente"):
        make_speaker_resamples(
            imbalanced_frame, n_resamples=10, seed=SEED, stratify_by="inexistente"
        )


# --------------------------------------------------------------------------- #
# 3. Pareamento — as mesmas reamostras nas duas condições
# --------------------------------------------------------------------------- #
def test_paired_frame_keeps_only_items_answered_in_both_conditions(cascade_frame):
    """Item respondido só em `reference` sai: senão a diferença deixa de ser Δ_ASR.

    Um item presente só na condição `reference` inflaria `acc(reference)` sem
    contrapartida em `acc(asr)`, misturando efeito de condição com efeito de
    composição da amostra.
    """
    orphan = pd.DataFrame(
        [
            {
                "item_id": "item_orfao",
                "speaker_code": f"{REGION_EXPOSED}_000",
                "model_key": "cascata",
                "condition": CONDITION_REFERENCE,
                "region": REGION_EXPOSED,
                "age": 40.0,
                "is_correct": True,
            }
        ]
    )
    with_orphan = pd.concat([cascade_frame, orphan], ignore_index=True)
    paired = paired_conditions_frame(with_orphan, "cascata")
    assert "item_orfao" not in set(paired["item_id"])
    assert paired["item_id"].nunique() == cascade_frame["item_id"].nunique()
    # Todo item sobrevivente tem exatamente as duas condições.
    assert set(paired.groupby("item_id")["condition"].nunique()) == {2}


def test_the_same_resamples_feed_both_conditions_and_narrow_the_interval(cascade_frame):
    """Reamostrar as duas condições JUNTAS é o que desconta a covariância.

    Δ_ASR = acc(reference) − acc(asr) é uma diferença medida nos MESMOS itens e
    nos MESMOS falantes. Sorteando as duas condições de forma independente,
    Var(Δ) = Var(ref) + Var(asr); sorteando pareado, Var(Δ) = Var(ref) +
    Var(asr) − 2·Cov, e a covariância é positiva (falante bom vai bem nas duas
    condições). O teste compara as duas construções nas mesmas 200 réplicas: a
    pareada tem de ser sensivelmente mais estreita.

    Se alguém trocar o `SpeakerResamples` compartilhado por dois sorteios
    independentes, o IC de Δ_ASR incha e ninguém percebe — é este teste que pega.
    """
    paired = paired_conditions_frame(cascade_frame, "cascata")
    resamples_a = make_speaker_resamples(cascade_frame, n_resamples=200, seed=SEED)
    resamples_b = make_speaker_resamples(cascade_frame, n_resamples=200, seed=SEED + 1)
    row_index = SpeakerRowIndex(resamples_a.speaker_codes, paired["speaker_code"])

    is_correct = paired["is_correct"].to_numpy(dtype=float)
    is_asr = (paired["condition"] == CONDITION_ASR).to_numpy()

    def _accuracies(rows: np.ndarray) -> tuple[float, float]:
        mask = is_asr[rows]
        values = is_correct[rows]
        return float(values[mask].mean()), float(values[~mask].mean())

    paired_deltas: list[float] = []
    unpaired_deltas: list[float] = []
    for index in range(resamples_a.n_resamples):
        rows_a = row_index.rows_for_draw(resamples_a.draws[index])
        rows_b = row_index.rows_for_draw(resamples_b.draws[index])
        acc_asr_a, acc_ref_a = _accuracies(rows_a)
        _, acc_ref_b = _accuracies(rows_b)
        paired_deltas.append(acc_ref_a - acc_asr_a)
        unpaired_deltas.append(acc_ref_b - acc_asr_a)

    assert np.std(paired_deltas) < 0.85 * np.std(unpaired_deltas)


def test_decompose_model_uses_the_paired_frame(cascade_frame):
    """`decompose_model` conta itens PAREADOS, não linhas soltas."""
    resamples = make_speaker_resamples(cascade_frame, n_resamples=50, seed=SEED)
    decomposition = decompose_model(
        cascade_frame, resamples, CeilingSpec("constructed", 1.0), model_key="cascata"
    )
    paired = paired_conditions_frame(cascade_frame, "cascata")
    assert decomposition.n_items_paired == paired["item_id"].nunique() == 500
    assert decomposition.n_speakers == 50
    # As acurácias são as BRUTAS do frame pareado — conferidas fora do módulo.
    expected_asr = paired.loc[paired["condition"] == CONDITION_ASR, "is_correct"].mean()
    expected_ref = paired.loc[paired["condition"] == CONDITION_REFERENCE, "is_correct"].mean()
    assert decomposition.acc_asr == pytest.approx(expected_asr)
    assert decomposition.acc_reference == pytest.approx(expected_ref)


def test_decomposition_finds_the_planted_asymmetry_between_regions(cascade_frame):
    """A fixture planta degradação de ASR maior no NE — a tabela tem de mostrar.

    `reference` é igual nas duas regiões (0,85); o ASR derruba o NE para 0,65 e o
    SE para 0,75. Logo Δ_ASR(NE) > Δ_ASR(SE), e é isso que responde à pergunta
    central do TCC: o gap vem do reconhecimento, não da compreensão.
    """
    resamples = make_speaker_resamples(cascade_frame, n_resamples=50, seed=SEED)
    table, warnings_out = decomposition_table(
        cascade_frame, resamples, CeilingSpec("constructed", 1.0)
    )
    assert warnings_out == []
    assert sorted(table["scope"]) == [REGION_EXPOSED, REGION_BASELINE, SCOPE_OVERALL]
    by_scope = table.set_index("scope")
    assert by_scope.loc[REGION_EXPOSED, "delta_asr"] > by_scope.loc[REGION_BASELINE, "delta_asr"]
    # E o Δ_raciocínio, por construção, NÃO deve carregar o gap.
    gap_reasoning = abs(
        by_scope.loc[REGION_EXPOSED, "delta_reasoning"]
        - by_scope.loc[REGION_BASELINE, "delta_reasoning"]
    )
    gap_asr = by_scope.loc[REGION_EXPOSED, "delta_asr"] - by_scope.loc[REGION_BASELINE, "delta_asr"]
    assert gap_asr > gap_reasoning


# --------------------------------------------------------------------------- #
# 4. Identidade da decomposição
# --------------------------------------------------------------------------- #
def test_decomposition_identity_holds(cascade_frame):
    """`acc(asr) + Δ_ASR + Δ_raciocínio == teto`, no geral e em cada região.

    É o que garante que a decomposição é uma PARTIÇÃO do erro e não duas
    quantidades soltas que por acaso moram na mesma tabela.
    """
    resamples = make_speaker_resamples(cascade_frame, n_resamples=50, seed=SEED)
    ceiling = CeilingSpec("constructed", 1.0)
    for scope in (SCOPE_OVERALL, REGION_EXPOSED, REGION_BASELINE):
        decomposition = decompose_model(
            cascade_frame, resamples, ceiling, model_key="cascata", scope=scope
        )
        check_identity(decomposition)
        total = decomposition.acc_asr + decomposition.delta_asr + decomposition.delta_reasoning
        assert total == pytest.approx(ceiling.value, abs=1e-12)


def test_the_identity_also_holds_for_a_ceiling_below_one(cascade_frame):
    """Com `human_sample` o teto é medido, não 1,0 — e a identidade continua."""
    resamples = make_speaker_resamples(cascade_frame, n_resamples=20, seed=SEED)
    ceiling = CeilingSpec("human_sample", 0.93)
    decomposition = decompose_model(cascade_frame, resamples, ceiling, model_key="cascata")
    check_identity(decomposition)
    assert decomposition.ceiling_mode == "human_sample"
    assert decomposition.delta_reasoning == pytest.approx(0.93 - decomposition.acc_reference)
    # O rótulo do modo viaja na linha: decomposições de modos diferentes não são
    # comparáveis, e a tabela precisa deixar isso à vista.
    assert decomposition.to_row()["ceiling_mode"] == "human_sample"


def test_check_identity_catches_a_broken_decomposition(cascade_frame):
    """Contraprova: adulterar um número faz `check_identity` falhar."""
    resamples = make_speaker_resamples(cascade_frame, n_resamples=20, seed=SEED)
    good = decompose_model(
        cascade_frame, resamples, CeilingSpec("constructed", 1.0), model_key="cascata"
    )
    tampered = dataclasses.replace(good, acc_asr=good.acc_asr + 0.05)
    with pytest.raises(ValueError, match="não fecha"):
        check_identity(tampered)


# --------------------------------------------------------------------------- #
# 5. Determinismo
# --------------------------------------------------------------------------- #
def test_same_seed_gives_bit_identical_intervals(planted_gap_frame, config):
    """Mesma semente ⇒ mesmo IC, bit a bit. Comparação com `==`, não `approx`.

    Reprodutibilidade é critério de aceite do plano: o número do TCC tem de sair
    igual em outra máquina, em outro dia.
    """
    first = make_speaker_resamples(planted_gap_frame, n_resamples=50, seed=SEED)
    second = make_speaker_resamples(planted_gap_frame, n_resamples=50, seed=SEED)
    assert np.array_equal(first.draws, second.draws)
    assert first.speaker_codes == second.speaker_codes

    a = estimate_gap(planted_gap_frame, first, covariates=config.covariates)
    b = estimate_gap(planted_gap_frame, second, covariates=config.covariates)
    assert (a.estimate, a.ci_low, a.ci_high) == (b.estimate, b.ci_low, b.ci_high)


def test_a_different_seed_moves_the_interval_but_not_the_point(planted_gap_frame, config):
    """A estimativa pontual não depende do sorteio; o IC depende — e só ele."""
    first = make_speaker_resamples(planted_gap_frame, n_resamples=50, seed=SEED)
    other = make_speaker_resamples(planted_gap_frame, n_resamples=50, seed=SEED + 1)
    a = estimate_gap(planted_gap_frame, first, covariates=config.covariates)
    b = estimate_gap(planted_gap_frame, other, covariates=config.covariates)
    assert a.estimate == b.estimate
    assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)


def test_speaker_order_is_canonical_and_independent_of_row_order(planted_gap_frame):
    """Embaralhar as linhas do frame não pode mudar as reamostras.

    Os falantes são ordenados por `(estrato, speaker_code)` antes de qualquer
    sorteio — sem isso, a ordem de iteração do pandas vazaria para dentro do IC.
    """
    shuffled = planted_gap_frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    a = make_speaker_resamples(planted_gap_frame, n_resamples=20, seed=SEED)
    b = make_speaker_resamples(shuffled, n_resamples=20, seed=SEED)
    assert a.speaker_codes == b.speaker_codes
    assert np.array_equal(a.draws, b.draws)


# --------------------------------------------------------------------------- #
# 6. G-computation ≠ coeficiente logit
# --------------------------------------------------------------------------- #
def test_average_marginal_effect_is_not_the_logit_coefficient():
    """O coeficiente está em log-odds; a saída do módulo está em PONTOS.

    Com 0,70 no NE e 0,80 no SE, os dois números são conferíveis à mão:

      coeficiente = log[(0,70/0,30) / (0,80/0,20)] = log(7/12) ≈ −0,5390
      AME         = 0,70 − 0,80                                 = −0,1000

    Um difere do outro por um fator de 5,4. Reportar o coeficiente como se fosse
    "pontos percentuais" — erro comum — quintuplicaria o gap do TCC.
    """
    frame = homogeneous_frame(n_correct_ne=14, n_correct_se=16)
    resamples = make_speaker_resamples(frame, n_resamples=10, seed=SEED)
    design = build_design(frame, resamples, covariates=("region", "age"))
    params = fit_logit(design.endog, design.exog)
    assert params is not None

    coefficient = float(params[design.region_column])
    assert coefficient == pytest.approx(np.log((0.70 / 0.30) / (0.80 / 0.20)), abs=1e-6)
    assert coefficient == pytest.approx(-0.5389965, abs=1e-6)

    contrast = average_marginal_effect(design.endog, design.exog, design.region_column)
    assert contrast is not None
    assert contrast.effect == pytest.approx(-0.10, abs=1e-12)
    assert abs(coefficient - contrast.effect) > 0.4

    # A saída está em pontos de acurácia: as duas pernas do contraste são
    # probabilidades, e o efeito é a diferença entre elas.
    assert 0.0 <= contrast.acc_exposed <= 1.0
    assert 0.0 <= contrast.acc_baseline <= 1.0
    assert contrast.effect == pytest.approx(contrast.acc_exposed - contrast.acc_baseline)


def test_the_coefficient_and_the_marginal_effect_can_rank_two_models_differently():
    """O caso que fecha a questão: os dois ordenam DOIS cenários ao contrário.

      cenário A: 0,70 × 0,80 ⇒ coeficiente −0,5390 | AME −0,10
      cenário B: 0,90 × 0,95 ⇒ coeficiente −0,7472 | AME −0,05

    Pelo coeficiente, B tem o gap MAIOR; em pontos de acurácia, B tem METADE do
    gap de A. Não é questão de escala — é inversão de ordenação. Quem relatasse o
    coeficiente logit escreveria a conclusão oposta no capítulo de resultados.
    """
    results: dict[str, tuple[float, float]] = {}
    for label, (k_ne, k_se) in {"A": (14, 16), "B": (18, 19)}.items():
        frame = homogeneous_frame(n_correct_ne=k_ne, n_correct_se=k_se)
        resamples = make_speaker_resamples(frame, n_resamples=10, seed=SEED)
        design = build_design(frame, resamples, covariates=("region", "age"))
        params = fit_logit(design.endog, design.exog)
        contrast = average_marginal_effect(design.endog, design.exog, design.region_column)
        assert params is not None and contrast is not None
        results[label] = (float(params[design.region_column]), contrast.effect)

    coef_a, ame_a = results["A"]
    coef_b, ame_b = results["B"]
    assert ame_a == pytest.approx(-0.10, abs=1e-9)
    assert ame_b == pytest.approx(-0.05, abs=1e-9)
    assert abs(coef_b) > abs(coef_a)  # o coeficiente diz "B é pior"
    assert abs(ame_b) < abs(ame_a)  # os pontos de acurácia dizem o contrário


def test_g_computation_standardizes_to_the_whole_sample():
    """Conferência independente da fórmula: média das diferenças contrafactuais.

    Reimplementamos aqui a g-computation com `numpy` puro — se o módulo estivesse
    avaliando as covariáveis na MÉDIA (outra prática comum, e diferente) em vez
    de observação a observação, os dois números divergiriam.
    """
    frame = homogeneous_frame(n_correct_ne=14, n_correct_se=16)
    resamples = make_speaker_resamples(frame, n_resamples=10, seed=SEED)
    design = build_design(frame, resamples, covariates=("region", "age"))
    params = fit_logit(design.endog, design.exog)
    assert params is not None

    exposed = design.exog.copy()
    exposed[:, design.region_column] = 1.0
    baseline = design.exog.copy()
    baseline[:, design.region_column] = 0.0
    expected = float(_expit(exposed @ params).mean() - _expit(baseline @ params).mean())

    contrast = average_marginal_effect(design.endog, design.exog, design.region_column)
    assert contrast is not None
    assert contrast.effect == pytest.approx(expected, abs=1e-12)


def test_adjusting_for_age_removes_a_gap_that_is_pure_composition(confounded_frame):
    """D6 em ação: idade é a única causa, e o NE é mais velho.

    A diferença BRUTA de acurácia é ≈ −0,16 e "significante"; o efeito regional
    verdadeiro é ZERO. Com `covariates=("region", "age")` o efeito ajustado cai
    para ~0 e o IC passa a conter o zero. Sem a idade no modelo, o mesmo dado
    produz um gap de −0,16 com IC que EXCLUI o zero — um falso positivo de livro.

    É a justificativa empírica da decisão D6: a covariável não é enfeite.
    """
    resamples = make_speaker_resamples(confounded_frame, n_resamples=N_RESAMPLES, seed=SEED)

    adjusted = estimate_gap(confounded_frame, resamples, covariates=("region", "age"))
    unadjusted = estimate_gap(confounded_frame, resamples, covariates=("region",))

    raw_gap = adjusted.acc_ne_raw - adjusted.acc_se_raw
    assert raw_gap == pytest.approx(-0.16, abs=0.02)

    # Sem ajuste, o estimador só reproduz a diferença bruta — e "acha" um efeito.
    assert unadjusted.estimate == pytest.approx(raw_gap, abs=1e-6)
    assert unadjusted.ci_high < 0.0

    # Com ajuste, o efeito some e o IC cobre o zero, que é a verdade da fixture.
    assert adjusted.estimate == pytest.approx(0.0, abs=0.02)
    assert adjusted.ci_low <= 0.0 <= adjusted.ci_high
    assert abs(adjusted.estimate) < abs(unadjusted.estimate) / 5


# --------------------------------------------------------------------------- #
# 7. `build_design` — codificação explícita, sem fórmulas
# --------------------------------------------------------------------------- #
def test_build_design_requires_region_among_the_covariates(planted_gap_frame):
    """D6: `region` é a EXPOSIÇÃO do estudo, não uma covariável opcional.

    Sem ela o modelo não tem o que estimar — e cairia num erro obscuro lá na
    frente, em vez de falhar com a explicação certa aqui.
    """
    resamples = make_speaker_resamples(planted_gap_frame, n_resamples=10, seed=SEED)
    with pytest.raises(ValueError, match="D6"):
        build_design(planted_gap_frame, resamples, covariates=("age",))
    with pytest.raises(ValueError, match="D6"):
        build_design(planted_gap_frame, resamples, covariates=())


def test_build_design_encodes_ne_as_the_exposed_level_with_se_as_reference(planted_gap_frame):
    """Coluna 1 é `regiao_NE`, SE é a referência: a direção reportada no texto.

    Deixar uma fórmula escolher a categoria de referência tornaria o SINAL do
    resultado dependente da ordem alfabética da versão da biblioteca.
    """
    resamples = make_speaker_resamples(planted_gap_frame, n_resamples=10, seed=SEED)
    design = build_design(planted_gap_frame, resamples, covariates=("region", "age"))
    assert design.names[0] == "const"
    assert design.names[1] == f"regiao_{REGION_EXPOSED}" == "regiao_NE"
    assert design.region_column == 1
    assert np.all(design.exog[:, 0] == 1.0)

    is_ne = (planted_gap_frame["region"] == REGION_EXPOSED).to_numpy()
    assert set(design.exog[is_ne, 1]) == {1.0}
    assert set(design.exog[~is_ne, 1]) == {0.0}  # SE é a referência: 0
    assert design.endog.tolist() == planted_gap_frame["is_correct"].astype(float).tolist()


def test_build_design_rejects_a_covariate_absent_from_the_frame(planted_gap_frame):
    resamples = make_speaker_resamples(planted_gap_frame, n_resamples=10, seed=SEED)
    with pytest.raises(ValueError, match="ausente"):
        build_design(planted_gap_frame, resamples, covariates=("region", "escolaridade"))


def test_build_design_dummies_a_categorical_covariate_with_the_lowest_level_as_reference():
    """Covariável categórica vira dummies; o menor nível fica de fora, por regra fixa."""
    frame = homogeneous_frame(n_correct_ne=14, n_correct_se=16)
    frame = frame.assign(
        genero=np.where(frame["speaker_code"].str[-1].astype(int) % 2 == 0, "f", "m")
    )
    resamples = make_speaker_resamples(frame, n_resamples=10, seed=SEED)
    design = build_design(frame, resamples, covariates=("region", "genero"))
    assert design.names == ("const", "regiao_NE", "genero_m")  # `f` é a referência


def test_fit_logit_returns_none_instead_of_raising_when_there_is_nothing_to_fit():
    """Falha de ajuste devolve `None`, e o chamador CONTA — não vira coeficiente ruim.

    Numa reamostra, coeficientes de um ajuste não convergido entrariam no IC como
    se fossem estimativas legítimas.
    """
    exog = np.column_stack([np.ones(20), np.repeat([0.0, 1.0], 10)])
    assert fit_logit(np.ones(20), exog) is None  # sem variação no desfecho
    assert fit_logit(np.empty(0), np.empty((0, 2))) is None  # amostra vazia
    # Colinearidade perfeita: a matriz não tem posto cheio.
    singular = np.column_stack([np.ones(20), np.ones(20)])
    assert fit_logit(np.repeat([0.0, 1.0], 10), singular) is None


# --------------------------------------------------------------------------- #
# 8. `CeilingSpec` — valida modo E faixa
# --------------------------------------------------------------------------- #
def test_ceiling_spec_validates_both_the_mode_and_the_range():
    """D4: o modo do teto é rotulado e validado; o valor é uma acurácia.

    Um teto fora de [0, 1] tornaria Δ_raciocínio negativo ou maior que 1 e a
    tabela publicaria uma "partição do erro" que não é partição de nada.
    """
    assert CeilingSpec("constructed", 1.0).value == 1.0
    assert CeilingSpec("human_sample", 0.0).value == 0.0

    with pytest.raises(ValueError, match="Modo de teto"):
        CeilingSpec("chute", 1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CeilingSpec("constructed", 1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CeilingSpec("human_sample", -0.01)


def test_ceiling_spec_mirrors_the_real_config(config):
    """O espelho local tem de aceitar exatamente o que o YAML declara."""
    ceiling = CeilingSpec(config.task_ceiling.mode, config.task_ceiling.value)
    assert (ceiling.mode, ceiling.value) == ("constructed", 1.0)


# --------------------------------------------------------------------------- #
# 9. Célula inestimável vira AVISO, não exceção
# --------------------------------------------------------------------------- #
def test_gap_table_turns_an_unestimable_cell_into_a_warning(planted_gap_frame, config):
    """Uma célula ruim não pode derrubar a tabela inteira — nem sumir calada.

    O modelo `so_se` só tem falantes do SE: a coluna de região fica toda zero, o
    logit não tem posto e o efeito não existe. A linha certa a tomar é registrar
    o motivo e seguir; um `KeyError` no meio da Etapa 6 perderia as células boas.
    """
    only_se = planted_gap_frame[planted_gap_frame["region"] == REGION_BASELINE].copy()
    only_se["model_key"] = "so_se"
    frame = pd.concat([planted_gap_frame, only_se], ignore_index=True)
    resamples = make_speaker_resamples(frame, n_resamples=20, seed=SEED)

    table, warnings_out = gap_table(frame, resamples, covariates=config.covariates)

    assert list(table["model_key"]) == ["cascata"]
    assert len(warnings_out) == 1
    assert warnings_out[0].startswith("so_se × asr: não estimável")


def test_decomposition_table_turns_an_unestimable_scope_into_a_warning(cascade_frame):
    """Escopo sem nenhum item pareado vira aviso; os escopos bons continuam."""
    only_se = cascade_frame[cascade_frame["region"] == REGION_BASELINE].reset_index(drop=True)
    resamples = make_speaker_resamples(only_se, n_resamples=20, seed=SEED)
    table, warnings_out = decomposition_table(only_se, resamples, CeilingSpec("constructed", 1.0))
    assert sorted(table["scope"]) == [REGION_BASELINE, SCOPE_OVERALL]
    assert len(warnings_out) == 1
    assert REGION_EXPOSED in warnings_out[0]


def test_decompose_model_refuses_a_model_without_the_two_conditions(cascade_frame):
    """Sem pareamento não há decomposição — e a mensagem diz isso."""
    only_asr = cascade_frame[cascade_frame["condition"] == CONDITION_ASR].reset_index(drop=True)
    resamples = make_speaker_resamples(only_asr, n_resamples=10, seed=SEED)
    with pytest.raises(ValueError, match="nenhum item com resposta nas duas condições"):
        decompose_model(only_asr, resamples, CeilingSpec("constructed", 1.0), model_key="cascata")


# --------------------------------------------------------------------------- #
# 10. A unidade de reamostragem é o FALANTE — reamostrar itens estreita o IC
# --------------------------------------------------------------------------- #
def test_resampling_items_instead_of_speakers_understates_the_interval(clustered_frame):
    """Reamostrar itens produz um IC MUITO mais estreito — e errado.

    Os 15 itens de um falante compartilham áudio, sotaque e qualidade de
    gravação: na fixture, metade dos falantes acerta ~0,25 e metade ~0,85. Tratar
    os 600 ensaios como 600 unidades independentes ignora essa dependência e
    infla o "n efetivo".

    O teste mede as duas larguras sobre a MESMA estatística (diferença bruta de
    acurácia NE − SE) e as mesmas 400 réplicas; só a unidade de reamostragem
    muda. O bootstrap de itens sai ~2,5× mais estreito — é assim que aparece
    "significância" espúria num desenho como este. Reamostrar itens NÃO é uma
    opção da API deste módulo, e este teste documenta por quê.
    """
    is_correct = clustered_frame["is_correct"].to_numpy(dtype=float)
    is_ne = (clustered_frame["region"] == REGION_EXPOSED).to_numpy()

    def _gap(rows: np.ndarray) -> float | None:
        mask = is_ne[rows]
        if mask.sum() == 0 or (~mask).sum() == 0:
            return None
        values = is_correct[rows]
        return float(values[mask].mean() - values[~mask].mean())

    resamples = make_speaker_resamples(clustered_frame, n_resamples=400, seed=5)
    row_index = SpeakerRowIndex(resamples.speaker_codes, clustered_frame["speaker_code"])
    by_speaker = bootstrap_over_speakers(
        _gap, lambda: _gap(np.arange(len(clustered_frame))), resamples, row_index
    )

    # Bootstrap de ITENS, feito à mão — o módulo, de propósito, não o oferece.
    rng = np.random.default_rng(5)
    n_rows = len(clustered_frame)
    item_replicates = [_gap(rng.integers(0, n_rows, n_rows)) for _ in range(resamples.n_resamples)]
    item_low, item_high = np.quantile([v for v in item_replicates if v is not None], [0.025, 0.975])

    width_speaker = by_speaker.ci_high - by_speaker.ci_low
    width_item = float(item_high - item_low)
    # Observado: ~0,39 (falantes) contra ~0,16 (itens). O limiar é folgado de
    # propósito — o que importa é a ORDEM de grandeza, não o terceiro decimal.
    assert width_item < 0.6 * width_speaker, (width_item, width_speaker)


def test_bootstrap_result_replicates_are_sorted_and_only_the_successful_ones(clustered_frame):
    """`replicates` vem ordenada e só com o que foi estimado de fato.

    Réplicas que falham são CONTADAS, nunca substituídas por zero — um IC
    calculado sobre um subconjunto enviesado das réplicas, sem que ninguém saiba,
    é exatamente o erro que este módulo existe para impedir.
    """
    resamples = make_speaker_resamples(clustered_frame, n_resamples=40, seed=SEED)
    row_index = SpeakerRowIndex(resamples.speaker_codes, clustered_frame["speaker_code"])
    calls = {"n": 0}

    def _sometimes_none(rows: np.ndarray) -> float | None:
        calls["n"] += 1
        return None if calls["n"] % 20 == 0 else float(rows.size)

    result = bootstrap_over_speakers(
        _sometimes_none, lambda: 1.0, resamples, row_index, max_failure_fraction=0.10
    )
    assert result.n_failed == 2
    assert len(result.replicates) == result.n_resamples - result.n_failed == 38
    assert np.all(np.diff(result.replicates) >= 0)
    assert result.failure_fraction == pytest.approx(2 / 40)
    assert result.point == 1.0


def test_bootstrap_refuses_an_interval_built_on_too_many_failures(clustered_frame):
    """Acima do limite de falhas, erro — não um IC que "parece bom"."""
    resamples = make_speaker_resamples(clustered_frame, n_resamples=20, seed=SEED)
    row_index = SpeakerRowIndex(resamples.speaker_codes, clustered_frame["speaker_code"])
    calls = {"n": 0}

    def _mostly_none(rows: np.ndarray) -> float | None:
        calls["n"] += 1
        return 1.0 if calls["n"] % 4 == 0 else None

    with pytest.raises(ValueError, match="não puderam ser"):
        bootstrap_over_speakers(
            _mostly_none, lambda: 1.0, resamples, row_index, max_failure_fraction=0.05
        )


def test_bootstrap_refuses_when_the_point_estimate_does_not_exist(clustered_frame):
    """Sem estimativa na amostra completa não há IC: falha alto, e explica."""
    resamples = make_speaker_resamples(clustered_frame, n_resamples=10, seed=SEED)
    row_index = SpeakerRowIndex(resamples.speaker_codes, clustered_frame["speaker_code"])
    with pytest.raises(ValueError, match="AMOSTRA COMPLETA"):
        bootstrap_over_speakers(lambda rows: 1.0, lambda: None, resamples, row_index)
    with pytest.raises(ValueError, match="ci_level"):
        bootstrap_over_speakers(lambda rows: 1.0, lambda: 1.0, resamples, row_index, ci_level=1.0)


def test_a_speaker_drawn_twice_contributes_its_rows_twice(clustered_frame):
    """Peso duplo do falante é o mecanismo do bootstrap; peso simples seria bug."""
    resamples = make_speaker_resamples(clustered_frame, n_resamples=5, seed=SEED)
    row_index = SpeakerRowIndex(resamples.speaker_codes, clustered_frame["speaker_code"])
    draw = np.array([0, 0, 1], dtype=np.int32)
    rows = row_index.rows_for_draw(draw)
    codes = clustered_frame["speaker_code"].to_numpy()[rows]
    first, second = resamples.speaker_codes[0], resamples.speaker_codes[1]
    assert int((codes == first).sum()) == 2 * int((codes == second).sum())
    # E as reamostras têm sempre o tamanho da amostra original, em falantes.
    assert all(
        len(resamples.codes_of(i)) == resamples.n_speakers for i in range(resamples.n_resamples)
    )


# --------------------------------------------------------------------------- #
# Montagem do frame de ensaios e recorte etário
# --------------------------------------------------------------------------- #
def test_build_trial_frame_counts_the_three_drop_reasons_separately():
    """Os três motivos são separados porque exigem providências diferentes.

    Resposta não interpretável é problema de formato (e NUNCA é imputada como
    erro: imputar penalizaria modelos verbosos por algo que não é compreensão);
    falante ausente da tabela é bug de junção a montante; região fora de escopo é
    propriedade do corpus. Somá-los num contador só esconderia o bug dentro do
    fato esperado.
    """

    def _record(item_id: str, speaker: str, label: str | None, correct: bool | None) -> RunRecord:
        return RunRecord(
            cache_key=f"k-{item_id}",
            item_id=item_id,
            speaker_code=speaker,
            model_key="m",
            condition=CONDITION_ASR,
            prompt_version="v1",
            permutation=(0, 1, 2, 3),
            raw_text="",
            parsed_label=label,
            is_correct=correct,
        )

    records = [
        _record("i1", "NE_01", "A", True),
        _record("i2", "NE_01", None, None),  # não interpretável
        _record("i3", "ZZ_99", "B", False),  # falante desconhecido
        _record("i4", "XX_01", "B", False),  # região fora de escopo
    ]
    speakers = pd.DataFrame(
        [
            {"speaker_code": "NE_01", "region": REGION_EXPOSED, "age": 33},
            {"speaker_code": "XX_01", "region": "N", "age": 40},
        ]
    )
    frame, ledger = build_trial_frame(records, speakers)

    assert list(frame.columns) == list(TRIAL_COLUMNS)
    assert list(frame["item_id"]) == ["i1"]
    assert ledger.total_in == 4
    assert ledger.kept == 1
    assert ledger.dropped == {
        "resposta não interpretável (parsed_label ausente)": 1,
        "falante ausente da tabela de falantes": 1,
        "região fora de escopo (N)": 1,
    }
    ledger.assert_balanced()


def test_build_trial_frame_requires_the_speaker_attributes():
    with pytest.raises(ValueError, match="obrigatória"):
        build_trial_frame([], pd.DataFrame({"speaker_code": []}))


def test_restrict_age_counts_who_fell_outside_on_each_side():
    """As duas pontas do recorte são contadas separadas: 20–69 é o recorte do texto."""
    frame = pd.DataFrame({"age": [19.0, 20.0, 45.0, 69.0, 70.0], "is_correct": [True] * 5})
    kept, ledger = restrict_age(frame, 20, 69)
    assert list(kept["age"]) == [20.0, 45.0, 69.0]  # inclusivo nas duas bordas
    assert ledger.dropped == {"idade < 20": 1, "idade > 69": 1}
    assert (ledger.total_in, ledger.kept) == (5, 3)
    ledger.assert_balanced()


# --------------------------------------------------------------------------- #
# Sensibilidade (D10) — o que está ligado, o que está desligado e por quê
# --------------------------------------------------------------------------- #
def test_subsample_items_per_speaker_is_deterministic_and_keeps_the_pairing(cascade_frame):
    """Sorteio semeado por `(seed, falante)`: mesma semente ⇒ mesmo subconjunto.

    E a seleção é no nível do ITEM: o item escolhido entra com TODAS as suas
    linhas (as duas condições), preservando o pareamento de que a decomposição
    depende. Selecionar linhas soltas quebraria a decomposição em silêncio.
    """
    subset, stats = subsample_items_per_speaker(cascade_frame, 4, seed=7)
    again, _ = subsample_items_per_speaker(cascade_frame, 4, seed=7)
    other, _ = subsample_items_per_speaker(cascade_frame, 4, seed=8)

    assert subset.equals(again)
    assert not subset.equals(other)
    assert set(subset.groupby("speaker_code")["item_id"].nunique()) == {4}
    assert stats["n_items_selecionados"] == 4 * cascade_frame["speaker_code"].nunique()
    assert stats["n_falantes_com_menos_itens_que_a_grade"] == 0
    # Pareamento intacto: todo item selecionado mantém as duas condições.
    assert set(subset.groupby("item_id")["condition"].nunique()) == {2}


def test_subsample_reports_speakers_with_fewer_items_than_the_grid(cascade_frame):
    """Pedir 25 de quem tem 10 não é erro — mas precisa aparecer no relatório.

    Sem essa nota, a variante "25 itens/falante" seria lida como se de fato
    tivesse 25 itens por falante.
    """
    subset, stats = subsample_items_per_speaker(cascade_frame, 25, seed=7)
    assert stats["n_falantes_com_menos_itens_que_a_grade"] == 50
    assert len(subset) == len(cascade_frame)  # todo mundo entrou com o que tem
    with pytest.raises(ValueError, match="n_items"):
        subsample_items_per_speaker(cascade_frame, 0, seed=7)


def test_sensitivity_n_items_reestimates_on_the_same_resamples(planted_gap_frame, config):
    """Cada `k` da grade vira uma variante; as reamostras são as MESMAS.

    Reusar as reamostras é o que torna os intervalos das variantes comparáveis
    entre si: a diferença entre eles vem do subconjunto de dados, não de um
    sorteio diferente.
    """
    resamples = make_speaker_resamples(planted_gap_frame, n_resamples=20, seed=SEED)
    outcome = sensitivity_n_items(
        planted_gap_frame,
        resamples,
        enabled=True,
        grid=config.sensitivity.n_items_grid[:2],
        covariates=config.covariates,
        seed=SEED,
    )
    assert outcome.enabled
    assert list(outcome.table["variant"]) == ["3 itens/falante", "5 itens/falante"]
    assert (outcome.table["n_items"] == [3 * 120, 5 * 120]).all()
    # O sinal do gap plantado sobrevive à subamostragem.
    assert (outcome.table["gap_ne_se"] < 0).all()
    assert any("falante(s) tinham" in note for note in outcome.notes)


def test_a_disabled_axis_returns_the_reason_not_an_empty_silence(planted_gap_frame, config):
    """Eixo desligado devolve o MOTIVO — o relatório registra que foi considerado.

    Uma análise de sensibilidade ausente sem explicação sugere que ninguém pensou
    nela; com o motivo escrito, a decisão fica auditável.
    """
    resamples = make_speaker_resamples(planted_gap_frame, n_resamples=10, seed=SEED)
    outcome = sensitivity_n_items(
        planted_gap_frame,
        resamples,
        enabled=False,
        grid=[3],
        covariates=config.covariates,
        seed=SEED,
    )
    assert outcome.enabled is False
    assert outcome.table.empty
    assert "desligado" in outcome.notes[0]


def test_n_alternatives_axis_carries_the_d10_rationale(planted_gap_frame, config):
    """D10 está desligado no YAML e o motivo (custo de revisão humana) vai junto."""
    assert config.sensitivity.n_alternatives_enabled is False
    resamples = make_speaker_resamples(planted_gap_frame, n_resamples=10, seed=SEED)
    outcome = sensitivity_n_alternatives(
        planted_gap_frame,
        resamples,
        enabled=config.sensitivity.n_alternatives_enabled,
        scope=config.sensitivity.n_alternatives_scope,
        covariates=config.covariates,
    )
    assert outcome.enabled is False
    assert outcome.table.empty
    assert "revisão humana" in outcome.notes[0]


def test_window_duration_bins_close_the_last_interval_on_the_right(config):
    """A última faixa é FECHADA: 60 s é o limite exato da janela (Etapa 1).

    Deixá-la aberta descartaria em silêncio justamente as janelas de 60 s.
    """
    labels = duration_bin_labels(config.sensitivity.window_duration_bins)
    assert [label for label, _, _ in labels] == ["[30, 40) s", "[40, 50) s", "[50, 60] s"]
    assert labels[-1][2] == 60.0
    with pytest.raises(ValueError, match="duas bordas"):
        duration_bin_labels([30.0])
    with pytest.raises(ValueError, match="crescente"):
        duration_bin_labels([60.0, 30.0])


def test_window_duration_axis_covers_every_trial_or_says_so(planted_gap_frame, config):
    """Janela fora de todas as faixas é CONTADA numa nota, nunca sumida.

    A fixture tem durações de 30 a 65 s; com as faixas do YAML (30–60), os
    ensaios de 61–65 s ficam de fora de todas elas e precisam aparecer.
    """
    durations = 30.0 + (np.arange(len(planted_gap_frame)) % 36)
    frame = planted_gap_frame.assign(**{DURATION_COLUMN: durations})
    resamples = make_speaker_resamples(frame, n_resamples=10, seed=SEED)
    outcome = sensitivity_window_duration(
        frame,
        resamples,
        enabled=True,
        bins=config.sensitivity.window_duration_bins,
        covariates=config.covariates,
    )
    assert outcome.enabled
    assert list(outcome.table["variant"]) == ["[30, 40) s", "[40, 50) s", "[50, 60] s"]
    assert any("fora de todas as faixas" in note for note in outcome.notes)


def test_window_duration_axis_explains_a_missing_duration_column(planted_gap_frame, config):
    """Sem a coluna de duração o eixo não roda — e diz de onde ela deveria vir."""
    resamples = make_speaker_resamples(planted_gap_frame, n_resamples=10, seed=SEED)
    outcome = sensitivity_window_duration(
        planted_gap_frame,
        resamples,
        enabled=True,
        bins=config.sensitivity.window_duration_bins,
        covariates=config.covariates,
    )
    assert outcome.enabled is False
    assert outcome.table.empty
    assert DURATION_COLUMN in outcome.notes[0]
    assert "windows.parquet" in outcome.notes[0]
