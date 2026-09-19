"""Logit `acerto ~ região + idade` e IC por bootstrap de falantes (Etapa 6).

Este é o módulo onde um erro estatístico silencioso passaria por uma suíte de
testes verde. Três decisões carregam todo o peso, e cada uma existe por um motivo
que o código não consegue adivinhar sozinho:

**1. O estimando é o efeito marginal médio, não o coeficiente logit.**
O coeficiente de `região` num logit é uma razão de chances em escala log —
inútil para a frase que o TCC precisa escrever ("o modelo acerta X pontos
percentuais a menos no NE"). O que se reporta aqui é a diferença de acurácia
**ajustada**: para cada observação da amostra, prediz-se a probabilidade de
acerto como se o falante fosse do NE e como se fosse do SE, mantendo as
covariáveis no valor observado, e tira-se a média das diferenças
(padronização / g-computation). O resultado está em pontos de acurácia, na mesma
unidade da tabela de resultados.

**2. O bootstrap é no nível do FALANTE, não do item.**
Itens do mesmo falante não são independentes: compartilham o áudio, o sotaque, a
qualidade de gravação e o estilo da entrevista. Reamostrar itens trataria ~10
observações correlacionadas como 10 unidades independentes e produziria um IC
estreito demais — a forma mais comum de "significância" espúria neste tipo de
estudo. Reamostrar falantes com reposição preserva a dependência interna: ou o
falante inteiro entra, ou nenhum item dele entra.

**3. O bootstrap é estratificado por região.**
O corpus tem 39 falantes do NE contra 193 do SE. Num bootstrap não estratificado
(232 sorteios do conjunto todo), o número de falantes do NE numa reamostra
varia — em torno de 39, mas com desvio de ~6. Essa variação NÃO é incerteza
sobre o efeito; é incerteza sobre o desenho amostral, que na prática é fixo
(sabemos que temos 39 falantes do NE). Deixá-la entrar alarga o IC por artefato.
Estratificar sorteia 39 de 39 e 193 de 193 separadamente, mantendo o desenho.

**4. As mesmas reamostras são reusadas entre condições.**
`SpeakerResamples` é construído UMA vez e passado a todas as condições e a todos
os modelos. Isso é o que permite que a decomposição (`decompose.py`) calcule o IC
de uma DIFERENÇA entre condições: reamostrar cada condição de forma independente
destruiria o pareamento item a item e inflaria a variância da diferença.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from nefair.schema import ExclusionLedger, RunRecord

# Região de exposição e de referência. `NE - SE` é a direção reportada no texto.
REGION_EXPOSED = "NE"
REGION_BASELINE = "SE"

# Colunas mínimas do frame de ensaios (uma linha = um item respondido por um
# modelo sob uma condição).
TRIAL_COLUMNS: tuple[str, ...] = (
    "item_id",
    "speaker_code",
    "model_key",
    "condition",
    "region",
    "age",
    "is_correct",
)


# --------------------------------------------------------------------------- #
# Montagem do frame de ensaios
# --------------------------------------------------------------------------- #
def build_trial_frame(
    records: Sequence[RunRecord],
    speakers: pd.DataFrame,
    *,
    extra_speaker_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, ExclusionLedger]:
    """Junta `RunRecord`s com os atributos do falante, contando toda exclusão.

    `speakers` precisa ter `speaker_code`, `region` e `age` (uma linha por
    falante). Colunas extras (p.ex. `speaker_gender`, para o D6) entram via
    `extra_speaker_columns`.

    Três motivos de exclusão, deliberadamente separados porque têm causas
    diferentes e exigem providências diferentes:

    - **resposta não interpretável** (`parsed_label is None`): o modelo respondeu
      algo que o parser não conseguiu ler. NUNCA é imputada como erro — imputar
      penalizaria modelos verbosos por um problema de formato, não de
      compreensão;
    - **falante ausente da tabela de falantes**: falha de junção, que indica bug
      a montante e não pode virar linha faltante silenciosa;
    - **região fora de escopo** (nem NE nem SE): informante nascido fora do
      Brasil ou em outra região.
    """
    ledger = ExclusionLedger(unit="ensaio (item × modelo × condição)")
    required = {"speaker_code", "region", "age"}
    missing = required - set(speakers.columns)
    if missing:
        raise ValueError(f"Tabela de falantes sem coluna(s) obrigatória(s): {sorted(missing)}.")

    keep_cols = ["speaker_code", "region", "age", *extra_speaker_columns]
    lookup = speakers.drop_duplicates("speaker_code").set_index("speaker_code")[keep_cols[1:]]

    rows: list[dict[str, object]] = []
    for record in records:
        ledger.total_in += 1
        if record.is_correct is None or record.parsed_label is None:
            ledger.drop("resposta não interpretável (parsed_label ausente)")
            continue
        if record.speaker_code not in lookup.index:
            ledger.drop("falante ausente da tabela de falantes")
            continue
        attrs = lookup.loc[record.speaker_code]
        region = str(attrs["region"])
        if region not in (REGION_EXPOSED, REGION_BASELINE):
            ledger.drop(f"região fora de escopo ({region})")
            continue
        row: dict[str, object] = {
            "item_id": record.item_id,
            "speaker_code": record.speaker_code,
            "model_key": record.model_key,
            "condition": record.condition,
            "region": region,
            "age": float(attrs["age"]),
            "is_correct": bool(record.is_correct),
        }
        for column in extra_speaker_columns:
            row[column] = attrs[column]
        rows.append(row)
        ledger.kept += 1

    ledger.assert_balanced()
    frame = pd.DataFrame(rows, columns=[*TRIAL_COLUMNS, *extra_speaker_columns])
    # Ordenação canônica: dois relatórios da mesma entrada são idênticos.
    frame = frame.sort_values(["model_key", "condition", "speaker_code", "item_id"])
    return frame.reset_index(drop=True), ledger


def restrict_age(
    frame: pd.DataFrame, age_min: int, age_max: int
) -> tuple[pd.DataFrame, ExclusionLedger]:
    """Restringe a amostra a `[age_min, age_max]`, contando quem ficou de fora.

    O recorte etário não é cosmético: a idade entra como covariável contínua e
    extrapolar a curva logística para faixas com um ou dois falantes produziria
    um ajuste dominado por esses poucos casos. O recorte é o mesmo declarado no
    texto (20–69 anos) e vem do config.
    """
    ledger = ExclusionLedger(unit="ensaio fora da faixa etária")
    ledger.total_in = len(frame)
    inside = frame["age"].between(age_min, age_max, inclusive="both")
    n_below = int((frame["age"] < age_min).sum())
    n_above = int((frame["age"] > age_max).sum())
    if n_below:
        ledger.drop(f"idade < {age_min}", n_below)
    if n_above:
        ledger.drop(f"idade > {age_max}", n_above)
    ledger.kept = int(inside.sum())
    ledger.assert_balanced()
    return frame.loc[inside].reset_index(drop=True), ledger


# --------------------------------------------------------------------------- #
# Reamostras de falantes (o coração do IC)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpeakerResamples:
    """B reamostras de falantes com reposição, opcionalmente estratificadas.

    `draws[b, j]` é um índice em `speaker_codes`. Guardar índices (e não strings)
    mantém o objeto compacto e a reamostragem vetorizada; `codes_of(b)` devolve a
    forma legível para inspeção e para os testes.

    O objeto é IMUTÁVEL e construído uma única vez por execução. Passá-lo a duas
    condições diferentes é o que garante, por construção, que elas vejam
    exatamente as mesmas reamostras.
    """

    speaker_codes: tuple[str, ...]
    strata: tuple[str, ...]
    stratum_of: tuple[str, ...]
    draws: np.ndarray
    seed: int
    stratify_by: str | None

    @property
    def n_resamples(self) -> int:
        return int(self.draws.shape[0])

    @property
    def n_speakers(self) -> int:
        return len(self.speaker_codes)

    def codes_of(self, index: int) -> tuple[str, ...]:
        """Códigos de falante da reamostra `index`, com repetição."""
        return tuple(self.speaker_codes[i] for i in self.draws[index])

    def stratum_counts(self, index: int) -> dict[str, int]:
        """Quantos falantes de cada estrato a reamostra `index` contém.

        Com estratificação, este dicionário é IDÊNTICO ao da amostra original em
        toda reamostra — é exatamente essa constância que o teste verifica.
        """
        counts = dict.fromkeys(self.strata, 0)
        for i in self.draws[index]:
            counts[self.stratum_of[i]] += 1
        return counts

    @property
    def observed_stratum_sizes(self) -> dict[str, int]:
        counts = dict.fromkeys(self.strata, 0)
        for stratum in self.stratum_of:
            counts[stratum] += 1
        return counts


def make_speaker_resamples(
    frame: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
    stratify_by: str | None = "region",
) -> SpeakerResamples:
    """Constrói as reamostras de falantes, em ordem canônica e com semente fixa.

    Determinismo: os falantes são ordenados por `(estrato, speaker_code)` e os
    estratos por nome antes de qualquer sorteio, e o gerador é
    `numpy.random.default_rng(seed)`. Mesma semente e mesma entrada ⇒ mesmos
    bytes de `draws`, em qualquer máquina.

    `stratify_by=None` produz o bootstrap NÃO estratificado. Ele existe para ser
    comparado com o estratificado num teste — e para deixar registrado o que se
    está evitando: sem estratificar, o número de falantes do NE por reamostra
    vira uma variável aleatória, e o IC passa a incluir incerteza sobre um
    desenho amostral que na verdade é fixo.
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples deve ser ≥ 1; recebido {n_resamples}.")
    if "speaker_code" not in frame.columns:
        raise ValueError("Frame sem coluna 'speaker_code'.")

    if stratify_by is None:
        speaker_table = frame[["speaker_code"]].drop_duplicates().copy()
        speaker_table["__stratum"] = "todos"
    else:
        if stratify_by not in frame.columns:
            raise ValueError(f"Coluna de estratificação '{stratify_by}' ausente do frame.")
        speaker_table = frame[["speaker_code", stratify_by]].drop_duplicates().copy()
        duplicated = speaker_table["speaker_code"].duplicated()
        if duplicated.any():
            offenders = sorted(speaker_table.loc[duplicated, "speaker_code"].unique())
            raise ValueError(
                f"Falante(s) com mais de um valor de '{stratify_by}': {offenders}. "
                "O estrato tem de ser atributo do FALANTE, não do ensaio."
            )
        speaker_table["__stratum"] = speaker_table[stratify_by].astype(str)

    speaker_table = speaker_table.sort_values(["__stratum", "speaker_code"]).reset_index(drop=True)
    codes = tuple(speaker_table["speaker_code"].astype(str))
    stratum_of = tuple(speaker_table["__stratum"])
    strata = tuple(sorted(set(stratum_of)))
    if not codes:
        raise ValueError("Nenhum falante no frame; impossível reamostrar.")

    # Posições de cada estrato na ordem canônica.
    positions: dict[str, np.ndarray] = {
        stratum: np.flatnonzero(np.array(stratum_of) == stratum) for stratum in strata
    }
    rng = np.random.default_rng(seed)
    draws = np.empty((n_resamples, len(codes)), dtype=np.int32)
    cursor = 0
    # Os estratos são percorridos em ordem alfabética e cada um preenche um bloco
    # fixo de colunas: a mesma semente sempre consome o gerador na mesma ordem.
    for stratum in strata:
        pool = positions[stratum]
        size = len(pool)
        picks = rng.integers(0, size, size=(n_resamples, size))
        draws[:, cursor : cursor + size] = pool[picks]
        cursor += size

    return SpeakerResamples(
        speaker_codes=codes,
        strata=strata,
        stratum_of=stratum_of,
        draws=draws,
        seed=seed,
        stratify_by=stratify_by,
    )


class SpeakerRowIndex:
    """Mapa falante → posições de linha, no formato que a reamostragem precisa.

    Guardar as linhas num vetor plano com deslocamentos permite expandir uma
    reamostra inteira com operações vetorizadas, em vez de 232 concatenações por
    reamostra × 2.000 reamostras. Não é microotimização gratuita: sem isso o
    bootstrap domina o tempo de execução da Etapa 6.
    """

    def __init__(self, speaker_codes: Sequence[str], frame_speakers: pd.Series) -> None:
        values = frame_speakers.to_numpy()
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        self._flat_rows = order.astype(np.int64)
        starts: list[int] = []
        counts: list[int] = []
        for code in speaker_codes:
            left = int(np.searchsorted(sorted_values, code, side="left"))
            right = int(np.searchsorted(sorted_values, code, side="right"))
            starts.append(left)
            counts.append(right - left)
        self._starts = np.asarray(starts, dtype=np.int64)
        self._counts = np.asarray(counts, dtype=np.int64)

    def rows_for_draw(self, draw: np.ndarray) -> np.ndarray:
        """Posições de linha correspondentes a uma reamostra de falantes.

        Um falante sorteado duas vezes contribui com suas linhas duas vezes — é
        isso que dá peso duplo ao falante na reamostra, como o bootstrap exige.
        """
        counts = self._counts[draw]
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        starts = self._starts[draw]
        # Expansão vetorizada de faixas: repete cada início e soma o offset
        # interno de cada elemento dentro da sua faixa.
        repeated_starts = np.repeat(starts, counts)
        block_begin = np.repeat(np.cumsum(counts) - counts, counts)
        offsets = np.arange(total, dtype=np.int64) - block_begin
        return self._flat_rows[repeated_starts + offsets]


# --------------------------------------------------------------------------- #
# Matriz de desenho e ajuste do logit
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Design:
    """Matriz de desenho pronta para o logit, com o índice de reamostragem."""

    endog: np.ndarray
    exog: np.ndarray
    names: tuple[str, ...]
    region_column: int
    row_index: SpeakerRowIndex
    speaker_codes: tuple[str, ...]
    stratum_of: tuple[str, ...]


def build_design(
    frame: pd.DataFrame,
    resamples: SpeakerResamples,
    *,
    covariates: Sequence[str],
) -> Design:
    """Monta a matriz de desenho à mão, sem fórmulas.

    Fórmulas (`patsy`/`formulaic`) escolheriam sozinhas a categoria de
    referência e a ordem das colunas, o que muda entre versões de biblioteca e
    quebra a comparabilidade byte a byte exigida pelo plano. Aqui a codificação é
    explícita: `região` vira o indicador `regiao_NE` (referência = SE, a direção
    reportada no texto) e cada covariável categórica vira dummies com a
    referência sendo o nível lexicograficamente menor.
    """
    if "region" not in covariates:
        raise ValueError(
            "D6: 'region' precisa estar em `covariates` — é a exposição do estudo, "
            f"não uma covariável opcional. Recebido: {list(covariates)}."
        )
    missing = [c for c in covariates if c not in frame.columns]
    if missing:
        raise ValueError(f"Covariável(is) ausente(s) do frame de ensaios: {missing}.")

    endog = frame["is_correct"].to_numpy(dtype=float)
    columns: list[np.ndarray] = [np.ones(len(frame), dtype=float)]
    names: list[str] = ["const"]

    region_values = frame["region"].astype(str).to_numpy()
    columns.append((region_values == REGION_EXPOSED).astype(float))
    names.append(f"regiao_{REGION_EXPOSED}")
    region_column = 1

    for covariate in covariates:
        if covariate == "region":
            continue
        series = frame[covariate]
        if pd.api.types.is_numeric_dtype(series):
            columns.append(series.to_numpy(dtype=float))
            names.append(covariate)
            continue
        levels = sorted(series.astype(str).unique())
        for level in levels[1:]:  # o menor nível é a referência
            columns.append((series.astype(str).to_numpy() == level).astype(float))
            names.append(f"{covariate}_{level}")

    return Design(
        endog=endog,
        exog=np.column_stack(columns),
        names=tuple(names),
        region_column=region_column,
        row_index=SpeakerRowIndex(resamples.speaker_codes, frame["speaker_code"]),
        speaker_codes=resamples.speaker_codes,
        stratum_of=resamples.stratum_of,
    )


def _expit(z: np.ndarray) -> np.ndarray:
    """Logística numericamente estável (evita `exp` de argumento grande)."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def fit_logit(endog: np.ndarray, exog: np.ndarray) -> np.ndarray | None:
    """Ajusta o logit e devolve os coeficientes, ou `None` se o ajuste falhar.

    Falha (separação perfeita, matriz singular, não convergência) devolve `None`
    em vez de coeficientes ruins: numa reamostra, coeficientes de um ajuste não
    convergido entrariam no IC como se fossem estimativas legítimas. O chamador
    CONTA as falhas e o relatório as mostra.
    """
    if endog.size == 0:
        return None
    if np.unique(endog).size < 2:
        # Sem variação no desfecho: a verossimilhança não tem máximo interior.
        return None
    if np.linalg.matrix_rank(exog) < exog.shape[1]:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = sm.Logit(endog, exog)
            result = model.fit(disp=0, method="newton", maxiter=100)
            if not result.mle_retvals.get("converged", False):
                result = model.fit(disp=0, method="bfgs", maxiter=500)
                if not result.mle_retvals.get("converged", False):
                    return None
        except (ValueError, RuntimeError, np.linalg.LinAlgError, OverflowError):
            return None
    params = np.asarray(result.params, dtype=float)
    if not np.all(np.isfinite(params)):
        return None
    return params


@dataclass(frozen=True)
class AdjustedContrast:
    """Acurácias ajustadas por região e a diferença entre elas.

    As duas acurácias são **padronizadas para a mesma população**: a distribuição
    de covariáveis do subconjunto analisado inteiro. Por isso `acc_ne - acc_se` é
    uma diferença "com tudo o mais igual" e não uma comparação entre dois grupos
    com composições etárias diferentes.
    """

    effect: float
    acc_exposed: float
    acc_baseline: float


def average_marginal_effect(
    endog: np.ndarray, exog: np.ndarray, region_column: int
) -> AdjustedContrast | None:
    """Efeito marginal médio da região sobre a acurácia (pontos, não log-odds).

    Contrafactual explícito: para CADA observação, prediz-se a probabilidade de
    acerto com `região = NE` e com `região = SE`, mantendo idade (e demais
    covariáveis) no valor observado; a estimativa é a média das diferenças.

    Não usamos `margeff` do statsmodels de propósito: para um regressor binário
    ele calcula por padrão a derivada parcial (`dy/dx`), que trata o indicador
    como contínuo — uma aproximação que não corresponde ao contraste discreto
    NE × SE que o texto reporta.
    """
    params = fit_logit(endog, exog)
    if params is None:
        return None
    exposed = exog.copy()
    exposed[:, region_column] = 1.0
    baseline = exog.copy()
    baseline[:, region_column] = 0.0
    p_exposed = float(_expit(exposed @ params).mean())
    p_baseline = float(_expit(baseline @ params).mean())
    return AdjustedContrast(
        effect=p_exposed - p_baseline,
        acc_exposed=p_exposed,
        acc_baseline=p_baseline,
    )


# --------------------------------------------------------------------------- #
# Bootstrap genérico sobre as reamostras de falantes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BootstrapResult:
    """Réplicas do bootstrap e o IC percentil derivado delas."""

    point: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_resamples: int
    n_failed: int
    replicates: np.ndarray

    @property
    def failure_fraction(self) -> float:
        return self.n_failed / self.n_resamples if self.n_resamples else 0.0


def bootstrap_over_speakers(
    statistic: Callable[[np.ndarray], float | None],
    point_statistic: Callable[[], float | None],
    resamples: SpeakerResamples,
    row_index: SpeakerRowIndex,
    *,
    ci_level: float = 0.95,
    max_failure_fraction: float = 0.05,
) -> BootstrapResult:
    """Aplica `statistic` a cada reamostra de falantes e devolve o IC percentil.

    `statistic` recebe as POSIÇÕES DE LINHA da reamostra (com repetição) e
    devolve o valor da estatística ou `None` se não for estimável naquela
    reamostra. Manter a estatística como parâmetro é o que permite que a
    regressão e a decomposição compartilhem exatamente o mesmo mecanismo — e,
    mais importante, exatamente as mesmas reamostras.

    Réplicas que falham são CONTADAS, nunca substituídas por zero nem ignoradas
    em silêncio: um IC calculado sobre 1.400 das 2.000 réplicas, sem que ninguém
    saiba, é o tipo de erro que este módulo existe para impedir. Acima de
    `max_failure_fraction` a função levanta erro em vez de devolver um IC que
    parece bom.

    Método: **bootstrap percentil**. Simples, sem suposição de simetria e
    diretamente interpretável; a alternativa (BCa) exigiria jackknife por falante
    e não muda a conclusão num desenho como este.
    """
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level deve estar em (0, 1); recebido {ci_level}.")
    point = point_statistic()
    if point is None:
        raise ValueError(
            "A estatística não pôde ser estimada na AMOSTRA COMPLETA. "
            "Sem estimativa pontual não há IC; verifique separação perfeita, "
            "colinearidade ou ausência de variação no desfecho."
        )

    values: list[float] = []
    n_failed = 0
    for index in range(resamples.n_resamples):
        rows = row_index.rows_for_draw(resamples.draws[index])
        value = statistic(rows)
        if value is None:
            n_failed += 1
            continue
        values.append(value)

    if n_failed and n_failed / resamples.n_resamples > max_failure_fraction:
        raise ValueError(
            f"{n_failed} de {resamples.n_resamples} reamostras não puderam ser "
            f"estimadas ({100 * n_failed / resamples.n_resamples:.1f}% > "
            f"{100 * max_failure_fraction:.1f}%). O IC seria calculado sobre um "
            "subconjunto enviesado das réplicas."
        )
    if not values:
        raise ValueError("Nenhuma reamostra estimável; IC indefinido.")

    replicates = np.asarray(sorted(values), dtype=float)
    alpha = (1.0 - ci_level) / 2.0
    ci_low, ci_high = np.quantile(replicates, [alpha, 1.0 - alpha], method="linear")
    return BootstrapResult(
        point=float(point),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        ci_level=ci_level,
        n_resamples=resamples.n_resamples,
        n_failed=n_failed,
        replicates=replicates,
    )


# --------------------------------------------------------------------------- #
# Estimando do TCC: diferença de acurácia ajustada NE − SE
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GapEstimate:
    """Diferença de acurácia ajustada NE − SE para uma configuração × condição."""

    model_key: str
    condition: str
    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    acc_ne_adjusted: float
    acc_se_adjusted: float
    acc_ne_raw: float
    acc_se_raw: float
    n_trials: int
    n_items: int
    n_speakers_ne: int
    n_speakers_se: int
    n_resamples: int
    n_failed_resamples: int
    covariates: tuple[str, ...]
    age_min: int
    age_max: int

    def to_row(self) -> dict[str, object]:
        return {
            "model_key": self.model_key,
            "condition": self.condition,
            "gap_ne_se": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "acc_ne_adjusted": self.acc_ne_adjusted,
            "acc_se_adjusted": self.acc_se_adjusted,
            "acc_ne_raw": self.acc_ne_raw,
            "acc_se_raw": self.acc_se_raw,
            "n_trials": self.n_trials,
            "n_items": self.n_items,
            "n_speakers_ne": self.n_speakers_ne,
            "n_speakers_se": self.n_speakers_se,
            "n_resamples": self.n_resamples,
            "n_failed_resamples": self.n_failed_resamples,
            "covariates": "+".join(self.covariates),
            "age_min": self.age_min,
            "age_max": self.age_max,
        }


def _mean_or_nan(series: pd.Series) -> float:
    """Média da série, ou NaN se ela estiver vazia.

    Região sem nenhum ensaio devolve NaN explícito e visível na tabela, nunca
    zero — zero seria lido como "acertou nada", e não como "não foi medido".
    """
    return float(series.mean()) if len(series) else float("nan")


def estimate_gap(
    frame: pd.DataFrame,
    resamples: SpeakerResamples,
    *,
    covariates: Sequence[str],
    ci_level: float = 0.95,
    age_min: int = 20,
    age_max: int = 69,
    model_key: str = "",
    condition: str = "",
    max_failure_fraction: float = 0.05,
) -> GapEstimate:
    """Ajusta o logit, calcula o efeito marginal médio e o IC por bootstrap.

    `frame` já deve estar restrito à faixa etária e a uma única combinação
    modelo × condição — `age_min`/`age_max` aqui só rotulam a saída, para que a
    tabela do TCC declare o recorte usado sem depender de memória.
    """
    if frame.empty:
        raise ValueError(f"Frame vazio para {model_key!r} × {condition!r}.")
    design = build_design(frame, resamples, covariates=covariates)
    full = average_marginal_effect(design.endog, design.exog, design.region_column)

    def _point() -> float | None:
        return None if full is None else full.effect

    def _replicate(rows: np.ndarray) -> float | None:
        if rows.size == 0:
            return None
        contrast = average_marginal_effect(
            design.endog[rows], design.exog[rows], design.region_column
        )
        return None if contrast is None else contrast.effect

    boot = bootstrap_over_speakers(
        _replicate,
        _point,
        resamples,
        design.row_index,
        ci_level=ci_level,
        max_failure_fraction=max_failure_fraction,
    )
    if full is None:  # pragma: no cover - `bootstrap_over_speakers` já teria falhado
        raise ValueError("Efeito marginal médio não estimável na amostra completa.")

    is_ne = frame["region"].astype(str) == REGION_EXPOSED
    return GapEstimate(
        model_key=model_key,
        condition=condition,
        estimate=boot.point,
        ci_low=boot.ci_low,
        ci_high=boot.ci_high,
        ci_level=ci_level,
        acc_ne_adjusted=full.acc_exposed,
        acc_se_adjusted=full.acc_baseline,
        acc_ne_raw=_mean_or_nan(frame.loc[is_ne, "is_correct"]),
        acc_se_raw=_mean_or_nan(frame.loc[~is_ne, "is_correct"]),
        n_trials=len(frame),
        n_items=int(frame["item_id"].nunique()),
        n_speakers_ne=int(frame.loc[is_ne, "speaker_code"].nunique()),
        n_speakers_se=int(frame.loc[~is_ne, "speaker_code"].nunique()),
        n_resamples=boot.n_resamples,
        n_failed_resamples=boot.n_failed,
        covariates=tuple(covariates),
        age_min=age_min,
        age_max=age_max,
    )


def gap_table(
    frame: pd.DataFrame,
    resamples: SpeakerResamples,
    *,
    covariates: Sequence[str],
    ci_level: float = 0.95,
    age_min: int = 20,
    age_max: int = 69,
    max_failure_fraction: float = 0.05,
) -> tuple[pd.DataFrame, list[str]]:
    """Uma linha por configuração × condição, em ordem canônica.

    Combinações que não puderem ser estimadas (separação perfeita, falta de
    variação, falantes de uma só região) entram na lista de avisos devolvida e
    aparecem no relatório — nunca somem da tabela sem explicação.
    """
    rows: list[dict[str, object]] = []
    warnings_out: list[str] = []
    pairs = frame[["model_key", "condition"]].drop_duplicates()
    keys = sorted((str(m), str(c)) for m, c in pairs.itertuples(index=False, name=None))
    for model_key, condition in keys:
        subset = frame[(frame["model_key"] == model_key) & (frame["condition"] == condition)]
        try:
            estimate = estimate_gap(
                subset.reset_index(drop=True),
                resamples,
                covariates=covariates,
                ci_level=ci_level,
                age_min=age_min,
                age_max=age_max,
                model_key=model_key,
                condition=condition,
                max_failure_fraction=max_failure_fraction,
            )
        except ValueError as exc:
            warnings_out.append(f"{model_key} × {condition}: não estimável — {exc}")
            continue
        rows.append(estimate.to_row())
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["model_key", "condition"]).reset_index(drop=True)
    return table, warnings_out


def summarize_resamples(resamples: SpeakerResamples) -> Mapping[str, object]:
    """Resumo do desenho de reamostragem, para o relatório da etapa."""
    first = resamples.stratum_counts(0)
    return {
        "n_resamples": resamples.n_resamples,
        "seed": resamples.seed,
        "stratify_by": resamples.stratify_by or "(não estratificado)",
        "n_speakers": resamples.n_speakers,
        "tamanho_por_estrato_observado": resamples.observed_stratum_sizes,
        "tamanho_por_estrato_na_reamostra_0": first,
    }
