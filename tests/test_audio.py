"""Testes do download seletivo de áudio (Etapa 2) — offline, sem tocar a HF.

A Etapa 2 existe por um único motivo: a coluna `audio` do CORAA-MUPE pesa
41,8 GB e as janelas da Etapa 1 precisam de uma fração minúscula disso. Toda a
lógica pura de `corpus/audio.py` serve a essa aposta — traduzir "quero estes
`audio_id`" em "baixe estes row groups" — e é ela que estes testes exercitam.

Três escolhas de método valem ser ditas:

1. O parquet de teste é escrito em `tmp_path` por `write_synthetic_parquet`, com
   a MESMA fixture usada na Etapa 1. Assim o `audio_id` de uma janela real
   localiza uma linha real do arquivo, e o plano de download pode ser conferido
   contra o footer do parquet — a fonte crua — e não contra as estruturas que o
   próprio `audio.py` construiu.
2. O teste central (`test_two_windows_touch_one_row_group_...`) mede a economia
   com números concretos. Se um dia ele passar a baixar metade do arquivo para
   16 segmentos, a estratégia inteira deixou de se pagar e o teste cai.
3. A metade de I/O de rede (`open_shard`, `read_shard_audio_ids`,
   `fetch_row_group_cells`, `index_shards`, `run_fetch_audio`) fica de fora: ela
   só exercita `RangeReader`/`SparseFile`, já cobertos pela Etapa 0, e exigiria
   rede e `HF_TOKEN`. O que sobra dela está registrado num teste com `skip`
   explícito no fim do arquivo, para que a ausência seja deliberada e visível.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from fixtures.synthetic_corpus import synthetic_frame, write_synthetic_parquet
from nefair.config import load_windows_config
from nefair.corpus.audio import (
    MANIFEST_COLUMNS,
    STATUS_DECODE_FAILED,
    STATUS_MISSING_SEGMENT,
    STATUS_OK,
    TARGET_SAMPLE_RATE,
    DownloadPlan,
    WindowAudio,
    attribute_bytes,
    build_manifest,
    build_plan,
    column_chunk_ranges,
    concatenate_segments,
    decode_segment,
    manifest_ledger,
    materialize_windows,
    plan_shard,
    required_audio_ids,
    resample,
    row_group_bounds,
    rows_to_row_groups,
    segment_bytes,
    sha256_of_file,
    window_wav_path,
)
from nefair.corpus.windows import build_windows

CONFIG_PATH = "configs/windows.yaml"

# Parâmetros do parquet sintético. Com 1226 linhas e 64 por row group saem 20
# row groups — número pequeno o bastante para ser conferido à mão e grande o
# bastante para que "poucos row groups" signifique alguma coisa.
ROW_GROUP_SIZE = 64
AUDIO_BYTES_PER_ROW = 4096
N_ROW_GROUPS = 20


# --------------------------------------------------------------------------- #
# Fixtures locais
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def config():
    """Config real do repositório: as janelas do teste são as que vão rodar."""
    return load_windows_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def frame():
    return synthetic_frame()


@pytest.fixture(scope="module")
def windows(frame, config):
    return build_windows(frame, config).windows


@pytest.fixture(scope="module")
def metadata(frame, tmp_path_factory):
    """Footer do parquet sintético — a fonte crua de todas as contas de bytes."""
    path = tmp_path_factory.mktemp("shard") / "shard-00000.parquet"
    write_synthetic_parquet(
        frame,
        path,
        row_group_size=ROW_GROUP_SIZE,
        audio_bytes_per_row=AUDIO_BYTES_PER_ROW,
    )
    return pq.ParquetFile(path).metadata


@pytest.fixture(scope="module")
def audio_ids(frame) -> np.ndarray:
    """Coluna `audio_id` NA ORDEM DAS LINHAS, como `plan_shard` a espera."""
    return frame["audio_id"].to_numpy().astype("int64")


@pytest.fixture(scope="module")
def row_of_audio_id(frame) -> dict[int, int]:
    """audio_id -> posição da linha no parquet. A referência independente."""
    return {int(audio_id): row for row, audio_id in enumerate(frame["audio_id"])}


def _expected_groups(row_of_audio_id, wanted) -> tuple[int, ...]:
    """Row groups esperados, calculados sem olhar para `audio.py`.

    Como o parquet é escrito com row groups de tamanho fixo, a linha `r` mora no
    grupo `r // ROW_GROUP_SIZE` — divisão inteira, e nada mais.
    """
    return tuple(sorted({row_of_audio_id[a] // ROW_GROUP_SIZE for a in wanted}))


def _wav_bytes(seconds: float, sample_rate: int = TARGET_SAMPLE_RATE, value: float = 0.25) -> bytes:
    """WAV PCM_16 em memória — o que uma célula da coluna `audio` carrega."""
    samples = np.full(int(round(seconds * sample_rate)), value, dtype="float32")
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, subtype="PCM_16", format="WAV")
    return buffer.getvalue()


def _cells_for(windows, frame) -> dict[int, bytes]:
    """Células de áudio com a duração EXATA que os metadados declaram.

    Fazer o áudio bater com o metadado é o que permite afirmar, no manifesto, que
    `duration_delta_s` é zero quando nada deu errado — e portanto que um delta
    diferente de zero significa divergência real, não ruído do teste.
    """
    durations = {int(row.audio_id): float(row.duration) for row in frame.itertuples()}
    return {
        audio_id: _wav_bytes(durations[audio_id])
        for window in windows
        for audio_id in window.segment_audio_ids
    }


# --------------------------------------------------------------------------- #
# Fronteiras de row group
# --------------------------------------------------------------------------- #
def test_row_group_bounds_covers_every_row_exactly_once(metadata):
    """`bounds` particiona [0, num_rows): monotônico e terminando em `num_rows`.

    Se a última fronteira não for exatamente `num_rows`, `searchsorted` joga as
    linhas finais num grupo inexistente — e o plano pede um row group que o
    arquivo não tem.
    """
    bounds = row_group_bounds(metadata)
    assert bounds.dtype == np.dtype("int64")
    assert len(bounds) == metadata.num_row_groups + 1 == N_ROW_GROUPS + 1
    assert bounds[0] == 0
    assert bounds[-1] == metadata.num_rows
    assert np.all(np.diff(bounds) > 0), "row group vazio ou fronteira fora de ordem"
    # Cada intervalo tem o tamanho que o footer declara para aquele row group.
    sizes = [metadata.row_group(g).num_rows for g in range(metadata.num_row_groups)]
    assert np.diff(bounds).tolist() == sizes
    assert sum(sizes) == metadata.num_rows


def test_rows_to_row_groups_matches_the_bounds_at_the_edges(metadata):
    """As bordas são onde um erro de `side=` ou de `-1` se esconde.

    Testamos a PRIMEIRA e a ÚLTIMA linha de cada grupo: um `side="left"` erraria
    a primeira, e um deslocamento de um erraria a última.
    """
    bounds = row_group_bounds(metadata)
    edges = []
    for group in range(metadata.num_row_groups):
        edges.extend([int(bounds[group]), int(bounds[group + 1]) - 1])
    grouped = rows_to_row_groups(bounds, edges)

    assert set(grouped) == set(range(metadata.num_row_groups))
    for group, rows in grouped.items():
        assert rows == (int(bounds[group]), int(bounds[group + 1]) - 1)
        for row in rows:
            assert bounds[group] <= row < bounds[group + 1]


def test_rows_to_row_groups_returns_sorted_rows_and_sorted_groups(metadata):
    """Ordem de entrada não pode vazar para o plano: o download é determinístico."""
    bounds = row_group_bounds(metadata)
    scrambled = [200, 3, 1225, 64, 0, 130]
    grouped = rows_to_row_groups(bounds, scrambled)
    assert list(grouped) == sorted(grouped)
    for rows in grouped.values():
        assert list(rows) == sorted(rows)
    # 0 e 3 caem no grupo 0; 64 no 1; 130 no 2; 200 no 3; 1225 é a última linha.
    assert grouped[0] == (0, 3)
    assert grouped[1] == (64,)
    assert grouped[2] == (130,)
    assert grouped[3] == (200,)
    assert grouped[metadata.num_row_groups - 1] == (1225,)


# --------------------------------------------------------------------------- #
# O coração da Etapa 2 — economia de bytes
# --------------------------------------------------------------------------- #
def test_two_windows_touch_one_row_group_and_avoid_almost_every_byte(
    metadata, audio_ids, windows, row_of_audio_id
):
    """Poucos segmentos ⇒ poucos row groups ⇒ quase nenhum byte baixado.

    É a justificativa inteira da etapa em um teste. Duas janelas (16 segmentos
    contíguos) tocam UM row group de 20, e isso custa menos de 10% dos bytes de
    áudio do shard. Na escala real, é a diferença entre baixar 41,8 GB e baixar
    alguns GB. Se esta desigualdade cair, a estratégia de row groups deixou de se
    pagar e alguém precisa saber disso antes de gastar a banda.
    """
    wanted = required_audio_ids(windows[:2])
    assert len(wanted) == 16, "duas janelas de 8 segmentos, sem sobreposição"

    plan = plan_shard(metadata, audio_ids, wanted, shard="shard-00000.parquet", split="train")

    assert plan.row_groups == _expected_groups(row_of_audio_id, wanted)
    assert len(plan.row_groups) == 1
    assert plan.n_row_groups_total == N_ROW_GROUPS
    assert plan.n_rows_targeted == 16
    assert not plan.is_empty

    # Os bytes conferidos contra o footer, grupo a grupo — não contra o módulo.
    measured = sum(column_chunk_ranges(metadata, "audio", [g])[1] for g in plan.row_groups)
    assert plan.audio_bytes_to_download == measured
    assert plan.audio_bytes_avoided == plan.audio_bytes_total - plan.audio_bytes_to_download

    assert plan.audio_bytes_to_download * 10 <= plan.audio_bytes_total
    assert plan.audio_bytes_avoided >= 0.9 * plan.audio_bytes_total


def test_each_window_lands_in_the_row_group_that_holds_its_rows(
    metadata, audio_ids, windows, row_of_audio_id
):
    """O mapa `audio_id -> row group` do plano bate linha a linha com o arquivo."""
    wanted = required_audio_ids(windows[:12])
    plan = plan_shard(metadata, audio_ids, wanted, shard="s", split="train")

    seen: set[int] = set()
    for group, ids_in_group in plan.audio_ids_by_group.items():
        for audio_id in ids_in_group:
            assert row_of_audio_id[audio_id] // ROW_GROUP_SIZE == group
            seen.add(audio_id)
        # E as linhas registradas para o grupo são as linhas desses `audio_id`.
        assert plan.rows_by_group[group] == tuple(sorted(row_of_audio_id[a] for a in ids_in_group))
    assert seen == set(wanted)


def test_asking_for_every_audio_id_downloads_the_whole_column(metadata, audio_ids):
    """O extremo oposto: pedir tudo não pode "economizar" nada.

    Um `audio_bytes_avoided` positivo aqui significaria que algum row group ficou
    de fora do plano — ou seja, segmento faltando no WAV final.
    """
    wanted = frozenset(int(a) for a in audio_ids)
    plan = plan_shard(metadata, audio_ids, wanted, shard="s", split="train")

    assert plan.row_groups == tuple(range(metadata.num_row_groups))
    assert plan.n_rows_targeted == metadata.num_rows == len(audio_ids)
    assert plan.audio_bytes_to_download == plan.audio_bytes_total
    assert plan.audio_bytes_avoided == 0


def test_asking_for_nothing_plans_nothing(metadata, audio_ids):
    """Shard sem nenhum segmento de interesse: nenhum row group, zero bytes."""
    plan = plan_shard(metadata, audio_ids, frozenset(), shard="s", split="train")
    assert plan.is_empty
    assert plan.row_groups == ()
    assert plan.rows_by_group == {}
    assert plan.n_rows_targeted == 0
    assert plan.audio_bytes_to_download == 0
    # Todo o shard foi evitado — é exatamente o que "evitado" quer dizer.
    assert plan.audio_bytes_avoided == plan.audio_bytes_total > 0


# --------------------------------------------------------------------------- #
# Intervalos de bytes no footer
# --------------------------------------------------------------------------- #
def test_column_chunk_ranges_matches_the_first_component_of_path_in_schema(tmp_path):
    """Coluna `struct<bytes, path>` aparece como DOIS chunks; ambos são `audio`.

    O dataset real usa o tipo `Audio` do `datasets`, que vira
    `audio.bytes` + `audio.path` no footer. Casar `path_in_schema` por igualdade
    simples devolveria zero intervalo — e o download traria um arquivo sem áudio.
    """
    n_rows, row_group_size = 8, 2
    table = pa.table(
        {
            "audio_id": pa.array(range(n_rows), pa.int64()),
            "audio": pa.array(
                [{"bytes": bytes((i,)) * 512, "path": f"data/{i}.wav"} for i in range(n_rows)],
                pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
        }
    )
    path = tmp_path / "struct.parquet"
    pq.write_table(table, path, row_group_size=row_group_size)
    metadata = pq.ParquetFile(path).metadata

    paths = [
        metadata.row_group(0).column(i).path_in_schema
        for i in range(metadata.row_group(0).num_columns)
    ]
    assert paths == ["audio_id", "audio.bytes", "audio.path"], "a fixture precisa ser struct"

    ranges, total = column_chunk_ranges(metadata, "audio", None)
    # Dois chunks por row group, 4 row groups.
    assert metadata.num_row_groups == n_rows // row_group_size == 4
    assert len(ranges) == 2 * metadata.num_row_groups == 8
    assert total == sum(end - start for start, end in ranges) > 0


def test_column_chunk_ranges_shrinks_when_row_groups_are_restricted(metadata):
    """Restringir os row groups é o que transforma o footer em economia real."""
    everything, total = column_chunk_ranges(metadata, "audio", None)
    subset, partial = column_chunk_ranges(metadata, "audio", [0, 1])

    assert len(everything) == metadata.num_row_groups == N_ROW_GROUPS
    assert len(subset) == 2
    assert set(subset) < set(everything)
    assert 0 < partial < total
    assert total == sum(end - start for start, end in everything)
    # Índices repetidos ou fora de ordem não podem duplicar o intervalo.
    assert column_chunk_ranges(metadata, "audio", [1, 0, 1])[0] == subset


def test_column_chunk_ranges_does_not_confuse_audio_with_audio_id(metadata):
    """`audio_id` começa com "audio" mas NÃO é subcoluna de `audio`.

    O casamento é por componente do caminho, não por prefixo de string: se fosse
    por prefixo, cada row group traria a coluna `audio_id` junto e os bytes
    reportados seriam maiores que os de fato baixados.
    """
    audio_ranges, audio_total = column_chunk_ranges(metadata, "audio", None)
    id_ranges, id_total = column_chunk_ranges(metadata, "audio_id", None)

    assert len(audio_ranges) == len(id_ranges) == metadata.num_row_groups
    assert not set(audio_ranges) & set(id_ranges)
    # A coluna de índice é ordens de grandeza mais barata — é por isso que a
    # indexação da Etapa 2 pode ler `audio_id` inteiro de cada shard.
    assert id_total * 10 < audio_total
    assert column_chunk_ranges(metadata, "coluna_inexistente", None) == ([], 0)


# --------------------------------------------------------------------------- #
# Plano completo
# --------------------------------------------------------------------------- #
def test_build_plan_reports_missing_audio_ids_instead_of_failing(metadata, audio_ids, windows):
    """Um `audio_id` inexistente é DEVOLVIDO, não levanta exceção nem some.

    Um segmento pedido pela Etapa 1 e ausente na revisão baixada é quebra de
    contrato entre etapas. A decisão (falhar? contabilizar?) é de quem chama, e
    só existe decisão se o fato chegar até lá.
    """
    real = required_audio_ids(windows[:3])
    ghost = 10**9  # nenhum audio_id da fixture chega perto disso
    assert ghost not in set(int(a) for a in audio_ids)
    wanted = real | {ghost}

    shard_plan = plan_shard(metadata, audio_ids, wanted, shard="s", split="train")
    plan = build_plan([shard_plan], wanted)

    assert plan.missing_audio_ids == (ghost,)
    assert shard_plan.n_rows_targeted == len(real), "o fantasma não vira linha"
    # E os ids encontrados continuam todos no plano.
    found = {a for ids in shard_plan.audio_ids_by_group.values() for a in ids}
    assert found == set(real)


def test_build_plan_adds_up_the_shards_and_ignores_the_empty_ones(metadata, audio_ids, windows):
    """Totais do plano = soma dos shards; `non_empty` filtra quem não será tocado."""
    wanted = required_audio_ids(windows[:2])
    touched = plan_shard(metadata, audio_ids, wanted, shard="shard-0.parquet", split="train")
    untouched = plan_shard(metadata, audio_ids, frozenset(), shard="shard-1.parquet", split="test")

    plan = build_plan([touched, untouched], wanted)

    assert plan.missing_audio_ids == ()
    assert plan.non_empty == (touched,)
    assert plan.n_row_groups == len(touched.row_groups) == 1
    assert plan.n_row_groups_total == 2 * N_ROW_GROUPS
    assert plan.bytes_to_download == touched.audio_bytes_to_download
    # O shard intocado contribui com TODOS os seus bytes para "evitados".
    assert plan.bytes_avoided == touched.audio_bytes_avoided + untouched.audio_bytes_total


def test_download_plan_frame_lists_only_the_shards_that_will_be_touched(
    metadata, audio_ids, windows
):
    """O CSV do plano é o que se confere antes de gastar banda: uma linha por
    shard que será de fato baixado, com os bytes VERDADEIROS daquele shard."""
    wanted = required_audio_ids(windows[:2])
    touched = plan_shard(metadata, audio_ids, wanted, shard="shard-0.parquet", split="train")
    untouched = plan_shard(metadata, audio_ids, frozenset(), shard="shard-1.parquet", split="test")
    frame = build_plan([touched, untouched], wanted).to_frame()

    assert list(frame.columns) == [
        "shard",
        "split",
        "n_row_groups_total",
        "n_row_groups_selected",
        "row_groups",
        "n_rows_targeted",
        "audio_bytes_to_download",
        "audio_bytes_avoided",
    ]
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["shard"] == "shard-0.parquet"
    assert row["split"] == "train"
    assert row["row_groups"] == ";".join(str(g) for g in touched.row_groups)
    assert row["n_rows_targeted"] == 16
    assert row["audio_bytes_to_download"] == touched.audio_bytes_to_download

    # Plano vazio ainda produz um frame com as colunas certas (CSV com cabeçalho).
    empty = DownloadPlan(shards=(), missing_audio_ids=()).to_frame()
    assert list(empty.columns) == list(frame.columns)
    assert empty.empty


def test_required_audio_ids_is_the_union_of_the_window_segments(windows):
    """Nem um segmento a mais (banda desperdiçada), nem um a menos (WAV truncado)."""
    union = {audio_id for window in windows for audio_id in window.segment_audio_ids}
    assert required_audio_ids(windows) == frozenset(union)
    # As janelas da Etapa 1 não se sobrepõem, então a união tem exatamente o
    # total de segmentos — se algum dia se sobrepuserem, este teste avisa.
    assert len(union) == sum(window.n_segments for window in windows)
    assert required_audio_ids([]) == frozenset()


# --------------------------------------------------------------------------- #
# Contabilidade de bytes (método do maior resto)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("weights", "total"),
    [
        ([1, 1, 1], 10),  # 10/3 não é inteiro: o clássico
        ([8, 8, 9, 15], 407767),  # bytes reais do shard sintético
        ([3, 1], 7),
        ([1], 5),
        ([5, 5], 0),
        ([0, 3, 0], 11),  # peso zero não pode receber resto
        ([1, 2, 3, 4, 5, 6, 7], 1),  # menos bytes que janelas
    ],
)
def test_attribute_bytes_sums_exactly_to_the_total(weights, total):
    """A coluna do manifesto tem de somar o total VERDADEIRO, sempre.

    É a razão de ser do método do maior resto: truncar perderia bytes e
    arredondar inventaria bytes, e em ambos os casos o manifesto passaria a
    contradizer o plano de download.
    """
    shares = attribute_bytes(weights, total)
    assert sum(shares) == total
    assert len(shares) == len(weights)
    assert all(isinstance(share, int) and share >= 0 for share in shares)
    # Ninguém recebe mais que seu teto nem menos que seu piso.
    for weight, share in zip(weights, shares, strict=True):
        exact = total * weight / sum(weights) if sum(weights) else 0
        assert int(exact) <= share <= int(exact) + 1


def test_attribute_bytes_is_zero_when_there_is_no_weight_to_divide_by():
    """Sem peso não há repartição possível — e dividir por zero não é opção."""
    assert attribute_bytes([], 100) == []
    assert attribute_bytes([0, 0, 0], 100) == [0, 0, 0]
    assert attribute_bytes([-1, 1], 100) == [0, 0]


def test_attribute_bytes_follows_the_weights():
    """Mais segmentos ⇒ mais bytes atribuídos. A convenção precisa ser monótona."""
    shares = attribute_bytes([1, 2, 3], 600)
    assert shares == [100, 200, 300]
    ordered = attribute_bytes([15, 8, 8], 1000)
    assert ordered[0] > ordered[1]
    assert attribute_bytes([15, 8, 8], 1000) == ordered, "a repartição é determinística"


# --------------------------------------------------------------------------- #
# Manifesto
# --------------------------------------------------------------------------- #
def test_build_manifest_has_exactly_the_declared_columns_sorted_by_window_id(windows):
    """Colunas fixas e ordem estável: o manifesto é artefato comparável entre
    execuções, e uma coluna a mais/a menos quebra quem o lê rio abaixo."""
    results = [
        WindowAudio(window=window, status=STATUS_OK, n_samples=16_000, sha256="f" * 64)
        for window in reversed(windows[:6])  # entra fora de ordem de propósito
    ]
    manifest = build_manifest(results, bytes_downloaded=1000, bytes_avoided=9000)

    assert tuple(manifest.columns) == MANIFEST_COLUMNS
    assert len(manifest) == 6
    assert manifest["window_id"].tolist() == sorted(manifest["window_id"])
    assert list(manifest.index) == list(range(6))
    assert (manifest["sample_rate"] == TARGET_SAMPLE_RATE).all()


def test_build_manifest_splits_the_plan_totals_without_losing_bytes(windows):
    """As colunas de bytes somam EXATAMENTE os totais do plano de download."""
    results = [
        WindowAudio(window=window, status=STATUS_OK, n_samples=1, sha256="a" * 64)
        for window in windows[:7]
    ]
    manifest = build_manifest(results, bytes_downloaded=407_767, bytes_avoided=41_800_000_000)

    assert manifest["bytes_downloaded"].sum() == 407_767
    assert manifest["bytes_avoided"].sum() == 41_800_000_000


def test_build_manifest_reports_the_two_durations_side_by_side(windows, frame):
    """Duração pedida e obtida em colunas separadas, com a diferença numa terceira.

    Quando o áudio bate com os metadados, o delta é zero; qualquer outra coisa é
    sinal de que os cortes do corpus não correspondem aos arquivos — e precisa
    ser visível no CSV sem recalcular nada.
    """
    window = windows[0]
    n_samples = int(round(window.duration_s * TARGET_SAMPLE_RATE))
    manifest = build_manifest([WindowAudio(window, STATUS_OK, n_samples=n_samples)], 0, 0)
    row = manifest.iloc[0]

    assert row["duration_requested_s"] == pytest.approx(window.duration_s, abs=1e-4)
    assert row["duration_obtained_s"] == pytest.approx(window.duration_s, abs=1e-4)
    assert row["duration_delta_s"] == pytest.approx(0.0, abs=1e-4)
    assert row["n_segments"] == window.n_segments
    assert row["segment_audio_ids"] == ";".join(str(a) for a in window.segment_audio_ids)
    assert row["region"] == window.region

    # Metade do áudio esperado ⇒ delta negativo de meia janela, sem esconder nada.
    truncated = build_manifest([WindowAudio(window, STATUS_OK, n_samples=n_samples // 2)], 0, 0)
    assert truncated.iloc[0]["duration_delta_s"] == pytest.approx(-window.duration_s / 2, abs=1e-3)


def test_manifest_ledger_counts_each_status_in_its_category(windows):
    """Nenhuma janela some sem motivo: `ok` retém, o resto entra pelo seu status."""
    results = [
        WindowAudio(windows[0], STATUS_OK, n_samples=10),
        WindowAudio(windows[1], STATUS_OK, n_samples=10),
        WindowAudio(windows[2], STATUS_MISSING_SEGMENT),
        WindowAudio(windows[3], STATUS_DECODE_FAILED),
        WindowAudio(windows[4], STATUS_DECODE_FAILED),
    ]
    ledger = manifest_ledger(results)

    ledger.assert_balanced()
    assert ledger.unit == "janelas"
    assert ledger.total_in == 5
    assert ledger.kept == 2
    assert ledger.dropped == {STATUS_MISSING_SEGMENT: 1, STATUS_DECODE_FAILED: 2}
    assert manifest_ledger([]).total_in == 0


# --------------------------------------------------------------------------- #
# Utilidades puras — caminhos, células e concatenação
# --------------------------------------------------------------------------- #
def test_window_wav_path_nests_the_wav_under_the_speaker(windows, tmp_path):
    """`<outputs>/audio/<speaker_code>/<window_id>.wav`.

    Um diretório por falante mantém a unidade de análise visível no disco: é por
    falante que a Etapa 6 agrupa, e é por falante que se audita à mão.
    """
    window = windows[0]
    path = window_wav_path(tmp_path, window)
    assert path == tmp_path / "audio" / window.speaker_code / f"{window.window_id}.wav"
    assert path.relative_to(tmp_path).parts == ("audio", window.speaker_code, path.name)
    # Aceita `str` além de `Path` — as chamadas vêm do CLI.
    assert window_wav_path(str(tmp_path), window) == path


def test_concatenate_segments_of_an_empty_list_is_an_empty_float32_array():
    """Janela sem segmento decodificado não pode virar exceção de `np.concatenate`:
    o caso é legítimo (janela descartada) e precisa fluir até o status."""
    empty = concatenate_segments([])
    assert empty.shape == (0,)
    assert empty.dtype == np.dtype("float32")


def test_concatenate_segments_preserves_order_and_length():
    """Sem crossfade nem silêncio: a duração é a soma exata das partes."""
    pieces = [np.full(3, 1.0, dtype="float32"), np.full(2, 2.0, dtype="float32")]
    joined = concatenate_segments(pieces)
    assert joined.dtype == np.dtype("float32")
    assert joined.tolist() == [1.0, 1.0, 1.0, 2.0, 2.0]
    assert len(joined) == sum(len(piece) for piece in pieces)


def test_segment_bytes_accepts_both_shapes_of_the_audio_cell():
    """`struct<bytes, path>` (dataset real) e `binary` puro (fixture sintética)."""
    assert segment_bytes({"bytes": b"abc", "path": "data/x.wav"}) == b"abc"
    assert segment_bytes(b"abc") == b"abc"
    assert segment_bytes(bytearray(b"abc")) == b"abc"
    assert segment_bytes(memoryview(b"abc")) == b"abc"


def test_segment_bytes_rejects_a_cell_without_embedded_audio():
    """Só `path` significa áudio NÃO embutido — baixar o parquet não traria o som.

    Falhar alto aqui é melhor que devolver `b""` e produzir um WAV mudo.
    """
    with pytest.raises(ValueError, match="sem campo `bytes`"):
        segment_bytes({"path": "data/x.wav"})


def test_segment_bytes_rejects_an_unexpected_type():
    with pytest.raises(TypeError, match="tipo inesperado"):
        segment_bytes(3.14)


# --------------------------------------------------------------------------- #
# Decodificação e reamostragem
# --------------------------------------------------------------------------- #
def test_decode_segment_averages_the_channels_instead_of_taking_the_left_one():
    """Mono por MÉDIA, não pelo canal esquerdo.

    Descartar um canal perderia energia de fala se o informante estiver mais
    forte de um lado — e isso entraria no WER como se fosse sotaque. Aqui o canal
    esquerdo vale 1,0 e o direito 0,0: a média dá 0,5 e o "pega o esquerdo" 1,0.
    """
    stereo = np.zeros((800, 2), dtype="float32")
    stereo[:, 0] = 1.0
    buffer = io.BytesIO()
    sf.write(buffer, stereo, 8_000, subtype="PCM_16", format="WAV")

    samples, sample_rate = decode_segment(buffer.getvalue())

    assert sample_rate == 8_000
    assert samples.shape == (800,)
    assert samples.dtype == np.dtype("float32")
    assert samples.mean() == pytest.approx(0.5, abs=1e-3)


def test_resample_keeps_the_duration_and_is_a_no_op_at_the_target_rate():
    """48 kHz → 16 kHz preserva a duração; na taxa alvo não se toca no sinal.

    `scipy` chega via `statsmodels` mas não está declarado em `pyproject.toml`:
    sem ele a reamostragem é impossível e o teste é pulado em vez de mentir.
    """
    pytest.importorskip("scipy", reason="reamostragem exige scipy (não declarado no projeto)")

    original = np.sin(np.arange(48_000, dtype="float32") * 2 * np.pi * 220 / 48_000)
    downsampled = resample(original, 48_000)
    assert len(downsampled) == TARGET_SAMPLE_RATE
    assert downsampled.dtype == np.dtype("float32")

    untouched = resample(original[:16_000], TARGET_SAMPLE_RATE)
    assert untouched.tolist() == original[:16_000].tolist()


# --------------------------------------------------------------------------- #
# Materialização das janelas (pura: recebe as células já baixadas)
# --------------------------------------------------------------------------- #
def test_materialize_windows_writes_one_wav_per_window_with_the_declared_duration(
    windows, frame, tmp_path
):
    """Caminho feliz: WAV por janela, duração igual à soma dos segmentos, hash real."""
    chosen = list(windows[:3])
    results = materialize_windows(chosen, _cells_for(chosen, frame), tmp_path)

    assert [result.status for result in results] == [STATUS_OK] * 3
    for result, window in zip(results, chosen, strict=True):
        path = window_wav_path(tmp_path, window)
        assert path.exists()
        assert result.wav_path == str(path.relative_to(tmp_path))
        assert result.n_samples == int(round(window.duration_s * TARGET_SAMPLE_RATE))
        # O hash é conferido contra o arquivo lido de uma vez só — o leitor em
        # blocos de `sha256_of_file` não pode divergir do hash direto.
        assert result.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert result.sha256 == sha256_of_file(path)

        samples, sample_rate = sf.read(path, dtype="float32")
        assert sample_rate == TARGET_SAMPLE_RATE
        assert len(samples) == result.n_samples


def test_materialize_windows_flags_a_window_whose_segment_was_not_downloaded(windows, frame):
    """Segmento ausente não vira WAV incompleto: vira status, e a janela sai inteira."""
    window = windows[0]
    cells = _cells_for([window], frame)
    del cells[window.segment_audio_ids[0]]

    results = materialize_windows([window], cells, "/caminho/que/nao/sera/usado")

    assert [result.status for result in results] == [STATUS_MISSING_SEGMENT]
    assert results[0].n_samples == 0
    assert results[0].wav_path == ""


def test_materialize_windows_flags_a_window_whose_bytes_do_not_decode(windows, tmp_path):
    """Falha de codec é contabilizada como motivo, não propagada como exceção."""
    window = windows[0]
    cells = {audio_id: b"isto nao e um wav" for audio_id in window.segment_audio_ids}

    results = materialize_windows([window], cells, tmp_path)

    assert [result.status for result in results] == [STATUS_DECODE_FAILED]
    assert not window_wav_path(tmp_path, window).exists()


# --------------------------------------------------------------------------- #
# Metade de I/O — registrada, não testada
# --------------------------------------------------------------------------- #
@pytest.mark.skip(
    reason="exige rede e HF_TOKEN: open_shard/read_shard_audio_ids/fetch_row_group_cells/"
    "index_shards/run_fetch_audio fazem range requests contra huggingface.co. O que essas "
    "funções acrescentam sobre a Etapa 0 é só a escolha de colunas e row groups, e essa "
    "escolha é a lógica pura já coberta acima."
)
def test_network_half_requires_huggingface():
    raise AssertionError("nunca executa")
