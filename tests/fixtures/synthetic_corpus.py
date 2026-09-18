"""Corpus sintético determinístico com a forma exata do CORAA-MUPE.

Existe para que TODA a suíte rode sem rede, sem `HF_TOKEN` e sem tocar os
41,8 GB de áudio do dataset real — e, ainda assim, exercite os casos que
realmente quebram o janelamento.

As 19 colunas e seus dtypes espelham o schema empírico confirmado na Etapa 0
(ver `outputs/corpus_audit_report.md`). Os casos-limite não são acidentais:
cada um corresponde a uma decisão do plano que precisa de teste.

| falante          | o que está plantado                    | testa                  |
|------------------|----------------------------------------|------------------------|
| `NE_LONG`        | 160 segmentos contíguos                | caminho feliz, 10 jan. |
| `NE_SHORT`       | 30 segmentos — não dá 10 janelas de ≥ 8 | D5                    |
| `NE_INTERRUPTED` | turnos `P/1` no meio da fala           | D1                     |
| `SE_GAP`         | salto temporal de 12 s no meio         | `gap_tolerance_s`      |
| `SE_MULTI_AUDIO` | dois `audio_name` distintos            | janela não cruza grav. |
| `SE_LOW_QUALITY` | metade dos segmentos `audio_quality` low | confundidor de gravação |
| `SE_OUT_OF_AGE`  | 72 anos                                | recorte 20–69          |
| `XX_FOREIGN`     | `birth_state == ''` (fora do Brasil)   | fora de escopo         |

Determinismo: nenhuma chamada a `random`; todo texto e toda idade derivam de
contadores e de um hash estável do código do falante.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

# Duração fixa por segmento. Com 4,0 s: 8 segmentos = 32 s e 15 segmentos = 60 s,
# ou seja, a faixa [8, 15] segmentos cai exatamente dentro de [30, 60] s. Isso
# torna os testes de limite legíveis à mão.
SEGMENT_DURATION = 4.0

# Vocabulário mínimo para gerar transcrições estáveis e distinguíveis.
_WORDS = (
    "a gente",
    "na época",
    "meu pai",
    "trabalhava",
    "na roça",
    "aí depois",
    "a família",
    "mudou",
    "pra cidade",
    "e foi",
    "assim mesmo",
    "que aconteceu",
)


@dataclass(frozen=True)
class SpeakerPlan:
    """Receita de um falante sintético."""

    speaker_code: str
    birth_state: str
    age: int
    n_segments: int
    gender: str = "F"
    education: str = "unknown"
    racial_category: str = "unknown"
    split: str = "train"
    birth_country: str = "Brazil"
    # Posições (índice do segmento) onde um turno de entrevistador é inserido.
    interviewer_at: tuple[int, ...] = ()
    # Posição a partir da qual se abre uma lacuna temporal, e seu tamanho.
    gap_at: int | None = None
    gap_seconds: float = 12.0
    # Nº de gravações distintas (`audio_name`) entre as quais os segmentos são
    # divididos igualmente.
    n_audios: int = 1
    # Fração inicial de segmentos marcados como `low`.
    low_quality_until: int = 0


# Plano fixo do corpus sintético. Alterar esta tupla altera os números esperados
# em vários testes de uma vez — por isso ela vive aqui, e não dentro de um teste.
DEFAULT_PLAN: tuple[SpeakerPlan, ...] = (
    SpeakerPlan("NE_LONG", "Bahia", 34, 160),
    SpeakerPlan("NE_SHORT", "Pernambuco", 41, 30),
    SpeakerPlan("NE_INTERRUPTED", "Ceará", 29, 120, interviewer_at=(20, 21, 60, 95)),
    SpeakerPlan("NE_MID", "Paraíba", 55, 96, gender="M"),
    SpeakerPlan("SE_LONG", "São Paulo", 38, 160, gender="M", split="train"),
    SpeakerPlan("SE_GAP", "Minas Gerais", 47, 120, gap_at=50),
    SpeakerPlan("SE_MULTI_AUDIO", "Rio De Janeiro", 33, 128, n_audios=2),
    SpeakerPlan("SE_LOW_QUALITY", "Espírito Santo", 26, 120, low_quality_until=60),
    SpeakerPlan("SE_OUT_OF_AGE", "São Paulo", 72, 96, gender="M"),
    SpeakerPlan("SE_TEST_SPLIT", "São Paulo", 31, 96, split="test"),
    SpeakerPlan("XX_FOREIGN", "", 44, 96, birth_country="Portugal"),
)


def _text_for(speaker_code: str, index: int, n_words: int = 6) -> str:
    """Transcrição estável e distinguível para um segmento.

    Deriva de um hash do par (falante, índice): dois segmentos nunca têm o mesmo
    texto por acaso, e o texto de um segmento nunca muda entre execuções.
    """
    digest = hashlib.sha256(f"{speaker_code}:{index}".encode()).digest()
    words = [_WORDS[digest[i] % len(_WORDS)] for i in range(n_words)]
    return " ".join(words)


def _rows_for_speaker(plan: SpeakerPlan, audio_id_start: int) -> list[dict]:
    """Linhas (segmentos) de um falante, em ordem temporal."""
    rows: list[dict] = []
    audio_id = audio_id_start
    clock = 0.0
    segments_per_audio = max(1, plan.n_segments // plan.n_audios)

    for index in range(plan.n_segments):
        audio_part = min(index // segments_per_audio, plan.n_audios - 1)
        audio_name = (
            f"{plan.speaker_code}_entrevista"
            if plan.n_audios == 1
            else f"{plan.speaker_code}_entrevista_{audio_part + 1}"
        )
        # Troca de gravação reinicia o relógio: `start_time` é relativo à gravação.
        if index > 0 and audio_part != min((index - 1) // segments_per_audio, plan.n_audios - 1):
            clock = 0.0

        # Lacuna temporal plantada (testa `gap_tolerance_s`).
        if plan.gap_at is not None and index == plan.gap_at:
            clock += plan.gap_seconds

        # Turno de entrevistador plantado (testa D1). Ocupa tempo e quebra a
        # contiguidade da fala do informante — mas só é visível ANTES do filtro.
        if index in plan.interviewer_at:
            rows.append(
                {
                    "audio_id": audio_id,
                    "audio_name": audio_name,
                    "file_path": f"data/{audio_name}/{audio_id}.wav",
                    "speaker_type": "P/1",
                    "speaker_code": f"{plan.speaker_code}_ENTREVISTADOR",
                    "speaker_gender": "M",
                    "education": "unknown",
                    "birth_state": "unknown",
                    "birth_country": "Brazil",
                    "age": 0,
                    "recording_year": 2015,
                    "audio_quality": "high",
                    "start_time": clock,
                    "end_time": clock + SEGMENT_DURATION,
                    "duration": SEGMENT_DURATION,
                    "normalized_text": "e como foi que aconteceu",
                    "original_text": "E como foi que aconteceu?",
                    "racial_category": "unknown",
                    "split": plan.split,
                }
            )
            audio_id += 1
            clock += SEGMENT_DURATION

        rows.append(
            {
                "audio_id": audio_id,
                "audio_name": audio_name,
                "file_path": f"data/{audio_name}/{audio_id}.wav",
                "speaker_type": "R",
                "speaker_code": plan.speaker_code,
                "speaker_gender": plan.gender,
                "education": plan.education,
                "birth_state": plan.birth_state,
                "birth_country": plan.birth_country,
                "age": plan.age,
                "recording_year": 2015,
                "audio_quality": "low" if index < plan.low_quality_until else "high",
                "start_time": clock,
                "end_time": clock + SEGMENT_DURATION,
                "duration": SEGMENT_DURATION,
                "normalized_text": _text_for(plan.speaker_code, index),
                "original_text": _text_for(plan.speaker_code, index).capitalize() + ".",
                "racial_category": plan.racial_category,
                "split": plan.split,
            }
        )
        audio_id += 1
        clock += SEGMENT_DURATION

    return rows


def synthetic_frame(plan: tuple[SpeakerPlan, ...] = DEFAULT_PLAN) -> pd.DataFrame:
    """DataFrame com as 19 colunas reais do corpus e os casos-limite plantados.

    Os dtypes replicam o schema real (`start_time`/`end_time`/`duration` em
    `float32`) porque a precisão de `float32` é justamente o que exige a
    tolerância de contiguidade — testar em `float64` esconderia o problema.
    """
    rows: list[dict] = []
    audio_id = 1
    for speaker in plan:
        speaker_rows = _rows_for_speaker(speaker, audio_id)
        rows.extend(speaker_rows)
        audio_id += len(speaker_rows)

    frame = pd.DataFrame(rows)
    for column in ("start_time", "end_time", "duration"):
        frame[column] = frame[column].astype("float32")
    for column in ("audio_id", "age", "recording_year"):
        frame[column] = frame[column].astype("int64")
    return frame


# Mapa de região suficiente para a fixture (mesmas grafias do config real).
REGION_MAP: dict[str, str] = {
    "Bahia": "NE",
    "Pernambuco": "NE",
    "Ceará": "NE",
    "Paraíba": "NE",
    "São Paulo": "SE",
    "Minas Gerais": "SE",
    "Rio De Janeiro": "SE",
    "Espírito Santo": "SE",
}


def write_synthetic_parquet(
    frame: pd.DataFrame,
    path,
    *,
    row_group_size: int = 64,
    audio_bytes_per_row: int = 4096,
) -> None:
    """Escreve a fixture como parquet COM uma coluna `audio`, para testar a
    leitura seletiva da Etapa 2.

    A coluna `audio` é preenchida com bytes sintéticos só para ocupar espaço
    real no arquivo: o que se testa é o cálculo de quais row groups precisam ser
    baixados, não o conteúdo do áudio.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    audio = pa.array(
        [bytes((i % 251,)) * audio_bytes_per_row for i in range(len(frame))],
        type=pa.binary(),
    )
    table = table.append_column("audio", audio)
    pq.write_table(table, path, row_group_size=row_group_size)
