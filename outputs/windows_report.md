# Janelamento de fala contínua (Etapa 1)

Artefato reprodutível e determinístico. Nenhum byte de áudio foi lido: as janelas são decididas inteiramente sobre metadados.

- **Gerado em (UTC):** 2026-10-04T14:50:03+00:00

## Proveniência e reprodutibilidade
- Dataset: `nilc-nlp/CORAA-MUPE-ASR`
- Revisão resolvida (commit SHA): `d3d5f9c699bd619f9e1c45681cabf214d1b080e1`
- Splits: train, validation, test
- SHA do config: `e2ac8f81d4e47f7e7f971332d09e2619aee3a2a5165568ef50c2aa6b91bc1957`
- Versão do código: `0.1.0`
- Origem do frame: prefetch paralelo de intervalos de bytes das colunas de metadados (range requests coalescidos sobre a URL de resolve da HF) + pyarrow em arquivo esparso de memória; coluna(s) de áudio nunca transferida(s)
- Versões das bibliotecas:
  - numpy: 2.4.6
  - pandas: 3.0.3
  - pyarrow: 24.0.0
  - PyYAML: 6.0.3
  - python: 3.11.9

## Configuração efetiva
Nenhum valor abaixo é default escondido no código: todos vêm do YAML e são ecoados aqui para que a tabela de janelas seja interpretável sozinha.

| chave | valor efetivo | decisão |
|---|---|---|
| `base_config` | `corpus_audit.yaml` | — |
| `seed` | 20260918 | — |
| `min_duration_s` | 30 | — |
| `max_duration_s` | 60 | — |
| `min_segments` | 8 | — |
| `max_segments` | 15 | — |
| `gap_tolerance_s` | 2 | — |
| `interviewer_turn` | `close_window` | **D1** |
| `n_windows_per_speaker` | 10 | — |
| `allow_fewer_windows` | true | **D5** |
| `reference_field` | `normalized_text` | **D7** |
| `age_range` | [20, 69] | — |
| `regions` | ['NE', 'SE'] | — |

## Trilha de exclusão — segmentos
Todo segmento do frame de entrada termina em exatamente uma linha abaixo. `sem_janela_candidata` são segmentos elegíveis que sobraram das bordas de uma corrida contígua (curta demais para fechar uma janela); `candidata_nao_sorteada` são segmentos de candidatas que a amostragem estratificada não sorteou.

- Entradas (segmentos): 317743
  - descartadas — candidata_nao_sorteada: 75961
  - descartadas — idade_fora_da_faixa: 69983
  - descartadas — nao_informante (P/1): 25954
  - descartadas — nao_informante (P/2): 1564
  - descartadas — nao_informante (P/3): 282
  - descartadas — regiao_fora_do_escopo: 50964
  - descartadas — sem_janela_candidata: 78713
  - subtotal descartado: 303421
- **Retidas: 14322**

## Trilha de exclusão — falantes informantes
Precedência fixa dos motivos (região antes de idade), para que um falante fora dos dois critérios seja contado uma única vez e sempre no mesmo.

- Entradas (falantes informantes): 289
  - descartadas — abaixo_da_meta_com_allow_fewer_windows_false: 0
  - descartadas — idade_fora_da_faixa: 67
  - descartadas — regiao_fora_do_escopo: 57
  - descartadas — sem_janela_candidata: 0
  - subtotal descartado: 124
- **Retidas: 165**

Falantes elegíveis (dentro de região e faixa etária), por região:
- **NE**: 33 falantes
- **SE**: 132 falantes

## Contiguidade: corridas e motivos de quebra
Uma *corrida* é a maior sequência de segmentos do informante que ainda conta como fala contínua. A detecção de turno de entrevistador acontece no frame COMPLETO, antes do filtro de informante — depois de filtrar, o turno `P/*` já não está no frame e a fala pareceria contínua.

- Corridas contíguas: 22.285
- Falantes elegíveis com ao menos um segmento: 165
- Quebras por motivo:
  - `troca_de_gravacao`: 0
  - `turno_de_entrevistador`: 7.989
  - `lacuna_temporal`: 14.131
  - total de quebras: 22.120
- Conferência: 165 falantes + 22.120 quebras = 22.285 corridas
- Turnos de entrevistador observados entre segmentos consecutivos do informante: 7.989 (D1 = `close_window`)

## Candidatas e seleção estratificada
As candidatas de um falante são DISJUNTAS e empacotadas pelo menor tamanho válido (maximiza o nº de candidatas, e com ele a cobertura da entrevista). A seleção divide as candidatas em 10 blocos de posição e sorteia uma por bloco, com semente derivada de `(seed, speaker_code)` por SHA-256.

Consequência a declarar: empacotar pelo menor tamanho válido concentra as janelas perto do piso de 30 s, não no meio da faixa [30, 60] s. É o preço de ter mais candidatas — e é o que a aritmética do D5 pressupõe ("com >= 8 segmentos por janela, N segmentos dão no máximo N/8 janelas"). A duração efetivamente obtida está tabulada em *Janelas selecionadas*, abaixo.

- Entradas (janelas candidatas): 10077
  - descartadas — candidata_nao_sorteada: 8475
  - subtotal descartado: 8475
- **Retidas: 1602**

Candidatas por falante — mín 1, mediana 55.0, média 61.1, máx 220.

## Janelas por falante (histograma)
| janelas por falante | falantes | |
|--:|--:|---|
| 1 (abaixo da meta) | 3 | ███ |
| 3 (abaixo da meta) | 2 | ██ |
| 7 (abaixo da meta) | 1 | █ |
| 8 (abaixo da meta) | 2 | ██ |
| 10 | 157 | ████████████████████████████████████████████████████████████ |

**D5 — falantes com menos de 10 candidatas: 8.** Com `allow_fewer_windows: true`, ficam com menos janelas; nunca se sobrepõem janelas para completar a meta (isso criaria pseudo-replicação dentro do próprio falante).
Falantes afetados (até 20): `MA_HV070`, `MA_HV151`, `MA_HV164`, `MA_HV165`, `MA_HV166`, `MA_HV169`, `MA_HV176`, `MB_HV093`

## Janelas selecionadas
- Total de janelas: 1.602
  - **NE**: 307
  - **SE**: 1.295
- Duração somada por janela — mín 30.01 s, mediana 35.89 s, média 39.03 s, máx 59.98 s
- Segmentos por janela — mín 8, mediana 8.0, máx 15
- Áudio a baixar na Etapa 2: 17.37 h
- Janelas com transcrição de referência vazia: 0

## Artefatos gerados
- `windows.parquet` — uma linha por janela (gitignorado: é derivado e regenerável a partir do config + revisão do dataset).
- `windows_report.md` — este relatório.
