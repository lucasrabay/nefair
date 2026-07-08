# Auditoria de metadados do corpus CORAA-MUPE

Artefato reprodutível e determinístico da Etapa 0. Nenhum áudio foi baixado ou decodificado; apenas metadados foram lidos.

- **Gerado em (UTC):** 2026-07-08T00:35:47+00:00

## Proveniência e reprodutibilidade
- Dataset: `nilc-nlp/CORAA-MUPE-ASR`
- Revisão pedida: `None`
- Revisão resolvida (commit SHA): `d3d5f9c699bd619f9e1c45681cabf214d1b080e1`
- Método de carga: prefetch paralelo de intervalos de bytes das colunas de metadados (range requests coalescidos sobre a URL de resolve da HF) + pyarrow em arquivo esparso de memória; coluna(s) de áudio nunca transferida(s)
- Shards parquet lidos: 85
- Footprint lido (metadados): 47.429 MB (colunas: audio_id, audio_name, file_path, speaker_type, speaker_code, speaker_gender, education, birth_state, birth_country, age, recording_year, audio_quality, start_time, end_time, duration, normalized_text, original_text, racial_category)
- Footprint evitado (áudio nunca transferido): 41.769 GB (colunas excluídas: audio)
- Versões das bibliotecas:
  - datasets: 5.0.0
  - huggingface_hub: 1.22.0
  - pandas: 3.0.3
  - pyarrow: 24.0.0
  - PyYAML: 6.0.3
  - python: 3.11.9

## Grão e splits
O grão é o SEGMENTO (não o falante). Linhas por split de origem (coluna `split` preservada na concatenação):
- `train`: 276.881 segmentos
- `validation`: 9.894 segmentos
- `test`: 30.968 segmentos
- **Total: 317.743 segmentos**

## Chave de falante escolhida
Contagem de falantes distintos por candidato (apenas informantes, `speaker_type == 'R'`):
- `speaker_code`: 289
- `speaker_code+audio_name`: 289

As duas contagens coincidem, então **`speaker_code` sozinho é a chave correta**: acrescentar `audio_name` não separa mais nenhum falante. Toda contagem de 'falantes' neste relatório usa essa chave.

## Sobreposição de falantes entre splits
- Falantes (por `speaker_code`) em mais de um split, no dataset completo: 10 (são códigos de entrevistador, reutilizados em várias entrevistas).
- Falantes informantes em mais de um split: 0 (sem vazamento entre train/validation/test no nível do informante).

## Filtragem (nenhuma linha dropada em silêncio)
- Total de segmentos: 317.743
- Excluídos por não serem informante (`speaker_type != 'R'`), por tipo:
  - `P/1`: 25.954
  - `P/2`: 1.564
  - `P/3`: 282
  - subtotal excluído: 27.800
  - (esses segmentos carregam as sentinelas `age == 0` e `birth_state == 'unknown'`, correlação 1:1)
- **Informantes retidos: 289.943 segmentos**
- Conferência: 27.800 + 289.943 = 317.743

## Mapeamento de região (deriva só de `birth_state`)
- **NE**: 39 falantes, 40.963 segmentos, 43.33 h
- **SE**: 193 falantes, 198.016 segmentos, 242.20 h
- **fora_de_escopo**: 57 falantes, 50.964 segmentos, 57.16 h

### Estados fora de escopo (nem NE nem SE) — listados, não dropados
| birth_state | falantes | segmentos | horas |
|---|--:|--:|--:|
| (vazio: nascido fora do Brasil) | 17 | 14.316 | 17.60 |
| Goiás | 2 | 1.901 | 2.61 |
| Mato Grosso Do Sul | 1 | 495 | 0.66 |
| Paraná | 2 | 1.702 | 2.78 |
| Pará | 19 | 18.451 | 19.46 |
| Rio Grande Do Sul | 5 | 5.180 | 5.95 |
| Rondônia | 11 | 8.919 | 8.08 |

O `birth_state` vazio corresponde a informantes nascidos fora do Brasil (`birth_country != 'Brazil'`, correlação 1:1): continuam contabilizados como informantes, mas sem região brasileira (fora de escopo).

## Distribuição etária por região (buckets provisórios)
Os buckets abaixo são PROVISÓRIOS. A distribuição etária BRUTA (idade a idade) está em `age_distribution_by_region.csv` — use-a para decidir as bordas depois de ver o confundidor (Nordeste tende a ser mais jovem).

| região | bucket | falantes | segmentos | horas |
|---|---|--:|--:|--:|
| NE | [16,25) | 2 | 1.766 | 1.57 |
| NE | [25,35) | 6 | 4.630 | 5.56 |
| NE | [35,45) | 8 | 9.131 | 9.06 |
| NE | [45,60) | 11 | 13.672 | 13.71 |
| NE | [60,120) | 12 | 11.764 | 13.44 |
| SE | [16,25) | 9 | 7.755 | 9.22 |
| SE | [25,35) | 13 | 8.906 | 11.55 |
| SE | [35,45) | 28 | 30.242 | 36.98 |
| SE | [45,60) | 58 | 60.976 | 77.66 |
| SE | [60,120) | 85 | 90.137 | 106.80 |

## Artefatos gerados
- `speaker_counts_by_state.csv` — falantes e horas por `birth_state`.
- `speaker_counts_by_region_age.csv` — cross-tab região × bucket etário.
- `age_distribution_by_region.csv` — distribuição etária bruta por região.
- `data_quality_report.csv` — sentinelas, `audio_quality`, `duration`, segmentos por falante.

## Inspeção de schema (bruta)
```
Linhas totais (splits concatenados): 317743
Colunas (19):

## Colunas, dtype e nulos
coluna                 dtype               n_nulos   % nulos
audio_id               int64                     0    0.000%
audio_name             str                       0    0.000%
file_path              str                       0    0.000%
speaker_type           str                       0    0.000%
speaker_code           str                       0    0.000%
speaker_gender         str                       0    0.000%
education              str                       0    0.000%
birth_state            str                       0    0.000%
birth_country          str                       0    0.000%
age                    int64                     0    0.000%
recording_year         int64                     0    0.000%
audio_quality          str                       0    0.000%
start_time             float32                   0    0.000%
end_time               float32                   0    0.000%
duration               float32                   0    0.000%
normalized_text        str                       0    0.000%
original_text          str                       0    0.000%
racial_category        str                       0    0.000%
split                  str                       0    0.000%

## Valores distintos das categóricas de baixa cardinalidade

### speaker_type: 4 distinto(s)
   'P/1'                                         25954
   'P/2'                                          1564
   'P/3'                                           282
   'R'                                          289943

### birth_state: 19 distinto(s)
   ''                                            14316
   'Alagoas'                                      3272
   'Bahia'                                       14736
   'Ceará'                                        4073
   'Espírito Santo'                               1388
   'Goiás'                                        1901
   'Mato Grosso Do Sul'                            495
   'Minas Gerais'                                17772
   'Paraná'                                       1702
   'Paraíba'                                      4276
   'Pará'                                        18451
   'Pernambuco'                                  13821
   'Piauí'                                         590
   'Rio De Janeiro'                              12002
   'Rio Grande Do Sul'                            5180
   'Rondônia'                                     8919
   'Sergipe'                                       195
   'São Paulo'                                  166854
   'unknown'                                     27800

### birth_country: 9 distinto(s)
   'Argentina'                                    1886
   'Brazil'                                     303427
   'Chile'                                        5037
   'Germany'                                      1821
   'Italy'                                         896
   'Japan'                                         615
   'Monaco'                                       1148
   'Nigeria'                                       396
   'Portugal'                                     2517

### education: 7 distinto(s)
   'college'                                     44483
   'elementary'                                  14201
   'high'                                         7686
   'master'                                       3855
   'none'                                         9633
   'phd'                                           674
   'unknown'                                    237211

### speaker_gender: 3 distinto(s)
   'F'                                          155956
   'M'                                          161384
   'X'                                             403

### audio_quality: 2 distinto(s)
   'high'                                       296330
   'low'                                         21413

### racial_category: 5 distinto(s)
   'Asian'                                        3988
   'Black'                                        4651
   'Pardo(mixed)'                                 4072
   'White'                                       27585
   'unknown'                                    277447
```
