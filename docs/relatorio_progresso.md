# Relatório de progresso — decisões e medições

**Data:** 2026-10-08
**Escopo:** Etapas 0–2 executadas sobre o corpus real; Etapas 3–6 implementadas e testadas, ainda não executadas.
**Finalidade:** servir de fonte única para a redação da Metodologia, da seção de Amostra e da seção de Limitações. Todo número abaixo é rastreável ao artefato que o produziu.

> **Convenção:** cada medição vem com o arquivo que a gerou. Nenhum número neste relatório foi estimado ou arredondado de cabeça — se um valor não puder ser reproduzido rodando o script indicado, é um defeito deste relatório.

---

## 1. Estado do pipeline

| Etapa | O que faz | Estado | Artefato |
|---|---|---|---|
| 0 | Auditoria de metadados do corpus | ✅ executada | `outputs/corpus_audit_report.md` |
| 1 | Janelamento de fala contínua | ✅ executada | `outputs/windows.parquet`, `outputs/windows_report.md` |
| 2 | Download seletivo de áudio | ✅ executada no piloto | `outputs/audio/` (120 WAV), `outputs/audio_manifest.csv` |
| 3 | Geração de itens + filtro textual + revisão | ⏸️ código pronto, não executada | — |
| 4–5 | Execução item × configuração × condição | ⏸️ falta `scripts/06_run_eval.py` | — |
| 6 | Regressão, decomposição, sensibilidade | ⏸️ falta `analysis/export.py` e `scripts/07_analyze.py` | — |

Suíte de testes: **320 passando, 1 ignorado**, toda offline (sem credencial, sem rede, sem SDK de provedor instalado).

---

## 2. Decisões fechadas

Todas vivem como chave nomeada em YAML. Nenhuma é default escondido no código — o relatório de cada etapa ecoa a configuração efetiva usada, para que a tabela de resultados seja interpretável isoladamente.

| Id | Questão | Decisão | Onde vive | Quando |
|---|---|---|---|---|
| **D1** | Turno do entrevistador no meio da janela | `close_window` — um turno `P/*` entre dois segmentos do informante fecha a janela | `configs/windows.yaml` | antes |
| **D2** | Mecanismo dos distratores | Extraídos de outras janelas da mesma entrevista, com proveniência obrigatória | `configs/items.yaml` | antes |
| **D3** | Controle textual | 3 modelos × 3 execuções, temperatura 0,7, ordem embaralhada; descarta se ≥ 2 de 3 acertarem | `configs/items.yaml` | antes |
| **D4** | Teto da tarefa | `constructed` = 1,0 | `configs/analysis.yaml` | **2026-10-07, orientador** |
| **D5** | Falantes sem 10 janelas disjuntas | `allow_fewer_windows: true` — aceitar menos e **contar**, nunca sobrepor | `configs/windows.yaml` | antes |
| **D6** | Gênero como covariável | **Entra** no modelo | `configs/analysis.yaml` | **2026-10-02, orientador** |
| **D7** | Campo de referência e normalização do WER | `normalized_text`, normalizador `v1`, aplicado igualmente a referência e hipótese | `configs/windows.yaml`, `configs/analysis.yaml` | antes |
| **D8** | Contaminação do ASR | ⚠️ **em aberto** — `contamination_checked: false` bloqueia a execução completa | `configs/models.yaml` | — |
| **D9** | Modelos sem áudio nativo | Entram como cascata extra, fora da comparação pareada | `configs/models.yaml` | antes |
| **D10** | Sensibilidade ao nº de alternativas | Desligada por padrão; se ligar, só no piloto | `configs/analysis.yaml` | antes |
| **D11** | LLM gerador | Fora do conjunto avaliado; validado por `model_id` na carga do config | `configs/items.yaml` | antes |
| **—** | Tolerância de pausa (`gap_tolerance_s`) | **2,0 s**, fixado com evidência | `configs/windows.yaml` | **2026-10-03** |
| **—** | Cobertura da revisão humana | 100% do piloto; percentual do conjunto completo a definir após o gate | — | **2026-10-07, orientador** |

### 2.1 Como escrever o D4 no texto

A decomposição do erro é:

```
Δ_ASR          = acc(reference) − acc(asr)
Δ_compreensão  = teto − acc(reference)
```

Assumir `teto = 1,0` **por construção** significa afirmar que todo item que sobreviveu ao filtro textual e à revisão humana é respondível sem ambiguidade a partir da transcrição de referência. É uma afirmação sobre o instrumento, não sobre o modelo — e é exatamente o que a revisão humana existe para sustentar. O modo usado (`constructed` vs `human_sample`) é **rotulado na saída**, porque `Δ_compreensão` não é comparável entre modos.

Consequência a declarar: se a revisão do piloto encontrar taxa de rejeição alta, a hipótese `teto = 1,0` fica frágil e `human_sample` volta à mesa.

### 2.2 Como escrever o D6 no texto

O modelo é `acerto ~ região + idade + gênero`. A justificativa empírica para incluir gênero e **não** escolaridade ou categoria racial está na cobertura dos metadados (§8): gênero tem cobertura completa, os outros dois são majoritariamente `unknown`.

---

## 3. A amostra: do corpus aos elegíveis

### 3.1 Corpus bruto

Fonte: `nilc-nlp/CORAA-MUPE-ASR`, revisão `d3d5f9c699bd619f9e1c45681cabf214d1b080e1`.

| | Valor |
|---|---|
| Segmentos totais | 317.743 |
| — `train` | 276.881 |
| — `validation` | 9.894 |
| — `test` | 30.968 |
| Falantes informantes (`speaker_type == 'R'`) | 289 |

**Chave de falante:** `speaker_code` sozinho. Verificado empiricamente — `speaker_code` dá 289 falantes distintos e `speaker_code + audio_name` dá os mesmos 289, logo acrescentar a gravação não separa ninguém.

**Vazamento entre splits:** zero no nível do informante. Dez `speaker_code` aparecem em mais de um split, e todos são códigos de **entrevistador**, reutilizados entre entrevistas.

### 3.2 Composição por região (antes de qualquer filtro do estudo)

| Região | Falantes | Segmentos | Horas |
|---|--:|--:|--:|
| NE | 39 | 40.963 | 43,33 |
| SE | 193 | 198.016 | 242,20 |
| fora de escopo | 57 | 50.964 | 57,16 |
| **Total** | **289** | **289.943** | — |

Região deriva **só** de `birth_state`. Os 57 fora de escopo (Norte, Sul, Centro-Oeste e 17 nascidos fora do Brasil) são listados e contados, nunca dropados em silêncio.

*Fonte: `outputs/corpus_audit_report.md`, `outputs/speaker_counts_by_state.csv`.*

### 3.3 Funil de exclusão — segmentos

Todo segmento de entrada termina em **exatamente uma** linha. A conferência fecha.

| Motivo | Segmentos |
|---|--:|
| Entrada | 317.743 |
| — não informante (`P/1`) | 25.954 |
| — não informante (`P/2`) | 1.564 |
| — não informante (`P/3`) | 282 |
| — região fora do escopo | 50.964 |
| — idade fora da faixa 20–69 | 69.983 |
| — sem janela candidata (bordas de corrida curta) | 78.713 |
| — candidata não sorteada pela amostragem | 75.961 |
| **Retidos** | **14.322** |

`303.421 + 14.322 = 317.743` ✓

### 3.4 Funil de exclusão — falantes

Precedência fixa dos motivos (região antes de idade), para que um falante fora dos dois critérios seja contado uma vez só e sempre no mesmo lugar.

| Motivo | Falantes |
|---|--:|
| Entrada (informantes) | 289 |
| — região fora do escopo | 57 |
| — idade fora da faixa 20–69 | 67 |
| — abaixo da meta com `allow_fewer_windows: false` | 0 |
| — sem janela candidata | 0 |
| **Elegíveis** | **165** |

**Elegíveis por região: NE 33, SE 132.**

### 3.5 ⚠️ Correção de um número que passei antes

Na conversa anterior eu afirmei que *"a restrição etária derruba proporcionalmente mais nordestinos"*. **Isso está errado.** A medição é o oposto:

| Região | Corpus | Elegíveis | Perda |
|---|--:|--:|--:|
| NE | 39 | 33 | −6 (−15,4%) |
| SE | 193 | 132 | −61 (−31,6%) |
| **Razão SE:NE** | **4,95 : 1** | **4,00 : 1** | — |

O recorte 20–69 remove proporcionalmente **mais sudestinos**, porque o SE é mais velho no corpus: 85 dos 193 falantes do SE (44%) estão na faixa de 60+, contra 12 dos 39 do NE (31%), e o teto de 69 anos corta mais daquele lado.

Ou seja: o recorte etário **atenua** levemente o desbalanço, de ~1:5 para ~1:4. Continua sendo um desbalanço de 4 para 1 e continua sendo limitação a declarar — mas a direção do efeito é a inversa da que eu descrevi, e isso muda a frase que entra no texto.

### 3.6 Idade dos elegíveis

| Região | n | média | sd | mín | p25 | p50 | p75 | máx |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| NE | 33 | 45,7 | 13,8 | 20 | 36 | 45 | 55 | 68 |
| SE | 132 | 48,3 | 12,2 | 21 | 40 | 49 | 58 | 69 |

Por década:

| Faixa | NE | SE |
|---|--:|--:|
| 20–29 | 5 | 13 |
| 30–39 | 6 | 17 |
| 40–49 | 8 | 39 |
| 50–59 | 8 | 35 |
| 60–69 | 6 | 28 |

As duas distribuições são comparáveis (diferença de médias ~2,6 anos, desvios próximos). Isso é favorável: a idade entra como covariável de controle sem sobreposição problemática entre os grupos.

---

## 4. A definição operacional de "fala contínua"

O texto do TCC diz "fala contínua" sem quantificar. Isso tornou necessária uma decisão metodológica nossa, e ela precisa aparecer na Metodologia.

### 4.1 O que foi medido

Distribuição das pausas entre segmentos consecutivos do mesmo informante — **168.831 lacunas medidas**:

| Percentil | Pausa |
|---|--:|
| p50 | 0,10 s |
| p75 | 1,02 s |
| p90 | 2,62 s |

### 4.2 A varredura

| Tolerância | Janelas | NE | SE | Falantes | Com as 10 | Duração mediana |
|--:|--:|--:|--:|--:|--:|--:|
| 0,5 s | 329 | 62 | 267 | 89 | 12 | 34,76 s |
| 1,0 s | 1.398 | 262 | 1.136 | 160 | 119 | 35,29 s |
| 1,5 s | 1.581 | 301 | 1.280 | 164 | 151 | 35,68 s |
| **2,0 s** | **1.602** | **307** | **1.295** | **165** | **157** | **35,89 s** |
| 3,0 s | 1.609 | 312 | 1.297 | 165 | 158 | 36,06 s |
| 5,0 s | 1.616 | 315 | 1.301 | 165 | 158 | 36,62 s |
| 10,0 s | 1.626 | 321 | 1.305 | 165 | 160 | 36,36 s |

*Fonte: `outputs/gap_tolerance_sweep.csv`.*

### 4.3 Por que 2,0 s

- **Cobre 87,4% das pausas observadas** e 97% do teto teórico de janelas (1.602 de 1.650).
- **A curva satura ali.** De 2 s a 10 s ganham-se 24 janelas (+1,5%), ao custo de "fala contínua" passar a admitir uma pausa de dez segundos — o que seria indefensável no texto.
- **O valor anterior (0,5 s) era inviável.** Cobria só 59,6% das pausas e picava a fala em corridas de ~2,5 segmentos, abaixo do mínimo de 8 para fechar uma janela: produzia 329 janelas, contra as 1.602 obtidas a 2,0 s. Pior, deixava só 12 falantes com as 10 janelas da meta.

Frase sugerida para a Metodologia: *"Dois segmentos consecutivos do informante são tratados como fala contínua quando a pausa entre eles não excede 2,0 s. O valor foi fixado a partir da distribuição empírica das 168.831 pausas do corpus (p90 = 2,62 s) e de uma varredura de sensibilidade, reportada em [tabela]; a 2,0 s o critério cobre 87,4% das pausas observadas e o número de janelas satura."*

---

## 5. As janelas obtidas

### 5.1 Configuração efetiva

| Chave | Valor |
|---|---|
| `min_duration_s` / `max_duration_s` | 30,0 / 60,0 |
| `min_segments` / `max_segments` | 8 / 15 |
| `gap_tolerance_s` | 2,0 |
| `interviewer_turn` | `close_window` (D1) |
| `n_windows_per_speaker` | 10 |
| `allow_fewer_windows` | `true` (D5) |
| `reference_field` | `normalized_text` (D7) |
| `age_range` | [20, 69] |
| `regions` | [NE, SE] |
| `seed` | 20260918 |

As duas condições — duração somada **e** contagem de segmentos — valem simultaneamente.

### 5.2 Contiguidade

| | Valor |
|---|--:|
| Corridas contíguas | 22.285 |
| Quebras por turno de entrevistador | 7.989 |
| Quebras por lacuna temporal | 14.131 |
| Quebras por troca de gravação | 0 |
| **Total de quebras** | **22.120** |

`165 falantes + 22.120 quebras = 22.285 corridas` ✓

Nota de implementação com consequência metodológica: a detecção de turno de entrevistador acontece no frame **completo**, antes do filtro de informante. Se filtrássemos primeiro, o turno `P/*` já não estaria no frame e a fala pareceria contínua — o D1 seria inoperante por construção.

### 5.3 Candidatas e seleção

As candidatas de um falante são **disjuntas** e empacotadas pelo menor tamanho válido, o que maximiza o número de candidatas e, com ele, a cobertura da entrevista.

| | Valor |
|---|--:|
| Candidatas geradas | 10.077 |
| Candidatas por falante — mín / mediana / média / máx | 1 / 55 / 61,1 / 220 |
| Não sorteadas | 8.475 |
| **Janelas selecionadas** | **1.602** |

A seleção divide as candidatas de cada falante em 10 blocos de posição e sorteia uma por bloco, com semente derivada de `(seed, speaker_code)` por SHA-256 — nunca de um RNG global compartilhado entre falantes.

**Consequência a declarar:** empacotar pelo menor tamanho válido concentra as janelas perto do piso de 30 s, não no meio da faixa. É o preço de ter mais candidatas, e aparece na distribuição de duração (mediana 35,89 s numa faixa de 30–60 s).

### 5.4 Totais

| | NE | SE | Total |
|---|--:|--:|--:|
| Falantes | 33 | 132 | 165 |
| Janelas | 307 | 1.295 | **1.602** |
| Horas de áudio | 3,16 | 14,21 | **17,37** |

- Duração por janela: mín 30,01 s, mediana 35,89 s, média 39,03 s, máx 59,98 s
- Segmentos por janela: mín 8, mediana 8, máx 15
- Janelas com transcrição de referência vazia: **0**
- Teto teórico (165 × 10): 1.650 → obtidas 1.602, **déficit de 48**

---

## 6. Assimetrias NE×SE medidas no instrumento

Esta seção é nova e, na minha leitura, é o material mais relevante para o artigo depois do funil da amostra. São três assimetrias entre as regiões que **não** vêm dos modelos avaliados — vêm do corpus e do nosso janelamento. Se não forem declaradas, qualquer diferença de acurácia que encontrarmos fica parcialmente atribuível a elas.

### 6.1 Falantes abaixo da meta de 10 janelas (D5)

Oito dos 165 falantes não comportam 10 janelas disjuntas. **Cinco são do NE.**

| Região | Falante | Janelas |
|---|---|--:|
| NE | `MA_HV169` | 1 |
| NE | `MA_HV151` | 3 |
| NE | `MA_HV070` | 7 |
| NE | `MA_HV165` | 8 |
| NE | `MB_HV093` | 8 |
| SE | `MA_HV164` | 1 |
| SE | `MA_HV166` | 1 |
| SE | `MA_HV176` | 3 |

| Região | Falantes abaixo da meta | % dos falantes da região | Déficit de janelas | % do déficit total |
|---|--:|--:|--:|--:|
| NE | 5 | **15,2%** | 23 | 47,9% |
| SE | 3 | **2,3%** | 25 | 52,1% |

O NE é 20% dos falantes elegíveis, mas concentra 62,5% dos falantes afetados. Em termos relativos, um falante do NE tem ~6,6× mais chance de não comportar as 10 janelas.

### 6.2 Os segmentos do NE são mais curtos

| Região | Duração média do segmento | Segmentos por janela (média) | Duração da janela (média) |
|---|--:|--:|--:|
| NE | 4,11 s | 9,49 | 37,03 s |
| SE | 4,64 s | 8,81 | 39,51 s |

Os segmentos do NE são ~11% mais curtos. Como o mínimo de 8 segmentos por janela é fixo, isso faz as janelas do NE empacotarem **mais** segmentos para menos duração total. Mais fronteiras de segmento por janela significa mais oportunidades de pausa acima da tolerância — o que é consistente com 6.1.

**O que isso pode ser:** diferença de prática de segmentação entre os responsáveis pela transcrição, diferença de taxa de fala, ou diferença de estilo de entrevista. Os metadados do corpus não permitem distinguir essas hipóteses. É limitação a declarar, e é um candidato natural a análise de sensibilidade — a grade por faixa de duração de janela (`sensitivity.window_duration`, já ligada em `configs/analysis.yaml`) cobre parcialmente isso.

### 6.3 ⚠️ A distribuição por split inviabiliza a mitigação prevista para o D8

| Split | Janelas NE | Janelas SE | Falantes NE | Falantes SE |
|---|--:|--:|--:|--:|
| `train` | 196 | 1.225 | 21 | 125 |
| `validation` | 20 | 30 | 2 | 3 |
| `test` | 91 | 40 | **10** | **4** |

O `test` do CORAA-MUPE é fortemente enviesado para o NE: 29,6% das janelas do NE vêm dele, contra 3,1% das do SE.

O `configs/models.yaml` prevê, como mitigação de contaminação do Whisper, `split_restriction: test`. **Essa mitigação não é viável.** Restringir ao `test` deixaria:

- 131 janelas (91 NE / 40 SE), 1,34 h de áudio
- **10 falantes do NE e 4 do SE**

Com 4 falantes num braço, o bootstrap no nível do falante estratificado por região perde qualquer sentido — o intervalo de confiança do SE seria dominado pela reamostragem de 4 pessoas. E o desbalanço se **inverte**, ficando 2,3:1 a favor do NE.

**Consequência para o D8:** a única via defensável é usar um ASR cujo treino esteja documentado — o checkpoint publicado `whisper-large-v3`, servido por um host que confirme servir os pesos publicados — e verificar o card de treino contra CORAA/MuPe. Se houver sobreposição, a alternativa é o Whisper base, não o recorte de split.

### 6.4 Por que criar um split próprio não resolve o D8

Pergunta levantada em 2026-10-08, e vale registrar a resposta porque o raciocínio é fácil de errar.

**Contaminação é propriedade do conjunto de treino do ASR, não da nossa partição.** A divisão que importa foi desenhada por quem treinou o modelo. Se o CORAA entrou no treino, ele entrou — rotular linhas como "nosso test" hoje não as remove de um treino já ocorrido. Um split nosso não tem efeito algum sobre o que o modelo viu.

O `test` publicado do CORAA tinha *algum* valor por ser a **convenção**: quem faz fine-tuning sobre um corpus tende a respeitar a divisão oficial. É uma aposta sobre o comportamento de terceiros, fraca mas não nula. Uma fronteira que nós inventamos não carrega essa propriedade.

**Nota de desenho:** este não é um estudo de treinamento. Não há ajuste em `train` com avaliação em `test` — as 1.602 janelas entram todas na análise, independentemente do split, e a coluna `split` é apenas proveniência. O split só entra na discussão por causa do D8.

#### A variante que funciona: holdout por pertinência documentada

Se o card do ASR documentar a pertinência ao treino (declarar o split usado, ou listar arquivos), então o holdout defensável é o **complemento daquela lista** — sob medida para aquele modelo, e estritamente melhor que o `test` do corpus. Isso é o que o D8 manda verificar. Se o card não documentar, não há recorte possível, porque não se sabe o que recortar.

#### Sonda de contaminação — medir em vez de assumir

O viés do `test` para o NE, que o inviabiliza como restrição (§6.3), o torna aproveitável como sonda.

| Região | Falantes em `train` | Falantes em `test` | Idade média (train / test) |
|---|--:|--:|---|
| NE | 21 | **10** | 45,8 / 46,3 |
| SE | 125 | 4 | 48,7 / 45,5 |

**Desenho:** comparar o WER sobre falantes de `train` contra falantes de `test`, **dentro da mesma região**. Se o ASR memorizou o `train`, o WER daquele lado deve ser visivelmente menor. No NE a comparação é 21 contra 10 falantes; a idade está equilibrada (45,8 vs 46,3), então esse confundidor não polui o contraste.

**Limites a reportar junto com o resultado:**

- **É entre falantes, não dentro do falante.** Zero informantes aparecem em mais de um split, então a diferença mistura contaminação com variação individual. Com 21 vs 10 falantes, só um efeito grande seria detectável.
- **Detecta, nunca descarta.** Se o treino cobriu o corpus inteiro, WER(train) ≈ WER(test) e a sonda não acusa nada. Resultado nulo é fracamente reconfortante, não prova de ausência.
- **O SE não serve** para a sonda: 4 falantes.

Ainda assim, troca "suponho que não há contaminação" por "procurei e não achei, com esta sensibilidade" — que é o máximo afirmável sem acesso ao treino.

#### Ordem recomendada para fechar o D8

1. **Não usar um fine-tune PT-BR.** A preocupação concentra-se nas variantes do Whisper ajustadas para português, várias das quais listam o CORAA no treino. O `whisper-large-v3` publicado não foi ajustado neste corpus e tem o treino descrito no artigo. Risco residual: o CORAA pode estar entre as 680k h de áudio web do pré-treino — supervisão fraca de varredura, não ajuste direcionado, e declarável como tal.
2. **Verificar o card** do checkpoint efetivamente servido pelo host contra CORAA/MuPe.
3. **Rodar a sonda** acima e reportar o número.
4. Só então `contamination_checked: true`.

#### Onde um split nosso seria legítimo

Por higiene de análise, não por contaminação: `gap_tolerance_s = 2,0` foi escolhido olhando o dado completo (§4). Para um estudo descritivo com a varredura publicada isso é defensável, mas um revisor rigoroso pode objetar. Um split nosso de desenvolvimento × final endereçaria essa objeção. Não tem relação com o D8.

---

## 7. O áudio do piloto

### 7.1 Seleção

12 falantes, 6 por região, sorteados entre os elegíveis com semente 20260918 (`items.yaml → pilot`). Todos com as 10 janelas completas.

| NE | SE |
|---|---|
| `MA_HV007`, `MA_HV038`, `MA_HV078`, `MA_HV132`, `MA_HV197`, `MA_HV266` | `MA_HV030`, `MA_HV034`, `MA_HV073`, `MA_HV086`, `MA_HV228`, `MA_HV264` |

**O piloto é balanceado por construção: 60 janelas de cada região.** Isso importa porque a taxa de descarte do filtro textual e o tempo de revisão serão medidos sem o desbalanço 4:1 do conjunto completo contaminar a estimativa.

Todas as 120 janelas do piloto caíram no split `train`.

### 7.2 Resultado do download

| | Valor |
|---|--:|
| Janelas solicitadas | 120 |
| Janelas com `status = ok` | **120 (100%)** |
| Falhas | 0 |
| Segmentos concatenados | 1.081 |
| Áudio obtido | 4.680,7 s = **78,0 min** |
| Em disco | 143 MB |
| Taxa de amostragem | 16.000 Hz em 120/120 |
| Duração por janela — mín / mediana / máx | 30,08 s / 36,27 s / 59,39 s |

**Fidelidade temporal.** Diferença entre a duração pedida (metadados) e a obtida (áudio decodificado):

| | Valor |
|---|--:|
| Mínimo | −0,0170 s |
| Máximo | +0,0200 s |
| Média | +0,00065 s |

O erro máximo é de 20 ms sobre janelas de 30–60 s, ou ~0,05%. Isso confirma que a reconstrução por concatenação de segmentos está correta e que os `start_time`/`end_time` dos metadados são confiáveis no grão do segmento.

**Unicidade.** 120 sha256 distintos em 120 arquivos. Isso não é redundância: é a verificação direta de que o bug descrito em §11.1 está morto. Com aquele bug, as 10 janelas de um mesmo falante teriam recebido o áudio idêntico da entrevista inteira, e veríamos 12 hashes, não 120.

### 7.3 Eficiência do download seletivo

| | Valor |
|---|--:|
| Bytes transferidos | 1,971 GB |
| Bytes evitados | 34,63 GB |
| Razão | **17,6×** |

O download opera em granularidade de *row group* do Parquet, via HTTP range requests sobre a URL de resolve da Hugging Face. Transferimos 1,97 GB para extrair 143 MB de áudio útil: a diferença é o resto de cada row group tocado. Baixar os shards inteiros custaria 36,6 GB.

Isso é relevante para o texto como nota de reprodutibilidade: **o estudo completo não exige baixar os 41,8 GB do dataset.** As 17,37 h de áudio das 1.602 janelas são obtíveis seletivamente.

---

## 8. Painel de confundidores — a justificativa do D6

A escolha de covariáveis não foi teórica: foi determinada pela cobertura real dos metadados.

### 8.1 Gênero (cobertura completa → entra)

| Região | F | M | X | Cobertura |
|---|--:|--:|--:|--:|
| NE | 21 | 18 | 0 | 100% |
| SE | 87 | 106 | 0 | 100% |

Nenhum `unknown`. Serve de controle. A proporção difere entre regiões (NE 54% F, SE 45% F), o que é exatamente o motivo de controlar.

### 8.2 Escolaridade (cobertura ruim → fora)

| Região | none | elementary | high | college | master | phd | unknown | % unknown |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| NE | 4 | 4 | 4 | 3 | 0 | 0 | 24 | **61,5%** |
| SE | 2 | 7 | 3 | 33 | 4 | 1 | 143 | **74,1%** |

### 8.3 Categoria racial (cobertura pior → fora)

| Região | White | Black | Pardo | Asian | unknown | % unknown |
|---|--:|--:|--:|--:|--:|--:|
| NE | 1 | 1 | 1 | 0 | 36 | **92,3%** |
| SE | 19 | 3 | 3 | 3 | 165 | **85,5%** |

Com 85–92% de ausência, incluir categoria racial no modelo não controlaria nada — estimaria um coeficiente sobre a minoria que declarou, e a taxa de declaração é ela própria correlacionada com região.

### 8.4 Qualidade de áudio (confundidor de gravação, a reportar)

| Região | Segmentos `high` | Segmentos `low` | % `low` | Falantes com ≥ 1 `low` |
|---|--:|--:|--:|--:|
| NE | 39.147 | 1.816 | **4,43%** | 38 de 39 |
| SE | 185.390 | 12.626 | **6,38%** | 192 de 193 |

O SE tem **mais** segmentos de baixa qualidade, em termos relativos. Isso é favorável ao desenho: se o NE tivesse áudio pior, qualquer perda de acurácia no NE seria confundida com qualidade de gravação. A direção observada é a inversa, então a qualidade de áudio não explicaria um resultado desfavorável ao NE. Vale uma frase no texto.

### 8.5 Concentração por falante (pseudo-replicação)

| Região | Falantes | mín | mediana | média | máx | Fração do falante mais prolífico |
|---|--:|--:|--:|--:|--:|--:|
| NE | 39 | 87 | 1.034 | 1.050 | 3.010 | **7,35%** |
| SE | 193 | 64 | 962 | 1.026 | 2.954 | **1,49%** |

No NE, um único falante responde por 7,35% dos segmentos da região. É o principal argumento empírico para o bootstrap ser **no nível do falante**, e não no nível do item: reamostrar itens trataria 1.034 segmentos de uma pessoa como 1.034 observações independentes.

---

## 9. Reprodutibilidade e proveniência

Material para a seção de Reprodutibilidade.

| | Valor |
|---|---|
| Dataset | `nilc-nlp/CORAA-MUPE-ASR` |
| Revisão resolvida (commit SHA) | `d3d5f9c699bd619f9e1c45681cabf214d1b080e1` |
| SHA do config da Etapa 1 | `e2ac8f81d4e47f7e7f971332d09e2619aee3a2a5165568ef50c2aa6b91bc1957` |
| Versão do código | `0.1.0` |
| Semente global | 20260918 |
| Python | 3.11.9 |
| numpy / pandas / pyarrow / PyYAML | 2.4.6 / 3.0.3 / 24.0.0 / 6.0.3 |

Garantias que o código sustenta:

1. **Determinismo total.** Nenhuma aleatoriedade não semeada; ordenação estável em toda agregação. Duas execuções da Etapa 1 produzem `windows.parquet` idêntico byte a byte.
2. **Semente por falante.** Derivada de `(seed, speaker_code)` por SHA-256. Acrescentar ou remover um falante não altera as janelas sorteadas dos outros.
3. **Nada é dropado em silêncio.** Toda linha excluída é contada com motivo, e `ExclusionLedger.assert_balanced()` falha se entradas ≠ descartados + retidos.
4. **Configuração ecoada.** Cada relatório reproduz a configuração efetiva que o gerou, para que a tabela seja interpretável sozinha.
5. **Resolução de revisão.** A revisão do dataset é resolvida para um commit SHA e registrada, não referenciada por `main`.

---

## 10. Para a seção de Limitações

Lista fechada do que as medições acima obrigam a declarar:

1. **Desbalanço regional de ~4:1** (33 NE / 132 SE). Vem do corpus (39/193), e o recorte etário o atenua levemente em vez de agravá-lo (§3.5). Nenhuma escolha estatística o conserta; a estratificação do bootstrap apenas evita que o IC fique largo por artefato de reamostragem.
2. **"Fala contínua" é uma operacionalização nossa**, fixada em 2,0 s com evidência empírica (§4). Outro valor daria outro conjunto de janelas — a varredura está versionada para que o leitor avalie a sensibilidade.
3. **As janelas concentram-se perto do piso de 30 s** (mediana 35,89 s), consequência de empacotar pelo menor tamanho válido para maximizar candidatas (§5.3).
4. **Falantes do NE têm ~6,6× mais chance de não comportar as 10 janelas** (15,2% vs 2,3%), o que torna o painel ligeiramente desbalanceado também no nível do falante (§6.1).
5. **Os segmentos do NE são ~11% mais curtos**, de causa indeterminável pelos metadados (§6.2).
6. **O `test` do corpus é enviesado para o NE**, o que inviabiliza o recorte de split como mitigação de contaminação (§6.3) — e criar um split próprio não substitui, porque contaminação é propriedade do treino do ASR, não da nossa partição (§6.4).
7. **Escolaridade e categoria racial não entram como controles** por ausência de 61–92% nos metadados (§8.2, §8.3).
8. **Concentração por falante no NE** — 7,35% dos segmentos num único informante (§8.5).
9. **Teto da tarefa assumido como 1,0 por construção** (D4), afirmação sobre o instrumento que depende da revisão humana para se sustentar (§2.1).
10. **Se o Qwen rodar por endpoint hospedado**, não há como garantir que não seja uma versão quantizada dos pesos Apache-2.0. Os pesos são públicos, então a reprodutibilidade está preservada em princípio, mas a execução específica não é auditável.

---

## 11. Correções de rumo com consequência metodológica

Três defeitos encontrados e corrigidos antes de qualquer dado ser gerado. Entram no texto só se houver seção de desenvolvimento do instrumento, mas importam para a defesa.

### 11.1 `audio_id` identifica a gravação, não o segmento

O código da Etapa 2 casava segmentos por `audio_id`, supondo que fosse a chave do segmento. **É a chave da entrevista inteira** — um único `audio_id` cobre mais de mil linhas. Verificado num shard: 4 `audio_id` distintos contra 3.871 `file_path` distintos em 3.871 linhas.

Se tivesse passado, cada janela teria recebido o áudio da entrevista completa: 25,77 GB de download e áudio errado em 100% das janelas, sem erro visível. A chave correta é `file_path`, única por linha.

**Como foi encontrado:** pelo `--dry-run` sobre o corpus real, não pelos 320 testes. Os testes concordaram com o bug porque eu escrevi a fixture com a mesma suposição errada do código — ela dava `audio_id` incremental por linha. A fixture foi corrigida para refletir a estrutura real (12 `audio_id` para 1.226 linhas), e com isso 6 testes passaram a falhar antes da correção do código.

**A lição metodológica:** teste contra fixture que você mesmo escreveu não valida suposição sobre o schema. Isso só se valida contra o dado real. É o motivo de o princípio "verificar o schema empiricamente antes de escrever lógica de análise" estar no topo das regras do projeto.

### 11.2 `gap_tolerance_s` em 0,5 s

Descrito em §4.3. Produzia 329 janelas (20% do viável) e deixava 153 dos 165 falantes abaixo da meta. Não era bug de código — era um valor plausível escolhido sem medir.

### 11.3 Números relatados a partir de um artefato obsoleto

Depois de um `--dry-run` que falhou por rede, eu li totais (183.531 linhas, 25,77 GB) de um CSV que a execução falha não havia reescrito, e quase os apresentei como atuais. Hoje `outputs/audio_download_plan.csv` é reescrito a cada execução e está no `.gitignore` — é derivado, não deve ser versionado.

---

## 12. Pendências

### 12.1 Código (não depende de credencial)

| Item | Onde |
|---|---|
| 8 adaptadores de provedor levantam `NotImplementedError` | `models/llm/clients.py` (3), `models/multimodal/clients.py` (3), `models/asr/whisper.py` (2) |
| `AnthropicTextQA` para o gerador não existe | registrar em `TEXT_QA_PROVIDERS` e `build_text_qa` |
| Guarda contra a string literal `PLACEHOLDER` | `validate_cross_config` em `src/nefair/config.py` |
| `scripts/06_run_eval.py` não existe | — |
| `src/nefair/analysis/export.py` (tabelas LaTeX) não existe | — |
| `scripts/07_analyze.py` não existe | — |
| Versão do prompt desconectada | `runner.DEFAULT_PROMPT_VERSION = "runner-default-v1"` vs `prompts.PROMPT_VERSION = "mcq-v1"` |

### 12.2 Rigor — a corrigir neste relatório e no código

- **Comentário obsoleto em `configs/analysis.yaml`:** diz que o bootstrap "reamostra 39 NE e 193 SE". Os números corretos são **33 e 132** (elegíveis, não corpus bruto). O código estratifica pelo que está no dado, então o comportamento está certo; o comentário é que engana quem ler.
- **`split_restriction: test` em `configs/models.yaml`** está listado como mitigação viável do D8. Pela §6.3, não é. O comentário precisa dizer isso.
- **D8 em aberto.** `contamination_checked: false` bloqueia a execução completa, por desenho. Caminho para fechar em §6.4: checkpoint sem fine-tune PT-BR, verificação do card, e a sonda train-vs-test dentro do NE.

### 12.3 Credenciais

`.env` tem **apenas** `HF_TOKEN`. Faltam: `ANTHROPIC_API_KEY` (gerador), `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `DASHSCOPE_API_KEY` + `QWEN_BASE_URL`.

13 `model_id`/`version` em `configs/models.yaml` e 2 em `configs/items.yaml` seguem como `PLACEHOLDER`, a fixar no dia da execução.

### 12.4 Dimensionamento restante

| Passo | Piloto (120 itens) | Completo (1.602 itens) |
|---|--:|--:|
| Geração | 120 chamadas | 1.602 |
| Filtro textual (3 modelos × 3 execuções) | 1.080 | 14.418 |
| ASR | 120 janelas / 78 min | 1.602 janelas / 17,37 h |
| Avaliação (9 células por item) | ~630 | ~8.400 |
| Revisão humana | ~85 itens | ~1.100 itens (~27 h a 1,5 min/item) |

Os números do conjunto completo caíram em relação ao plano original, que supunha 2.320 itens — reflexo dos 165 falantes elegíveis contra os 232 estimados antes do recorte etário.

**A revisão humana segue sendo o caminho crítico.** É por isso que o gate do piloto existe: medir a taxa de aceitação e os minutos por item em 120 itens balanceados antes de comprometer ~27 h de trabalho.
