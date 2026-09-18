# Plano de implementação — pipeline de avaliação (Etapas 1–6)

Base: `tcc/capitulos/principal.tex`, Seção "Pipeline de avaliação" (§ Visão geral,
Construção da camada de compreensão, Configurações avaliadas, Condições de controle
e decomposição do erro, Métricas). Cada decisão de código abaixo rastreia uma frase
do texto; onde o texto não especifica o suficiente para implementar, a lacuna está
listada em **Decisões pendentes** e o código a expõe como parâmetro de config em vez
de fixá-la em silêncio.

Princípio herdado da Etapa 0: tudo determinístico, versionado e com trilha de
exclusão explícita (nenhuma linha, janela ou item descartado sem contagem).

---

## 1. Estrutura de módulos

```
src/nefair/
  corpus/            # existe (load, audit)
    windows.py       # janelas contíguas de 30–60 s por falante (só metadados)
    audio.py         # download seletivo do áudio das janelas + concatenação
  items/
    generate.py      # LLM gerador: pergunta + correta + 3 distratores
    filter.py        # controle textual (sem áudio, sem transcrição), maioria de 3
    review.py        # export/import da planilha de revisão humana
    schema.py        # dataclass Item (id, speaker, window, alternativas, gabarito, proveniência)
  models/
    base.py          # interfaces: ASR, TextQA, AudioQA
    asr/             # Whisper (local) e ASR via API
    llm/             # clientes de texto (OpenAI, Google, Qwen local/API)
    multimodal/      # clientes áudio→resposta (mesmas famílias)
    prompts.py       # template único de MCQ, versionado
  run/
    runner.py        # item × configuração × condição → resposta; cache JSONL por hash
    parse.py         # extração da letra escolhida; resposta inválida contada, nunca imputada
  metrics/
    normalize.py     # normalizador de texto PT-BR para WER (fixo, testado)
    wer.py           # WER (S+I+D)/N por janela e agregado
  analysis/
    regression.py    # logit: acerto ~ região + idade (contínua); bootstrap por falante
    decompose.py     # Δ_ASR = acc(ref) − acc(ASR); Δ_raciocínio = teto − acc(ref)
    sensitivity.py   # reestimação sob variações de janela / nº itens / nº alternativas
    export.py        # CSV → tabelas LaTeX (booktabs/siunitx, \fonte{})
configs/
  windows.yaml  items.yaml  models.yaml  eval.yaml  analysis.yaml
scripts/
  01_build_windows.py  02_fetch_audio.py  03_generate_items.py
  04_text_only_filter.py  05_review.py  06_run_eval.py  07_analyze.py
```

Novas dependências (adicionadas por etapa, não de uma vez): `jiwer`, `statsmodels`,
`numpy`, `soundfile`/`librosa`, SDKs dos provedores escolhidos, `transformers`/`torch`
só quando a configuração open-weight entrar.

---

## 2. Etapas

### Etapa 1 — Janelas (só metadados, sem áudio, sem API)
Texto: "janela de trinta a sessenta segundos de fala contínua, … entre oito e quinze
segmentos consecutivos"; "dez itens por falante".

- Ordenar segmentos de informante por (`speaker_code`, `audio_name`, `start_time`).
- Janela = sequência de segmentos consecutivos do mesmo `audio_name`, duração
  somada em [30, 60] s, sem sobreposição entre janelas do mesmo falante.
- Selecionar 10 janelas por falante com semente fixa, espalhadas ao longo da
  entrevista (amostragem estratificada por posição, não as 10 primeiras).
- Transcrição de referência da janela = concatenação de `normalized_text` (ver D7).
- Saídas: `outputs/windows.parquet` + `outputs/windows_report.md` com a trilha:
  falantes elegíveis, janelas candidatas, falantes com < 10 janelas possíveis.
- Testes: contiguidade, não sobreposição, limites de duração, determinismo.

Alerta já visível nos dados: o mínimo é 64 segmentos por falante (SE) e 87 (NE).
Com ≥ 8 segmentos por janela, um falante com 64 segmentos comporta no máximo 8
janelas disjuntas. A Etapa 1 conta quantos falantes estão nessa situação (D5).

### Etapa 2 — Áudio das janelas
- Download seletivo por shard/row group (reaproveitar o `_RangeReader` da Etapa 0),
  apenas para os segmentos das janelas selecionadas; nunca os 41,8 GB inteiros.
- Concatenação dos segmentos em um WAV por janela (16 kHz mono), com manifesto
  (janela → segmentos, duração, hash do arquivo).
- Estimativa: ~2.320 janelas × ~45 s ≈ 29 h de áudio.

### Etapa 3 — Itens (piloto primeiro)
Texto: LLM recebe a transcrição da janela e propõe pergunta, correta e três
incorretas; distratores "extraídos de outros trechos da mesma entrevista";
revisão humana de todo item; controle textual com "maioria de três execuções".

1. **Piloto**: 12 falantes (6 NE, 6 SE, dentro de 20–69 anos) → ~120 itens.
2. Geração: prompt recebe a janela-alvo **e** trechos de outras janelas da mesma
   entrevista, com instrução de ancorar cada distrator num desses trechos; a
   proveniência de cada distrator é salva no item (D2).
3. Controle textual (`filter.py`): só pergunta + 4 alternativas, ordem das
   alternativas embaralhada por execução, 3 execuções; descarta se acertar ≥ 2/3.
4. Revisão humana apenas dos itens que sobreviveram ao filtro (reduz a carga); a
   planilha registra decisão, motivo e tempo gasto.
5. Métricas do piloto, reportadas: taxa de descarte do filtro por região, taxa de
   rejeição na revisão por região, minutos por item revisado → projeção para o
   conjunto completo.

**Gate**: só seguir para a geração completa se (a) a taxa de descarte não for ~0
nem > 60%, (b) não houver diferença grosseira de descarte entre NE e SE, e (c) o
tempo de revisão projetado couber no cronograma.

Nota estatística para o texto: um item que de fato exige áudio e é respondido ao
acaso (p = 0,25) é acertado em ≥ 2 de 3 execuções com probabilidade
3·0,25²·0,75 + 0,25³ ≈ 0,156. O filtro, portanto, descarta ~16% dos itens válidos
por puro acaso — isso é esperado e deve constar junto à taxa de descarte.

### Etapa 4 — Fatia vertical (antes de escalar)
Sobre os itens do piloto:
- 1 cascata: Whisper → LLM X, condições **ASR** e **referência**.
- 1 multimodal nativo da mesma família do LLM X (pareamento, §Configurações).
- WER por janela; acurácia por condição; decomposição; regressão + bootstrap
  rodando ponta a ponta (os números do piloto não têm poder, servem para validar
  o código e o custo por item).
- Runner com cache: chave = hash(item, config, condição, versão do prompt,
  versão do modelo). Reexecução nunca paga duas vezes.

### Etapa 5 — Execução completa
Itens completos (gerados → filtrados → revisados) × todas as configurações.

Proposta de configurações que satisfaz o critério de pareamento do texto
(mesmo LLM nas duas famílias):

| Família do LLM | Nativo (áudio → resposta) | Cascata (ASR → LLM texto) |
|---|---|---|
| OpenAI | GPT-4o Audio (ou sucessor) | Whisper → mesmo modelo, entrada texto |
| Google | Gemini (áudio) | Whisper → mesmo Gemini, entrada texto |
| Open-weight | Qwen2.5-Omni (ou Qwen2-Audio) | Whisper → mesmo backbone, entrada texto |

- Um **único** ASR compartilhado entre as cascatas, para que a condição ASR varie
  só o LLM. Versões exatas dos modelos fixadas em `models.yaml` no dia da execução
  e reportadas no texto.
- Claude não tem entrada de áudio nativa → não pode ser pareado; se entrar, entra
  como cascata extra fora da comparação pareada (D9).
- Opcional: pedir ao modelo nativo também a transcrição, só para reportar WER dele
  (não entra na decomposição).

### Etapa 6 — Análise
Texto: logit com região + idade contínua; incerteza agrupada por falante;
bootstrap no nível do falante; estimativas restritas a 20–69 anos.

- Modelo por configuração × condição: `acerto ~ região + idade`.
- Estimando reportado: diferença de acurácia ajustada NE − SE (efeito marginal
  médio) dentro de 20–69, com IC por bootstrap de falantes **estratificado por
  região** (reamostra 39 NE e 193 SE separadamente), B = 2.000, semente fixa.
- Decomposição por cascata com IC pelo mesmo bootstrap (mesmas reamostras para as
  duas condições, preservando o pareamento item a item).
- Sensibilidade: nº de itens por falante via subamostragem (barato); duração da
  janela via reestimação em subconjuntos por faixa de duração (barato);
  nº de alternativas exige regerar e revisar itens (caro — ver D10).
- Export direto para tabelas LaTeX do `tcc`.

---

## 3. Decisões pendentes (bloqueiam código específico)

| # | Decisão | Por que o texto não resolve | Proposta |
|---|---|---|---|
| D1 | Janela quando o entrevistador interrompe | "fala contínua" vs. segmentos `R` intercalados com `P/1` | Janela só com segmentos `R` sem turno de entrevistador no meio; interrupção fecha a janela |
| D2 | Mecanismo dos distratores | O LLM "propõe" as incorretas, mas elas são "extraídas de outros trechos" | LLM recebe outros trechos da mesma entrevista e precisa ancorar cada distrator num deles; proveniência salva |
| D3 | Modelo(s) do controle textual e temperatura | "maioria de três execuções" com temperatura 0 dá 3 respostas iguais | Rodar com **cada** LLM avaliado, temperatura > 0 e alternativas embaralhadas; descartar se qualquer um passar no critério |
| D4 | O que é "o teto da tarefa" | Δ_raciocínio = teto − acc(ref) precisa de um teto definido | 100% por construção (itens revisados sobre a referência), ou acurácia humana numa amostra — decidir com o Jorge |
| D5 | Falantes que não comportam 10 janelas | 64 segmentos → no máximo 8 janelas de ≥ 8 segmentos | Aceitar < 10 itens para esses falantes e reportar a contagem, em vez de sobrepor janelas |
| D6 | Gênero como covariável | O texto mostra cobertura completa de gênero mas não o inclui no modelo | Decidir com o Jorge; se entrar, entra também na seção de Métricas |
| D7 | Referência do WER e normalização | `normalized_text` vs. `original_text`; marcas de hesitação, números | WER sobre `normalized_text` e a hipótese passada pelo mesmo normalizador |
| D8 | Contaminação do ASR | Whisper PT-BR ajustado pode ter sido treinado em CORAA/MuPe | Verificar o card de treino; se houver sobreposição, usar Whisper base ou restringir ao split `test` |
| D9 | Claude no conjunto | Viola o critério de pareamento | Tirar da comparação pareada; se entrar, como cascata extra |
| D10 | Sensibilidade ao nº de alternativas | Regerar com 3/5 alternativas duplica a revisão humana | Fazer só no subconjunto do piloto, ou tirar esse parâmetro do texto |
| D11 | LLM gerador vs. LLMs avaliados | O texto não diz qual modelo gera os itens | Não usar como gerador um modelo avaliado (viés a favor do próprio gerador); declarar o gerador |

---

## 4. Pendências fora do código (encontradas nesta leitura)

- `configs/corpus_audit.yaml` tem `revision: null`, mas o texto diz que a revisão
  "foi fixada". Fixar `d3d5f9c699bd619f9e1c45681cabf214d1b080e1` no config e herdar
  esse SHA em todas as etapas novas.
- `principal.tex`, §O corpus CORAA-MUPE-ASR, ainda diz "do CORAA ASR, do qual
  constitui um subconjunto" — a formulação de "subconjunto" já estava marcada como
  factualmente incorreta. Corrigir antes da versão final.
- `nefair` está 6 commits à frente do GitHub; fazer o push antes de começar a Etapa 1.

---

## 5. Cronograma (defesa em dezembro)

| Semana | Entrega |
|---|---|
| 21/09 | D1, D5, D7 fechadas; Etapa 1 (janelas) + testes; push pendente |
| 28/09 | Etapa 2 (áudio) no piloto; Etapa 3 no piloto; D2, D3, D11 fechadas |
| 05/10 | Revisão do piloto + gate; Etapa 4 (fatia vertical) |
| 12/10 – 26/10 | Geração completa + filtro + **revisão humana (caminho crítico)** |
| 02/11 – 09/11 | Etapa 5: todas as configurações |
| 16/11 | Etapa 6: análise, sensibilidade, tabelas LaTeX |
| 23/11 → | Resultados e discussão no texto |

A revisão humana de ~2.300 itens é o maior risco de prazo. O gate do piloto existe
justamente para medir minutos por item antes de assumir esse custo.
