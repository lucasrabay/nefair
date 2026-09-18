# Plano de implementação de CÓDIGO — Etapas 1–6

Derivado de `docs/plano_implementacao.md`, restrito ao que é **código**. O que é
processo (revisão humana, gate do piloto, cronograma, correções no `.tex`,
conversas com o orientador) fica fora deste plano e só aparece aqui quando vira
um parâmetro, um artefato ou uma asserção.

Restrições herdadas da Etapa 0 (não negociáveis):

1. **Determinismo total** — sem aleatoriedade não semeada, ordenação estável,
   duas execuções produzem bytes idênticos (timestamps isolados num único campo).
2. **Nada dropado em silêncio** — toda linha, janela ou item descartado é contado
   e reportado com o motivo.
3. **Proveniência da fonte** — SHA do dataset, versões de libs e versões de
   modelo/prompt gravadas em todo artefato.
4. **Decisão pendente nunca vira default escondido** — cada uma das D1–D11 é uma
   chave explícita de YAML, com o valor efetivo ecoado no relatório da etapa.

### Restrição operacional que molda todo o plano

Não há `.env` no repositório: **sem chave de API de nenhum provedor** e sem
`HF_TOKEN`. Logo, nenhum agente pode validar código contra um modelo real. O
plano é, portanto, **offline-first**:

- toda lógica de negócio é pura e testável sem rede;
- todo provedor (ASR, LLM de texto, multimodal) fica atrás de um `Protocol`, com
  uma implementação **falsa determinística** usada nos testes;
- os clientes reais são adaptadores finos, importados **preguiçosamente**, de modo
  que a ausência do SDK não quebra nem o import do pacote nem a suíte de testes;
- os scripts `01`–`07` rodam ponta a ponta sobre um **corpus sintético de
  fixture**, sem tocar a rede.

Quando as chaves existirem, o que muda é o YAML — não o código.

---

## 0. Camada de contratos (pré-requisito, sem paralelismo)

Esta camada existe para que os blocos seguintes possam ser escritos em paralelo
sem disputar arquivo. Ela congela: dataclasses, protocolos, schema dos YAMLs,
formato dos artefatos em disco e dependências.

| Arquivo | Conteúdo |
|---|---|
| `pyproject.toml` | **todas** as dependências novas de uma vez (`jiwer`, `statsmodels`, `numpy`, `soundfile`); SDKs de provedor e `torch`/`transformers` como extras opcionais (`[project.optional-dependencies] providers / local`) |
| `src/nefair/config.py` | `load_yaml` compartilhado + `WindowsConfig`, `ItemsConfig`, `ModelsConfig`, `EvalConfig`, `AnalysisConfig`. `CorpusAuditConfig` fica **intocado** (Etapa 0 está entregue) |
| `src/nefair/schema.py` | `Segment`, `Window`, `Alternative`, `Item`, `Provenance`, `RunRecord` — dataclasses congeladas + I/O JSONL/parquet com ordenação de colunas fixa |
| `src/nefair/models/base.py` | `ModelSpec`; `Protocol`s `ASR`, `TextQA`, `AudioQA`; `ASRResult`, `QAResult` (inclui `raw_text`, `finish_reason`, `usage`); `FakeASR`, `FakeTextQA`, `FakeAudioQA` determinísticos |
| `configs/windows.yaml` `items.yaml` `models.yaml` `eval.yaml` `analysis.yaml` | schema completo, com D1–D11 como chaves nomeadas e comentadas |
| `tests/fixtures/synthetic_corpus.py` | gerador determinístico de um frame com a mesma forma do CORAA-MUPE (as 19 colunas reais), com casos-limite plantados: falante curto demais para 10 janelas, interrupção de entrevistador, lacuna temporal, segmento `low` |
| `src/nefair/**/__init__.py` | todos os pacotes novos criados vazios |

**Invariante de contrato exigida em código:** `ItemsConfig.generator_model` não
pode aparecer em `ModelsConfig` (D11). Violação levanta `ValueError` na carga do
config, não numa revisão manual.

---

## 1. Blocos de implementação

Quatro blocos com conjuntos de arquivos **disjuntos**, todos dependendo apenas da
camada 0.

### Bloco W — janelas e áudio (Etapas 1 e 2)

`src/nefair/corpus/windows.py`, `src/nefair/corpus/audio.py`,
`scripts/01_build_windows.py`, `scripts/02_fetch_audio.py`,
`tests/test_windows.py`, `tests/test_audio_manifest.py`

**`windows.py`** — janelamento só sobre metadados.

- Ordenação canônica: `(speaker_code, audio_name, start_time, audio_id)` —
  `audio_id` como desempate para estabilidade total.
- **Contiguidade é avaliada no frame COMPLETO, antes do filtro de informante.**
  Este é o ponto sutil de D1: um turno `P/*` entre dois segmentos `R` some quando
  filtramos, e a "fala contínua" pareceria intacta. O detector recebe o frame não
  filtrado, marca a interrupção e só então descarta as linhas `P/*`.
- Quebra de janela por: mudança de `audio_name`, turno de entrevistador no meio
  (D1), lacuna `start_time[i+1] - end_time[i] > gap_tolerance_s`.
- Janela válida: duração somada em `[30, 60] s` **e** `[8, 15]` segmentos —
  as duas condições do texto, não uma só.
- Seleção de `n_windows_per_speaker` (10) por **amostragem estratificada por
  posição na entrevista** (divide as candidatas disjuntas em 10 blocos iguais e
  sorteia uma por bloco com `numpy.random.default_rng(seed + hash estável do
  speaker_code)`), jamais "as 10 primeiras". Falante com menos candidatas do que
  isso fica com menos janelas e **é contado** (D5); nunca se sobrepõe janela.
- Transcrição de referência = `join` de `normalized_text` (D7), campo
  configurável.
- Saídas: `outputs/windows.parquet` (gitignorado) + `outputs/windows_report.md`
  (commitado) com a trilha completa: falantes elegíveis, candidatas por falante,
  motivo de cada quebra contabilizado, histograma de janelas por falante,
  quantos falantes ficaram abaixo de 10.
- Testes: contiguidade, não sobreposição, limites de duração **e** de contagem de
  segmentos, interrupção de entrevistador fecha janela, determinismo
  (duas chamadas → frames idênticos), falante curto produz < 10 sem erro.

**`audio.py`** — download seletivo.

- Reaproveita `_RangeReader` / `_SparseFile` de `corpus/load.py` (que passam a ser
  exportados com nome público em vez de privado).
- Mapeia `(shard, row_group)` → linhas-alvo pelo footer, baixa **somente** os
  chunks da coluna `audio` dos row groups que contêm segmentos selecionados.
- Decodifica, reamostra para 16 kHz mono, concatena por janela, escreve
  `outputs/audio/<speaker>/<window_id>.wav`.
- `outputs/audio_manifest.csv`: janela → segmentos, duração pedida vs. obtida,
  `sha256` do WAV, bytes baixados, bytes evitados.
- Testes: o manifesto e o cálculo de row groups são testados offline com parquet
  sintético escrito em `tmp_path`; a decodificação real fica atrás de
  `pytest.mark.network`, desligada por padrão.

### Bloco I — itens (Etapa 3)

`src/nefair/items/generate.py`, `filter.py`, `review.py`,
`src/nefair/models/prompts.py`, `scripts/03_generate_items.py`,
`04_text_only_filter.py`, `05_review.py`, `tests/test_items.py`,
`tests/test_prompts.py`

- `prompts.py`: template único de MCQ, **versionado** (`PROMPT_VERSION = "mcq-v1"`),
  com renderização pura `render(...) -> str` e teste de snapshot. A versão entra
  na chave de cache e no relatório.
- `generate.py`: monta o prompt com a janela-alvo **e** trechos de outras janelas
  da mesma entrevista; exige que cada distrator declare a janela de origem (D2).
  Resposta do LLM validada contra um schema estrito; item malformado é
  **contado e descartado**, nunca reparado silenciosamente.
- `filter.py`: controle textual sem áudio nem transcrição. `n_runs` execuções por
  modelo com a ordem das alternativas embaralhada por execução (permutação
  derivada de `(item_id, run_index, seed)` — reprodutível), maioria de
  `discard_if_correct_at_least` de `n_runs`; se **qualquer** modelo passar no
  critério, o item cai (D3). A taxa de descarte é reportada por região.
- `review.py`: `export_review_sheet()` → CSV estável para revisão humana;
  `import_review_sheet()` → valida colunas, rejeita decisões fora do vocabulário,
  concilia com os itens por `item_id` e **falha** se houver item não revisado.
- Saída `outputs/items_report.md`: taxa de descarte por região, taxa de rejeição
  na revisão por região, minutos por item, projeção para o conjunto completo, e a
  linha de base de acaso (≈ 15,6% de descarte esperado sob `p = 0,25`)
  **calculada em código**, não hardcoded.
- Testes: todos com `FakeTextQA`. Determinismo do embaralhamento, contagem do
  descarte, rejeição de distrator sem proveniência, round-trip da planilha.

### Bloco R — modelos e execução (Etapas 4 e 5)

`src/nefair/models/asr/`, `llm/`, `multimodal/`, `src/nefair/run/runner.py`,
`parse.py`, `scripts/06_run_eval.py`, `tests/test_parse.py`, `tests/test_runner.py`

- `parse.py`: extrai a letra escolhida do texto livre. Cascata de estratégias
  ordenada e documentada; **resposta inválida vira `None` e é contada**, jamais
  imputada como erro nem como acerto. Testado contra um corpus de strings
  patológicas (letra minúscula, "Alternativa C.", resposta em prosa, vazia,
  duas letras, letra fora de A–D).
- `runner.py`: produto `item × configuração × condição` → `RunRecord`.
  Chave de cache = `sha256` de `(item_id, model_spec, condition, prompt_version,
  decoding_params, alternatives_permutation)`. Cache em JSONL append-only;
  reexecução lê do cache e nunca paga duas vezes. Retomada após interrupção.
  Concorrência limitada e com backoff, no mesmo espírito de `load.py`.
- `asr/`: `WhisperLocal` (lazy `transformers`/`torch`) e `WhisperAPI`; um **único**
  ASR compartilhado por todas as cascatas, conforme §Etapa 5.
- `llm/`, `multimodal/`: adaptadores OpenAI / Google / Qwen, todos implementando
  os `Protocol`s e **nenhum importado no topo do pacote**.
- Condições: `asr` (hipótese do ASR), `reference` (transcrição de referência),
  `audio` (multimodal nativo). A decomposição da Etapa 6 depende de `asr` e
  `reference` serem o **mesmo item e a mesma permutação** — o runner garante isso
  por construção e o teste verifica.
- Testes: tudo com `FakeASR`/`FakeTextQA`/`FakeAudioQA`. Cache hit não chama o
  modelo (contador de chamadas), chave de cache muda quando a versão do prompt
  muda, pareamento item-a-item entre condições.

### Bloco A — métricas e análise (Etapa 6)

`src/nefair/metrics/normalize.py`, `wer.py`, `src/nefair/analysis/regression.py`,
`decompose.py`, `sensitivity.py`, `export.py`, `scripts/07_analyze.py`,
`tests/test_normalize.py`, `test_wer.py`, `test_analysis.py`

- `normalize.py`: normalizador PT-BR **fixo e versionado** (`NORMALIZER_VERSION`),
  aplicado igualmente à referência e à hipótese (D7): caixa baixa, pontuação,
  marcas de hesitação, números por extenso, espaços. Cada regra é uma função
  pequena com teste próprio e um caso real do corpus.
- `wer.py`: `(S + I + D) / N` via `jiwer`, por janela e agregado; devolve também
  as contagens brutas, para que o agregado seja soma de contagens e não média de
  razões. Teste com exemplos calculados à mão.
- `regression.py`: logit `acerto ~ região + idade` (idade contínua) por
  configuração × condição, restrito a 20–69 anos; efeito marginal médio de
  NE − SE; IC por **bootstrap de falantes estratificado por região** (reamostra
  39 NE e 193 SE separadamente), `B = 2000`, semente fixa. As mesmas reamostras
  são reusadas entre condições, preservando o pareamento.
- `decompose.py`: `Δ_ASR = acc(ref) − acc(ASR)`; `Δ_raciocínio = teto − acc(ref)`,
  com o teto vindo do config (D4) e **rotulado na saída** com o modo usado.
- `sensitivity.py`: nº de itens por falante por subamostragem; duração da janela
  por reestimação em faixas; nº de alternativas apenas se habilitado (D10).
- `export.py`: DataFrame → tabela LaTeX `booktabs`/`siunitx` com `\fonte{}`,
  saída determinística, comparável byte a byte.
- Testes: dados sintéticos com efeito **conhecido** (gap NE−SE plantado) e
  verificação de que a estimativa o recupera dentro do IC; determinismo do
  bootstrap sob a mesma semente; LaTeX byte a byte.

---

## 2. D1–D11 → superfície de configuração

Nenhuma dessas decisões é tomada pelo código. Cada uma é uma chave, com o valor
efetivo ecoado no relatório da etapa correspondente.

| # | Arquivo | Chave | Default proposto |
|---|---|---|---|
| D1 | `windows.yaml` | `interviewer_turn: close_window` \| `ignore` | `close_window` |
| D2 | `items.yaml` | `distractors.source`, `distractors.require_provenance` | `other_windows_same_interview`, `true` |
| D3 | `items.yaml` | `text_filter.{models, temperature, n_runs, discard_if_correct_at_least, any_model_discards}` | todos os avaliados, `0.7`, `3`, `2`, `true` |
| D4 | `analysis.yaml` | `task_ceiling.{mode, value}` | `constructed`, `1.0` |
| D5 | `windows.yaml` | `allow_fewer_windows: true` | `true` (nunca sobrepor) |
| D6 | `analysis.yaml` | `covariates: [region, age]` | sem gênero |
| D7 | `analysis.yaml` | `wer.reference_field`, `wer.normalizer_version` | `normalized_text`, `v1` |
| D8 | `models.yaml` | `asr.contamination_checked`, `asr.split_restriction` | `false` (bloqueia execução completa até ser preenchido), `null` |
| D9 | `models.yaml` | `paired: [...]`, `unpaired_extra: [...]` | Claude fora do pareado |
| D10 | `analysis.yaml` | `sensitivity.n_alternatives.enabled` | `false` |
| D11 | `items.yaml` | `generator_model` | validado ≠ qualquer modelo avaliado |

D8 merece destaque: `contamination_checked: false` faz o script `06` **recusar a
execução completa** com mensagem explícita. É barato e evita rodar a Etapa 5
inteira sobre um ASR possivelmente contaminado.

---

## 3. Critérios de aceite

Um bloco só está pronto quando, na raiz do repositório:

1. `uv run ruff check .` e `uv run ruff format --check .` passam;
2. `uv run pytest` passa, **sem rede** e sem nenhuma chave de API;
3. o script da etapa roda ponta a ponta sobre a fixture sintética;
4. rodar o script duas vezes produz artefatos idênticos byte a byte (exceto o
   campo de timestamp isolado);
5. o relatório `.md` da etapa contabiliza 100% das unidades de entrada
   (retidas + descartadas por motivo = total);
6. nenhum import de SDK de provedor acontece no topo de um módulo.

## 4. Fora do escopo deste plano

- Executar qualquer modelo real (não há credenciais).
- Baixar os ~29 h de áudio da Etapa 2 (o código fica pronto e testado; a execução
  é um comando separado, com custo de rede real).
- A revisão humana em si — o código entrega a planilha e o importador.
- Fechar D1–D11: são decisões do autor e do orientador. O código as expõe,
  valida e reporta.
