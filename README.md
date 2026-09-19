# nefair

Auditoria de fairness de modelos de fala/linguagem em português brasileiro,
comparando **Nordeste** (agregado) vs. **Sudeste** (agregado), com idade como
covariável de controle. Corpus base: [`nilc-nlp/CORAA-MUPE-ASR`](https://huggingface.co/datasets/nilc-nlp/CORAA-MUPE-ASR).

## Setup

O projeto é gerenciado por [`uv`](https://docs.astral.sh/uv/), com compatibilidade
com `pip`.

### Com uv (recomendado)

```bash
uv sync                      # cria o venv e instala runtime + dev
```

### Com pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # runtime + dev; use `pip install -e .` para só runtime
```

### Token do HuggingFace

O dataset pode exigir login (aceitar os termos na página do dataset). Copie o
`.env.example` para `.env` e preencha o token:

```bash
cp .env.example .env
# edite .env e preencha HF_TOKEN=hf_...
```

O token é lido da variável de ambiente `HF_TOKEN` (o `.env` é carregado
automaticamente pelo script). Nunca é escrito em código nem no relatório.

## Etapa 0: auditoria do corpus

Transforma a exploração do corpus em um artefato reprodutível e determinístico.
**Não baixa nem decodifica áudio** — lê apenas os metadados via projeção de
colunas dos parquets.

```bash
uv run scripts/run_corpus_audit.py --config configs/corpus_audit.yaml
```

Saídas (em `outputs/`):

- `corpus_audit_report.md` — relatório em prosa com os números-chave, a chave de
  falante escolhida, contagens de filtragem, estados fora de escopo, footprint de
  download, revisão do dataset e versões das libs. **Commitado.**
- `speaker_counts_by_state.csv` — falantes distintos e horas por `birth_state`.
- `speaker_counts_by_region_age.csv` — cross-tab região × bucket etário.
- `age_distribution_by_region.csv` — distribuição etária bruta por região.
- `data_quality_report.csv` — sentinelas/faltantes, `audio_quality`, `duration`,
  segmentos por falante.

Os CSVs de agregados pequenos são commitados; dumps grandes por segmento são
gitignorados.

Rodar duas vezes gera saídas idênticas byte a byte (exceto o campo de timestamp,
isolado no relatório).

## Etapas 1-6: pipeline de avaliação

O plano de origem está em `docs/plano_implementacao.md` (inclui as decisões
pendentes D1-D11); o plano restrito ao código, em `docs/plano_codigo.md`.

**Nenhuma decisão pendente virou default escondido.** Cada uma é uma chave
nomeada num YAML de `configs/`, com o valor efetivo ecoado no relatório da
etapa. Duas são aplicadas pelo próprio código:

- **D11** — `validate_cross_config` recusa a execução se o LLM gerador dos itens
  estiver entre os modelos avaliados (o gerador partiria na frente).
- **D8** — a execução completa se recusa a rodar enquanto a contaminação do ASR
  não for verificada em `models.yaml`.

### Estado atual

| Etapa | Módulos | Script | Testes |
|---|---|---|---|
| 1 — janelas | `corpus/windows.py` | `01_build_windows.py` | 33 |
| 2 — áudio | `corpus/audio.py` | `02_fetch_audio.py` | 38 |
| 3 — itens | `items/{generate,filter,review}.py`, `models/prompts.py` | `03`, `04`, `05` | 56 |
| 4-5 — execução | `run/{parse,runner}.py` | — (falta o `06`) | 67 |
| 6 — análise | `metrics/*`, `analysis/{regression,decompose,sensitivity}.py` | — (falta o `07`) | 108 |

Pendentes: `src/nefair/analysis/export.py` (tabelas LaTeX), `scripts/06_run_eval.py`
e `scripts/07_analyze.py`.

### Provedores

Não há chave de API no projeto. Todo provedor (ASR, LLM de texto, multimodal)
fica atrás de um `Protocol` em `src/nefair/models/base.py`, com implementações
**falsas determinísticas** usadas nos testes. Os adaptadores reais são stubs com
o contrato documentado: sem credencial não há como validá-los, e código não
verificável envelhece mal. Nenhum SDK é importado no topo de módulo.

Consequência prática: **a suíte inteira roda offline, sem credencial nenhuma.**

### Testes

```bash
uv run pytest -q          # 320 testes, ~3 s, sem rede
uv run ruff check . && uv run ruff format --check .
```

A suíte é determinística: duas execuções produzem resultado idêntico teste a
teste, e o resultado não depende de `PYTHONHASHSEED`.
