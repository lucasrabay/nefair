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
