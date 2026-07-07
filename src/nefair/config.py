"""Loader do YAML de configuração e utilitário mínimo de `.env`.

Sem dependências além de `pyyaml`: o `.env` é lido por um parser simples de
linhas `CHAVE=VALOR` para não introduzir `python-dotenv` (deps mínimas).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    """Identificação e escopo do dataset a auditar."""

    id: str
    revision: str | None
    splits: tuple[str, ...]


@dataclass(frozen=True)
class CorpusAuditConfig:
    """Configuração completa da auditoria de metadados (Etapa 0)."""

    dataset: DatasetConfig
    audio_columns: tuple[str, ...]
    region_map: dict[str, str]
    age_bucket_edges: tuple[int, ...]

    @property
    def dataset_id(self) -> str:
        return self.dataset.id


def load_dotenv(path: str | Path = ".env") -> None:
    """Carrega pares CHAVE=VALOR de um `.env` no ambiente (sem sobrescrever).

    Silencioso se o arquivo não existir. Não sobrescreve variáveis já definidas
    no ambiente, para que o ambiente do processo tenha precedência.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str | Path) -> CorpusAuditConfig:
    """Lê e valida o YAML de configuração da auditoria."""
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Config inválido em {path}: esperado mapa no topo.")

    dataset_raw = raw.get("dataset")
    if not isinstance(dataset_raw, dict):
        raise ValueError("Config inválido: seção 'dataset' ausente ou malformada.")

    splits = dataset_raw.get("splits")
    if not isinstance(splits, list) or not splits:
        raise ValueError("Config inválido: 'dataset.splits' deve ser lista não vazia.")

    dataset = DatasetConfig(
        id=str(dataset_raw["id"]),
        revision=dataset_raw.get("revision"),
        splits=tuple(str(s) for s in splits),
    )

    region_map_raw = raw.get("region_map")
    if not isinstance(region_map_raw, dict) or not region_map_raw:
        raise ValueError("Config inválido: 'region_map' deve ser mapa não vazio.")
    region_map = {str(k): str(v) for k, v in region_map_raw.items()}

    audio_columns = tuple(str(c) for c in raw.get("audio_columns", ["audio"]))

    age_buckets_raw = raw.get("age_buckets", {})
    edges = age_buckets_raw.get("edges", []) if isinstance(age_buckets_raw, dict) else []
    age_bucket_edges = tuple(int(e) for e in edges)

    return CorpusAuditConfig(
        dataset=dataset,
        audio_columns=audio_columns,
        region_map=region_map,
        age_bucket_edges=age_bucket_edges,
    )
