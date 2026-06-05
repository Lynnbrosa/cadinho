"""Typed configuration loader + global determinism.

A single YAML (`config/config.yaml`) is the source of truth. We parse it into typed
pydantic models so a typo in a key fails loudly instead of silently changing behaviour.
`set_global_seeds` is the one place determinism is established.
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

# Repo root = two levels up from this file (src/cadinho/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


class _Base(BaseModel):
    # Allow extra keys so adding a knob to YAML doesn't require touching this file,
    # but still type-check the ones we rely on.
    model_config = ConfigDict(extra="allow")


class ProjectCfg(_Base):
    name: str = "cadinho"
    seed: int = 42
    genome_build: str = "GRCh38"


class DataSourcesCfg(_Base):
    mode: Literal["fixture", "live", "provided"] = "fixture"
    fixture_seed: int = 1234
    fixture_n_per_gene: int = 90
    clinvar: dict = Field(default_factory=dict)
    gnomad: dict = Field(default_factory=dict)
    dbnsfp: dict = Field(default_factory=dict)
    uniprot: dict = Field(default_factory=dict)


class GenesCfg(_Base):
    panel: list[str]
    focus: str
    meta: dict[str, dict]


class LabelsCfg(_Base):
    pathogenic: list[str]
    benign: list[str]
    vus_holdout: list[str]
    drop_if_unmatched: bool = True
    min_review_stars: int = 1


class VariantClassCfg(_Base):
    truncating_consequences: list[str]
    splice_canonical_offset: int = 2
    truncating_rule_label: str = "pathogenic"


class FeaturesCfg(_Base):
    conservation: list[str]
    substitution: list[str]
    predictors: list[str]
    population: list[str]
    positional: list[str]
    ablation_groups: dict[str, list[str]]

    def all_feature_names(self) -> list[str]:
        names: list[str] = []
        for group in (self.conservation, self.substitution, self.predictors,
                      self.population, self.positional):
            names.extend(group)
        # stable de-dup
        seen: set[str] = set()
        out: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out


class DecisionCfg(_Base):
    pathogenic_threshold: float = 0.90
    benign_threshold: float = 0.10


class ModelCfg(_Base):
    type: str = "lightgbm"
    params: dict = Field(default_factory=dict)
    calibration: Literal["isotonic", "sigmoid"] = "isotonic"
    decision: DecisionCfg = Field(default_factory=DecisionCfg)


class CVCfg(_Base):
    scheme: Literal["gene_disjoint", "position_kfold"] = "gene_disjoint"
    folds: int = 4
    repeats: int = 20


class BaselinesCfg(_Base):
    reference_predictors: list[str]
    thresholds: dict[str, float]


class EvaluationCfg(_Base):
    cv: CVCfg
    metrics: list[str]
    ablations: list[str]
    baselines: BaselinesCfg


class PathsCfg(_Base):
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    models: str = "models"
    reports: str = "reports"


class Config(_Base):
    project: ProjectCfg
    data_sources: DataSourcesCfg
    genes: GenesCfg
    labels: LabelsCfg
    variant_class: VariantClassCfg
    features: FeaturesCfg
    model: ModelCfg
    evaluation: EvaluationCfg
    paths: PathsCfg

    # --- path helpers (always absolute, parents created on demand) ---
    def _abs(self, rel: str) -> Path:
        p = (REPO_ROOT / rel).resolve()
        return p

    def path(self, kind: Literal["raw", "interim", "processed", "models", "reports"],
             *parts: str, mkdir: bool = True) -> Path:
        base = self._abs(getattr(self.paths, kind))
        p = base.joinpath(*parts) if parts else base
        target_dir = p.parent if parts and "." in p.name else p
        if mkdir:
            target_dir.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached default config for convenience in CLI/pipeline code."""
    return load_config()


def set_global_seeds(seed: int) -> None:
    """Establish determinism across stdlib, numpy, and hashing.

    Call once at the start of any entrypoint. Library-specific seeds (e.g. LightGBM
    `random_state`) are passed explicitly where the estimator is built.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
