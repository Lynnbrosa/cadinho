"""Train the missense classifier: deterministic gradient boosting + honest calibration.

Calibration is fit on **gene-disjoint out-of-fold** predictions (not resubstitution), so
the reported calibration reflects generalisation. The shipped model trains on all labelled
missense and applies the OOF-derived calibration map. A probability in the middle band
(benign_threshold, pathogenic_threshold) is reported as "Uncertain" — never coerced.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from cadinho.config import Config, set_global_seeds
from cadinho.evaluate.splits import make_folds
from cadinho.features.build import get_xy

log = logging.getLogger("cadinho.train")


def build_estimator(cfg: Config, seed: int):
    """Deterministic gradient-boosting estimator; LightGBM with a HistGB fallback."""
    p = dict(cfg.model.params)
    if cfg.model.type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                random_state=seed, n_jobs=1, verbose=-1, deterministic=True,
                force_row_wise=True, subsample_freq=1, **p), "lightgbm"
        except Exception as e:  # pragma: no cover - environment dependent
            log.warning("LightGBM unavailable (%s); using HistGradientBoosting fallback", e)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=p.get("learning_rate", 0.05),
        max_iter=p.get("n_estimators", 400),
        max_leaf_nodes=p.get("num_leaves", 31),
        min_samples_leaf=p.get("min_child_samples", 15),
        random_state=seed), "hist_gradient_boosting"


class _SigmoidCalibrator:
    def __init__(self, lr: LogisticRegression):
        self.lr = lr

    def predict(self, raw):
        return self.lr.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]


def fit_calibrator(oof_raw: np.ndarray, y: np.ndarray, method: str):
    mask = ~np.isnan(oof_raw)
    raw, yy = oof_raw[mask], y[mask]
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw, yy)
        return ir
    lr = LogisticRegression()
    lr.fit(raw.reshape(-1, 1), yy)
    return _SigmoidCalibrator(lr)


def gene_disjoint_oof(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                      cfg: Config, seed: int) -> np.ndarray:
    """One pass of gene-disjoint OOF probabilities (each gene scored by a model trained
    on the others). Used both for calibration and for the reported calibration curve."""
    folds, _ = make_folds(meta, "gene_disjoint", cfg.evaluation.cv.folds)
    oof = np.full(len(y), np.nan)
    yv = y.to_numpy()
    for tr, te in folds:
        if len(np.unique(yv[tr])) < 2:
            continue
        est, _ = build_estimator(cfg, seed)
        est.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = est.predict_proba(X.iloc[te])[:, 1]
    return oof


@dataclass
class ModelBundle:
    estimator: object
    backend: str
    calibrator: object
    calibration_method: str
    feature_names: list[str]
    decision: dict
    metadata: dict = field(default_factory=dict)

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X[self.feature_names])[:, 1]

    def calibrated_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.calibrator.predict(self.raw_proba(X)))

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path


def load_bundle(path: Path) -> ModelBundle:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def decide_label(prob, decision: dict) -> str:
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return "Uncertain"
    if prob >= decision["pathogenic_threshold"]:
        return "Likely pathogenic"
    if prob <= decision["benign_threshold"]:
        return "Likely benign"
    return "Uncertain"


def train_model(cfg: Config, df: pd.DataFrame) -> tuple[ModelBundle, pd.DataFrame]:
    set_global_seeds(cfg.project.seed)
    seed = cfg.project.seed
    X, y, meta = get_xy(df, cfg)  # full feature set, labelled missense
    if len(y) == 0 or y.nunique() < 2:
        raise ValueError("Not enough labelled missense variants with both classes to train.")

    oof = gene_disjoint_oof(X, y, meta, cfg, seed)
    calibrator = fit_calibrator(oof, y.to_numpy(), cfg.model.calibration)

    estimator, backend = build_estimator(cfg, seed)
    estimator.fit(X, y)

    bundle = ModelBundle(
        estimator=estimator, backend=backend, calibrator=calibrator,
        calibration_method=cfg.model.calibration, feature_names=list(X.columns),
        decision={"pathogenic_threshold": cfg.model.decision.pathogenic_threshold,
                  "benign_threshold": cfg.model.decision.benign_threshold},
        metadata={"n_train": int(len(y)), "n_pathogenic": int(y.sum()),
                  "n_benign": int((1 - y).sum()), "genes": cfg.genes.panel,
                  "data_mode": cfg.data_sources.mode, "genome_build": cfg.project.genome_build,
                  "synthetic": cfg.data_sources.mode == "fixture"},
    )
    bundle.save(cfg.path("models", "model.pkl"))

    oof_df = meta.copy()
    oof_df["oof_raw"] = oof
    oof_df["oof_calibrated"] = np.where(~np.isnan(oof), calibrator.predict(np.nan_to_num(oof)), np.nan)
    oof_df.to_csv(cfg.path("processed", "oof_predictions.csv"), index=False)
    return bundle, oof_df
