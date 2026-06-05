"""Leakage-aware metrics: repeated CV with CIs, pooled OOF, calibration bins."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from cadinho.config import Config
from cadinho.evaluate.splits import make_folds
from cadinho.features.build import get_xy
from cadinho.model.train import build_estimator


def _fit_predict(X_tr, y_tr, X_te, cfg, seed):
    est, _ = build_estimator(cfg, seed)
    est.fit(X_tr, y_tr)
    return est.predict_proba(X_te)[:, 1]


def cv_scores(df: pd.DataFrame, cfg: Config, drop_groups: tuple[str, ...] = (),
              repeats: int | None = None) -> dict:
    """Repeated, leakage-aware CV. Returns per-(repeat,fold) scores + summary with CIs."""
    X, y, meta = get_xy(df, cfg, drop_groups)
    folds, _ = make_folds(meta, cfg.evaluation.cv.scheme, cfg.evaluation.cv.folds)
    repeats = repeats if repeats is not None else cfg.evaluation.cv.repeats
    yv = y.to_numpy()
    rows = []
    for rep in range(repeats):
        seed = cfg.project.seed + rep
        for fi, (tr, te) in enumerate(folds):
            if len(np.unique(yv[tr])) < 2 or len(np.unique(yv[te])) < 2:
                continue  # a fold with one class can't yield AUC; skip honestly
            p = _fit_predict(X.iloc[tr], y.iloc[tr], X.iloc[te], cfg, seed)
            rows.append({"repeat": rep, "fold": fi, "n": int(len(te)),
                         "auc": roc_auc_score(yv[te], p),
                         "ap": average_precision_score(yv[te], p),
                         "brier": brier_score_loss(yv[te], p)})
    sc = pd.DataFrame(rows)

    def agg(metric: str) -> dict:
        v = sc[metric].to_numpy()
        if len(v) == 0:
            return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
        return {"mean": float(np.mean(v)),
                "lo": float(np.percentile(v, 2.5)),
                "hi": float(np.percentile(v, 97.5))}

    return {"per_fold": sc, "n_samples": len(sc),
            "summary": {m: agg(m) for m in ("auc", "ap", "brier")}}


def pooled_oof(df: pd.DataFrame, cfg: Config, drop_groups: tuple[str, ...] = ()):
    """Single-pass OOF probabilities (one seed) for calibration assessment."""
    X, y, meta = get_xy(df, cfg, drop_groups)
    folds, _ = make_folds(meta, cfg.evaluation.cv.scheme, cfg.evaluation.cv.folds)
    yv = y.to_numpy()
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        if len(np.unique(yv[tr])) < 2:
            continue
        oof[te] = _fit_predict(X.iloc[tr], y.iloc[tr], X.iloc[te], cfg, cfg.project.seed)
    return yv, oof, meta


def calibration_bins(y: np.ndarray, prob: np.ndarray, n_bins: int = 10):
    mask = ~np.isnan(prob)
    frac_pos, mean_pred = calibration_curve(y[mask], prob[mask], n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y[mask], prob[mask])
    return frac_pos, mean_pred, brier
