"""Head-to-head against the reference predictors (REVEL, AlphaMissense).

We compare the model's gene-disjoint OOF ranking against each reference predictor on the
*same* held-out variants. REVEL/AlphaMissense were themselves trained on ClinVar/HGMD, so
they are strong, partly-circular baselines: beating them is not guaranteed and not the point.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cadinho.config import Config
from cadinho.evaluate.metrics import pooled_oof
from cadinho.features.build import get_xy


def compare_to_baselines(df: Config, cfg: Config) -> dict:
    yv, oof, _ = pooled_oof(df, cfg)
    X, _, _ = get_xy(df, cfg)
    model_mask = ~np.isnan(oof)

    out = {"model_oof": {
        "auc": float(roc_auc_score(yv[model_mask], oof[model_mask])),
        "ap": float(average_precision_score(yv[model_mask], oof[model_mask])),
        "n": int(model_mask.sum()),
    }}
    for pred in cfg.evaluation.baselines.reference_predictors:
        if pred not in X.columns:
            continue
        s = X[pred].to_numpy()
        m = model_mask & ~np.isnan(s)
        if m.sum() == 0 or len(np.unique(yv[m])) < 2:
            continue
        out[pred] = {
            "auc": float(roc_auc_score(yv[m], s[m])),
            "ap": float(average_precision_score(yv[m], s[m])),
            "n": int(m.sum()),
            "threshold": cfg.evaluation.baselines.thresholds.get(pred),
        }
    return out
