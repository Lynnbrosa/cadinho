"""SHAP attribution so explanations are grounded in the model, not narrated.

Wraps shap.TreeExplainer over the gradient-boosting model. Degrades gracefully to global
feature importances if SHAP can't explain the backend.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class Explainer:
    def __init__(self, bundle):
        self.bundle = bundle
        self.features = bundle.feature_names
        self.explainer = None
        try:
            import shap
            self.explainer = shap.TreeExplainer(bundle.estimator)
        except Exception:
            self.explainer = None

    def attribute(self, X_row: pd.DataFrame, top: int = 6) -> list[dict]:
        """Return top features by |contribution| toward the pathogenic class."""
        Xf = X_row[self.features]
        values = None
        if self.explainer is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sv = self.explainer.shap_values(Xf)
                arr = np.array(sv)
                if arr.ndim == 3:  # (classes,n,feat) or (n,feat,classes)
                    arr = arr[1] if arr.shape[0] == 2 else arr[..., -1]
                values = arr.reshape(-1, len(self.features))[0]
            except Exception:
                values = None
        if values is None:  # fallback: global importances * sign-agnostic
            imp = getattr(self.bundle.estimator, "feature_importances_", None)
            values = (np.asarray(imp, dtype=float) if imp is not None
                      else np.zeros(len(self.features)))
        fv = Xf.iloc[0].to_dict()
        out = [{"feature": f, "contribution": float(v), "value": fv.get(f)}
               for f, v in zip(self.features, values)]
        out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return out[:top]
