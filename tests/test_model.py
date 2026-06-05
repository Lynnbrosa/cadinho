"""Model determinism + the honest 'Uncertain' decision band."""

from __future__ import annotations

import numpy as np

from cadinho.model.train import decide_label, train_model
from cadinho.normalize.reconcile import build_interim


def test_training_is_deterministic(cfg):
    df = build_interim(cfg, force=True)
    _, oof1 = train_model(cfg, df)
    _, oof2 = train_model(cfg, df)
    m = oof1["oof_raw"].notna()
    assert np.allclose(oof1["oof_raw"][m].to_numpy(), oof2["oof_raw"][m].to_numpy())


def test_decision_band_never_coerces_middle(cfg):
    decision = {"pathogenic_threshold": 0.9, "benign_threshold": 0.1}
    assert decide_label(0.95, decision) == "Likely pathogenic"
    assert decide_label(0.05, decision) == "Likely benign"
    assert decide_label(0.5, decision) == "Uncertain"
    assert decide_label(float("nan"), decision) == "Uncertain"
    assert decide_label(None, decision) == "Uncertain"
