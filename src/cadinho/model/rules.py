"""Rule-based track for truncating variants.

For a recessive loss-of-function gene like SGCA, nonsense / frameshift / canonical ±1,2
splice variants are generally pathogenic. We classify them by an explicit, flagged rule
and NEVER feed them to the missense ML model. This is a heuristic, clearly labelled as
such — not a calibrated probability.
"""

from __future__ import annotations

import pandas as pd

from cadinho.config import Config

_BASIS = {
    "stop_gained": "Premature stop codon (nonsense)",
    "frameshift_variant": "Frameshift",
    "splice_variant": "Canonical ±1/2 splice-site disruption",
}


def classify_truncating(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    trunc = df[df["variant_class"] == "truncating"].copy()
    rule_label = cfg.variant_class.truncating_rule_label
    trunc["prediction_track"] = "rule"
    trunc["predicted_class"] = (
        "Likely pathogenic" if rule_label == "pathogenic" else rule_label.capitalize()
    )
    trunc["calibrated_probability"] = pd.NA  # rules do not produce a calibrated probability
    trunc["rule_basis"] = trunc["consequence"].map(_BASIS).fillna("Loss of function")
    trunc["note"] = (
        "Predicted by rule (recessive loss-of-function gene), not the ML model. "
        "Confirm zygosity/segregation clinically; this is a heuristic."
    )
    return trunc
