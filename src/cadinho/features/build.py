"""Assemble the missense feature matrix.

Features = dbNSFP conservation + predictor scores (already in the table) + gnomAD AF +
UniProt positional context + substitution properties computed here from the residue pair.
LightGBM/HistGradientBoosting consume NaN natively, so missing dbNSFP values are left as
NaN rather than imputed. `drop_groups` powers the honest with/without ablations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cadinho.config import Config
from cadinho.features.aa import AA_SET, substitution_features

SUBSTITUTION = ["Grantham", "BLOSUM62", "delta_hydrophobicity", "delta_volume"]
META_COLS = ["gene", "chrom", "pos", "ref", "alt", "protein_position", "hgvs_c", "hgvs_p",
             "aaref", "aaalt", "clinvar_sig", "review_stars", "label_class", "label",
             "is_vus", "region_name", "in_known_domain"]


def add_substitution_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols: dict[str, list] = {k: [] for k in SUBSTITUTION}
    for ref, alt in zip(df.get("aaref", []), df.get("aaalt", [])):
        if isinstance(ref, str) and isinstance(alt, str) and ref in AA_SET and alt in AA_SET:
            f = substitution_features(ref, alt)
        else:
            f = {k: np.nan for k in SUBSTITUTION}
        for k in SUBSTITUTION:
            cols[k].append(f[k])
    for k in SUBSTITUTION:
        df[k] = cols[k]
    return df


def feature_names(cfg: Config, drop_groups: tuple[str, ...] = ()) -> list[str]:
    drop: set[str] = set()
    for g in drop_groups:
        drop |= set(cfg.features.ablation_groups.get(g, []))
    return [f for f in cfg.features.all_feature_names() if f not in drop]


def get_xy(df: pd.DataFrame, cfg: Config, drop_groups: tuple[str, ...] = (),
           labelled_only: bool = True) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """Return (X, y, meta) for the missense track.

    labelled_only=True  -> P/B training/eval set (y in {0,1}).
    labelled_only=False -> all missense incl. VUS (y is None) for end-stage prediction.
    """
    mis = df[df["variant_class"] == "missense"].copy()
    if labelled_only:
        mis = mis[mis["label"].notna()]
    mis = add_substitution_features(mis).reset_index(drop=True)

    feats = feature_names(cfg, drop_groups)
    X = mis[feats].apply(pd.to_numeric, errors="coerce")
    y = mis["label"].astype(int) if labelled_only else None
    meta = mis[[c for c in META_COLS if c in mis.columns]].copy()
    return X, y, meta
