"""Leakage-aware cross-validation splits.

  gene_disjoint  : leave-one-gene-out (train on some sarcoglycans, test on a held-out one).
                   The primary scheme — prevents gene/position memorisation.
  position_kfold : GroupKFold where the group is (gene, protein position), so variants at
                   the SAME residue never span train and test.

Both return (folds, group_ids) where folds is a list of (train_idx, test_idx).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


def gene_groups(meta: pd.DataFrame) -> np.ndarray:
    return meta["gene"].to_numpy()


def position_groups(meta: pd.DataFrame) -> np.ndarray:
    return (meta["gene"].astype(str) + ":" + meta["protein_position"].astype(str)).to_numpy()


def make_folds(meta: pd.DataFrame, scheme: str, n_folds: int):
    n = len(meta)
    if scheme == "gene_disjoint":
        groups = gene_groups(meta)
        folds = list(LeaveOneGroupOut().split(np.zeros(n), groups=groups))
    elif scheme == "position_kfold":
        groups = position_groups(meta)
        k = min(n_folds, len(np.unique(groups)))
        folds = list(GroupKFold(n_splits=k).split(np.zeros(n), groups=groups))
    else:
        raise ValueError(f"unknown CV scheme: {scheme}")
    return folds, groups
