"""The anti-leakage guarantees: no gene or residue position spans train and test."""

from __future__ import annotations

from cadinho.evaluate.splits import gene_groups, make_folds, position_groups
from cadinho.features.build import get_xy
from cadinho.normalize.reconcile import build_interim


def test_gene_disjoint_has_no_gene_overlap(cfg):
    df = build_interim(cfg, force=True)
    _, _, meta = get_xy(df, cfg)
    folds, _ = make_folds(meta, "gene_disjoint", cfg.evaluation.cv.folds)
    g = gene_groups(meta)
    assert len(folds) == len(set(g))            # one fold per gene
    held = set()
    for tr, te in folds:
        assert set(g[tr]).isdisjoint(set(g[te]))  # no gene in both sides
        held |= set(g[te])
    assert held == set(g)                        # every gene held out exactly once


def test_position_kfold_has_no_position_overlap(cfg):
    df = build_interim(cfg, force=True)
    _, _, meta = get_xy(df, cfg)
    folds, _ = make_folds(meta, "position_kfold", cfg.evaluation.cv.folds)
    pg = position_groups(meta)
    assert len(folds) > 0
    for tr, te in folds:
        # the same (gene, residue) must never appear in train and test
        assert set(pg[tr]).isdisjoint(set(pg[te]))
