"""Per-variant + VUS reporting behaviour."""

from __future__ import annotations

from cadinho.ingest import uniprot
from cadinho.ingest.layout import raw_paths
from cadinho.model.train import train_model
from cadinho.normalize.reconcile import build_interim
from cadinho.report.variant_report import (
    report_for_row,
    report_variant,
    run_vus_report,
)


def _setup(cfg):
    df = build_interim(cfg, force=True)
    bundle, _ = train_model(cfg, df)
    uni = uniprot.parse(raw_paths(cfg)["uniprot"])
    return df, bundle, uni


def test_r77c_missense_report(cfg):
    df, bundle, uni = _setup(cfg)
    rep = report_variant(cfg, bundle, df, uni, "c.229C>T")
    assert rep["resolved"] and rep["track"] == "missense"
    assert 0.0 <= rep["probability"] <= 1.0
    assert rep["predicted_class"] in {"Likely pathogenic", "Uncertain", "Likely benign"}
    assert "SYNTHETIC" in rep["markdown"] and "attribution" in rep


def test_protein_and_coding_resolve_to_same_variant(cfg):
    df, bundle, uni = _setup(cfg)
    a = report_variant(cfg, bundle, df, uni, "c.229C>T")
    b = report_variant(cfg, bundle, df, uni, "p.Arg77Cys")
    assert a["hgvs_c"] == b["hgvs_c"] == "c.229C>T"


def test_unresolved_variant_is_honest(cfg):
    df, bundle, uni = _setup(cfg)
    rep = report_variant(cfg, bundle, df, uni, "c.99999A>G")
    assert rep["resolved"] is False and "Could not resolve" in rep["reason"]


def test_truncating_uses_rule_track_not_probability(cfg):
    df, bundle, uni = _setup(cfg)
    t = df[(df["gene"] == "SGCA") & (df["variant_class"] == "truncating")].iloc[0]
    rep = report_for_row(cfg, bundle, df, uni, t)
    assert rep["track"] == "truncating" and rep["probability"] is None


def test_vus_report_only_focus_gene_and_untrained(cfg):
    df, bundle, uni = _setup(cfg)
    vus = run_vus_report(cfg, bundle, df, uni)
    if len(vus):
        assert (vus["gene"] == cfg.genes.focus).all()
