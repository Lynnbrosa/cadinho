"""dbNSFP-optional plumbing + section-5 validation (tested on fixtures)."""

from __future__ import annotations

from cadinho.features.build import available_feature_names, get_xy
from cadinho.ingest.dbnsfp import gene_intervals_from_clinvar
from cadinho.ingest.run import load_sources
from cadinho.model.train import train_model
from cadinho.normalize.reconcile import (
    build_interim,
    has_dbnsfp_features,
    reconcile,
)
from cadinho.validate import run_validation


def test_pipeline_runs_without_dbnsfp(cfg):
    src = load_sources(cfg, force=True)
    src["dbnsfp"] = None  # simulate a first real run with no dbNSFP enrichment
    df = reconcile(src, cfg)

    assert not has_dbnsfp_features(df)
    feats = available_feature_names(df, cfg)
    assert "REVEL_score" not in feats and "GERPpp_RS" not in feats  # dbNSFP gone
    assert "Grantham" in feats and "gnomad_AF" in feats             # biochem + AF remain

    # residue must come from ClinVar HGVS when dbNSFP is absent
    r77c = df[(df["gene"] == "SGCA") & (df["hgvs_c"] == "c.229C>T")].iloc[0]
    assert r77c["aaref"] == "R" and r77c["aaalt"] == "C" and r77c["protein_position"] == 77

    X, y, _ = get_xy(df, cfg)
    assert X.shape[1] == len(feats) and y.nunique() == 2
    bundle, _ = train_model(cfg, df)            # must train on the reduced feature set
    assert len(bundle.feature_names) == len(feats)


def test_validation_checks(cfg):
    df = build_interim(cfg, force=True)
    res = run_validation(df, cfg)
    assert res["r77c"]["present"] and res["r77c"]["ok"]
    assert res["r77c"]["gnomad_AF"] <= 1e-3
    counts = res["per_gene"]
    assert set(counts[counts["gene"] != "TOTAL"]["gene"]) == set(cfg.genes.panel)
    assert res["consistency"]["declared_build"] == "GRCh38"


def test_gene_intervals_from_clinvar(cfg):
    from cadinho.ingest import clinvar
    from cadinho.ingest.layout import raw_paths
    load_sources(cfg, force=True)  # ensure fixtures written
    clin = clinvar.parse(raw_paths(cfg)["clinvar"], cfg)
    intervals = gene_intervals_from_clinvar(clin, cfg)
    assert set(intervals) == set(cfg.genes.panel)
    for gene, (chrom, start, end) in intervals.items():
        assert chrom == str(cfg.genes.meta[gene]["chrom"])
        assert start < end
