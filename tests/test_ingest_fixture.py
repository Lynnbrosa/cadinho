"""Fixture ingest + reconcile invariants (build filtering, routing, R77C resolution)."""

from __future__ import annotations

from cadinho.normalize.reconcile import build_interim


def test_build_interim_invariants(cfg):
    df = build_interim(cfg, force=True)

    # all four panel genes present
    assert set(df["gene"].unique()) == set(cfg.genes.panel)

    # truncating variants are routed away from the missense track and carry no binary label
    trunc = df[df["variant_class"] == "truncating"]
    assert len(trunc) > 0
    assert trunc["consequence"].isin(
        ["stop_gained", "frameshift_variant", "splice_variant"]).all()

    # VUS are never given a binary label
    vus = df[df["label_class"] == "vus"]
    assert vus["label"].isna().all()

    # labelled missense has both classes for each gene (gene-disjoint CV needs this)
    mis = df[(df["variant_class"] == "missense") & df["label"].notna()]
    for gene in cfg.genes.panel:
        labels = set(mis[mis["gene"] == gene]["label"].unique())
        assert labels == {0, 1}, f"{gene} missing a class: {labels}"


def test_r77c_resolves(cfg):
    df = build_interim(cfg, force=True)
    r = df[(df["gene"] == "SGCA") & (df["hgvs_c"] == "c.229C>T")]
    assert len(r) == 1
    r = r.iloc[0]
    assert r["protein_position"] == 77
    assert r["variant_class"] == "missense"
    assert r["label_class"] == "pathogenic"
    assert r["in_known_domain"] == 1  # extracellular domain in the fixture topology


def test_grch37_rows_filtered_out(cfg):
    # fixtures inject GRCh37 duplicate rows; reconcile must keep only GRCh38 coordinates.
    df = build_interim(cfg, force=True)
    # GRCh37 fixture rows sit 500kb downstream; none should survive
    assert (df["pos"] < 600000).sum() == 0
