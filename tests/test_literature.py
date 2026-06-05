"""Literature module: query building + the never-fabricate / offline contract."""

from __future__ import annotations

from cadinho.config import load_config
from cadinho.literature.pubmed import (
    build_query,
    literature_for_variant,
    variant_aliases,
)
from cadinho.normalize.hgvs import parse_variant

CFG = load_config()


def test_variant_aliases_for_r77c():
    pv = parse_variant("NM_000023.4(SGCA):c.229C>T (p.Arg77Cys)")
    aliases = variant_aliases(pv)
    assert "p.Arg77Cys" in aliases
    assert "Arg77Cys" in aliases
    assert "R77C" in aliases
    assert "c.229C>T" in aliases


def test_build_query_includes_gene_and_terms():
    q = build_query("SGCA", ["R77C", "p.Arg77Cys"])
    assert q.startswith("SGCA AND (")
    assert '"R77C"' in q and '"p.Arg77Cys"' in q


def test_offline_never_fabricates():
    # fixture mode -> defaults offline; must return the query but zero (invented) results
    res = literature_for_variant(CFG, "p.Arg77Cys", gene="SGCA")
    assert res.online is False
    assert res.results == []
    assert "SGCA" in res.query
    assert "fabricat" in res.note.lower()
