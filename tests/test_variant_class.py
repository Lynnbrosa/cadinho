"""Variant-class routing: truncating -> rule track, missense -> ML track, rest excluded."""

from __future__ import annotations

import pytest

from cadinho.config import load_config
from cadinho.normalize.hgvs import parse_variant
from cadinho.variants.classify_class import derive_consequence, route

VC = load_config().variant_class


@pytest.mark.parametrize("name,consequence,track", [
    ("NM_000023.4(SGCA):c.229C>T (p.Arg77Cys)", "missense_variant", "missense"),
    ("NM_000023.4(SGCA):c.229C>T (p.Arg77Ter)", "stop_gained", "truncating"),
    ("NM_000023.4(SGCA):c.583delA (p.Arg195fs)", "frameshift_variant", "truncating"),
    ("NM_000023.4(SGCA):c.334+1G>A", "splice_variant", "truncating"),       # canonical ±1
    ("NM_000023.4(SGCA):c.334+5G>A", "splice_region_variant", "other"),     # deep intronic
    ("NM_000023.4(SGCA):c.229C>T (p.Arg77=)", "synonymous_variant", "other"),
])
def test_route(name, consequence, track):
    pv = parse_variant(name, VC.splice_canonical_offset)
    cons = derive_consequence(pv)
    assert cons == consequence
    assert route(cons, VC) == track
