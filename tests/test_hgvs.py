"""HGVS parsing + CDS->protein mapping. These are the silent-bug hotspots."""

from __future__ import annotations

import pytest

from cadinho.normalize.hgvs import cdna_to_protein, parse_c, parse_p, parse_variant


@pytest.mark.parametrize("cdna,expected", [
    (1, (1, 1)), (2, (1, 2)), (3, (1, 3)), (4, (2, 1)),
    (229, (77, 1)),   # R77C: c.229 -> residue 77, codon position 1
    (231, (77, 3)),
])
def test_cdna_to_protein(cdna, expected):
    assert cdna_to_protein(cdna) == expected


def test_cdna_to_protein_rejects_nonpositive():
    with pytest.raises(ValueError):
        cdna_to_protein(0)


def test_parse_c_snv():
    c = parse_c("c.229C>T")
    assert (c.cdna_pos, c.ref_nt, c.alt_nt) == (229, "C", "T")
    assert c.event == "substitution" and c.is_coding_snv and c.intron_offset is None


@pytest.mark.parametrize("s,offset,canonical", [
    ("c.334+1G>A", 1, True),
    ("c.100-2A>G", -2, True),
    ("c.100+5G>A", 5, False),
])
def test_parse_c_splice(s, offset, canonical):
    c = parse_c(s)
    assert c.intron_offset == offset
    assert c.is_splice_canonical is canonical
    assert not c.is_coding_snv  # intronic -> not a coding SNV


def test_parse_c_indel():
    assert parse_c("c.583delA").event == "indel"


@pytest.mark.parametrize("s", ["p.Arg77Cys", "p.(Arg77Cys)", "p.R77C"])
def test_parse_p_missense(s):
    p = parse_p(s)
    assert (p.aaref, p.pos, p.aaalt, p.consequence) == ("R", 77, "C", "missense")


@pytest.mark.parametrize("s", ["p.Arg77Ter", "p.Arg77*"])
def test_parse_p_nonsense(s):
    p = parse_p(s)
    assert p.aaalt == "*" and p.consequence == "nonsense"


def test_parse_p_frameshift():
    p = parse_p("p.Gln155fs")
    assert p.pos == 155 and p.consequence == "frameshift"


@pytest.mark.parametrize("s", ["p.Arg77=", "p.(=)"])
def test_parse_p_synonymous(s):
    assert parse_p(s).consequence == "synonymous"


def test_parse_variant_clinvar_name():
    pv = parse_variant("NM_000023.4(SGCA):c.229C>T (p.Arg77Cys)")
    assert pv.transcript == "NM_000023.4" and pv.gene == "SGCA"
    assert pv.hgvs_c == "c.229C>T" and pv.hgvs_p == "p.Arg77Cys"
    assert pv.protein_position == 77 and pv.aaref == "R" and pv.aaalt == "C"


def test_parse_variant_bare_c_derives_position():
    # from c. alone we know the position (77) but not the residues (no transcript offline)
    pv = parse_variant("c.229C>T")
    assert pv.protein_position == 77 and pv.aaref is None


def test_parse_variant_bare_p():
    pv = parse_variant("p.Arg77Cys")
    assert pv.protein_position == 77 and pv.aaalt == "C"
