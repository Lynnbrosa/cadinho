"""Guard the verified sarcoglycan UniProt accessions (regression test for a real bug:
SGCB/SGCG were previously swapped/duplicated and SGCG was null)."""

from __future__ import annotations

from cadinho.config import load_config

# Verified accessions (UniProtKB), GRCh38 chromosomes.
EXPECTED = {
    "SGCA": ("Q16586", "17"),
    "SGCB": ("Q16585", "4"),
    "SGCG": ("Q13326", "13"),
    "SGCD": ("Q92629", "5"),
}


def test_accessions_match_verified_values():
    meta = load_config().genes.meta
    for gene, (acc, chrom) in EXPECTED.items():
        assert meta[gene]["uniprot"] == acc, f"{gene} accession should be {acc}"
        assert str(meta[gene]["chrom"]) == chrom, f"{gene} chrom should be {chrom}"


def test_all_four_accessions_present_and_distinct():
    meta = load_config().genes.meta
    accs = [meta[g]["uniprot"] for g in ("SGCA", "SGCB", "SGCG", "SGCD")]
    assert all(accs), "every sarcoglycan must have a non-null accession"
    assert len(set(accs)) == 4, f"accessions must be distinct, got {accs}"
