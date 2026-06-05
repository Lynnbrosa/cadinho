"""Map a residue position to its UniProt protein region for the explanation."""

from __future__ import annotations

from cadinho.ingest import uniprot


def describe(uni: dict, gene: str, position: int | None) -> dict:
    rf = uniprot.region_features(uni, gene, position)
    regions = uniprot._regions_for_gene(uni, gene)
    if position is None or rf["region_ordinal"] == 0:
        sentence = f"Residue {position} could not be mapped to an annotated region."
    else:
        kind = "an annotated functional region" if rf["in_known_domain"] else "a region"
        sentence = (f"Residue {position} falls in {kind}: “{rf['region_name']}” "
                    f"(region {rf['region_ordinal']} of {len(regions)}).")
    return {**rf, "n_regions": len(regions), "sentence": sentence}
