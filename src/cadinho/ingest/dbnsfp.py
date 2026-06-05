"""dbNSFP ingest: the feature backbone (conservation, AA props, predictor scores).

dbNSFP is a precomputed, genome-wide table of every possible non-synonymous SNV. We
slice by gene, never recompute conservation. Missing values are '.' in dbNSFP -> NaN.

There is no anonymous bulk URL for dbNSFP (academic download). `download_live` therefore
points you at `provided` mode: drop a gene/chr-sliced dbNSFP file at the expected path
with the documented columns (or slice the academic release with tabix using the
per-gene chromosomes in config.genes.meta).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cadinho.config import Config

# dbNSFP column -> internal name (only what we use). '++' is not identifier-safe.
_RENAME = {
    "#chr": "chrom", "pos(1-based)": "pos", "ref": "ref", "alt": "alt",
    "aaref": "aaref", "aaalt": "aaalt", "aapos": "protein_position",
    "genename": "gene", "GERP++_RS": "GERPpp_RS",
}
_NUMERIC = [
    "SIFT_score", "Polyphen2_HVAR_score", "MetaSVM_score", "CADD_phred",
    "REVEL_score", "AlphaMissense_score", "GERPpp_RS", "phyloP100way_vertebrate",
    "phastCons100way_vertebrate", "SiPhy_29way_logOdds",
]


def parse(path: Path, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, na_values=["."])
    df = df.rename(columns=_RENAME)
    df["chrom"] = df["chrom"].astype(str)
    # dbNSFP packs multiple transcripts as ';'-joined values; take the first token.
    for col in ("protein_position", "aaref", "aaalt", *(_NUMERIC)):
        if col in df.columns:
            df[col] = df[col].astype(str).str.split(";").str[0].replace({"nan": None})
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    df["protein_position"] = pd.to_numeric(df["protein_position"], errors="coerce").astype("Int64")
    for col in _NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def download_live(cfg: Config) -> Path:
    raise NotImplementedError(
        "dbNSFP has no anonymous bulk download URL. Use `data_sources.mode: provided`: "
        "place a gene/chr-sliced dbNSFP file (documented columns, incl. GERP++_RS, "
        "phyloP/phastCons, REVEL_score, AlphaMissense_score, ...) at "
        f"data/raw/dbnsfp_slice.{'_'.join(cfg.genes.panel)}.tsv, or slice the academic "
        "dbNSFP release with tabix using config.genes.meta[*].chrom. See README."
    )
