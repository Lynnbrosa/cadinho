"""dbNSFP ingest: the feature backbone (conservation, AA props, predictor scores).

dbNSFP is a precomputed, genome-wide table of every possible non-synonymous SNV. We
slice by gene, never recompute conservation. Missing values are '.' in dbNSFP -> NaN.

The full release (~30 GB) is never downloaded. `fetch_live` does REMOTE tabix region
queries over the 4 gene intervals if a bgzip/tabix build that supports HTTP range
requests is configured (data_sources.dbnsfp.file_url) and pysam is installed; otherwise
it returns None and the pipeline runs without dbNSFP. `provided` mode also works: drop a
gene/chr-sliced file at the expected path with the documented columns.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cadinho.config import Config
from cadinho.ingest import manifest
from cadinho.ingest.layout import raw_paths

log = logging.getLogger("cadinho.dbnsfp")

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


def gene_intervals_from_clinvar(clinvar_df, cfg: Config, pad: int = 2000) -> dict[str, tuple]:
    """Derive per-gene (chrom, start, end) GRCh38 intervals from the ClinVar rows we
    already have (min/max position per gene). Avoids hardcoding coordinates; used to scope
    dbNSFP region queries to a few kb instead of the whole 30 GB file."""
    intervals: dict[str, tuple] = {}
    for gene, g in clinvar_df.groupby("gene"):
        pos = g["pos"].dropna().astype(int)
        if len(pos) == 0:
            continue
        chrom = str(cfg.genes.meta.get(gene, {}).get("chrom") or g["chrom"].iloc[0])
        intervals[gene] = (chrom, int(pos.min()) - pad, int(pos.max()) + pad)
    return intervals


def supports_range_requests(url: str, timeout: int = 20) -> bool:
    """HEAD-probe whether the dbNSFP file host supports HTTP range requests."""
    import requests

    r = requests.head(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.headers.get("Accept-Ranges", "").lower() == "bytes"


def fetch_live(cfg: Config, clinvar_df=None) -> Path | None:
    """Best-effort dbNSFP enrichment via REMOTE tabix region queries (no 30 GB download).

    Returns the written slice path, or None if not viable (so the caller proceeds without
    dbNSFP). Never downloads the full file. Requires:
      * data_sources.dbnsfp.file_url -> a bgzip+tabix (.tbi) dbNSFP build that supports ranges
      * pysam installed (optional extra: `uv pip install '.[dbnsfp]'`)
    """
    url = cfg.data_sources.dbnsfp.get("file_url")
    if not url:
        log.warning("dbNSFP file_url not configured; skipping enrichment.")
        return None
    try:
        if not supports_range_requests(url):
            log.warning("dbNSFP host does not advertise Accept-Ranges; skipping (won't pull 30 GB).")
            return None
    except Exception as e:
        log.warning("dbNSFP range probe failed (%s); skipping.", e)
        return None
    try:
        import pysam  # optional
    except Exception:
        log.warning("pysam not installed; cannot do remote tabix region queries. "
                    "Install with `uv pip install '.[dbnsfp]'`. Skipping dbNSFP.")
        return None

    if clinvar_df is None:
        from cadinho.ingest import clinvar as _cv
        clinvar_df = _cv.parse(raw_paths(cfg)["clinvar"], cfg)
    intervals = gene_intervals_from_clinvar(clinvar_df, cfg)

    out = raw_paths(cfg)["dbnsfp"]
    tbx = pysam.TabixFile(url)  # reads index remotely; fetches only requested byte ranges
    header = "\t".join(tbx.header[-1].split("\t")) if tbx.header else None
    n = 0
    with open(out, "w") as fh:
        if header:
            fh.write(header + "\n")
        for gene, (chrom, start, end) in intervals.items():
            for row in tbx.fetch(chrom, max(0, start), end):
                fh.write(row + "\n")
                n += 1
    manifest.record_source(cfg.path("raw"), "dbnsfp", mode="live", path=out, url=url,
                           version=str(cfg.data_sources.dbnsfp.get("version")),
                           n_rows=n, note="remote-tabix region slice (4 gene intervals)")
    log.info("dbNSFP remote slice: %d rows over %d genes -> %s", n, len(intervals), out)
    return out
