"""ClinVar ingest: download/parse variant_summary, map labels, filter to GRCh38 + panel.

Output (parse): one row per unique GRCh38 variant with HGVS, label class/binary, review
stars. VUS/Conflicting are kept (flagged is_vus) for end-stage prediction but carry no
binary label. Below `min_review_stars` labelled rows are dropped; VUS are kept regardless.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cadinho.config import Config
from cadinho.ingest import manifest
from cadinho.ingest.layout import raw_paths
from cadinho.normalize.hgvs import parse_variant
from cadinho.normalize.labels import (
    classify_significance,
    review_status_to_stars,
    to_binary,
)


def parse(path: Path, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    df = df[(df["Assembly"] == "GRCh38") & (df["GeneSymbol"].isin(cfg.genes.panel))]
    recs = []
    for _, r in df.iterrows():
        pv = parse_variant(r["Name"], cfg.variant_class.splice_canonical_offset)
        klass = classify_significance(r["ClinicalSignificance"], cfg.labels)
        pos = r.get("PositionVCF") or r.get("Start")
        ref = r.get("ReferenceAlleleVCF") or r.get("ReferenceAllele")
        alt = r.get("AlternateAlleleVCF") or r.get("AlternateAllele")
        recs.append({
            "gene": r["GeneSymbol"], "chrom": str(r["Chromosome"]),
            "pos": int(pos) if str(pos).isdigit() else None,
            "ref": ref, "alt": alt,
            "transcript": pv.transcript or "", "hgvs_c": pv.hgvs_c or "",
            "hgvs_p": pv.hgvs_p or "",
            "clinvar_sig": r["ClinicalSignificance"],
            "review_status": r["ReviewStatus"],
            "review_stars": review_status_to_stars(r["ReviewStatus"]),
            "label_class": klass, "label": to_binary(klass), "is_vus": klass == "vus",
            "variation_id": r.get("VariationID"),
        })
    out = pd.DataFrame(recs).dropna(subset=["pos"])
    if cfg.labels.drop_if_unmatched:
        out = out[out["label_class"].notna()]
    # stars floor applies to labelled (P/B); VUS kept regardless (predicted, not trusted)
    keep = (out["label_class"] == "vus") | (out["review_stars"] >= cfg.labels.min_review_stars)
    out = out[keep]
    # one row per variant: prefer the highest-confidence assertion
    out = (out.sort_values("review_stars", ascending=False)
              .drop_duplicates(subset=["chrom", "pos", "ref", "alt"])
              .reset_index(drop=True))
    return out


def download_live(cfg: Config) -> Path:
    import requests  # local import: only needed in live mode

    url = cfg.data_sources.clinvar["url"]
    raw = cfg.path("raw")
    gz = raw / "variant_summary.txt.gz"
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        last_mod = resp.headers.get("Last-Modified")
        with open(gz, "wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
    df = pd.read_csv(gz, sep="\t", dtype=str, compression="gzip")
    sub = df[(df["Assembly"] == "GRCh38") & (df["GeneSymbol"].isin(cfg.genes.panel))]
    out = raw_paths(cfg)["clinvar"]
    sub.to_csv(out, sep="\t", index=False)
    manifest.record_source(raw, "clinvar", mode="live", path=out, version=last_mod,
                           url=url, n_rows=len(sub),
                           note="GRCh38 + panel slice of ClinVar variant_summary")
    gz.unlink(missing_ok=True)
    return out
