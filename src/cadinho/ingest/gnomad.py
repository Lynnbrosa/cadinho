"""gnomAD v4 (GRCh38) ingest: per-variant population allele frequency.

parse() reads a cached GraphQL-style JSON and flattens to (chrom,pos,ref,alt) + AF,
popmax AF, homozygote count. download_live() does a gene-scoped GraphQL pull and caches it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cadinho.config import Config
from cadinho.ingest import manifest
from cadinho.ingest.layout import raw_paths

_QUERY = """
query VariantsInGene($gene: String!, $dataset: DatasetId!) {
  gene(gene_symbol: $gene, reference_genome: GRCh38) {
    variants(dataset: $dataset) {
      variant_id pos ref alt
      genome { ac an af homozygote_count populations { id ac an } }
    }
  }
}
"""


def _variants(obj: dict) -> list[dict]:
    data = obj.get("data", obj)
    if "variants" in data:
        return data["variants"]
    return (data.get("gene") or {}).get("variants") or []


def parse(path: Path, cfg: Config) -> pd.DataFrame:
    obj = json.loads(Path(path).read_text())
    recs = []
    for v in _variants(obj):
        g = v.get("genome") or {}
        af = float(g.get("af") or 0.0)
        pop_afs = []
        for p in g.get("populations") or []:
            pa = p.get("af")
            if pa is None and p.get("an"):
                pa = (p.get("ac") or 0) / p["an"]
            if pa is not None:
                pop_afs.append(float(pa))
        chrom = str(v.get("chrom") or v["variant_id"].split("-")[0])
        recs.append({
            "chrom": chrom, "pos": int(v["pos"]), "ref": v["ref"], "alt": v["alt"],
            "gnomad_AF": af,
            "gnomad_AF_popmax": max(pop_afs) if pop_afs else af,
            "gnomad_nhomalt": int(g.get("homozygote_count") or 0),
        })
    return pd.DataFrame(recs, columns=["chrom", "pos", "ref", "alt", "gnomad_AF",
                                       "gnomad_AF_popmax", "gnomad_nhomalt"])


def download_live(cfg: Config) -> Path:
    import requests

    api = cfg.data_sources.gnomad["api"]
    dataset = cfg.data_sources.gnomad.get("dataset", "gnomad_r4")
    all_variants = []
    for gene in cfg.genes.panel:
        resp = requests.post(api, json={"query": _QUERY,
                                        "variables": {"gene": gene, "dataset": dataset}},
                             timeout=180)
        resp.raise_for_status()
        payload = resp.json()
        for v in (payload.get("data") or {}).get("gene", {}).get("variants") or []:
            v["chrom"] = v["variant_id"].split("-")[0]
            for p in (v.get("genome") or {}).get("populations") or []:
                if p.get("an"):
                    p["af"] = (p.get("ac") or 0) / p["an"]
            all_variants.append(v)
    obj = {"data": {"meta": {"dataset": dataset, "build": "GRCh38"},
                    "variants": all_variants}}
    out = raw_paths(cfg)["gnomad"]
    out.write_text(json.dumps(obj, indent=1))
    manifest.record_source(cfg.path("raw"), "gnomad", mode="live", path=out,
                           version=dataset, url=api, n_rows=len(all_variants))
    return out
