"""UniProt ingest: protein domains / topology for positional context + the report.

parse() -> {gene: {accession, length, features:[{type,description,start,end}]}}.
region_features() maps a residue to its containing region (used by feature build) and a
human-readable region name (used by the report's domain mapping). Only SGCA's accession
(Q16586) is verified; a null accession degrades to "unknown region".
"""

from __future__ import annotations

import json
from pathlib import Path

from cadinho.config import Config
from cadinho.ingest import manifest
from cadinho.ingest.layout import raw_paths

# region types we treat as "in a known structured/functional region"
_DOMAIN_TYPES = {"Domain", "Region", "Repeat", "Topological domain", "Transmembrane"}
# UniProt feature types worth keeping for context
_KEEP_TYPES = {"Signal", "Chain", "Domain", "Region", "Repeat", "Transmembrane",
               "Topological domain", "Disulfide bond", "Binding site", "Active site"}


def parse(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _regions_for_gene(uni: dict, gene: str) -> list[dict]:
    entry = uni.get(gene) or {}
    out = []
    for f in entry.get("features", []):
        loc = f.get("location", {})
        try:
            start = int(loc["start"]["value"])
            end = int(loc["end"]["value"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"type": f.get("type"), "description": f.get("description", ""),
                    "start": start, "end": end})
    return sorted(out, key=lambda r: (r["start"], r["end"]))


def region_features(uni: dict, gene: str, position: int | None) -> dict:
    """Positional features for a residue: ordinal of containing region + domain flag + name."""
    if position is None:
        return {"region_ordinal": 0, "in_known_domain": 0, "region_name": "unknown"}
    regions = _regions_for_gene(uni, gene)
    best = None
    for i, r in enumerate(regions, start=1):
        if r["start"] <= position <= r["end"]:
            # prefer the most specific (smallest) containing region
            if best is None or (r["end"] - r["start"]) < (best[1]["end"] - best[1]["start"]):
                best = (i, r)
    if best is None:
        return {"region_ordinal": 0, "in_known_domain": 0, "region_name": "unknown"}
    ordinal, r = best
    name = r["description"] or r["type"] or "region"
    return {
        "region_ordinal": ordinal,
        "in_known_domain": int(r["type"] in _DOMAIN_TYPES),
        "region_name": name,
    }


def download_live(cfg: Config) -> Path:
    import requests

    api = cfg.data_sources.uniprot["api"]
    res = {}
    for gene in cfg.genes.panel:
        acc = cfg.genes.meta[gene].get("uniprot")
        if not acc:
            res[gene] = {"primaryAccession": None, "features": [],
                         "note": "accession unverified -> unknown region"}
            continue
        r = requests.get(f"{api}/{acc}.json", timeout=120)
        r.raise_for_status()
        j = r.json()
        feats = []
        for f in j.get("features", []):
            if f.get("type") not in _KEEP_TYPES:
                continue
            loc = f.get("location", {})
            try:
                feats.append({
                    "type": f["type"],
                    "description": (f.get("description") or ""),
                    "location": {"start": {"value": loc["start"]["value"]},
                                 "end": {"value": loc["end"]["value"]}},
                })
            except (KeyError, TypeError):
                continue
        res[gene] = {"primaryAccession": acc,
                     "sequence_length": (j.get("sequence") or {}).get("length"),
                     "features": feats}
    out = raw_paths(cfg)["uniprot"]
    out.write_text(json.dumps(res, indent=1))
    manifest.record_source(cfg.path("raw"), "uniprot", mode="live", path=out,
                           version="uniprotkb", url=api,
                           n_rows=sum(len(v.get("features", [])) for v in res.values()))
    return out
