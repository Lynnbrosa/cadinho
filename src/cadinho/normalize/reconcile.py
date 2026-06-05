"""Reconcile sources into one unified, deduplicated GRCh38 variant table.

The universe is the ClinVar slice (labelled P/B + held-out VUS). We left-join dbNSFP
features and gnomAD AF on (chrom,pos,ref,alt), derive the functional consequence + track
(missense/truncating/other) from HGVS, and attach UniProt positional context.
"""

from __future__ import annotations

import pandas as pd

from cadinho.config import Config
from cadinho.ingest import uniprot
from cadinho.ingest.run import load_sources
from cadinho.normalize.hgvs import ParsedVariant, parse_c, parse_p
from cadinho.variants.classify_class import derive_consequence, route

_DBNSFP_FEATURES = [
    "SIFT_score", "Polyphen2_HVAR_score", "MetaSVM_score", "CADD_phred",
    "REVEL_score", "AlphaMissense_score", "GERPpp_RS", "phyloP100way_vertebrate",
    "phastCons100way_vertebrate", "SiPhy_29way_logOdds",
]
_KEY = ["chrom", "pos", "ref", "alt"]


def has_dbnsfp_features(df: pd.DataFrame) -> bool:
    """True if the table carries any dbNSFP conservation/predictor column."""
    return any(c in df.columns for c in _DBNSFP_FEATURES)


def _normalize_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["chrom"] = df["chrom"].astype(str)
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    df["ref"] = df["ref"].astype(str)
    df["alt"] = df["alt"].astype(str)
    return df


def _consequence_and_class(row, cfg: Config) -> tuple[str, str]:
    c = parse_c(row["hgvs_c"], cfg.variant_class.splice_canonical_offset) if row["hgvs_c"] else None
    p = parse_p(row["hgvs_p"]) if row["hgvs_p"] else None
    pv = ParsedVariant(raw="", c=c, p=p)
    cons = derive_consequence(pv)
    return cons, route(cons, cfg.variant_class)


def reconcile(sources: dict, cfg: Config) -> pd.DataFrame:
    clin = _normalize_key(sources["clinvar"])
    uni = sources["uniprot"]
    df = clin.copy()
    for col in ("aaref", "aaalt", "protein_position"):
        if col not in df.columns:
            df[col] = pd.NA

    # --- dbNSFP feature layer (OPTIONAL: may be absent for a first real run) ---
    db = sources.get("dbnsfp")
    has_db = db is not None and len(db) > 0
    if has_db:
        db = _normalize_key(db).drop_duplicates(subset=_KEY)
        db_cols = _KEY + [c for c in ["aaref", "aaalt", "protein_position", *_DBNSFP_FEATURES]
                          if c in db.columns]
        df = df.merge(db[db_cols], on=_KEY, how="left", suffixes=("", "_db"))
        # dbNSFP residue annotation is authoritative where present; else keep ClinVar's
        for col in ("aaref", "aaalt", "protein_position"):
            dbcol = f"{col}_db"
            if dbcol in df.columns:
                df[col] = df[dbcol].where(df[dbcol].notna(), df[col])
                df = df.drop(columns=[dbcol])

    # --- gnomAD AF (OPTIONAL per gene; missing => absent => 0, flagged) ---
    gno = sources.get("gnomad")
    if gno is not None and len(gno) > 0:
        gno = _normalize_key(gno).drop_duplicates(subset=_KEY)
        df = df.merge(gno, on=_KEY, how="left")
        df["in_gnomad"] = df["gnomad_AF"].notna()
    else:
        df["gnomad_AF"] = df["gnomad_AF_popmax"] = df["gnomad_nhomalt"] = pd.NA
        df["in_gnomad"] = False
    for col in ("gnomad_AF", "gnomad_AF_popmax", "gnomad_nhomalt"):
        df[col] = df[col].fillna(0)

    # consequence + track from HGVS
    cc = df.apply(lambda r: _consequence_and_class(r, cfg), axis=1, result_type="expand")
    df["consequence"], df["variant_class"] = cc[0], cc[1]

    # UniProt positional context
    pos_feats = df.apply(
        lambda r: uniprot.region_features(
            uni, r["gene"],
            int(r["protein_position"]) if pd.notna(r["protein_position"]) else None),
        axis=1, result_type="expand")
    for col in ("region_ordinal", "in_known_domain", "region_name"):
        df[col] = pos_feats[col]

    return df.reset_index(drop=True)


def build_interim(cfg: Config, force: bool = False) -> pd.DataFrame:
    sources = load_sources(cfg, force=force)
    df = reconcile(sources, cfg)
    out = cfg.path("interim", "variants.csv")
    df.to_csv(out, index=False)
    return df


def load_interim(cfg: Config) -> pd.DataFrame:
    path = cfg.path("interim", "variants.csv")
    if not path.exists():
        return build_interim(cfg)
    df = pd.read_csv(path, dtype={"chrom": str})
    # restore "" for string columns that pandas read back as NaN (preserve feature NaNs)
    for col in ("hgvs_c", "hgvs_p", "transcript", "clinvar_sig", "consequence",
                "variant_class", "label_class", "region_name", "aaref", "aaalt"):
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df
