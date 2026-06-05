"""Deterministic SYNTHETIC data generator (fixture mode).

Emits raw files that mirror the *real* schemas of ClinVar variant_summary, dbNSFP,
gnomAD (GraphQL), and UniProt, so the entire downstream pipeline runs unchanged on
either fixtures or real data. A latent per-variant "propensity" drives BOTH the label
and the features (conserved+rare+high-predictor-score => more likely pathogenic, with
noise), so the model, calibration, and leakage-aware CV are genuinely exercised.

NONE OF THIS IS REAL. Every artifact is recorded as synthetic in the manifest, and
reports carry a SYNTHETIC banner. The numbers say nothing about real variants.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cadinho.config import Config
from cadinho.features.aa import AA1, ONE_TO_THREE, grantham_distance
from cadinho.ingest import manifest
from cadinho.ingest.layout import raw_paths

NUCS = list("ACGT")

# Approximate UniProt-style topology per gene (fixture). SGCA mirrors the known
# alpha-sarcoglycan layout (signal peptide / large extracellular domain / TM / cytoplasmic);
# the others use a generic 3-region split. Live mode fetches the real features.
_FIXTURE_REGIONS = {
    "SGCA": [
        ("Signal", 1, 23, "Signal peptide (fixture approximation)"),
        ("Topological domain", 24, 290, "Extracellular / cadherin-like (fixture approximation)"),
        ("Transmembrane", 291, 318, "Helical (fixture approximation)"),
        ("Topological domain", 319, 387, "Cytoplasmic (fixture approximation)"),
    ],
}


def _generic_regions(aa_len: int) -> list[tuple[str, int, int, str]]:
    a = max(1, aa_len // 12)
    b = int(aa_len * 0.85)
    return [
        ("Signal", 1, a, "Signal peptide (generic fixture)"),
        ("Topological domain", a + 1, b, "Extracellular (generic fixture)"),
        ("Transmembrane", b + 1, aa_len, "TM + cytoplasmic (generic fixture)"),
    ]


def _stars_to_status(stars: int, conflicting: bool = False) -> str:
    if conflicting:
        return "criteria provided, conflicting classifications"
    return {
        3: "reviewed by expert panel",
        2: "criteria provided, multiple submitters, no conflicts",
        1: "criteria provided, single submitter",
        0: "no assertion criteria provided",
    }[stars]


def _position_constraint(rng: np.random.RandomState, aa_len: int) -> np.ndarray:
    """Per-residue evolutionary constraint in [0,1] with a few high-constraint domains."""
    base = rng.uniform(0.05, 0.35, size=aa_len)
    n_hot = rng.randint(2, 4)
    for _ in range(n_hot):
        center = rng.randint(0, aa_len)
        width = rng.randint(aa_len // 12 + 3, aa_len // 5 + 5)
        lo, hi = max(0, center - width), min(aa_len, center + width)
        base[lo:hi] += rng.uniform(0.4, 0.6)
    return np.clip(base, 0.0, 1.0)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _gene_rows(cfg: Config, gene: str, rng: np.random.RandomState) -> list[dict]:
    meta = cfg.genes.meta[gene]
    chrom = str(meta["chrom"])
    aa_len = int(meta["aa_len"])
    transcript = meta["transcript"]
    base_g = 1_000_000  # synthetic genomic anchor (chrom differs per gene -> keys unique)
    constraint = _position_constraint(rng, aa_len)
    n = cfg.data_sources.fixture_n_per_gene
    rows: list[dict] = []

    def make_missense(pos: int, aaref: str, aaalt: str, off: int,
                      forced_sig: str | None = None, forced_prop: float | None = None) -> dict:
        cdna = (pos - 1) * 3 + 1 + off
        gpos = base_g + cdna
        ref_nt = NUCS[rng.randint(4)]
        alt_nt = NUCS[(NUCS.index(ref_nt) + 1 + rng.randint(3)) % 4]
        grant = grantham_distance(aaref, aaalt)
        if forced_prop is not None:
            prop = forced_prop
        else:
            prop = float(np.clip(0.55 * constraint[pos - 1] + 0.35 * (grant / 180.0)
                                 + rng.normal(0, 0.12), 0, 1))
        # features (independent noise so they are informative but not collinear)
        revel = float(np.clip(prop + rng.normal(0, 0.13), 0, 1))
        am = float(np.clip(prop + rng.normal(0, 0.14), 0, 1))
        pph = float(np.clip(prop + rng.normal(0, 0.16), 0, 1))
        sift = float(np.clip(1 - prop + rng.normal(0, 0.13), 0, 1))
        metasvm = float((prop - 0.5) * 4 + rng.normal(0, 0.8))
        cadd = float(np.clip(10 + 25 * prop + rng.normal(0, 3), 0, 50))
        gerp = float(np.clip(-2 + 8 * constraint[pos - 1] + rng.normal(0, 1.2), -12.3, 6.17))
        phylop = float(np.clip(-3 + 12 * constraint[pos - 1] + rng.normal(0, 1.0), -20, 10))
        phastcons = float(np.clip(constraint[pos - 1] + rng.normal(0, 0.12), 0, 1))
        siphy = float(np.clip(18 * constraint[pos - 1] + rng.normal(0, 2), 0, 37))
        # population frequency: anti-correlated with propensity; often absent if constrained
        absent = rng.uniform() < (0.25 + 0.55 * prop)
        if absent:
            af = popmax = 0.0
            nhom = 0
            in_gnomad = rng.uniform() < 0.2
        else:
            af = float(10 ** (-2.0 - 3.0 * prop + rng.normal(0, 0.4)))
            af = min(af, 0.25)
            popmax = float(min(af * rng.uniform(1.0, 3.0), 0.5))
            nhom = int(rng.poisson(af * 2000)) if af > 1e-3 else 0
            in_gnomad = True
        # label / VUS assignment. A forced_sig (e.g. canonical R77C) is respected as-is.
        if forced_sig is not None:
            sig = forced_sig
            klass = ("pathogenic" if sig in cfg.labels.pathogenic
                     else "benign" if sig in cfg.labels.benign else "vus")
            stars = 3 if klass != "vus" else 1
            conflicting = False
        else:
            mid = 0.35 <= prop <= 0.65
            if (mid and rng.uniform() < 0.30) or rng.uniform() < 0.06:
                klass = "vus"
            else:
                p_path = _sigmoid(7 * (prop - 0.5))
                klass = "pathogenic" if rng.uniform() < p_path else "benign"
            if klass == "pathogenic":
                sig = rng.choice(["Pathogenic", "Likely pathogenic"], p=[0.6, 0.4])
                stars = int(rng.choice([1, 2, 3], p=[0.45, 0.35, 0.20]))
                conflicting = False
            elif klass == "benign":
                sig = rng.choice(["Benign", "Likely benign"], p=[0.5, 0.5])
                stars = int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))
                conflicting = False
            else:
                conflicting = rng.uniform() < 0.4
                sig = ("Conflicting interpretations of pathogenicity" if conflicting
                       else "Uncertain significance")
                stars = int(rng.choice([0, 1, 2], p=[0.45, 0.45, 0.10]))
        return {
            "gene": gene, "chrom": chrom, "gpos": gpos, "ref_nt": ref_nt, "alt_nt": alt_nt,
            "transcript": transcript, "aaref": aaref, "aaalt": aaalt, "protein_position": pos,
            "hgvs_c": f"c.{cdna}{ref_nt}>{alt_nt}",
            "hgvs_p": f"p.{ONE_TO_THREE[aaref]}{pos}{ONE_TO_THREE[aaalt]}",
            "consequence": "missense_variant", "is_missense": True,
            "clinvar_sig": sig, "review_status": _stars_to_status(stars, conflicting),
            "clinsig_simple": {"pathogenic": 1, "benign": 0, "vus": -1}[klass],
            "vtype": "single nucleotide variant",
            "SIFT_score": sift, "Polyphen2_HVAR_score": pph, "MetaSVM_score": round(metasvm, 4),
            "CADD_phred": round(cadd, 2), "REVEL_score": round(revel, 4),
            "AlphaMissense_score": round(am, 4), "GERP++_RS": round(gerp, 3),
            "phyloP100way_vertebrate": round(phylop, 3),
            "phastCons100way_vertebrate": round(phastcons, 4),
            "SiPhy_29way_logOdds": round(siphy, 3),
            "gnomad_AF": af, "gnomad_AF_popmax": popmax, "gnomad_nhomalt": nhom,
            "in_gnomad": in_gnomad,
        }

    # ---- canonical SGCA test case: R77C (c.229C>T / p.Arg77Cys), pathogenic ----
    if gene == "SGCA":
        r77c = make_missense(77, "R", "C", off=0, forced_sig="Pathogenic", forced_prop=0.93)
        r77c["ref_nt"], r77c["alt_nt"] = "C", "T"
        r77c["gpos"] = base_g + 229
        r77c["hgvs_c"] = "c.229C>T"
        r77c["review_status"] = _stars_to_status(3)
        r77c["gnomad_AF"] = r77c["gnomad_AF_popmax"] = 0.0
        r77c["gnomad_nhomalt"] = 0
        r77c["in_gnomad"] = False
        rows.append(r77c)

    # ---- missense variants; positions sampled WITH repeats to exercise position-disjoint CV ----
    positions = rng.randint(2, aa_len, size=n)
    pos_counter: dict[int, int] = {}
    for pos in positions:
        pos = int(pos)
        aaref = AA1[rng.randint(20)]
        aaalt = AA1[(AA1.index(aaref) + 1 + rng.randint(19)) % 20]
        off = pos_counter.get(pos, 0) % 3
        pos_counter[pos] = pos_counter.get(pos, 0) + 1
        rows.append(make_missense(pos, aaref, aaalt, off))

    # ---- a handful of truncating variants (rule track; no dbNSFP/gnomAD features) ----
    for _ in range(rng.randint(4, 7)):
        pos = int(rng.randint(10, aa_len - 5))
        aaref = AA1[rng.randint(20)]
        cdna = (pos - 1) * 3 + 1
        kind = rng.choice(["stop_gained", "frameshift_variant", "splice_variant"])
        sig = rng.choice(["Pathogenic", "Likely pathogenic", "Uncertain significance"],
                         p=[0.7, 0.2, 0.1])
        stars = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
        if kind == "stop_gained":
            hgvs_c, hgvs_p = f"c.{cdna}C>T", f"p.{ONE_TO_THREE[aaref]}{pos}Ter"
            vtype = "single nucleotide variant"
        elif kind == "frameshift_variant":
            hgvs_c, hgvs_p = f"c.{cdna}delA", f"p.{ONE_TO_THREE[aaref]}{pos}fs"
            vtype = "Deletion"
        else:
            hgvs_c, hgvs_p = f"c.{cdna}+1G>A", ""
            vtype = "single nucleotide variant"
        rows.append({
            "gene": gene, "chrom": chrom, "gpos": base_g + cdna, "ref_nt": "C", "alt_nt": "T",
            "transcript": transcript, "aaref": aaref, "aaalt": "", "protein_position": pos,
            "hgvs_c": hgvs_c, "hgvs_p": hgvs_p, "consequence": kind, "is_missense": False,
            "clinvar_sig": sig, "review_status": _stars_to_status(stars),
            "clinsig_simple": 1 if "athogenic" in sig else -1, "vtype": vtype,
            "in_gnomad": False,
        })
    return rows


def _balance_classes(df: pd.DataFrame, labels_cfg, min_per_class: int = 10,
                     rng: np.random.RandomState | None = None) -> pd.DataFrame:
    """Ensure each gene's labelled missense has both classes (avoids degenerate folds)."""
    path_set, benign_set = set(labels_cfg.pathogenic), set(labels_cfg.benign)
    for gene in df["gene"].unique():
        m = (df["gene"] == gene) & df["is_missense"]
        sub = df[m]
        n_path = sub["clinvar_sig"].isin(path_set).sum()
        n_benign = sub["clinvar_sig"].isin(benign_set).sum()
        # relabel highest-REVEL VUS/benign -> pathogenic, lowest -> benign, as needed
        if n_path < min_per_class:
            cand = sub.sort_values("REVEL_score", ascending=False).index
            for idx in cand[: min_per_class - n_path]:
                df.loc[idx, ["clinvar_sig", "clinsig_simple", "review_status"]] = [
                    "Likely pathogenic", 1, "criteria provided, single submitter"]
        if n_benign < min_per_class:
            cand = sub.sort_values("REVEL_score", ascending=True).index
            for idx in cand[: min_per_class - n_benign]:
                df.loc[idx, ["clinvar_sig", "clinsig_simple", "review_status"]] = [
                    "Likely benign", 0, "criteria provided, single submitter"]
    return df


def generate_universe(cfg: Config) -> pd.DataFrame:
    rng = np.random.RandomState(cfg.data_sources.fixture_seed)
    rows: list[dict] = []
    for gene in cfg.genes.panel:
        rows.extend(_gene_rows(cfg, gene, rng))
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["chrom", "gpos", "ref_nt", "alt_nt"]).reset_index(drop=True)
    df = _balance_classes(df, cfg.labels, rng=rng)
    df.insert(0, "allele_id", np.arange(100000, 100000 + len(df)))
    df.insert(1, "variation_id", np.arange(500000, 500000 + len(df)))
    return df


# ----------------------------- raw-file writers -----------------------------

def _write_clinvar(df: pd.DataFrame, path: Path, rng: np.random.RandomState) -> int:
    cols = ["AlleleID", "Type", "Name", "GeneSymbol", "ClinicalSignificance",
            "ClinSigSimple", "ReviewStatus", "Assembly", "Chromosome", "Start", "Stop",
            "ReferenceAllele", "AlternateAllele", "PositionVCF", "ReferenceAlleleVCF",
            "AlternateAlleleVCF", "VariationID"]
    out = []
    for _, r in df.iterrows():
        name = f"{r['transcript']}({r['gene']}):{r['hgvs_c']}"
        if r["hgvs_p"]:
            name += f" ({r['hgvs_p']})"
        out.append([r["allele_id"], r["vtype"], name, r["gene"], r["clinvar_sig"],
                    r["clinsig_simple"], r["review_status"], "GRCh38", r["chrom"],
                    r["gpos"], r["gpos"], r["ref_nt"], r["alt_nt"], r["gpos"],
                    r["ref_nt"], r["alt_nt"], r["variation_id"]])
        # add a GRCh37 duplicate row for ~8% of variants (must be filtered out downstream)
        if rng.uniform() < 0.08:
            out.append([r["allele_id"], r["vtype"], name, r["gene"], r["clinvar_sig"],
                        r["clinsig_simple"], r["review_status"], "GRCh37", r["chrom"],
                        r["gpos"] - 500000, r["gpos"] - 500000, r["ref_nt"], r["alt_nt"],
                        r["gpos"] - 500000, r["ref_nt"], r["alt_nt"], r["variation_id"]])
    pd.DataFrame(out, columns=cols).to_csv(path, sep="\t", index=False)
    return len(out)


def _write_dbnsfp(df: pd.DataFrame, path: Path, rng: np.random.RandomState) -> int:
    mis = df[df["is_missense"]].copy()
    cols = ["#chr", "pos(1-based)", "ref", "alt", "aaref", "aaalt", "genename", "aapos",
            "SIFT_score", "Polyphen2_HVAR_score", "MetaSVM_score", "CADD_phred",
            "REVEL_score", "AlphaMissense_score", "GERP++_RS", "phyloP100way_vertebrate",
            "phastCons100way_vertebrate", "SiPhy_29way_logOdds"]
    out = []
    for _, r in mis.iterrows():
        def maybe_missing(v, p=0.06):
            return "." if rng.uniform() < p else v  # dbNSFP uses '.' for missing
        out.append([r["chrom"], r["gpos"], r["ref_nt"], r["alt_nt"], r["aaref"], r["aaalt"],
                    r["gene"], r["protein_position"], r["SIFT_score"], r["Polyphen2_HVAR_score"],
                    r["MetaSVM_score"], r["CADD_phred"], maybe_missing(r["REVEL_score"]),
                    maybe_missing(r["AlphaMissense_score"]), r["GERP++_RS"],
                    r["phyloP100way_vertebrate"], r["phastCons100way_vertebrate"],
                    r["SiPhy_29way_logOdds"]])
    pd.DataFrame(out, columns=cols).to_csv(path, sep="\t", index=False)
    return len(out)


def _write_gnomad(df: pd.DataFrame, path: Path) -> int:
    variants = []
    pops = ["afr", "amr", "asj", "eas", "fin", "nfe", "sas"]
    for _, r in df[df["in_gnomad"]].iterrows():
        af = float(r.get("gnomad_AF", 0.0) or 0.0)
        an = 150000
        populations = [{"id": p, "ac": int(af * an / len(pops)), "an": an // len(pops),
                        "af": float(min(af * (0.5 + (i % 3)), 0.5))}
                       for i, p in enumerate(pops)]
        variants.append({
            "variant_id": f"{r['chrom']}-{r['gpos']}-{r['ref_nt']}-{r['alt_nt']}",
            "chrom": str(r["chrom"]), "pos": int(r["gpos"]),
            "ref": r["ref_nt"], "alt": r["alt_nt"],
            "genome": {"ac": int(af * an), "an": an, "af": af,
                       "homozygote_count": int(r.get("gnomad_nhomalt", 0) or 0),
                       "populations": populations},
        })
    obj = {"data": {"meta": {"dataset": "SYNTHETIC", "build": "GRCh38"},
                    "variants": variants}}
    path.write_text(json.dumps(obj, indent=1))
    return len(variants)


def _write_uniprot(cfg: Config, df: pd.DataFrame, path: Path) -> int:
    out = {}
    for gene in cfg.genes.panel:
        meta = cfg.genes.meta[gene]
        aa_len = int(meta["aa_len"])
        regions = _FIXTURE_REGIONS.get(gene) or _generic_regions(aa_len)
        out[gene] = {
            "primaryAccession": meta.get("uniprot") or f"FIXTURE_{gene}",
            "sequence_length": aa_len,
            "_synthetic": True,
            "features": [
                {"type": t, "description": d,
                 "location": {"start": {"value": s}, "end": {"value": e}}}
                for (t, s, e, d) in regions
            ],
        }
    path.write_text(json.dumps(out, indent=1))
    return sum(len(v["features"]) for v in out.values())


def write_fixtures(cfg: Config) -> dict[str, Path]:
    """Generate all four synthetic raw files + manifest entries. Deterministic."""
    raw = cfg.path("raw")
    rng = np.random.RandomState(cfg.data_sources.fixture_seed + 7)
    uni = generate_universe(cfg)

    paths = raw_paths(cfg)
    n_cv = _write_clinvar(uni, paths["clinvar"], rng)
    n_db = _write_dbnsfp(uni, paths["dbnsfp"], rng)
    n_gn = _write_gnomad(uni, paths["gnomad"])
    n_up = _write_uniprot(cfg, uni, paths["uniprot"])

    note = "SYNTHETIC fixture data — mirrors real schema, NOT real biology."
    manifest.record_source(raw, "clinvar", mode="fixture", path=paths["clinvar"],
                           version="fixture", n_rows=n_cv, note=note)
    manifest.record_source(raw, "dbnsfp", mode="fixture", path=paths["dbnsfp"],
                           version="fixture", n_rows=n_db, note=note)
    manifest.record_source(raw, "gnomad", mode="fixture", path=paths["gnomad"],
                           version="fixture", n_rows=n_gn, note=note)
    manifest.record_source(raw, "uniprot", mode="fixture", path=paths["uniprot"],
                           version="fixture", n_rows=n_up, note=note)
    return paths
