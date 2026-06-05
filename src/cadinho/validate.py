"""Section-5 ground-truth validation, to run after (live) ingest before trusting anything.

  * R77C present & correct: SGCA c.229C>T / p.Arg77Cys should be pathogenic with very low
    gnomAD AF. Fails loudly otherwise.
  * Per-gene label counts: Pathogenic / Benign / VUS / truncating per gene (expected small
    and imbalanced on real data — that is normal).
  * Build/source consistency: everything is GRCh38; report gnomAD/dbNSFP match rates.
"""

from __future__ import annotations

import pandas as pd

from cadinho.config import Config
from cadinho.normalize.reconcile import has_dbnsfp_features

R77C_MAX_AF = 1e-3  # R77C is a recurrent pathogenic allele; must be rare in gnomAD


def r77c_check(df: pd.DataFrame) -> dict:
    sgca = df[df["gene"] == "SGCA"]
    hit = sgca[sgca["hgvs_c"] == "c.229C>T"]
    if len(hit) == 0:  # fallback to residue identity
        hit = sgca[(sgca["protein_position"] == 77) & (sgca["aaref"] == "R")
                   & (sgca["aaalt"] == "C")]
    if len(hit) == 0:
        return {"present": False, "ok": False,
                "detail": "SGCA R77C (c.229C>T/p.Arg77Cys) NOT found in the data."}
    r = hit.iloc[0]
    af = float(r.get("gnomad_AF") or 0.0)
    is_path = r.get("label_class") == "pathogenic"
    af_ok = af <= R77C_MAX_AF
    return {"present": True, "ok": bool(is_path and af_ok),
            "label_class": r.get("label_class"), "clinvar_sig": r.get("clinvar_sig"),
            "gnomad_AF": af, "is_pathogenic": bool(is_path), "af_low_enough": bool(af_ok),
            "detail": f"label={r.get('label_class')} sig='{r.get('clinvar_sig')}' AF={af:.2e}"}


def per_gene_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene in sorted(df["gene"].unique()):
        g = df[df["gene"] == gene]
        mis = g[g["variant_class"] == "missense"]
        rows.append({
            "gene": gene,
            "pathogenic": int((mis["label_class"] == "pathogenic").sum()),
            "benign": int((mis["label_class"] == "benign").sum()),
            "vus": int((mis["label_class"] == "vus").sum()),
            "truncating": int((g["variant_class"] == "truncating").sum()),
            "total": int(len(g)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out.loc["TOTAL"] = out.sum(numeric_only=True)
        out.loc["TOTAL", "gene"] = "TOTAL"
    return out


def build_consistency(df: pd.DataFrame, build: str = "GRCh38") -> dict:
    mis = df[df["variant_class"] == "missense"]
    return {
        "declared_build": build,
        "n_variants": int(len(df)),
        "gnomad_match_rate": round(float(mis["in_gnomad"].mean()), 3) if len(mis) else 0.0,
        "dbnsfp_present": has_dbnsfp_features(df),
        "dbnsfp_match_rate": (round(float(mis["REVEL_score"].notna().mean()), 3)
                              if "REVEL_score" in mis.columns else None),
        "residue_resolved_rate": round(float(mis["protein_position"].notna().mean()), 3) if len(mis) else 0.0,
    }


def run_validation(df: pd.DataFrame, cfg: Config, strict: bool = False) -> dict:
    result = {
        "data_mode": cfg.data_sources.mode,
        "r77c": r77c_check(df),
        "per_gene": per_gene_counts(df),
        "consistency": build_consistency(df, cfg.project.genome_build),
    }
    if strict and not result["r77c"]["ok"]:
        raise ValueError(f"R77C ground-truth check FAILED: {result['r77c']['detail']}")
    return result


def format_validation(result: dict) -> str:
    r = result["r77c"]
    flag = "✅" if r["ok"] else ("⚠️" if r["present"] else "❌")
    L = [f"# Ingest validation (mode: {result['data_mode']})", "",
         f"## R77C ground-truth check {flag}", f"- {r['detail']}"]
    if not r["ok"]:
        L.append("- **CHECK FAILED** — investigate before trusting results.")
    L += ["", "## Per-gene counts", "", result["per_gene"].to_string(index=False), "",
          "## Build / source consistency", ""]
    for k, v in result["consistency"].items():
        L.append(f"- {k}: {v}")
    return "\n".join(L) + "\n"
