"""Per-variant interpretable report.

For a missense variant: calibrated probability + honest uncertainty band, the *why*
(UniProt region, substitution nature, conservation, gnomAD frequency, REVEL/AlphaMissense),
and SHAP attribution. For a truncating variant: the flagged rule call. VUS get an explicit
"the model is the only signal here" note. Everything is stamped SYNTHETIC in fixture mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cadinho import SYNTHETIC_BANNER
from cadinho.config import Config
from cadinho.features.build import get_xy
from cadinho.model.train import decide_label
from cadinho.normalize.hgvs import parse_variant
from cadinho.report import domain_map
from cadinho.report.explain import Explainer


def resolve(df: pd.DataFrame, cfg: Config, hgvs_text: str, gene: str | None = None):
    """Resolve an HGVS string to a variant row in the table (focus gene by default)."""
    pv = parse_variant(hgvs_text, cfg.variant_class.splice_canonical_offset)
    gene = gene or pv.gene or cfg.genes.focus
    cand = df[df["gene"] == gene]
    if pv.hgvs_c:
        hit = cand[cand["hgvs_c"] == pv.hgvs_c]
        if len(hit):
            return hit.iloc[0], pv, None
    if pv.protein_position and pv.aaref and pv.aaalt:
        hit = cand[(cand["protein_position"] == pv.protein_position)
                   & (cand["aaref"] == pv.aaref) & (cand["aaalt"] == pv.aaalt)]
        if len(hit):
            return hit.iloc[0], pv, None
    reason = (f"Could not resolve '{hgvs_text}' for {gene} in the current dataset. "
              "Offline, only variants present in the dbNSFP/ClinVar slice can be scored "
              "(a transcript sequence would be needed to map an arbitrary c. change). "
              "In live/provided mode dbNSFP covers every possible missense in the gene.")
    return None, pv, reason


def _missense_matrix(df: pd.DataFrame, cfg: Config):
    return get_xy(df, cfg, labelled_only=False)  # (X, None, meta)


def _band_text(prob: float, decision: dict) -> str:
    label = decide_label(prob, decision)
    if label == "Uncertain":
        return (f"**Uncertain** (calibrated p(pathogenic) = {prob:.2f}; between the "
                f"benign ≤ {decision['benign_threshold']} and pathogenic ≥ "
                f"{decision['pathogenic_threshold']} thresholds — the model abstains).")
    return f"**{label}** (calibrated p(pathogenic) = {prob:.2f})."


def report_for_row(cfg: Config, bundle, df: pd.DataFrame, uni: dict, row: pd.Series,
                   explainer: Explainer | None = None) -> dict:
    synthetic = cfg.data_sources.mode == "fixture"
    head = []
    if synthetic:
        head.append(f"> ⚠️ **{SYNTHETIC_BANNER}**\n")
    title = f"{row['gene']} {row.get('hgvs_c') or ''} {row.get('hgvs_p') or ''}".strip()
    head.append(f"## Variant report — {title}\n")
    head.append(f"- ClinVar: **{row.get('clinvar_sig') or 'n/a'}** "
                f"({int(row.get('review_stars') or 0)}★)  |  consequence: "
                f"`{row['consequence']}`  |  track: `{row['variant_class']}`")

    out = {"gene": row["gene"], "hgvs_c": row.get("hgvs_c"), "hgvs_p": row.get("hgvs_p"),
           "track": row["variant_class"], "synthetic": synthetic}

    if row["variant_class"] == "truncating":
        from cadinho.model.rules import _BASIS
        basis = _BASIS.get(row["consequence"], "Loss of function")
        cls = "Likely pathogenic" if cfg.variant_class.truncating_rule_label == "pathogenic" else "?"
        out.update(predicted_class=cls, probability=None, basis=basis)
        head += ["", f"### Prediction (rule track): **{cls}**",
                 f"- Basis: {basis} in a recessive loss-of-function gene.",
                 "- This is a **heuristic rule**, not a calibrated model probability. "
                 "Confirm zygosity and segregation clinically."]
        out["markdown"] = "\n".join(head) + "\n"
        return out

    if row["variant_class"] != "missense":
        head += ["", f"### Not scored: consequence `{row['consequence']}` is outside both "
                 "tracks (synonymous / in-frame / deep-intronic). No prediction made."]
        out.update(predicted_class="Not scored", probability=None)
        out["markdown"] = "\n".join(head) + "\n"
        return out

    # ---- missense ML track ----
    X, _, meta = _missense_matrix(df, cfg)
    locus = meta.index[(meta["gene"] == row["gene"]) & (meta["hgvs_c"] == row["hgvs_c"])]
    Xr = X.loc[[locus[0]]]
    raw = float(bundle.raw_proba(Xr)[0])
    prob = float(bundle.calibrated_proba(Xr)[0])
    cls = decide_label(prob, bundle.decision)
    out.update(predicted_class=cls, probability=prob, raw_probability=raw)

    dm = domain_map.describe(uni, row["gene"], int(row["protein_position"])
                             if pd.notna(row["protein_position"]) else None)

    def g(col):
        v = row.get(col)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v

    head += ["", f"### Prediction (ML track): {_band_text(prob, bundle.decision)}", ""]
    if row.get("is_vus"):
        head.append("> **ClinVar calls this a VUS** — the model is the only signal here. "
                    "Treat the score as a hypothesis, not a classification.\n")
    head += ["### Why", "",
             f"- **Location**: {dm['sentence']}",
             f"- **Substitution**: {row.get('aaref')}→{row.get('aaalt')} "
             f"(Grantham {g('Grantham') if g('Grantham') is not None else _sub(row,'Grantham')}, "
             f"BLOSUM62 {_sub(row,'BLOSUM62')}).",
             f"- **Conservation**: GERP++ {g('GERPpp_RS')}, phyloP {g('phyloP100way_vertebrate')}, "
             f"phastCons {g('phastCons100way_vertebrate')}.",
             f"- **Population (gnomAD)**: AF {g('gnomad_AF')}, popmax {g('gnomad_AF_popmax')}, "
             f"homozygotes {g('gnomad_nhomalt')}"
             + ("  *(rare/absent — consistent with pathogenic)*" if (g('gnomad_AF') or 0) == 0 else "") + ".",
             f"- **Reference predictors**: REVEL {g('REVEL_score')}, "
             f"AlphaMissense {g('AlphaMissense_score')}."]

    if explainer is not None:
        attr = explainer.attribute(Xr)
        out["attribution"] = attr
        head += ["", "### Feature attribution (SHAP, toward pathogenic)", "",
                 "| feature | value | contribution |", "|---|---|---|"]
        for a in attr:
            val = a["value"]
            vtxt = f"{val:.3f}" if isinstance(val, (int, float)) and val is not None and not (isinstance(val, float) and np.isnan(val)) else "NA"
            head.append(f"| {a['feature']} | {vtxt} | {a['contribution']:+.3f} |")

    head += ["", "### Honest note",
             "- Calibrated probability from gene-disjoint out-of-fold calibration; see "
             "`evaluation.md` for how reliable that calibration is.",
             "- REVEL/AlphaMissense are inputs **and** baselines; they are partly circular "
             "w.r.t. ClinVar. The model's own signal is the `no_predictor_derived` ablation."]
    if synthetic:
        head.append("- **Synthetic data: this score is meaningless for the real variant.**")
    out["markdown"] = "\n".join(head) + "\n"
    return out


def _sub(row, name):
    """Substitution feature may not be on the raw row (computed in feature build); recompute."""
    from cadinho.features.aa import AA_SET, substitution_features
    ref, alt = row.get("aaref"), row.get("aaalt")
    if isinstance(ref, str) and isinstance(alt, str) and ref in AA_SET and alt in AA_SET:
        return substitution_features(ref, alt).get(name)
    return None


def report_variant(cfg: Config, bundle, df: pd.DataFrame, uni: dict, hgvs_text: str,
                   gene: str | None = None) -> dict:
    row, pv, reason = resolve(df, cfg, hgvs_text, gene)
    if row is None:
        return {"resolved": False, "reason": reason,
                "markdown": f"## Variant report — {hgvs_text}\n\n**Unresolved.** {reason}\n"}
    explainer = Explainer(bundle)
    rep = report_for_row(cfg, bundle, df, uni, row, explainer)
    rep["resolved"] = True
    return rep


def run_vus_report(cfg: Config, bundle, df: pd.DataFrame, uni: dict) -> pd.DataFrame:
    """Score the focus-gene VUS missense set (never trained on) and write a report."""
    synthetic = cfg.data_sources.mode == "fixture"
    focus = cfg.genes.focus
    X, _, meta = _missense_matrix(df, cfg)
    is_vus = meta["is_vus"].fillna(False).to_numpy()
    is_focus = (meta["gene"] == focus).to_numpy()
    sel = np.where(is_vus & is_focus)[0]

    rows = []
    explainer = Explainer(bundle)
    for i in sel:
        Xr = X.iloc[[i]]
        prob = float(bundle.calibrated_proba(Xr)[0])
        m = meta.iloc[i]
        top = explainer.attribute(Xr, top=3)
        rows.append({"gene": m["gene"], "hgvs_c": m["hgvs_c"], "hgvs_p": m["hgvs_p"],
                     "protein_position": m["protein_position"], "region": m.get("region_name"),
                     "calibrated_p_pathogenic": round(prob, 3),
                     "model_call": decide_label(prob, bundle.decision),
                     "REVEL_score": m.get("REVEL_score") if "REVEL_score" in meta else None,
                     "top_features": ", ".join(a["feature"] for a in top)})
    vus = pd.DataFrame(rows).sort_values("calibrated_p_pathogenic", ascending=False)
    vus.to_csv(cfg.path("processed", "vus_predictions.csv"), index=False)

    L = []
    if synthetic:
        L += [f"> ⚠️ **{SYNTHETIC_BANNER}**", ""]
    L += [f"# {focus} VUS predictions", "",
          f"The {len(vus)} {focus} missense variants ClinVar calls Uncertain/Conflicting — "
          "**never used in training**. The model adds a calibrated hypothesis where ClinVar "
          "currently has none. A score in the middle band is reported as Uncertain by design.",
          ""]
    if len(vus):
        L += ["| variant | residue | region | calibrated p(path) | model call |",
              "|---|---|---|---|---|"]
        for _, r in vus.iterrows():
            L.append(f"| {r['hgvs_c']} {r['hgvs_p'] or ''} | {r['protein_position']} | "
                     f"{r['region']} | {r['calibrated_p_pathogenic']:.2f} | {r['model_call']} |")
    else:
        L.append("_No focus-gene VUS missense variants in the current dataset._")
    L += ["", "**Caveat:** VUS predictions are hypotheses, not classifications; calibration "
          "and the predictor-circularity caveats in `evaluation.md` apply."]
    if synthetic:
        L.append("\n**Synthetic data — these VUS scores are not real.**")
    cfg.path("reports", "vus_predictions.md").write_text("\n".join(L) + "\n")
    return vus
