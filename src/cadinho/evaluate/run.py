"""Orchestrate honest evaluation -> reports/evaluation.md + calibration plot + JSON.

Reports leakage-aware CV with CIs, the required with/without-feature ablations, a REVEL /
AlphaMissense head-to-head, and a calibration curve + Brier score. In fixture mode every
artifact is stamped SYNTHETIC.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cadinho import SYNTHETIC_BANNER  # noqa: E402
from cadinho.config import Config  # noqa: E402
from cadinho.evaluate.baselines import compare_to_baselines  # noqa: E402
from cadinho.evaluate.metrics import calibration_bins, cv_scores, pooled_oof  # noqa: E402

_ABLATION_DROPS = {
    "full": (),
    "no_predictor_derived": ("predictor_derived",),
    "no_allele_frequency": ("allele_frequency",),
    "no_predictor_and_no_af": ("predictor_derived", "allele_frequency"),
}


def _fmt(d: dict) -> str:
    return f"{d['mean']:.3f} [{d['lo']:.3f}–{d['hi']:.3f}]"


def run_evaluation(cfg: Config, df: pd.DataFrame, repeats: int | None = None) -> dict:
    synthetic = cfg.data_sources.mode == "fixture"

    # 1) ablations (leakage-aware repeated CV)
    ablations = {}
    for name in cfg.evaluation.ablations:
        ablations[name] = cv_scores(df, cfg, drop_groups=_ABLATION_DROPS[name],
                                    repeats=repeats)["summary"]

    # 2) baselines (model OOF vs REVEL/AlphaMissense on same variants)
    baselines = compare_to_baselines(df, cfg)

    # 3) calibration on pooled OOF
    yv, oof, _ = pooled_oof(df, cfg)
    frac_pos, mean_pred, brier = calibration_bins(yv, oof)
    _plot_calibration(cfg, frac_pos, mean_pred, brier, synthetic)

    results = {"synthetic": synthetic, "data_mode": cfg.data_sources.mode,
               "cv_scheme": cfg.evaluation.cv.scheme, "genes": cfg.genes.panel,
               "ablations": ablations, "baselines": baselines,
               "calibration": {"brier": float(brier),
                               "bins": [{"mean_pred": float(p), "frac_pos": float(f)}
                                        for p, f in zip(mean_pred, frac_pos)]}}
    cfg.path("processed", "eval_metrics.json").write_text(json.dumps(results, indent=2))
    _write_markdown(cfg, results, synthetic)
    return results


def _plot_calibration(cfg, frac_pos, mean_pred, brier, synthetic):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label=f"model (Brier={brier:.3f})")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction pathogenic")
    title = "Calibration (gene-disjoint OOF)"
    if synthetic:
        title += "\nSYNTHETIC DATA — not real"
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(cfg.path("reports", "calibration.png"), dpi=110)
    plt.close(fig)


def _write_markdown(cfg, r, synthetic):
    L = []
    if synthetic:
        L += [f"> ⚠️ **{SYNTHETIC_BANNER}**", ""]
    L += ["# Evaluation", "",
          f"- Data mode: `{r['data_mode']}`  |  CV: `{r['cv_scheme']}`  |  "
          f"genes: {', '.join(r['genes'])}",
          "- Metrics are mean [2.5–97.5 percentile] over repeated leakage-aware CV.", ""]

    L += ["## Ablations (model's own signal vs. partly-circular features)", "",
          "| Feature set | ROC-AUC | AUPRC | Brier |", "|---|---|---|---|"]
    for name, s in r["ablations"].items():
        L.append(f"| {name} | {_fmt(s['auc'])} | {_fmt(s['ap'])} | {_fmt(s['brier'])} |")
    L += ["",
          "*`no_predictor_derived` removes REVEL/AlphaMissense/SIFT/PolyPhen/CADD/MetaSVM "
          "(themselves ClinVar/HGMD-trained). `no_allele_frequency` removes gnomAD AF "
          "(partly circular: ACMG uses AF). The gap to `full` is the borrowed signal.*", ""]

    L += ["## Baselines — model vs. reference predictors (same held-out variants)", "",
          "| Predictor | ROC-AUC | AUPRC | n |", "|---|---|---|---|"]
    for name, b in r["baselines"].items():
        L.append(f"| {name} | {b['auc']:.3f} | {b['ap']:.3f} | {b['n']} |")
    L += ["",
          "*REVEL/AlphaMissense are strong, partly-circular baselines (ClinVar/HGMD-trained). "
          "On a single gene family the model may not beat them — that is reported, not hidden.*",
          ""]

    L += ["## Calibration (gene-disjoint OOF)", "",
          f"- Brier score: **{r['calibration']['brier']:.3f}** (lower is better).",
          "- See `calibration.png`. A well-calibrated 0.8 should mean ~80% observed.", "",
          "| mean predicted | observed fraction |", "|---|---|"]
    for b in r["calibration"]["bins"]:
        L.append(f"| {b['mean_pred']:.3f} | {b['frac_pos']:.3f} |")
    L += ["",
          "## Honest caveats",
          "- Predictor-derived features and AF are partly circular w.r.t. ClinVar labels; "
          "the ablations above quantify this.",
          "- Sarcoglycan-only training is small; CIs are wide and gene-disjoint folds are few.",
          "- Decisions use a deliberately high bar (path ≥ "
          f"{cfg.model.decision.pathogenic_threshold}, benign ≤ "
          f"{cfg.model.decision.benign_threshold}); the middle band is reported as Uncertain."]
    if synthetic:
        L += ["", "- **These numbers are from synthetic fixtures and mean nothing for real "
              "variants.** Re-run with `data_sources.mode: live` or `provided` for real data."]
    cfg.path("reports", "evaluation.md").write_text("\n".join(L) + "\n")
