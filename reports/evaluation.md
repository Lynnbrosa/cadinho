> ⚠️ **SYNTHETIC FIXTURE DATA — results are about made-up variants and say nothing about real biology. Not for any clinical or research conclusion.**

# Evaluation

- Data mode: `fixture`  |  CV: `gene_disjoint`  |  genes: SGCA, SGCB, SGCG, SGCD
- Metrics are mean [2.5–97.5 percentile] over repeated leakage-aware CV.

## Ablations (model's own signal vs. partly-circular features)

| Feature set | ROC-AUC | AUPRC | Brier |
|---|---|---|---|
| full | 0.831 [0.698–0.938] | 0.839 [0.663–0.957] | 0.180 [0.122–0.241] |
| no_predictor_derived | 0.774 [0.664–0.909] | 0.784 [0.596–0.918] | 0.216 [0.126–0.297] |
| no_allele_frequency | 0.846 [0.698–0.954] | 0.846 [0.667–0.967] | 0.162 [0.102–0.235] |
| no_predictor_and_no_af | 0.783 [0.670–0.916] | 0.788 [0.592–0.920] | 0.211 [0.117–0.289] |

*`no_predictor_derived` removes REVEL/AlphaMissense/SIFT/PolyPhen/CADD/MetaSVM (themselves ClinVar/HGMD-trained). `no_allele_frequency` removes gnomAD AF (partly circular: ACMG uses AF). The gap to `full` is the borrowed signal.*

## Baselines — model vs. reference predictors (same held-out variants)

| Predictor | ROC-AUC | AUPRC | n |
|---|---|---|---|
| model_oof | 0.837 | 0.832 | 288 |
| REVEL_score | 0.832 | 0.843 | 267 |
| AlphaMissense_score | 0.810 | 0.817 | 269 |

*REVEL/AlphaMissense are strong, partly-circular baselines (ClinVar/HGMD-trained). On a single gene family the model may not beat them — that is reported, not hidden.*

## Calibration (gene-disjoint OOF)

- Brier: **raw 0.183** (genuinely out-of-fold) → **calibrated 0.147** (after the shipped isotonic map; in-sample, so modestly optimistic).
- See `calibration.png`. A well-calibrated 0.8 should mean ~80% observed. The table below is the **raw** OOF model (the conservative, no-optimism view).

| mean predicted (raw) | observed fraction |
|---|---|
| 0.000 | 0.207 |
| 0.001 | 0.172 |
| 0.010 | 0.172 |
| 0.083 | 0.179 |
| 0.282 | 0.414 |
| 0.711 | 0.483 |
| 0.954 | 0.786 |
| 0.993 | 0.828 |
| 0.999 | 1.000 |
| 1.000 | 0.862 |

## Honest caveats
- Predictor-derived features and AF are partly circular w.r.t. ClinVar labels; the ablations above quantify this.
- Sarcoglycan-only training is small; CIs are wide and gene-disjoint folds are few.
- Decisions use a deliberately high bar (path ≥ 0.9, benign ≤ 0.1); the middle band is reported as Uncertain.

- **These numbers are from synthetic fixtures and mean nothing for real variants.** Re-run with `data_sources.mode: live` or `provided` for real data.
