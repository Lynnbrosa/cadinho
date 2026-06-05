> ⚠️ **SYNTHETIC FIXTURE DATA — results are about made-up variants and say nothing about real biology. Not for any clinical or research conclusion.**

# Evaluation

- Data mode: `fixture`  |  CV: `gene_disjoint`  |  genes: SGCA, SGCB, SGCG, SGCD
- Metrics are mean [2.5–97.5 percentile] over repeated leakage-aware CV.

## Ablations (model's own signal vs. partly-circular features)

| Feature set | ROC-AUC | AUPRC | Brier |
|---|---|---|---|
| full | 0.830 [0.697–0.937] | 0.839 [0.668–0.955] | 0.180 [0.123–0.239] |
| no_predictor_derived | 0.773 [0.667–0.907] | 0.784 [0.600–0.914] | 0.216 [0.128–0.301] |
| no_allele_frequency | 0.845 [0.697–0.952] | 0.845 [0.671–0.965] | 0.162 [0.104–0.234] |
| no_predictor_and_no_af | 0.782 [0.672–0.912] | 0.786 [0.593–0.916] | 0.213 [0.120–0.290] |

*`no_predictor_derived` removes REVEL/AlphaMissense/SIFT/PolyPhen/CADD/MetaSVM (themselves ClinVar/HGMD-trained). `no_allele_frequency` removes gnomAD AF (partly circular: ACMG uses AF). The gap to `full` is the borrowed signal.*

## Baselines — model vs. reference predictors (same held-out variants)

| Predictor | ROC-AUC | AUPRC | n |
|---|---|---|---|
| model_oof | 0.837 | 0.832 | 288 |
| REVEL_score | 0.832 | 0.843 | 267 |
| AlphaMissense_score | 0.810 | 0.817 | 269 |

*REVEL/AlphaMissense are strong, partly-circular baselines (ClinVar/HGMD-trained). On a single gene family the model may not beat them — that is reported, not hidden.*

## Calibration (gene-disjoint OOF)

- Brier score: **0.183** (lower is better).
- See `calibration.png`. A well-calibrated 0.8 should mean ~80% observed.

| mean predicted | observed fraction |
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
