# cadinho — SGCA variant pathogenicity classifier

> ## ⚠️ NOT A CLINICAL OR DIAGNOSTIC TOOL
> This is a **personal research / learning project**. Nothing it produces is medical
> advice, a diagnosis, or a substitute for a qualified clinical geneticist. Do not use
> any output here to make health decisions. Variant interpretation for patient care must
> follow ACMG/AMP guidelines under professional oversight.

> ## ⚠️ THIS REPOSITORY CURRENTLY RUNS ON SYNTHETIC FIXTURE DATA
> The execution environment this was built in **blocks the real data hosts** (NCBI/ClinVar,
> gnomAD, UniProt, EBI). To make the pipeline runnable and testable today, every ingest
> step can run in **`fixture` mode**, which generates *clearly-labelled synthetic data that
> mirrors the real schemas*. **All metrics, reports, and predictions produced in fixture
> mode are about made-up data and say NOTHING about real biology.** Every generated artifact
> is stamped `SYNTHETIC`. The real download code is written and runs unchanged the moment the
> hosts are reachable or real files are placed in `data/raw/` (see *Switching to real data*).

---

## What this is

A reproducible, local-first pipeline that takes a missense variant in **SGCA**
(α-sarcoglycan; the gene behind **LGMD R3 / LGMD2D**, an autosomal-recessive
sarcoglycanopathy) and predicts **pathogenic vs benign** with an *interpretable,
honestly-calibrated* explanation: where the residue sits in the protein, what kind of
substitution it is, how conserved the position is, its population frequency, how
established predictors score it, and **how confident the model actually is**.

### Design choice: train broad, interpret narrow
SGCA alone has too few high-confidence labelled variants to train a stable model. So the
model is trained on the **four sarcoglycan genes (SGCA, SGCB, SGCG, SGCD)** — one protein
complex, shared recessive loss-of-function mechanism — and then **applied and explained
specifically on SGCA**. The model is a *general missense* classifier; SGCA is the focus.

### Two tracks
- **Truncating variants** (nonsense, frameshift, canonical ±1/2 splice) → **rule-based track**.
  For a recessive LoF gene these are generally pathogenic; we classify by rule, **flag them
  clearly, and never feed them to the ML model.**
- **Missense variants** → **ML track** (gradient-boosting classifier).

---

## Honesty constraints (the point of the project)

This project is built to **not lie with inflated confidence**. Concretely:

1. **Leakage-aware evaluation.** Variants at the *same protein position* never span
   train/test, and the primary scheme is **gene-disjoint CV** (train on some sarcoglycan
   genes, test on a held-out one). See `src/cadinho/evaluate/splits.py` + `tests/test_splits.py`.
2. **Circularity is reported, not hidden.** Some features (REVEL, AlphaMissense, MetaSVM, …)
   were themselves trained on ClinVar/HGMD. We (a) report metrics **with and without** those
   predictor-derived features to show the model's *own* signal, and (b) use REVEL and
   AlphaMissense as **reference baselines** to beat, not just as inputs.
3. **Allele-frequency caveat.** Benign variants are often common and pathogenic ones rare,
   and ACMG uses AF — so AF is partly circular. We report **with and without AF**.
4. **Calibration, not just ranking.** We report a **calibration curve + Brier score** so a
   "0.8" means ~80%, alongside ROC-AUC / AUPRC with **confidence intervals from repeated CV**.
5. **VUS stay VUS.** Variants of Uncertain Significance / Conflicting are **never trained on**;
   they are held out and predicted *last*, and the report surfaces honest uncertainty rather
   than coercing them into a confident class. A score in the middle band is reported as
   **Uncertain**, by design.
6. **Synthetic ≠ real.** In fixture mode every artifact is stamped `SYNTHETIC`.

When the data can't support a claim, the output says so in plain language.

---

## Non-goals

No disease-progression or patient-level modelling, no imaging, no "cure/target discovery",
no web deployment / API server. Local-first, reproducible, done.

---

## Data sources (all public; pinned at ingest into `data/raw/manifest.json`)

| Source | Role | Build | Access |
|---|---|---|---|
| **ClinVar** `variant_summary.txt.gz` | labels (Path/Benign + VUS holdout) | GRCh38 | NCBI FTP |
| **gnomAD v4** | population allele frequency | GRCh38 | GraphQL API |
| **dbNSFP v4.x** (academic) | feature backbone: conservation, AA properties, predictor scores | GRCh38 | gene/chr slice |
| **UniProt Q16586** | protein domains/topology for positional context | — | REST |
| *(optional)* LOVD SGCA | extra reported variants | GRCh38 only | — |

**Genome build: GRCh38 only.** Every source is filtered to its GRCh38 records; no liftover
(legacy hg19-only sources are out of scope for now).

---

## Reproduce end-to-end (one command)

```bash
uv sync                 # create env from pinned uv.lock
make all                # ingest(fixture) -> normalize -> features -> train -> evaluate -> report
```

Or step by step:

```bash
make ingest     # generate/download raw sources + manifest
make normalize  # parse HGVS, map labels, dedup/reconcile, split classes
make features   # build missense feature matrix
make train      # deterministic gradient-boosting + calibration
make evaluate   # leakage-aware CV, ablations, REVEL/AlphaMissense baselines, calibration
make report     # SHAP + UniProt domain mapping; per-variant reports incl. SGCA VUS set
make test       # parsing/normalization/leakage tests
```

Single-variant report (R77C, the most common SGCA pathogenic variant — the end-to-end test case):

```bash
uv run cadinho report-variant --hgvs "c.229C>T"      # or --hgvs "p.Arg77Cys"
```

---

## Switching to real data

Two ways, no code change to the science core:

1. **Allow the hosts** (e.g. open this session's network policy to `ftp.ncbi.nlm.nih.gov`,
   `gnomad.broadinstitute.org`, `rest.uniprot.org`, `www.ebi.ac.uk`), set
   `data_sources.mode: live` in `config/config.yaml`, then `make ingest`. Versions/dates/sha256
   are pinned into `data/raw/manifest.json`.
2. **Drop real files** (gene-sliced) into `data/raw/` matching the documented schemas and set
   `data_sources.mode: provided`.

---

## Layout

```
config/config.yaml      single source of truth (paths, gene panel, build, thresholds, seeds)
src/cadinho/
  ingest/   clinvar gnomad dbnsfp uniprot  (real download + synthetic fixtures)
  normalize/ hgvs  reconcile               (label mapping, dedup, cross-source join)
  variants/ classify_class                 (truncating->rule, missense->ML)
  features/ build
  model/    train  calibrate  rules        (rules = truncating track)
  evaluate/ splits metrics ablations baselines
  report/   explain(SHAP) domain_map variant_report
  literature/ pubmed                       (OPTIONAL, isolated)
tests/  notebooks/  reports/  cli.py
```

## Caveats, in one place
- Fixture results are synthetic and meaningless for real variants (see top).
- Predictor-derived features and AF are partly circular w.r.t. ClinVar; we report ablations.
- Sarcoglycan-only training → the model may **not** beat REVEL/AlphaMissense on this gene
  family. If it doesn't, we say so. The goal is understanding, not a leaderboard win.
- HGVS handling targets the SNV/missense + truncating cases we route on; it is not a full
  HGVS engine.

## License
MIT (code). Data sources retain their own licenses; usage here is non-commercial/research.
