"""Map ClinVar ClinicalSignificance + ReviewStatus to our label scheme.

Rules (see config.labels):
  {Pathogenic, Likely pathogenic, Pathogenic/Likely pathogenic} -> 1 (pathogenic)
  {Benign, Likely benign, Benign/Likely benign}                 -> 0 (benign)
  {Uncertain significance, Conflicting...}                        -> VUS holdout (never trained)
  anything else                                                  -> dropped (not guessed)
Review status -> ACMG-style star count; rows below `min_review_stars` are dropped.
"""

from __future__ import annotations

from cadinho.config import LabelsCfg

# ClinVar ReviewStatus -> star rating (the standard mapping).
_REVIEW_STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, conflicting classifications": 1,
    "no assertion criteria provided": 0,
    "no assertion provided": 0,
    "no classification provided": 0,
    "no classifications from unflagged records": 0,
    "flagged submission": 0,
}


def normalize_significance(sig: str) -> str:
    return " ".join(str(sig).strip().split())


def _norm_set(values: list[str]) -> set[str]:
    return {normalize_significance(v).lower() for v in values}


def classify_significance(sig: str, labels: LabelsCfg) -> str | None:
    """Return 'pathogenic' | 'benign' | 'vus' | None (drop)."""
    s = normalize_significance(sig).lower()
    if s in _norm_set(labels.pathogenic):
        return "pathogenic"
    if s in _norm_set(labels.benign):
        return "benign"
    if s in _norm_set(labels.vus_holdout):
        return "vus"
    return None


def to_binary(klass: str | None) -> int | None:
    return {"pathogenic": 1, "benign": 0}.get(klass) if klass else None


def review_status_to_stars(status: str) -> int:
    key = normalize_significance(status).lower()
    return _REVIEW_STARS.get(key, 0)
