"""ClinVar significance/review-status mapping."""

from __future__ import annotations

import pytest

from cadinho.config import load_config
from cadinho.normalize.labels import (
    classify_significance,
    review_status_to_stars,
    to_binary,
)

LABELS = load_config().labels


@pytest.mark.parametrize("sig,expected", [
    ("Pathogenic", "pathogenic"),
    ("Likely pathogenic", "pathogenic"),
    ("Pathogenic/Likely pathogenic", "pathogenic"),
    ("Benign", "benign"),
    ("Likely benign", "benign"),
    ("Uncertain significance", "vus"),
    ("Conflicting interpretations of pathogenicity", "vus"),
    ("drug response", None),      # not guessed -> dropped
    ("association", None),
])
def test_classify_significance(sig, expected):
    assert classify_significance(sig, LABELS) == expected


def test_classify_significance_is_case_and_space_insensitive():
    assert classify_significance("  pathogenic ", LABELS) == "pathogenic"


@pytest.mark.parametrize("klass,expected", [
    ("pathogenic", 1), ("benign", 0), ("vus", None), (None, None)])
def test_to_binary(klass, expected):
    assert to_binary(klass) == expected


@pytest.mark.parametrize("status,stars", [
    ("practice guideline", 4),
    ("reviewed by expert panel", 3),
    ("criteria provided, multiple submitters, no conflicts", 2),
    ("criteria provided, single submitter", 1),
    ("no assertion criteria provided", 0),
    ("something unknown", 0),
])
def test_review_status_to_stars(status, stars):
    assert review_status_to_stars(status) == stars
