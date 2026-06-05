"""Shared test fixtures. Points config paths at a tmp dir and shrinks fixture size."""

from __future__ import annotations

import pytest

from cadinho.config import load_config, set_global_seeds


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    # absolute tmp paths (joining REPO_ROOT / abs_path yields abs_path)
    c.paths.raw = str(tmp_path / "raw")
    c.paths.interim = str(tmp_path / "interim")
    c.paths.processed = str(tmp_path / "processed")
    c.paths.models = str(tmp_path / "models")
    c.paths.reports = str(tmp_path / "reports")
    c.data_sources.fixture_n_per_gene = 40  # smaller -> faster tests
    set_global_seeds(c.project.seed)
    return c
