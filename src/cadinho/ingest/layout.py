"""Canonical raw-file paths, shared by fixture writer, live downloaders, and parsers.

Keeping these in one place means the downstream pipeline reads the same filenames
regardless of whether the data came from fixtures, a live download, or user-provided files.
"""

from __future__ import annotations

from pathlib import Path

from cadinho.config import Config


def panel_tag(cfg: Config) -> str:
    return "_".join(cfg.genes.panel)


def raw_paths(cfg: Config) -> dict[str, Path]:
    raw = cfg.path("raw")
    t = panel_tag(cfg)
    return {
        "clinvar": raw / f"clinvar_variant_summary.{t}.tsv",
        "dbnsfp": raw / f"dbnsfp_slice.{t}.tsv",
        "gnomad": raw / f"gnomad_v4.{t}.json",
        "uniprot": raw / f"uniprot.{t}.json",
    }
