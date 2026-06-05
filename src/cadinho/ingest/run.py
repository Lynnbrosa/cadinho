"""Ingest orchestration: ensure raw files exist (per mode), then parse them.

mode=fixture  -> generate synthetic raw files
mode=live     -> download from real hosts
mode=provided -> expect user-supplied raw files at the documented paths
"""

from __future__ import annotations

from pathlib import Path

from cadinho.config import Config
from cadinho.ingest import clinvar, dbnsfp, gnomad, manifest, synthetic, uniprot
from cadinho.ingest.layout import raw_paths


def run_ingest(cfg: Config, force: bool = False) -> dict[str, Path]:
    mode = cfg.data_sources.mode
    paths = raw_paths(cfg)

    if mode == "fixture":
        if force or not all(p.exists() for p in paths.values()):
            return synthetic.write_fixtures(cfg)
        return paths

    if mode == "provided":
        missing = {k: str(p) for k, p in paths.items() if not p.exists()}
        if missing:
            raise FileNotFoundError(
                "mode=provided but raw files are missing. Place real, GRCh38, "
                f"gene-sliced files at: {missing}. See README 'Switching to real data'."
            )
        raw = cfg.path("raw")
        for name, p in paths.items():
            manifest.record_source(raw, name, mode="provided", path=p,
                                   version="user-provided", note="provided by user")
        return paths

    if mode == "live":
        clinvar.download_live(cfg)
        gnomad.download_live(cfg)
        uniprot.download_live(cfg)
        dbnsfp.download_live(cfg)  # raises with guidance unless wired to a local release
        return paths

    raise ValueError(f"unknown data_sources.mode: {mode}")


def load_sources(cfg: Config, force: bool = False) -> dict:
    """Ensure raw files exist, then parse each source into its standardized form."""
    paths = run_ingest(cfg, force=force)
    return {
        "clinvar": clinvar.parse(paths["clinvar"], cfg),
        "dbnsfp": dbnsfp.parse(paths["dbnsfp"], cfg),
        "gnomad": gnomad.parse(paths["gnomad"], cfg),
        "uniprot": uniprot.parse(paths["uniprot"]),
        "paths": paths,
    }
