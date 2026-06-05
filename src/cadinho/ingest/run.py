"""Ingest orchestration: ensure raw files exist (per mode), then parse them.

mode=fixture  -> generate synthetic raw files
mode=live     -> download from real hosts
mode=provided -> expect user-supplied raw files at the documented paths
"""

from __future__ import annotations

import logging
from pathlib import Path

from cadinho.config import Config
from cadinho.ingest import clinvar, dbnsfp, gnomad, manifest, synthetic, uniprot
from cadinho.ingest.layout import raw_paths

log = logging.getLogger("cadinho.ingest")


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
        # ClinVar (labels), gnomAD (AF), UniProt (domains) are REQUIRED — fail loudly.
        clinvar.download_live(cfg)
        gnomad.download_live(cfg)
        uniprot.download_live(cfg)
        # dbNSFP is OPTIONAL enrichment: never let it block a first real run.
        if cfg.data_sources.dbnsfp.get("enabled", False):
            try:
                got = dbnsfp.fetch_live(cfg)
                log.info("dbNSFP enrichment: %s", got or "skipped (not viable)")
            except Exception as e:  # never fatal
                log.warning("dbNSFP enrichment skipped: %s", e)
        else:
            log.info("dbNSFP disabled (data_sources.dbnsfp.enabled=false); running without it.")
        return paths

    raise ValueError(f"unknown data_sources.mode: {mode}")


def load_sources(cfg: Config, force: bool = False) -> dict:
    """Ensure raw files exist, then parse each source into its standardized form.

    dbNSFP is optional: if its raw file is absent the feature layer is simply None and the
    pipeline runs on ClinVar labels + gnomAD AF + amino-acid biochemistry + UniProt context.
    """
    paths = run_ingest(cfg, force=force)
    db_path = paths["dbnsfp"]
    return {
        "clinvar": clinvar.parse(paths["clinvar"], cfg),
        "dbnsfp": dbnsfp.parse(db_path, cfg) if db_path.exists() else None,
        "gnomad": gnomad.parse(paths["gnomad"], cfg) if paths["gnomad"].exists() else None,
        "uniprot": uniprot.parse(paths["uniprot"]),
        "paths": paths,
    }
