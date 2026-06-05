"""Provenance manifest for data/raw/.

Every raw artifact (fixture or real) is recorded with its mode, source, version/date,
row count and a sha256 so the pipeline is reproducible and an auditor can tell at a
glance whether results came from synthetic or real data.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "manifest.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / MANIFEST_NAME


def load_manifest(raw_dir: Path) -> dict:
    p = _manifest_path(raw_dir)
    if p.exists():
        return json.loads(p.read_text())
    return {"generated_at": None, "sources": {}}


def record_source(
    raw_dir: Path,
    name: str,
    *,
    mode: str,
    path: Path,
    version: str | None = None,
    url: str | None = None,
    n_rows: int | None = None,
    note: str | None = None,
    extra: dict | None = None,
) -> dict:
    manifest = load_manifest(raw_dir)
    now = datetime.now(timezone.utc).isoformat()
    entry = {
        "mode": mode,
        "synthetic": mode == "fixture",
        "path": str(path.relative_to(raw_dir.parent.parent)) if path.is_absolute() else str(path),
        "version": version,
        "url": url,
        "retrieved_at": now,
        "n_rows": n_rows,
        "sha256": sha256_of(path) if path.exists() else None,
        "note": note,
    }
    if extra:
        entry.update(extra)
    manifest["sources"][name] = entry
    manifest["generated_at"] = now
    _manifest_path(raw_dir).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return entry
