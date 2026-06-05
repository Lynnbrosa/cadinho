"""Route variants to the right track.

  truncating  (nonsense / frameshift / canonical ±1,2 splice) -> rule-based track
  missense    (single-residue substitution)                   -> ML track
  other       (synonymous / in-frame indel / deep-intronic / unknown) -> excluded

For a recessive loss-of-function gene like SGCA, truncating variants are handled by an
explicit, flagged rule (model/rules.py) and are never fed to the missense model.
"""

from __future__ import annotations

from cadinho.config import VariantClassCfg
from cadinho.normalize.hgvs import ParsedVariant

TRUNCATING = {"truncating"}
MISSENSE = {"missense"}


def derive_consequence(parsed: ParsedVariant) -> str:
    """Best-effort functional consequence from parsed HGVS."""
    p, c = parsed.p, parsed.c
    if p is not None:
        if p.consequence == "nonsense":
            return "stop_gained"
        if p.consequence == "frameshift":
            return "frameshift_variant"
        if p.consequence == "missense":
            return "missense_variant"
        if p.consequence == "synonymous":
            return "synonymous_variant"
        if p.consequence == "inframe_indel":
            return "inframe_indel"
    if c is not None:
        if c.is_splice_canonical:
            return "splice_variant"
        if c.intron_offset is not None:
            return "splice_region_variant"
        if c.event == "indel":
            return "inframe_indel"
    return "unknown"


def route(consequence: str, vc: VariantClassCfg) -> str:
    """Map a consequence string to a track: 'truncating' | 'missense' | 'other'."""
    truncating = set(vc.truncating_consequences) | {"splice_variant"}
    if consequence in truncating:
        return "truncating"
    if consequence == "missense_variant":
        return "missense"
    return "other"
