"""Minimal, well-tested HGVS handling for the cases we route on.

This is intentionally NOT a full HGVS engine (which needs transcript sequences / a UTA
database we cannot reach offline). It robustly handles:
  * coding SNV substitutions      c.229C>T            (+ intronic ±N splice offsets)
  * protein consequences          p.Arg77Cys / p.R77C / p.Arg77Ter / p.Gln155fs / p.(=)
  * ClinVar `Name` strings         NM_000023.4(SGCA):c.229C>T (p.Arg77Cys)
and derives protein position from a CDS coordinate. Anything outside this scope is
reported as `event="other"` rather than silently mis-parsed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cadinho.features.aa import AA_SET, THREE_TO_ONE

# ---- ClinVar Name: "NM_000023.4(SGCA):c.229C>T (p.Arg77Cys)" -------------------
_NAME_RE = re.compile(
    r"^(?P<tx>[NXLM][MRP]_\d+\.\d+)?\s*(?:\((?P<gene>[A-Za-z0-9\-]+)\))?\s*:?\s*"
    r"(?P<c>c\.[^ ]+)?\s*(?:\((?P<p>p\.[^)]+)\))?\s*$"
)

# ---- coding SNV: c.229C>T  or  c.229+1G>A  or  c.230-2A>G -----------------------
_C_SNV_RE = re.compile(
    r"^c\.(?P<pos>\d+)(?P<offset>[+-]\d+)?(?P<ref>[ACGT])>(?P<alt>[ACGT])$"
)
_C_FS_RE = re.compile(r"^c\.\d+.*(del|dup|ins)", re.IGNORECASE)


@dataclass
class HGVSc:
    raw: str
    cdna_pos: int | None = None
    ref_nt: str | None = None
    alt_nt: str | None = None
    intron_offset: int | None = None  # signed; None when exonic
    event: str = "other"              # substitution | indel | other
    is_splice_canonical: bool = False  # intronic ±1/±2

    @property
    def is_coding_snv(self) -> bool:
        return self.event == "substitution" and self.intron_offset is None


@dataclass
class HGVSp:
    raw: str
    aaref: str | None = None   # 1-letter
    pos: int | None = None
    aaalt: str | None = None   # 1-letter; '*' for stop
    consequence: str = "unknown"  # missense | nonsense | frameshift | synonymous | inframe_indel | unknown


def _aa_to_one(token: str) -> str | None:
    """Accept 1- or 3-letter residue token -> 1-letter, or None."""
    if token in ("Ter", "*"):
        return "*"
    if len(token) == 1 and token in AA_SET:
        return token
    return THREE_TO_ONE.get(token)


def cdna_to_protein(cdna_pos: int) -> tuple[int, int]:
    """CDS coordinate -> (protein_position, codon_position 1..3).

    c.1 is the A of the start codon -> protein residue 1, codon position 1.
    c.229 -> residue 77, codon position 1.
    """
    if cdna_pos < 1:
        raise ValueError(f"CDS position must be >= 1, got {cdna_pos}")
    protein_position = (cdna_pos - 1) // 3 + 1
    codon_position = (cdna_pos - 1) % 3 + 1
    return protein_position, codon_position


def parse_c(c_str: str, splice_canonical_offset: int = 2) -> HGVSc:
    raw = c_str.strip()
    m = _C_SNV_RE.match(raw)
    if m:
        offset = int(m.group("offset")) if m.group("offset") else None
        canonical = offset is not None and abs(offset) <= splice_canonical_offset
        return HGVSc(
            raw=raw,
            cdna_pos=int(m.group("pos")),
            ref_nt=m.group("ref"),
            alt_nt=m.group("alt"),
            intron_offset=offset,
            event="substitution",
            is_splice_canonical=canonical,
        )
    if _C_FS_RE.match(raw):
        return HGVSc(raw=raw, event="indel")
    return HGVSc(raw=raw, event="other")


def parse_p(p_str: str) -> HGVSp:
    raw = p_str.strip()
    body = raw[2:] if raw.startswith("p.") else raw
    body = body.strip("()")
    out = HGVSp(raw=raw)

    if body in ("=", ""):  # p.(=) synonymous, no residue detail
        out.consequence = "synonymous"
        return out

    is_fs = "fs" in body
    # leading residue (3- or 1-letter) + position
    m = re.match(r"^([A-Z][a-z]{2}|[A-Z])(\d+)(.*)$", body)
    if not m:
        return out
    aaref = _aa_to_one(m.group(1))
    pos = int(m.group(2))
    rest = m.group(3)
    out.aaref, out.pos = aaref, pos

    if is_fs:
        out.consequence = "frameshift"
        return out
    if rest in ("=",):
        out.consequence = "synonymous"
        out.aaalt = aaref
        return out
    # trailing residue token
    m2 = re.match(r"^(Ter|\*|[A-Z][a-z]{2}|[A-Z])", rest)
    if m2:
        aaalt = _aa_to_one(m2.group(1))
        out.aaalt = aaalt
        if aaalt == "*":
            out.consequence = "nonsense"
        elif aaalt and aaref:
            out.consequence = "synonymous" if aaalt == aaref else "missense"
    elif rest.startswith("del") or rest.startswith("ins") or rest.startswith("dup"):
        out.consequence = "inframe_indel"
    return out


@dataclass
class ParsedVariant:
    raw: str
    transcript: str | None = None
    gene: str | None = None
    hgvs_c: str | None = None
    hgvs_p: str | None = None
    c: HGVSc | None = None
    p: HGVSp | None = None
    protein_position: int | None = None
    aaref: str | None = None
    aaalt: str | None = None


def parse_variant(text: str, splice_canonical_offset: int = 2) -> ParsedVariant:
    """Parse a full ClinVar Name, or a bare c./p. string, into a ParsedVariant."""
    raw = text.strip()
    pv = ParsedVariant(raw=raw)

    c_str = p_str = None
    if raw.startswith("c."):
        c_str = raw
    elif raw.startswith("p."):
        p_str = raw
    else:
        m = _NAME_RE.match(raw)
        if m:
            pv.transcript = m.group("tx")
            pv.gene = m.group("gene")
            c_str = m.group("c")
            p_str = m.group("p")

    if c_str:
        pv.hgvs_c = c_str
        pv.c = parse_c(c_str, splice_canonical_offset)
        if pv.c.is_coding_snv and pv.c.cdna_pos is not None:
            pv.protein_position, _ = cdna_to_protein(pv.c.cdna_pos)
    if p_str:
        pv.hgvs_p = p_str
        pv.p = parse_p(p_str)
        if pv.p.pos is not None:
            pv.protein_position = pv.p.pos
        pv.aaref, pv.aaalt = pv.p.aaref, pv.p.aaalt
    return pv
