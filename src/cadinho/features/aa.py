"""Amino-acid biochemistry: reference tables + substitution features.

These are *published reference constants* (Grantham 1974; BLOSUM62; Kyte-Doolittle),
not learned parameters and not recomputed conservation. Used both to build features for
real variants and to generate biologically-plausible synthetic fixtures.
"""

from __future__ import annotations

import math

AA1 = "ARNDCQEGHILKMFPSTWYV"  # canonical 20, BLOSUM62 row/col order
AA_SET = set(AA1)

ONE_TO_THREE = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln",
    "E": "Glu", "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys",
    "M": "Met", "F": "Phe", "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp",
    "Y": "Tyr", "V": "Val", "*": "Ter",
}
THREE_TO_ONE = {v: k for k, v in ONE_TO_THREE.items()}
THREE_TO_ONE["Ter"] = "*"

# Grantham (1974) composition / polarity / volume per residue.
GRANTHAM_CPV = {
    "A": (0.00, 8.1, 31.0), "R": (0.65, 10.5, 124.0), "N": (1.33, 11.6, 56.0),
    "D": (1.38, 13.0, 54.0), "C": (2.75, 5.5, 55.0), "Q": (0.89, 10.5, 85.0),
    "E": (0.92, 12.3, 83.0), "G": (0.74, 9.0, 3.0), "H": (0.58, 10.4, 96.0),
    "I": (0.00, 5.2, 111.0), "L": (0.00, 4.9, 111.0), "K": (0.33, 11.3, 119.0),
    "M": (0.00, 5.7, 105.0), "F": (0.00, 5.0, 132.0), "P": (0.39, 8.0, 32.5),
    "S": (1.42, 9.2, 32.0), "T": (0.71, 8.6, 61.0), "W": (0.13, 5.4, 170.0),
    "Y": (0.20, 6.2, 136.0), "V": (0.00, 5.9, 84.0),
}
# Grantham distance constants (reproduce e.g. R->C = 180).
_G_ALPHA, _G_BETA, _G_GAMMA, _G_RHO = 1.833, 0.1018, 0.000399, 50.723

# Kyte-Doolittle hydropathy index.
KD_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# BLOSUM62 (NCBI), rows/cols in AA1 order.
_BLOSUM62_ROWS = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -2],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -2, 4],
]
BLOSUM62 = {
    a: {b: _BLOSUM62_ROWS[i][j] for j, b in enumerate(AA1)}
    for i, a in enumerate(AA1)
}


def grantham_distance(aaref: str, aaalt: str) -> float:
    """Grantham (1974) physicochemical distance between two residues."""
    ci, pi, vi = GRANTHAM_CPV[aaref]
    cj, pj, vj = GRANTHAM_CPV[aaalt]
    d2 = _G_ALPHA * (ci - cj) ** 2 + _G_BETA * (pi - pj) ** 2 + _G_GAMMA * (vi - vj) ** 2
    return round(_G_RHO * math.sqrt(d2), 2)


def blosum62(aaref: str, aaalt: str) -> int:
    return BLOSUM62[aaref][aaalt]


def delta_hydrophobicity(aaref: str, aaalt: str) -> float:
    return round(abs(KD_HYDRO[aaref] - KD_HYDRO[aaalt]), 2)


def delta_volume(aaref: str, aaalt: str) -> float:
    return round(abs(GRANTHAM_CPV[aaref][2] - GRANTHAM_CPV[aaalt][2]), 2)


def substitution_features(aaref: str, aaalt: str) -> dict[str, float]:
    """The four substitution-property features used by the model.

    Raises KeyError on non-standard residues (e.g. '*') so callers must route
    truncating/nonsense changes elsewhere rather than silently scoring them.
    """
    return {
        "Grantham": grantham_distance(aaref, aaalt),
        "BLOSUM62": float(blosum62(aaref, aaalt)),
        "delta_hydrophobicity": delta_hydrophobicity(aaref, aaalt),
        "delta_volume": delta_volume(aaref, aaalt),
    }
