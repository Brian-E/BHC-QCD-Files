# shared.py
"""
BHC+QCD=TOE — shared constants and core toy relations.

This module is the single source of truth for:
- locked canonical parameters (spin inheritance, Kerr gate)
- z_acc observational default used for *consistency testing* (not precision fitting)
- small helper functions used by BHfromUniverse.py and UniverseFromBH.py

Paper alignment:
- Spin inheritance: chi_n = chi0 * rho^n  (Section 5)
- Kerr gate: chi >= chi_min (Section 5)
- Acceleration onset observable: z_acc_obs = 0.67 ± 0.10 (Diagnostic default)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Optional, Tuple, Dict

def ladder_stage_label(n: int) -> str:
    if n <= 1:
        return "early ladder (near Base-1)"
    if n <= 7:
        return "mid ladder"
    if n <= 11:
        return "late ladder (near-terminal)"
    return "terminal / beyond Kerr gate"


def bh_condition_summary(chi: float, chi_min: float) -> str:
    if chi < chi_min:
        return f"Sub-threshold spin (chi={chi:.3f} < chi_min={chi_min:.3f}): Kerr-enabled boundary effects are absent."
    return f"Kerr-viable spin (chi={chi:.3f} ≥ chi_min={chi_min:.3f}): boundary-enabled effects are permitted."


def universe_condition_summary(
    *,
    z_acc_pred: Optional[float],
    dm_band: Optional[str],
    dm_ratio_band: Optional[Tuple[float, float]],
    relic_expect: Optional[str],
    Pi1_compatible: bool,
) -> List[str]:
    lines: List[str] = []

    if z_acc_pred is None:
        lines.append("Expansion: no late-time acceleration is predicted under the Kerr gate (terminal behavior).")
    else:
        lines.append(f"Expansion: late-time acceleration is permitted; illustrative onset z_acc ≈ {z_acc_pred:.3f}.")

    if relic_expect is not None:
        lines.append(f"Structure-growth proxy: relic expectation band = {relic_expect!r} (coarse).")

    if dm_band is not None and dm_ratio_band is not None:
        lo, hi = dm_ratio_band
        lines.append(f"Dark matter: dm_band={dm_band!r}, implying Ω_DM/Ω_b ∈ [{lo:.1f}, {hi:.1f}] (toy band).")

    if Pi1_compatible:
        lines.append("Parity: Pi=1 would be compatible (coherent spin inheritance allowed).")
    else:
        lines.append("Parity: Pi=1 would be incompatible under current Kerr gate (if Pi=1 is observed, this case is disfavored).")

    return lines


def channel_story(channel: str) -> List[str]:
    """
    Human-readable birth-channel expectations used by UniverseFromBH.
    """
    ch = channel.lower().strip()
    if ch == "collapse":
        return [
            "Birth channel: core-collapse–like. Expect rapid, high-entropy reheating and strong equilibration.",
            "Dark sector tendency: micro/meso-dominated; macroscopic components are not favored.",
        ]
    if ch == "merger":
        return [
            "Birth channel: merger–like. Expect anisotropic/prolonged formation conditions and stronger dynamical imprint.",
            "Dark sector tendency: broader spectrum; macroscopic components more likely to survive (curvature-stabilized).",
        ]
    if ch == "vacuum":
        return [
            "Birth channel: vacuum/fluctuation–like. Expect gentler reheating and reduced density contrasts at birth.",
            "Dark sector tendency: lighter components favored; macroscopic remnants suppressed.",
        ]
    return ["Birth channel: unspecified. No channel-specific expectations applied."]


def format_interpretation_block(title: str, bullets: List[str]) -> str:
    """
    Returns a preformatted block for console output.
    """
    out = []
    out.append(f"\n{title}")
    out.append("-" * len(title))
    for b in bullets:
        out.append(f"- {b}")
    return "\n".join(out)

DM_BANDS: Dict[str, Tuple[float, float]] = {
    "low": (1.0, 4.0),
    "medium": (4.0, 8.0),
    "high": (8.0, 15.0),
}

DM_COMPOSITION_HINT: Dict[str, str] = {
    "low": "micro-dominated; negligible macro component",
    "medium": "mixed micro/meso; rare macros",
    "high": "macro-enhanced; broad spectrum",
}

_DM_ORDER = ["low", "medium", "high"]


def _clamp_band(band: str) -> str:
    if band not in DM_BANDS:
        raise ValueError(f"Invalid dm_band {band!r}")
    return band


def _shift_band(band: str, delta: int) -> str:
    """delta: -1 (down), +1 (up), 0 (no shift)"""
    i = _DM_ORDER.index(band)
    j = max(0, min(len(_DM_ORDER) - 1, i + delta))
    return _DM_ORDER[j]


@dataclass(frozen=True)
class DMPrediction:
    dm_band: str                       # low | medium | high
    dm_ratio_band: Tuple[float, float] # (min,max) for Omega_DM/Omega_b
    dm_composition_hint: str
    macro_survival_hint: str           # e.g. "neutral" / "slightly favored"
    dm_rationale: str                  # 1–2 sentence explanation


def predict_dm_band(
    *,
    channel: Optional[str] = None,     # collapse|merger|vacuum (UniverseFromBH)
    n: Optional[int] = None,           # ladder generation index
    relic: Optional[str] = None,       # low|medium|high (BHfromUniverse)
    Pi: int = 0,                       # 0|1
) -> DMPrediction:
    """
    Canonical toy DM-band mapping (BHUT+QCD=TOE):

    Bands are for R = Omega_DM/Omega_b:
      low:    [1,4]
      medium: [4,8]
      high:   [8,15]

    Mapping rules:
      A) Base band by channel:
           collapse -> medium
           merger   -> high
           vacuum   -> low
         If channel is None, default base=medium (agnostic).

      B) Late ladder shift:
           if n >= 8: shift down by 1 band
           else: no shift

      C) Relic shift (modest weight):
           relic=high: shift up by 1 band
           relic=medium/low/None: no shift

      D) Pi:
           Pi=1 does not shift band; adds macro_survival_hint="slightly favored"
    """
    rationale_parts = []

    # A) base by channel (or agnostic)
    base = "medium"
    if channel is not None:
        ch = channel.lower().strip()
        if ch == "collapse":
            base = "medium"
            rationale_parts.append("channel=collapse → efficient equilibration → medium DM band")
        elif ch == "merger":
            base = "high"
            rationale_parts.append("channel=merger → prolonged/aniso birth → macro survival → high DM band")
        elif ch == "vacuum":
            base = "low"
            rationale_parts.append("channel=vacuum → gentle/low-contrast birth → low DM band")
        else:
            raise ValueError("channel must be one of: collapse, merger, vacuum")
    else:
        rationale_parts.append("channel unspecified → base DM band set to medium (agnostic)")

    band = base

    # B) ladder shift
    if n is not None:
        if n < 0:
            raise ValueError("n must be >= 0")
        if n >= 8:
            band = _shift_band(band, -1)
            rationale_parts.append("n>=8 → late-ladder filter → shift DM band down by one")
        else:
            rationale_parts.append("n<8 → no ladder-driven DM band shift")

    # C) relic shift
    if relic is not None:
        r = relic.lower().strip()
        if r not in ("low", "medium", "high"):
            raise ValueError("relic must be one of: low, medium, high")
        if r == "high":
            band = _shift_band(band, +1)
            rationale_parts.append("relic=high → stronger relic survival proxy → shift DM band up by one")
        else:
            rationale_parts.append(f"relic={r} → no relic-driven DM band shift")

    # D) parity hint (no shift)
    macro_hint = "neutral"
    if Pi == 1:
        macro_hint = "slightly favored"
        rationale_parts.append("Pi=1 → coherent spin imprint → macro survival slightly favored (no band shift)")

    band = _clamp_band(band)
    dm_range = DM_BANDS[band]
    comp = DM_COMPOSITION_HINT[band]

    # Keep rationale short (1–2 sentences). We’ll join with '; ' and let caller print full if desired.
    rationale = "; ".join(rationale_parts)

    return DMPrediction(
        dm_band=band,
        dm_ratio_band=dm_range,
        dm_composition_hint=comp,
        macro_survival_hint=macro_hint,
        dm_rationale=rationale,
    )



# -------------------------
# Locked canonical defaults
# -------------------------

CHI_MIN_DEFAULT: float = 0.2
CHI0_DEFAULT: float = 0.7
RHO_DEFAULT: float = 0.9

# Diagnostic default (consistency testing only; conservative tolerance)
ZACC_OBS_DEFAULT: float = 0.67
ZACC_TOL_DEFAULT: float = 0.10


# -------------------------
# Config container
# -------------------------

@dataclass(frozen=True)
class Canon:
    chi_min: float = CHI_MIN_DEFAULT
    chi0: float = CHI0_DEFAULT
    rho: float = RHO_DEFAULT
    z_acc_obs: float = ZACC_OBS_DEFAULT
    z_acc_tol: float = ZACC_TOL_DEFAULT

import argparse
class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter,
):
    pass
    
# -------------------------
# Core ladder relations
# -------------------------

def chi_n(n: int, canon: Canon = Canon()) -> float:
    """Inherited spin at generation n (n=0 is Base-1)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    return canon.chi0 * (canon.rho ** n)


def kerr_viable(chi: float, canon: Canon = Canon()) -> bool:
    """Kerr gate viability."""
    return chi >= canon.chi_min


def n_max_kerr_viable(canon: Canon = Canon()) -> int:
    """
    Maximum generation index n such that chi_n >= chi_min.
    With locked canon, this should return 11.
    """
    # Solve chi0 * rho^n >= chi_min -> n <= ln(chi_min/chi0)/ln(rho)
    if not (0.0 < canon.rho < 1.0):
        raise ValueError("rho must be in (0,1) for decay per generation.")
    if canon.chi0 <= 0 or canon.chi_min <= 0:
        raise ValueError("chi0 and chi_min must be > 0")
    bound = log(canon.chi_min / canon.chi0) / log(canon.rho)
    return int(bound)  # floor for positive/negative logs works here given rho<1


# -------------------------
# Acceleration onset toy mapping
# -------------------------

def z_acc_from_lambda_ratio(lambda_to_m0_ratio: float) -> Optional[float]:
    """
    Toy mapping for acceleration onset when rho_m(z_acc) = 2 rho_Lambda,
    with rho_m(z) = rho_m0 (1+z)^3.

    Define r = rho_Lambda / rho_m0. Then:
      1 + z_acc = (2r)^(1/3)
      z_acc = (2r)^(1/3) - 1

    Returns None if r <= 0 (no acceleration).
    """
    r = lambda_to_m0_ratio
    if r <= 0:
        return None
    return (2.0 * r) ** (1.0 / 3.0) - 1.0


def within_tolerance(value: Optional[float], target: float, tol: float) -> bool:
    """Conservative tolerance check; None is always a fail for numerical targets."""
    if value is None:
        return False
    return abs(value - target) <= tol


# -------------------------
# Parity indicator (optional observable)
# -------------------------

def parity_mismatch(Pi: int, chi: float, canon: Canon = Canon()) -> int:
    """
    Boolean penalty:
      Pi = 1 (preferred rotation detected) requires Kerr-viable chi.
      Pi = 0 imposes no penalty.
    """
    if Pi not in (0, 1):
        raise ValueError("Pi must be 0 or 1")
    if Pi == 1 and not kerr_viable(chi, canon):
        return 1
    return 0
