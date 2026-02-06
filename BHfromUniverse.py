# BHfromUniverse.py  (UPDATED: optional explicit Dark Energy knobs via lambda-ratio + eta)

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from shared import (
    Canon,
    RawDefaultsHelpFormatter,
    chi_n,
    kerr_viable,
    n_max_kerr_viable,
    parity_mismatch,
    within_tolerance,
    z_acc_from_lambda_ratio,
    predict_dm_band,
    ladder_stage_label,
    format_interpretation_block,
    )
from shared_help import COMMON_HELP

# -------------------------
# Relic consistency (LOCKED)
# -------------------------

"""
relic : str, optional
    Qualitative indicator of the abundance of relic (early-formed, compact) galaxy
    populations at late times.

    Allowed values:
      - "low"    : relic systems are rare; prolonged structure growth/merging erases
                   early compact populations.
      - "medium" : relic systems are present but not dominant; indicates moderate
                   suppression of late-time growth.
      - "high"   : relic systems are abundant; indicates early/strong suppression
                   of late-time growth.

    Locked coarse thresholds (consistency discriminator, modest weight):
      - relic="high"   expects z_acc_pred >= 0.8
      - relic="medium" expects 0.5 <= z_acc_pred < 0.8
      - relic="low"    expects z_acc_pred < 0.5

    Edge cases:
      - If z_acc_pred is None (no acceleration), then:
          * relic="low" is treated as consistent
          * relic="medium"/"high" are treated as inconsistent
"""
BH_QUICKSTART = r"""
Quick start (BHfromUniverse):
  This tool filters ladder generations n for consistency with *universe observations*.

  You should almost always specify:
    --zacc and --zacc-tol          (primary observable; defaults match our universe)
  You may optionally specify:
    --relic {low,medium,high}     (coarse structure-history discriminator)
    --Pi {0,1}                    (optional parity discriminator)
    --lambda-ratio and --eta      (advanced: explicit DE amplitude; otherwise proxy drift)

Examples:
  1) Our universe-style run:
     python BHfromUniverse.py --zacc 0.67 --zacc-tol 0.10 --relic medium

  2) Add explicit DE amplitude (our-universe-ish r0≈2.33):
     python BHfromUniverse.py --zacc 0.67 --zacc-tol 0.10 --relic medium --lambda-ratio 2.33 --eta 0.95
"""


RELIC_THRESHOLDS = {
    "high": (0.8, None),
    "medium": (0.5, 0.8),
    "low": (None, 0.5),
}
  
def relic_mismatch(relic: Optional[str], z_acc_pred: Optional[float]) -> int:
    """Returns 1 if relic category is inconsistent with z_acc_pred; 0 otherwise."""
    if relic is None:
        return 0
    relic = relic.lower().strip()
    if relic not in RELIC_THRESHOLDS:
        raise ValueError(f"relic must be one of {list(RELIC_THRESHOLDS.keys())}, got {relic!r}")

    if z_acc_pred is None:
        return 0 if relic == "low" else 1

    lo, hi = RELIC_THRESHOLDS[relic]
    if lo is not None and z_acc_pred < lo:
        return 1
    if hi is not None and z_acc_pred >= hi:
        return 1
    return 0


# -------------------------
# z_acc prediction
# -------------------------

def z_acc_pred_from_generation(
    n: int,
    canon: Canon,
    r0: Optional[float],
    eta: float,
) -> Optional[float]:
    """
    Predict z_acc for generation n.

    If r0 is provided (explicit DE amplitude):
      - If Kerr gate fails: None
      - Else r(n) = r0 * eta^n and z_acc_pred = (2 r(n))^(1/3) - 1

    If r0 is not provided:
      - If Kerr gate fails: None
      - Else use conservative monotonic proxy:
          z_acc_pred(n) = canon.z_acc_obs * (canon.rho ** (n/3))
    """
    chi = chi_n(n, canon)
    if not kerr_viable(chi, canon):
        return None

    if r0 is not None:
        if r0 <= 0:
            return None
        if not (0.0 < eta <= 1.0):
            raise ValueError("eta must be in (0,1]")
        r_n = r0 * (eta ** n)
        return z_acc_from_lambda_ratio(r_n)

    # Fallback proxy (no explicit DE amplitude provided)
    return canon.z_acc_obs * (canon.rho ** (n / 3.0))


# -------------------------
# CLI / runner
# -------------------------

# Assumes you already added: HELP_EPILOG = """..."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="BHfromUniverse.py",
        usage="python BHfromUniverse.py [OPTIONS]",
        description="BHfromUniverse: Universe → candidate parent BH classes (consistency test).",
        epilog=COMMON_HELP,                # shows at the BOTTOM (deep reference)
        formatter_class=RawDefaultsHelpFormatter,  # preserves formatting + shows defaults
    )

    # -------------------------
    # Basic usage parameters
    # -------------------------
    basic = p.add_argument_group("Basic usage parameters (most users should start here)")
    basic.add_argument(
        "--zacc",
        type=float,
        default=None,
        help="Observed acceleration-onset redshift z_acc^obs (defaults to canonical 0.67).",
    )
    basic.add_argument(
        "--zacc-tol",
        type=float,
        default=None,
        dest="zacc_tol",
        help="Tolerance on z_acc consistency (defaults to canonical ±0.10).",
    )
    basic.add_argument(
        "--relic",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Relic abundance category (low/medium/high). Optional discriminator.",
    )
    basic.add_argument(
        "--Pi",
        type=int,
        default=0,
        choices=[0, 1],
        help="Parity indicator: 1=preferred early-galaxy rotation direction; 0=off/absent.",
    )

    # --------------------------------
    # Optional dark-energy amplitude knobs
    # --------------------------------
    de = p.add_argument_group("Optional dark-energy amplitude knobs (advanced)")
    de.add_argument(
        "--lambda-ratio",
        type=float,
        default=None,
        dest="lambda_ratio",
        help="r0 = rho_Lambda_eff / rho_m0 (dimensionless). Enables explicit z_acc_pred.",
    )
    de.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="Per-generation weakening for r(n)=r0*eta^n (used only if --lambda-ratio is supplied).",
    )

    # -----------------------------
    # Ladder / model parameters
    # -----------------------------
    model = p.add_argument_group("Ladder / model parameters (advanced; defaults are canonical)")
    model.add_argument("--chi0", type=float, default=None, help="Seed spin at n=0 (Base-1).")
    model.add_argument("--rho", type=float, default=None, help="Spin retention per generation in chi_n=chi0*rho^n.")
    model.add_argument(
        "--chi-min",
        type=float,
        default=None,
        dest="chi_min",
        help="Kerr threshold chi_min. If chi_n < chi_min, terminal (no acceleration).",
    )
    model.add_argument(
        "--n-scan-max",
        type=int,
        default=30,
        dest="n_scan_max",
        help="Maximum ladder generation index n to scan (n=0 is Base-1).",
    )

    # -----------------------------
    # Output controls
    # -----------------------------
    out = p.add_argument_group("Output controls")
    out.add_argument(
        "--json-out",
        type=str,
        default=None,
        dest="json_out",
        help="Write full per-n diagnostics and allowed_n results to this JSON file path.",
    )

    return p


def make_canon(args: argparse.Namespace) -> Canon:
    base = Canon()
    return Canon(
        chi_min=args.chi_min if args.chi_min is not None else base.chi_min,
        chi0=args.chi0 if args.chi0 is not None else base.chi0,
        rho=args.rho if args.rho is not None else base.rho,
        z_acc_obs=args.zacc if args.zacc is not None else base.z_acc_obs,
        z_acc_tol=args.zacc_tol if args.zacc_tol is not None else base.z_acc_tol,
    )


def classify_parent_bh_requirements(Pi: int, acceleration_observed: bool) -> List[str]:
    reqs: List[str] = []
    if acceleration_observed:
        reqs.append("Kerr-enabled origin required (acceleration observed).")
    if Pi == 1:
        reqs.append("Kerr-enabled origin required (parity signal implies coherent spin inheritance).")
    if not reqs:
        reqs.append("Schwarzschild-allowed (no Kerr-specific observables supplied).")
    return reqs


def main() -> None:
    args = build_parser().parse_args()
    canon = make_canon(args)

    r0 = args.lambda_ratio
    eta = float(args.eta)

    # Determine scan range (ensure we scan past terminal boundary for completeness)
    scan_max = max(args.n_scan_max, n_max_kerr_viable(canon) + 5)

    results: List[Dict[str, Any]] = []
    allowed_n: List[int] = []

    # Universe-level constraints
    zacc_target = canon.z_acc_obs
    zacc_tol = canon.z_acc_tol
    relic = args.relic
    Pi = args.Pi

    for n in range(0, scan_max + 1):
        chi = chi_n(n, canon)
        kerr_ok = kerr_viable(chi, canon)

        zacc_pred = z_acc_pred_from_generation(n=n, canon=canon, r0=r0, eta=eta)

        # Core constraint: match observed z_acc (requires acceleration to exist)
        passes_zacc = within_tolerance(zacc_pred, zacc_target, zacc_tol)

        # Optional constraints
        pm = parity_mismatch(Pi, chi, canon)  # 1 iff Pi=1 and Kerr gate fails
        rm = relic_mismatch(relic, zacc_pred)

        passes_parity = (pm == 0)
        passes_relic = (rm == 0)

        passes_all = passes_zacc and passes_parity and passes_relic
        if passes_all:
            allowed_n.append(n)

        results.append(
            {
                "n": n,
                "chi_n": chi,
                "kerr_viable": kerr_ok,
                "z_acc_pred": zacc_pred,
                "passes": {
                    "z_acc": passes_zacc,
                    "parity": passes_parity if Pi in (0, 1) else None,
                    "relic": passes_relic if relic is not None else None,
                },
                "fails": {
                    "z_acc": (not passes_zacc),
                    "parity": (not passes_parity) if Pi == 1 else False,
                    "relic": (not passes_relic) if relic is not None else False,
                },
            }
        )

    parent_bh_reqs = classify_parent_bh_requirements(Pi=Pi, acceleration_observed=True)

    payload = {
        "canon": asdict(canon),
        "inputs": {
            "relic": relic,
            "Pi": Pi,
            "lambda_ratio_r0": r0,
            "eta": eta,
            "n_scan_max_used": scan_max,
        },
        "outputs": {
            "allowed_n": allowed_n,
            "parent_bh_requirements": parent_bh_reqs,
        },
        "results": results,
    }
    
    def _band_order(b: str) -> int:
        return {"low": 0, "medium": 1, "high": 2}[b]

    dm_by_n = []
    for n in allowed_n:
        dm_n = predict_dm_band(channel=None, n=n, relic=relic, Pi=Pi)
        dm_by_n.append({"n": n, "dm_band": dm_n.dm_band, "dm_ratio_band": dm_n.dm_ratio_band})    

    # Summarize overall implied band range across allowed n (conservative envelope)
    if dm_by_n:
        bands = [x["dm_band"] for x in dm_by_n]
        lo_band = min(bands, key=_band_order)
        hi_band = max(bands, key=_band_order)
        lo_rng = min(x["dm_ratio_band"][0] for x in dm_by_n)
        hi_rng = max(x["dm_ratio_band"][1] for x in dm_by_n)
    else:
        lo_band = hi_band = None
        lo_rng = hi_rng = None

    payload["outputs"].update(
        {
            "dm_band_by_allowed_n": dm_by_n,  # list with per-n band
            "dm_band_envelope": {"min_band": lo_band, "max_band": hi_band},
            "dm_ratio_envelope": None if lo_rng is None else {"min": lo_rng, "max": hi_rng},
        }
    )
    
    bullets = []

    if not allowed_n:
        bullets.append("No ladder generations satisfy the provided constraints under current assumptions.")
    else:
        nmin, nmax = min(allowed_n), max(allowed_n)
        bullets.append(f"Allowed ladder generations: n ∈ [{nmin}, {nmax}] (may be non-contiguous).")
        bullets.append(f"Ladder staging: {ladder_stage_label(nmin)} through {ladder_stage_label(nmax)}.")

        # Parent BH implication (Kerr requirement)
        bullets.append("Parent BH class: Kerr-enabled origin is required if acceleration is observed (and/or Pi=1 is supplied).")

        # Relic + parity inputs
        if relic is not None:
            bullets.append(f"Relic discriminator: input relic={relic!r} applied as a coarse consistency check.")
        bullets.append(f"Parity discriminator: Pi={Pi} {'applied' if Pi==1 else 'not constraining'}.")

        # DM envelope summary (if present)
        if lo_band is not None and hi_band is not None and lo_rng is not None and hi_rng is not None:
            bullets.append(f"Implied dark matter band across allowed n: dm_band ∈ [{lo_band}, {hi_band}].")
            bullets.append(f"Implied Ω_DM/Ω_b envelope: [{lo_rng:.1f}, {hi_rng:.1f}] (toy bands).")

        # zacc interpretation
        bullets.append(f"Acceleration-onset constraint: z_acc_obs={canon.z_acc_obs:.2f}±{canon.z_acc_tol:.2f} matched by z_acc_pred(n).")
        if r0 is not None:
            bullets.append(f"DE mapping: used explicit r(n)=r0*eta^n with r0={r0:.3f}, eta={eta:.3f} (Kerr-gated).")
        else:
            bullets.append("DE mapping: used conservative monotonic proxy for z_acc_pred drift (no explicit r0 supplied).")

    print(format_interpretation_block("Interpretation (human-readable)", bullets))
    payload["outputs"]["interpretation"] = bullets
     
    # Human-readable summary
    print("\nBHfromUniverse — Consistency Results")
    print("-----------------------------------")
    print(f"z_acc_obs = {canon.z_acc_obs:.2f} ± {canon.z_acc_tol:.2f}")
    if r0 is not None:
        print(f"DE knobs: lambda_ratio r0={r0:.3f}, eta={eta:.3f} (r(n)=r0*eta^n, Kerr-gated)")
    else:
        print("DE knobs: not supplied (using conservative monotonic proxy for z_acc_pred drift)")
    if relic is not None:
        print("relic thresholds: high>=0.8, medium:[0.5,0.8), low<0.5 (None→low)")
        print(f"relic = {relic!r}")
    print(f"Pi = {Pi}  (Pi=1 requires Kerr-viable chi >= {canon.chi_min})")

    print("\nParent BH class constraints:")
    for r in parent_bh_reqs:
        print(f"  - {r}")

    if allowed_n:
        print(f"\nAllowed ladder generations (n): {allowed_n}")
        print("Sample allowed rows:")
        for n in allowed_n[: min(5, len(allowed_n))]:
            row = next(rr for rr in results if rr["n"] == n)
            ztxt = "None" if row["z_acc_pred"] is None else f"{row['z_acc_pred']:.3f}"
            print(f"  n={n:2d}: chi={row['chi_n']:.3f}, z_acc_pred={ztxt}")
    else:
        print("\nNo ladder generations satisfy the provided constraints under current assumptions.")
        print("Try relaxing optional constraints (relic/Pi) or widening z_acc tolerance (zacc-tol).")

    print("\nDark matter (toy output bands):")
    if not dm_by_n:
        print("  (no allowed n; DM band not reported)")
    else:
        print(f"  Envelope across allowed n: dm_band ∈ [{lo_band}, {hi_band}]")
        print(f"  Envelope across allowed n: Omega_DM/Omega_b ∈ [{lo_rng:.1f}, {hi_rng:.1f}]")
        print("  Per allowed n:")
        for x in dm_by_n[: min(10, len(dm_by_n))]:
            lo, hi = x["dm_ratio_band"]
            print(f"    n={x['n']:2d}: dm_band={x['dm_band']:<6s}  Omega_DM/Omega_b ∈ [{lo:.1f}, {hi:.1f}]")
        if len(dm_by_n) > 10:
            print(f"    ... ({len(dm_by_n)-10} more)")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON output to: {args.json_out}")


if __name__ == "__main__":
    main()
