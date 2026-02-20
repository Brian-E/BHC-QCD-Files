# UniverseFromBH.py  (UPDATED: explicit Dark Energy component via lambda-ratio + eta)

from __future__ import annotations

import argparse
import json
import  math
from dataclasses import asdict
from typing import Any, Dict, Optional

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
    bh_condition_summary,
    universe_condition_summary,
    channel_story,
    format_interpretation_block,    
)
from shared_help import COMMON_HELP

# -------------------------
# Relic consistency (LOCKED)
# -------------------------

UB_QUICKSTART = r"""
Quick start (UniverseFromBH):
  This tool predicts *universe outcome classes* from a specified parent BH.

  You must specify:
    --chi                         (parent BH spin; controls Kerr viability)
  You may optionally specify:
    --channel {collapse,merger,vacuum}   (qualitative birth channel)
    --lambda-ratio r0                    (enables numeric z_acc_pred; otherwise qualitative)
    --n and --eta                         (optional generational weakening if ladder position is known)

Examples:
  1) Qualitative outcome (no DE amplitude):
     python UniverseFromBH.py --chi 0.6 --channel merger

  2) Compute z_acc_pred with r0 (our-universe-ish r0≈2.33):
     python UniverseFromBH.py --chi 0.6 --channel collapse --lambda-ratio 2.33

  3) Add ladder generation n and weakening eta:
     python UniverseFromBH.py --chi 0.6 --channel merger --n 3 --lambda-ratio 2.33 --eta 0.9
"""

RELIC_THRESHOLDS = {
    "high": (0.8, None),
    "medium": (0.5, 0.8),
    "low": (None, 0.5),
}


def relic_band_from_zacc(z_acc_pred: Optional[float]) -> str:
    """Returns the relic category most consistent with z_acc_pred under locked thresholds."""
    if z_acc_pred is None:
        return "low"
    if z_acc_pred >= 0.8:
        return "high"
    if z_acc_pred >= 0.5:
        return "medium"
    return "low"


def archetype(channel: str, kerr_ok: bool) -> str:
    """
    Named archetypes (qualitative, paper-aligned):
      - terminal: Kerr gate fails (no boundary-enabled acceleration)
      - fertile: collapse-born + strong structure era before acceleration
      - drifting: merger-born + dynamical imprint / composite skew
      - sterile: vacuum-born + low-entropy / marginal tendencies
      - reproductive: Kerr-enabled capability for generational propagation
    """
    if not kerr_ok:
        return "terminal"

    ch = channel.lower().strip()
    if ch == "collapse":
        return "fertile/reproductive"
    if ch == "merger":
        return "drifting/reproductive"
    if ch == "vacuum":
        return "sterile-or-marginal/reproductive"
    return "reproductive (unspecified channel)"


def z_acc_pred_from_bh_inputs(
    *,
    chi: float,
    r0: Optional[float],
    n: Optional[int],
    eta: float,
    canon: Canon,
) -> Optional[float]:
    """
    Compute z_acc_pred if possible.

    Option 2 (locked):
      - --chi is interpreted as the BH spin at birth for the specified case.
      - Kerr viability depends ONLY on chi.
      - If n is provided, it weakens the effective vacuum amplitude via
        r(n) = r0 * eta^n.
      - Spin is NOT recomputed internally from chi0/rho.

    Rules:
      1) If Kerr gate fails (chi < chi_min): return None (no acceleration).
      2) If r0 is not provided: return None (we refuse to invent a normalization).
      3) If n is provided: apply r(n)=r0*eta^n.
         Otherwise: use r=r0.
    """
    # Kerr viability gate
    if not kerr_viable(chi, canon):
        return None

    # Require a vacuum-to-matter normalization
    if r0 is None or r0 <= 0:
        return None

    # No ladder position given
    if n is None:
        return z_acc_from_lambda_ratio(r0)

    # Ladder position given
    if n < 0:
        raise ValueError("n must be >= 0 (n=0 is Base-1)")

    if not (0.0 < eta <= 1.0):
        raise ValueError("eta must be in (0,1]")

    r_n = r0 * (eta ** n)
    return z_acc_from_lambda_ratio(r_n)

import math

def infer_n_range_from_zacc(z_obs: float, z_tol: float, r0: float, eta: float, n_max: int):
    # Convert z -> required ratio r_req = ((1+z)^3)/2
    def r_req(z):
        return ((1.0 + z) ** 3) / 2.0

    if not (0.0 < eta < 1.0):
        raise ValueError("infer-n requires eta in (0,1).")
    if r0 <= 0:
        raise ValueError("infer-n requires lambda-ratio (r0) > 0.")

    r_lo = r_req(max(0.0, z_obs - z_tol))
    r_hi = r_req(z_obs + z_tol)

    # r(n) = r0 * eta^n must fall within [r_lo, r_hi]
    # Solve for n bounds:
    # eta^n >= r_lo/r0 and eta^n <= r_hi/r0 (note eta<1)
    # Use logs carefully; clamp to feasible range
    ln_eta = math.log(eta)

    # If the target ratios exceed r0 (i.e., require eta^n > 1), then only very small n could work (often none)
    # We'll compute raw bounds and then scan integer n for robustness.
    candidates = []
    for n in range(0, n_max + 1):
        r_n = r0 * (eta ** n)
        z_n = (2.0 * r_n) ** (1.0 / 3.0) - 1.0
        if abs(z_n - z_obs) <= z_tol:
            candidates.append(n)

    if not candidates:
        return None, None, []
    return min(candidates), max(candidates), candidates


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="UniverseFromBH.py",
        usage="python UniverseFromBH.py [OPTIONS]",
        description="UniverseFromBH: Parent BH → candidate universe outcomes (phenomenological).",
        epilog=COMMON_HELP,                # shows at the BOTTOM (deep reference)
        formatter_class=RawDefaultsHelpFormatter,  # preserves formatting + shows defaults
    )

    # -------------------------
    # Basic usage parameters
    # -------------------------
    basic = p.add_argument_group("Basic usage parameters (most users should start here)")
    basic.add_argument(
        "--chi",
        type=float,
        required=True,
        help="Parent BH spin at universe birth (dimensionless). Primary control parameter.",
    )
    basic.add_argument(
        "--channel",
        type=str,
        default="collapse",
        choices=["collapse", "merger", "vacuum"],
        help="Qualitative BH birth channel label (affects archetype labeling).",
    )

    # --------------------------------
    # Optional basic parameters
    # --------------------------------
    opt = p.add_argument_group("Optional basic parameters (common follow-ups)")
    opt.add_argument(
        "--n",
        type=int,
        default=None,
        help="Optional ladder generation index (n=0 is Base-1). Enables generational weakening via eta.",
    )
    opt.add_argument(
        "--lambda-ratio",
        type=float,
        default=None,
        dest="lambda_ratio",
        help="r0 = rho_Lambda_eff / rho_m0 (dimensionless). Enables numeric z_acc_pred.",
    )
    opt.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="Per-generation weakening factor for r(n)=r0*eta^n (used when --n is provided).",
    )
    opt.add_argument("--infer-n", action="store_true",
               help="Infer a compatible range of n from z_acc_obs/tol given --lambda-ratio and --eta.")
    opt.add_argument("--n-max", type=int, default=30,
               help="Max n to consider when using --infer-n (default: 30).")

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
        help="Kerr threshold chi_min. If chi_n < chi_min, universe is terminal (no acceleration).",
    )

    # -----------------------------
    # Output controls
    # -----------------------------
    out = p.add_argument_group("Output controls")
    out.add_argument("--json-out", type=str, default=None, help="Write JSON output to this file path.")

    return p


def make_canon(args: argparse.Namespace) -> Canon:
    base = Canon()
    return Canon(
        chi_min=args.chi_min if args.chi_min is not None else base.chi_min,
        chi0=args.chi0 if args.chi0 is not None else base.chi0,
        rho=args.rho if args.rho is not None else base.rho,
        z_acc_obs=base.z_acc_obs,   # UniverseFromBH doesn’t fit to obs; keep canon anchor
        z_acc_tol=base.z_acc_tol,
    )


def main() -> None:
    args = build_parser().parse_args()
    canon = make_canon(args)

    chi = float(args.chi)
    n = args.n
    channel = args.channel
    r0 = args.lambda_ratio
    eta = float(args.eta)

    kerr_ok = kerr_viable(chi, canon)

    zacc_pred = z_acc_pred_from_bh_inputs(chi=chi, n=n, r0=r0, eta=eta, canon=canon)
    relic_expect = relic_band_from_zacc(zacc_pred)

    if args.infer_n:
        if args.zacc is None or args.zacc_tol is None:
            raise ValueError("--infer-n requires --zacc and --zacc-tol")
        if args.lambda_ratio is None:
            raise ValueError("--infer-n requires --lambda-ratio (r0)")
        nmin, nmax, ns = infer_n_range_from_zacc(args.zacc, args.zacc_tol, args.lambda_ratio, args.eta, args.n_max)
        if ns:
            print(f"\nInferred n range matching z_acc={args.zacc:.3f}±{args.zacc_tol:.3f}: n={nmin}..{nmax}")
            print(f"Representative n values: {ns[:10]}{' ...' if len(ns)>10 else ''}")
        else:
            print(f"\nNo n in [0,{args.n_max}] matches z_acc={args.zacc:.3f}±{args.zacc_tol:.3f} for r0={args.lambda_ratio} eta={args.eta}")

    # Parity compatibility: Pi=1 requires Kerr viability
    Pi1_compatible = bool(kerr_ok)

    payload: Dict[str, Any] = {
        "canon": asdict(canon),
        "inputs": {"chi": chi, "channel": channel, "n": n, "lambda_ratio_r0": r0, "eta": eta},
        "outputs": {
            "kerr_viable": kerr_ok,
            "z_acc_pred": zacc_pred,
            "relic_expectation": relic_expect,
            "parity_Pi1_compatible": Pi1_compatible,
            "archetype": archetype(channel, kerr_ok),
            "relic_thresholds_locked": {
                "high": "z_acc_pred >= 0.8",
                "medium": "0.5 <= z_acc_pred < 0.8",
                "low": "z_acc_pred < 0.5 (or z_acc_pred is None)",
            },
        },
    }
    
    dm = predict_dm_band(channel=channel, n=n, relic=None, Pi=1 if Pi1_compatible else 0)

    payload["outputs"].update(
        {
            "dm_band": dm.dm_band,
            "dm_ratio_band": {"min": dm.dm_ratio_band[0], "max": dm.dm_ratio_band[1]},  # Omega_DM/Omega_b
            "dm_composition_hint": dm.dm_composition_hint,
            "macro_survival_hint": dm.macro_survival_hint,
            "dm_rationale": dm.dm_rationale,
        }
    )    

    stage = ladder_stage_label(n) if n is not None else "ladder position not specified (n not provided)"
    bh_line = bh_condition_summary(chi, canon.chi_min)

    bullets = [
        f"Ladder: {stage}.",
        bh_line,
    ]
    bullets += channel_story(channel)
    bullets += universe_condition_summary(
        z_acc_pred=zacc_pred,
        dm_band=dm.dm_band,
        dm_ratio_band=dm.dm_ratio_band,
        relic_expect=None,               # UniverseFromBH doesn't take relic as input
        Pi1_compatible=Pi1_compatible,
    )

    print(format_interpretation_block("Interpretation (human-readable)", bullets))

    payload["outputs"]["interpretation"] = bullets

    # Human-readable summary
    print("\nUniverseFromBH — Predicted Outcome Class")
    print("--------------------------------------")
    print(f"Input BH spin chi = {chi:.3f}")
    if n is not None:
        print(f"Generation index n = {n} (n=0 is Base-1)")
    print(f"Birth channel = {channel!r}")
    print("")
    print(f"Kerr viable (chi >= {canon.chi_min})?  {kerr_ok}")

    if r0 is None:
        if kerr_ok:
            print("Acceleration: possible in principle (Kerr-enabled), but z_acc not computed (no --lambda-ratio supplied).")
        else:
            print("Acceleration: not possible under Kerr gate (terminal/no acceleration).")
    else:
        if zacc_pred is None:
            print("Acceleration: not possible under Kerr gate or non-positive lambda ratio (terminal/no acceleration).")
        else:
            if n is None:
                print(f"Acceleration: z_acc_pred = {zacc_pred:.3f} (from r0={r0:.3f}).")
            else:
                print(f"Acceleration: z_acc_pred = {zacc_pred:.3f} (from r0={r0:.3f}, eta={eta:.3f}, n={n}).")

    print(f"Relic expectation band: {relic_expect!r}  (locked thresholds)")
    print(f"Parity (Pi=1) compatible?  {Pi1_compatible}")
    print(f"Archetype: {payload['outputs']['archetype']}")
    print("\nDark matter (toy output band):")
    print(f"  dm_band = {dm.dm_band!r}")
    print(f"  Omega_DM/Omega_b ∈ [{dm.dm_ratio_band[0]:.1f}, {dm.dm_ratio_band[1]:.1f}]")
    print(f"  composition: {dm.dm_composition_hint}")
    if dm.macro_survival_hint != "neutral":
        print(f"  macro survival hint: {dm.macro_survival_hint}")
    print(f"  rationale: {dm.dm_rationale}")
    
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON output to: {args.json_out}")


if __name__ == "__main__":
    main()
