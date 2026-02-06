# shared_help.py
"""
Shared help text for BHC+QCD=TOE scripts.

Designed to be used with argparse.RawTextHelpFormatter (or a combined Raw+Defaults formatter).
"""

COMMON_HELP = r"""
BHC+QCD=TOE — Shared Parameter Guide (applies to BHfromUniverse and UniverseFromBH)

What these tools are:
  These scripts are *toy-model consistency and exploration tools* for the BHC+QCD=TOE framework.
  They do NOT identify a unique origin, do NOT perform precision fitting, and do NOT claim
  any parameter is measured unless explicitly provided as an input.

Core ladder concepts (shared):
  n (generation index):
    n=0 corresponds to a Base-1 universe (first interior universe seeded by a Base-0 BH).
    Larger n means later generations further down the ladder.

  chi (dimensionless BH spin, 0..1):
    chi controls whether Kerr-enabled inheritance/boundary effects are active.

  Spin inheritance law:
    chi_n = chi0 * rho^n
    - chi0 : seed spin at n=0
    - rho  : per-generation spin retention (0<rho<1 for decay)

  Kerr gate (terminality threshold):
    If chi_n < chi_min, the model treats the universe as "terminal":
      - boundary-enabled acceleration is absent
      - z_acc_pred is undefined (None)
      - parity Pi=1 is incompatible
    chi_min is a toy threshold encoding the idea that below some rotation,
    Kerr-enabled effects are too weak to matter.

Acceleration-onset observable (shared):
  z_acc:
    Redshift when cosmic acceleration begins (ddot(a)=0). This is distinct from
    the epoch when dark energy density exceeds matter density.

  Canonical observational anchor (used in BHfromUniverse consistency tests):
    z_acc_obs = 0.67 ± 0.10 (conservative; not precision fitting)

Optional population/structure discriminator (shared):
  relic (low | medium | high):
    Qualitative indicator of how many relic (early-formed, compact) galaxies survive to late times.
    This acts as a proxy for how strongly late-time structure growth was suppressed.

    LOCKED coarse thresholds (modest-weight discriminator; not a precision constraint):
      - relic="high"   expects z_acc_pred >= 0.8
      - relic="medium" expects 0.5 <= z_acc_pred < 0.8
      - relic="low"    expects z_acc_pred < 0.5
    Edge case:
      - If z_acc_pred is None (no acceleration), relic is treated as "low".

Optional parity discriminator (shared):
  Pi (0 | 1):
    Boolean indicator for whether early galaxies show a preferred rotation direction.
      Pi=1 : preferred handedness detected (parity signal)
      Pi=0 : absent/not used

    Consistency rule:
      - If Pi=1, Kerr viability is required (chi or chi_n >= chi_min).
      - If Pi=0, no penalty is applied (absence is not evidence against Kerr).

Optional Dark Energy amplitude knobs (shared; advanced):
  These knobs allow computing a numeric z_acc_pred without inventing units.

  lambda-ratio (r0):
    r0 = (boundary-sourced vacuum-like energy) / (ordinary matter energy)  [dimensionless]
    Intuition:
      - small r0  → matter dominates longer → acceleration late or absent
      - r0 ~ 2–3  → our-universe-like onset z ~ 0.5–1
      - large r0  → early acceleration → strong suppression of structure growth

    Mapping used when provided:
      z_acc = (2 r)^(1/3) - 1

    Reference point:
      r0 ≈ Omega_Lambda/Omega_m ≈ 0.7/0.3 ≈ 2.33 → z_acc ≈ 0.67

  eta:
    Per-generation weakening factor for the DE amplitude:
      r(n) = r0 * eta^n   (used only when n is provided)
    eta=1.0 means no weakening with generation; smaller eta makes acceleration drift
    to lower redshift more rapidly at larger n.

Typical ranges (shared):
  chi, chi0, chi_min in [0,1]
  rho in (0,1) for decay-per-generation scenarios
  eta in (0,1]
  n is an integer >= 0
"""
