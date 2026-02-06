# Samples: Our Universe (BHfromUniverse ↔ UniverseFromBH)

This file provides canonical sample runs that reproduce “our universe” constraints
in both directions using the BHUT+QCD=TOE ladder diagnostics.

Locked canonical assumptions:
- Kerr viability threshold: chi_min = 0.2
- Typical seed spin: chi0 = 0.7
- Per-generation spin retention: rho = 0.9
- Mergers count as new-universe birth events (Option B)
- Observational anchor: z_acc_obs = 0.67 with tolerance ±0.10
- Relic-consistency mapping:
  - relic="high"   expects z_acc_pred ≥ 0.8
  - relic="medium" expects 0.5 ≤ z_acc_pred < 0.8
  - relic="low"    expects z_acc_pred < 0.5

Notes:
- “Our universe” is treated as z_acc ≈ 0.67, relic="medium".
- Parity/handedness indicator is optional; set it only if you want to force Kerr-enabled solutions.

---

## A) BHfromUniverse — infer candidate host BHs for our universe

### A1. Baseline: our universe (no parity constraint)
Expected: allowed solutions should include n ≈ 0 with chi ≈ 0.7 and z_acc_pred ≈ 0.67.

Command:
python BHfromUniverse.py \
  --z-acc-obs 0.67 --z-acc-tol 0.10 \
  --relic medium

Interpretation:
- Output rows represent (n, chi_n, z_acc_pred) combinations consistent with the observational anchor.
- For relic="medium", solutions with z_acc_pred ~ 0.5–0.8 are preferred.

### A2. Baseline + parity present (forces Kerr-enabled)
Use this if you want to require inherited spin imprint consistency (Pi=1).
Expected: same band as A1, but terminal/near-terminal solutions filtered out.

Command:
python BHfromUniverse.py \
  --z-acc-obs 0.67 --z-acc-tol 0.10 \
  --relic medium \
  --parity 1

### A3. Sensitivity: slightly earlier onset (upper edge of tolerance)
Expected: favors smaller n (earlier ladder) and/or larger effective r.

Command:
python BHfromUniverse.py \
  --z-acc-obs 0.77 --z-acc-tol 0.05 \
  --relic medium

---

## B) UniverseFromBH — infer example universes consistent with our observations

### B1. Canonical host BH: base generation
This is “our universe as a base-1 interior” reference case.
Expected: z_acc_pred should be ~0.67 when the effective vacuum ratio matches the anchor.

Command:
python UniverseFromBH.py \
  --n 0 \
  --chi0 0.7 --rho 0.9 --chi-min 0.2 \
  --z-acc-obs 0.67 --z-acc-tol 0.10 \
  --relic medium

Interpretation:
- UniverseFromBH should report a Kerr-enabled universe and return diagnostic outputs
  consistent with our anchor band.

### B2. “Slightly down the ladder but still viable”
This probes whether our observed z_acc could occur at modest n if the model allows it.
Expected: still Kerr-enabled; depending on eta (if used), may drift z_acc downward.

Command:
python UniverseFromBH.py \
  --n 3 \
  --chi0 0.7 --rho 0.9 --chi-min 0.2 \
  --z-acc-obs 0.67 --z-acc-tol 0.10 \
  --relic medium

### B3. Terminal check (should fail Kerr viability)
Expected: classified terminal OR “no acceleration” / filtered-out depending on script behavior.

Command:
python UniverseFromBH.py \
  --n 13 \
  --chi0 0.7 --rho 0.9 --chi-min 0.2 \
  --z-acc-obs 0.67 --z-acc-tol 0.10 \
  --relic medium

---

## C) “Golden outputs” (qualitative)

These runs are considered correct if:
- BHfromUniverse includes an allowed row near:
  - n ≈ 0, chi ≈ 0.7, z_acc_pred ≈ 0.67
- UniverseFromBH for n=0 reports:
  - Kerr-enabled (chi >= chi_min)
  - z_acc_pred within [0.57, 0.77]
  - relic consistency: “medium” compatible
- UniverseFromBH for n=13 reports:
  - terminal (chi < chi_min) and no acceleration
