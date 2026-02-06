#!/usr/bin/env bash
set -e

echo "BHfromUniverse baseline..."
python BHfromUniverse.py --z-acc-obs 0.67 --z-acc-tol 0.10 --relic medium

echo "BHfromUniverse baseline + parity=1..."
python BHfromUniverse.py --z-acc-obs 0.67 --z-acc-tol 0.10 --relic medium --parity 1

echo "UniverseFromBH n=0..."
python UniverseFromBH.py --n 0 --chi0 0.7 --rho 0.9 --chi-min 0.2 --z-acc-obs 0.67 --z-acc-tol 0.10 --relic medium

echo "UniverseFromBH n=3..."
python UniverseFromBH.py --n 3 --chi0 0.7 --rho 0.9 --chi-min 0.2 --z-acc-obs 0.67 --z-acc-tol 0.10 --relic medium

echo "UniverseFromBH n=13 (terminal check)..."
python UniverseFromBH.py --n 13 --chi0 0.7 --rho 0.9 --chi-min 0.2 --z-acc-obs 0.67 --z-acc-tol 0.10 --relic medium

echo "Done."
