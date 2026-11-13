#!/usr/bin/env python3
# polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, standard library only).
#
# Forecast: for each market of Polymarket's "Next CEO of NVIDIA?" event, the
# Yes price at every minute of the 240 minutes after the checkpoint, as five
# quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of that minute's price.
#
# Method (measured on the six practice cases with a local scorer built from
# Polymarket's public price history; the scorer reproduces the house
# baseline's published practice loss of 0.363589 exactly):
#
#   * Persistence median: each market's price stays at its checkpoint price.
#     The pre-checkpoint returns do NOT predict the next 240 minutes here:
#     they are negatively correlated with the realized drift (mean reversion),
#     and every drift term tried made the practice loss worse, so no drift.
#   * Spread: a random-walk spread, sigma * sqrt(h), where sigma is the
#     per-market standard deviation of one-minute price changes over the
#     last 120 minutes, winsorized at +-0.01. Winsorizing matters: a single
#     one-minute jump (Colette Kress 0.3535 -> 0.59 in a minute) otherwise
#     inflates the plain 6h/24h/72h standard deviation by an order of
#     magnitude, and long windows are poisoned by old regimes (markets that
#     opened near 0.5 and decayed). Floored at 0.0003, capped at 0.02.
#   * Cross-market normalization: the Yes prices of mutually exclusive
#     outcomes should sum to about 1. At each horizon minute the whole
#     forecast row of every market is scaled so the medians sum to a target
#     halfway between the checkpoint sum and 1.0. The realized sum reverts
#     toward 1 in most practice cases; halfway hedges the cases where it
#     does not. This is the one cross-market term; it improves the median
#     error as well as the loss.
#
# Practice loss (local scorer, mean pinball loss in percentage points):
#   this entry 0.271567 vs house baseline 0.363589 (-25%), better on all
#   six practice cases (p00 0.4182, p01 0.3102, p02 0.3464, p03 0.2626,
#   p04 0.2075, p05 0.1383 vs baseline 0.5361, 0.3804, 0.4084, 0.3686,
#   0.2988, 0.2382). Variants tried and rejected on the same scorer:
#   plain 6h/24h/72h sigma (worse than baseline), EWMA sigma (0.486),
#   median-absolute-deviation sigma (0.332), any drift term (linear or
#   exponentially-decaying mean reversion, 0.280-1.983), full normalization
#   to sum 1.0 (0.278, wins one case big, loses another), wider outer
#   quantiles (no gain), sub-diffusive spread growth (no gain).
#
# The program is deterministic: no network, no clock, no randomness. Two
# runs on the same case write byte-identical output.
#
# Usage: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# (reads $MAKEFUN_INPUT_DIR, else /input; writes $MAKEFUN_OUTPUT_DIR, else
# /output; one <id>.out per <id>.in, in name order).

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

# The quantile levels the contest scores, in the order each row lists them.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level, to 6 decimals (as the
# house baseline uses them).
Z_TABLE = {0.05: -1.644854, 0.25: -0.67449, 0.5: 0.0, 0.75: 0.67449,
           0.95: 1.644854}
# One-minute changes are winsorized at +-WINSOR before the sigma estimate:
# a single jump must not set a market's spread for the next four hours.
WINSOR = 0.01
# How many minutes of one-minute changes feed the sigma estimate.
SIGMA_WINDOW = 120
# Floor and cap on the per-minute sigma, in price units.
SIGMA_FLOOR = 0.0003
SIGMA_CAP = 0.02
# Cross-market normalization: medians are scaled toward this blend of the
# checkpoint sum and 1.0 (0.5 = halfway).
NORM_BLEND = 0.5


def norm_ppf(p):
    """Standard normal quantile (Acklam's approximation); only used for
    levels outside the fixed table, so the five scored levels stay exact."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("quantile level out of (0, 1): %r" % p)
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def z_for(tau):
    return Z_TABLE.get(tau, norm_ppf(tau))


def sigma_winsorized(prices):
    """Per-minute sigma: sample std of the one-minute price changes over
    the last SIGMA_WINDOW minutes, each change winsorized to +-WINSOR."""
    n = len(prices)
    if n < 2:
        return SIGMA_FLOOR
    w = prices[-(SIGMA_WINDOW + 1):] if n > SIGMA_WINDOW + 1 else prices
    d = [w[i + 1] - w[i] for i in range(len(w) - 1)]
    d = [max(-WINSOR, min(WINSOR, x)) for x in d]
    m = len(d)
    if m < 2:
        return SIGMA_FLOOR
    mu = sum(d) / m
    var = sum((x - mu) * (x - mu) for x in d) / (m - 1)
    sig = math.sqrt(var)
    return max(SIGMA_FLOOR, min(SIGMA_CAP, sig))


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)), 6) + 0.0


def forecast_case(case):
    horizon = case["horizon_minutes"]
    taus = case["quantiles"]
    zs = [z_for(t) for t in taus]
    markets = case["markets"]
    # First pass: persistence median with a winsorized random-walk spread.
    per_market = []
    for m in markets:
        prices = m["prices"]
        if not prices:
            raise ValueError("market %s has no prices" % m["id"])
        x = prices[-1]
        sig = sigma_winsorized(prices)
        rows = []
        for h in range(1, horizon + 1):
            sp = sig * math.sqrt(h)
            rows.append([x + z * sp for z in zs])
        per_market.append({"id": m["id"], "q": rows})
    # Cross-market normalization: scale every market's row at each horizon
    # minute so the medians sum to a target halfway between the checkpoint
    # sum and 1.0 (the outcomes are mutually exclusive, so the Yes prices
    # should sum to about 1; the realized sum reverts toward 1 in most
    # practice cases).
    # The median is the quantile level closest to 0.5 (the middle one for
    # the five scored levels); do not assume 0.5 is listed exactly, so a
    # case with different levels still forecasts instead of crashing.
    mid = min(range(len(taus)), key=lambda i: abs(taus[i] - 0.5))
    s_at = sum(m["prices"][-1] for m in markets)
    target = s_at + NORM_BLEND * (1.0 - s_at)
    for h in range(horizon):
        s = sum(pm["q"][h][mid] for pm in per_market)
        if s > 0:
            f = target / s
            if f != 1.0:
                for pm in per_market:
                    pm["q"][h] = [q * f for q in pm["q"][h]]
    # Clamp, round, enforce non-decreasing quantiles.
    out = []
    for pm in per_market:
        rows = []
        for row in pm["q"]:
            r = []
            prev = 0.0
            for q in row:
                v = price6(q)
                if v < prev:
                    v = prev
                r.append(v)
                prev = v
            rows.append(r)
        out.append({"id": pm["id"], "q": rows})
    return {"forecasts": out}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        cid = fname[:-len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        fc = forecast_case(case)
        with open(os.path.join(OUTPUT_DIR, cid + ".out"), "w", encoding="utf-8") as f:
            json.dump(fc, f)
            f.write("\n")


if __name__ == "__main__":
    main()
