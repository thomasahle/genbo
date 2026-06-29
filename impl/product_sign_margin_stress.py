"""Stress the product-sign posterior-rank margin target.

The frozen finite-channel target reduces product-sign rank margins to the bit
score

    Psi_s(x) = sum_a s_a u_a(x) - log cosh u_a(x).

This script tests that target in the ideal Gaussian bit-logit model.  It keeps
the normal form fixed and reports whether the top-eta score quantile is already
too close to the ceiling r log 2 for a fixed margin to be possible.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Callable

import numpy as np


FIELDNAMES = [
    "m",
    "c",
    "eta",
    "alpha",
    "r",
    "log10_B",
    "sigma",
    "corr",
    "margin",
    "quantile_samples",
    "pair_trials",
    "label_samples",
    "q_eta",
    "ceiling",
    "ceiling_gap",
    "near_ceiling_prob",
    "near_ceiling_lower",
    "margin_blocked_empirical",
    "margin_blocked_lower",
    "score_level",
    "mean_gap_q01",
    "mean_gap_q05",
    "mean_gap_median",
    "tilted_variance_q95",
    "cantelli_good_q01",
    "cantelli_good_q05",
    "cantelli_bound_over_alpha_q05",
    "good_mass_q01",
    "good_mass_q05",
    "good_mass_median",
    "affinity_q05",
    "bound_q05",
    "bound_over_alpha_q05",
]


def parse_csv_list(text: str, cast: Callable[[str], object] = str) -> list[object]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    if not out:
        raise ValueError("list must contain at least one item")
    return out


def log_cosh(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.logaddexp(x, -x) - math.log(2.0)


def bit_score(z: np.ndarray) -> np.ndarray:
    """Return z - log cosh(z), stably, with ceiling log 2."""
    z = np.asarray(z, dtype=float)
    return z - np.logaddexp(z, -z) + math.log(2.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _normal_upper_tail(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def ceiling_gap_lower_tail(*, r_bits: int, sigma: float, gap: float) -> float:
    """A conservative lower bound for Pr[Psi >= r log 2 - gap].

    If every fixed-label Gaussian bit logit is at least

        t = -1/2 log(exp(gap / r) - 1),

    then each bit loses at most gap / r from its log-2 ceiling.  Independence
    gives the displayed lower bound.  This is intentionally one-sided: when it
    already exceeds eta, the fixed-margin certificate is provably blocked.
    """
    if r_bits < 1:
        raise ValueError("r_bits must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if gap <= 0:
        return 0.0
    per_bit = math.expm1(gap / r_bits)
    if not math.isfinite(per_bit) or per_bit <= 0:
        return 0.0
    threshold = -0.5 * math.log(per_bit)
    return _normal_upper_tail(threshold / sigma) ** r_bits


def score_quantile(
    *,
    r_bits: int,
    sigma: float,
    eta: float,
    samples: int,
    margin: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("samples must be positive")
    z = sigma * rng.standard_normal((samples, r_bits))
    scores = bit_score(z).sum(axis=1)
    q_eta = float(np.quantile(scores, 1.0 - eta))
    near_ceiling = float(np.mean(scores >= r_bits * math.log(2.0) - margin))
    return q_eta, near_ceiling


def _log_affinity(up: np.ndarray, uq: np.ndarray, alpha: float) -> np.ndarray:
    z = alpha * up + (1.0 - alpha) * uq
    return np.sum(log_cosh(z) - alpha * log_cosh(up) - (1.0 - alpha) * log_cosh(uq), axis=1)


def tilted_score_mean_variance(
    up: np.ndarray,
    uq: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conditional mean and variance of Psi_s(p) under tilted labels."""
    if up.shape != uq.shape:
        raise ValueError("up and uq must have the same shape")
    tilt = alpha * up + (1.0 - alpha) * uq
    mean_sign = np.tanh(tilt)
    mean = np.sum(up * mean_sign - log_cosh(up), axis=1)
    variance = np.sum(up * up * (1.0 - mean_sign * mean_sign), axis=1)
    return mean, variance


def cantelli_good_mass_lower_bound(
    mean: np.ndarray,
    variance: np.ndarray,
    threshold: float,
) -> np.ndarray:
    gap = mean - threshold
    positive = gap > 0.0
    out = np.zeros_like(mean, dtype=float)
    out[positive] = (gap[positive] * gap[positive]) / (
        variance[positive] + gap[positive] * gap[positive])
    return out


def tilted_good_mass_samples(
    *,
    up: np.ndarray,
    uq: np.ndarray,
    alpha: float,
    threshold: float,
    label_samples: int,
    rng: np.random.Generator,
    chunk_size: int = 128,
) -> np.ndarray:
    if label_samples < 1:
        raise ValueError("label_samples must be positive")
    if up.shape != uq.shape:
        raise ValueError("up and uq must have the same shape")

    out = np.empty(up.shape[0], dtype=float)
    log_cosh_up = log_cosh(up)
    for start in range(0, up.shape[0], chunk_size):
        stop = min(start + chunk_size, up.shape[0])
        up_chunk = up[start:stop]
        tilt = alpha * up_chunk + (1.0 - alpha) * uq[start:stop]
        prob_plus = _sigmoid(2.0 * tilt)
        signs = np.where(
            rng.random((stop - start, label_samples, up.shape[1])) < prob_plus[:, None, :],
            1.0,
            -1.0,
        )
        scores = np.sum(signs * up_chunk[:, None, :] - log_cosh_up[start:stop, None, :], axis=2)
        out[start:stop] = np.mean(scores >= threshold, axis=1)
    return out


def run_trial(
    *,
    m: int,
    c: float,
    r_bits: int,
    sigma: float,
    corr: float,
    margin: float,
    quantile_samples: int,
    pair_trials: int,
    label_samples: int,
    seed: int,
) -> dict[str, object]:
    if m < 3:
        raise ValueError("m must be at least 3")
    if c <= 1:
        raise ValueError("c must be greater than 1")
    if r_bits < 1:
        raise ValueError("r_bits must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not (-1.0 < corr < 1.0):
        raise ValueError("corr must lie in (-1, 1)")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if pair_trials < 1:
        raise ValueError("pair_trials must be positive")

    rng = np.random.default_rng(seed)
    eta = m ** (-1.0 / (2.0 * c * c))
    alpha = 1.0 / math.log(m)
    ceiling = r_bits * math.log(2.0)
    q_eta, near_ceiling_prob = score_quantile(
        r_bits=r_bits,
        sigma=sigma,
        eta=eta,
        samples=quantile_samples,
        margin=margin,
        rng=rng,
    )
    near_ceiling_lower = ceiling_gap_lower_tail(r_bits=r_bits, sigma=sigma, gap=margin)
    threshold = q_eta + margin

    up = sigma * rng.standard_normal((pair_trials, r_bits))
    noise = sigma * rng.standard_normal((pair_trials, r_bits))
    uq = corr * up + math.sqrt(1.0 - corr * corr) * noise
    good_mass = tilted_good_mass_samples(
        up=up,
        uq=uq,
        alpha=alpha,
        threshold=threshold,
        label_samples=label_samples,
        rng=rng,
    )
    tilted_mean, tilted_variance = tilted_score_mean_variance(up, uq, alpha)
    mean_gap = tilted_mean - threshold
    cantelli_good = cantelli_good_mass_lower_bound(tilted_mean, tilted_variance, threshold)
    affinity = np.exp(_log_affinity(up, uq, alpha))
    margin_factor = 1.0 - math.exp(-alpha * margin)
    bound = margin_factor * affinity * good_mass
    cantelli_bound = margin_factor * affinity * cantelli_good

    return {
        "m": m,
        "c": c,
        "eta": eta,
        "alpha": alpha,
        "r": r_bits,
        "log10_B": r_bits * math.log10(2.0),
        "sigma": sigma,
        "corr": corr,
        "margin": margin,
        "quantile_samples": quantile_samples,
        "pair_trials": pair_trials,
        "label_samples": label_samples,
        "q_eta": q_eta,
        "ceiling": ceiling,
        "ceiling_gap": ceiling - q_eta,
        "near_ceiling_prob": near_ceiling_prob,
        "near_ceiling_lower": near_ceiling_lower,
        "margin_blocked_empirical": float(near_ceiling_prob >= eta),
        "margin_blocked_lower": float(near_ceiling_lower >= eta),
        "score_level": threshold / r_bits,
        "mean_gap_q01": float(np.quantile(mean_gap, 0.01)),
        "mean_gap_q05": float(np.quantile(mean_gap, 0.05)),
        "mean_gap_median": float(np.median(mean_gap)),
        "tilted_variance_q95": float(np.quantile(tilted_variance, 0.95)),
        "cantelli_good_q01": float(np.quantile(cantelli_good, 0.01)),
        "cantelli_good_q05": float(np.quantile(cantelli_good, 0.05)),
        "cantelli_bound_over_alpha_q05": float(np.quantile(cantelli_bound / alpha, 0.05)),
        "good_mass_q01": float(np.quantile(good_mass, 0.01)),
        "good_mass_q05": float(np.quantile(good_mass, 0.05)),
        "good_mass_median": float(np.median(good_mass)),
        "affinity_q05": float(np.quantile(affinity, 0.05)),
        "bound_q05": float(np.quantile(bound, 0.05)),
        "bound_over_alpha_q05": float(np.quantile(bound / alpha, 0.05)),
    }


def write_rows(rows: list[dict[str, object]], output: str | None) -> None:
    target = open(output, "w", newline="") if output else sys.stdout
    try:
        writer = csv.DictWriter(target, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if output:
            target.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress the ideal Gaussian product-sign bounded-margin target.")
    parser.add_argument("--m", default="1000000,1000000000,1000000000000")
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--r", default="4,8,12,16,24,32")
    parser.add_argument("--sigma", default="3.5")
    parser.add_argument("--corr", type=float, default=0.9)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--quantile-samples", type=int, default=200000)
    parser.add_argument("--pair-trials", type=int, default=1000)
    parser.add_argument("--label-samples", type=int, default=512)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    rows = []
    for m in parse_csv_list(args.m, int):
        for sigma in parse_csv_list(args.sigma, float):
            for r_bits in parse_csv_list(args.r, int):
                for seed in parse_csv_list(args.seeds, int):
                    rows.append(run_trial(
                        m=m,
                        c=args.c,
                        r_bits=r_bits,
                        sigma=sigma,
                        corr=args.corr,
                        margin=args.margin,
                        quantile_samples=args.quantile_samples,
                        pair_trials=args.pair_trials,
                        label_samples=args.label_samples,
                        seed=seed,
                    ))
    write_rows(rows, args.csv)


if __name__ == "__main__":
    main()
