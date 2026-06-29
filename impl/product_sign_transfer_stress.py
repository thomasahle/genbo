"""Stress transfer from ideal Gaussian bits to actual product-sign panels.

The ideal product-sign argument is phrased in the fixed normal form

    Psi_s(x) = sum_a s_a u_a(x) - log cosh u_a(x).

This diagnostic keeps that normal form and measures the missing transfer
facts for the concrete panels used by the finite-channel experiments: fitted
bit variance, cross-bit dependence, near-pair bit correlation, fixed-label
score thresholds, and tilted bounded-margin good mass.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Callable

import numpy as np

from finite_channel_diagnostic import (
    _data_whitening,
    is_power_of_two,
    make_product_sign_basis,
    product_sign_labels,
    synthetic_near_pairs,
)
from product_sign_margin_stress import (
    gaussian_rate_gap_summary,
    log_cosh,
    score_quantile,
    tilted_good_mass_samples,
    tilted_score_mean_variance,
)


FIELDNAMES = [
    "n",
    "d",
    "c",
    "queries",
    "near_corr",
    "panel",
    "scale",
    "bit_sigma_target",
    "seed",
    "B",
    "r_bits",
    "eta",
    "alpha",
    "margin",
    "ref_samples",
    "threshold_labels",
    "tilted_label_samples",
    "enumerated_thresholds",
    "bit_mean_abs_max",
    "bit_var_mean",
    "bit_var_q05",
    "bit_var_q95",
    "bit_sigma_fit",
    "ref_cov_rel_op",
    "ref_cov_lambda_min_rel",
    "ref_cov_lambda_max_rel",
    "cross_corr_abs_q95",
    "cross_corr_abs_max",
    "metric_sigma_near_mean",
    "metric_sigma_query_mean",
    "metric_pair_corr_mean",
    "metric_pair_corr_q05",
    "metric_pair_corr_q95",
    "pair_corr_mean",
    "pair_corr_q05",
    "pair_corr_q95",
    "ideal_q_eta",
    "threshold_ref_median",
    "threshold_ref_q95",
    "threshold_ref_max",
    "threshold_node_q95",
    "threshold_ref_node_abs_q50",
    "threshold_ref_node_abs_q90",
    "threshold_q95_minus_ideal",
    "threshold_max_minus_ideal",
    "tilted_mean_q05",
    "tilted_variance_q95",
    "mean_gap_q01",
    "mean_gap_q05",
    "cantelli_good_q01",
    "cantelli_good_q05",
    "sampled_good_mass_q01",
    "sampled_good_mass_q05",
    "exact_good_mass_q01",
    "exact_good_mass_q05",
    "affinity_q05",
    "sampled_bound_over_alpha_q05",
    "exact_bound_over_alpha_q05",
    "cantelli_bound_over_alpha_q05",
    "ideal_score_level",
    "actual_score_level_q95",
    "score_level_gap_q95",
    "gaussian_rate",
    "min_log_b_exponent",
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


def sphere_points(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, d))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def upper_tail_thresholds(scores: np.ndarray, eta: float) -> np.ndarray:
    """Return the empirical top-eta lower endpoint for each score column."""
    if not (0.0 < eta <= 1.0):
        raise ValueError("eta must lie in (0, 1]")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be a two-dimensional array")
    n = scores.shape[0]
    if n < 1:
        raise ValueError("scores must have at least one row")
    top_k = max(1, int(math.ceil(eta * n)))
    kth_index = n - top_k
    return np.partition(scores, kth_index, axis=0)[kth_index]


def _label_sample(
    *,
    b_count: int,
    r_bits: int,
    label_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    if label_samples < 1:
        raise ValueError("label_samples must be positive")
    if b_count <= label_samples:
        return product_sign_labels(b_count), True
    return rng.choice([-1.0, 1.0], size=(label_samples, r_bits)), False


def _score_matrix(bit_logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return bit_logits @ labels.T - np.sum(log_cosh(bit_logits), axis=1, keepdims=True)


def _abs_offdiag_correlations(bit_logits: np.ndarray) -> np.ndarray:
    if bit_logits.shape[1] <= 1:
        return np.array([0.0])
    centered = bit_logits - np.mean(bit_logits, axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(centered), 1)
    var = np.maximum(np.diag(cov), 1e-300)
    corr = cov / np.sqrt(var[:, None] * var[None, :])
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return np.abs(corr[mask])


def covariance_spectral_summary(bit_logits: np.ndarray) -> tuple[float, float, float]:
    """Return relative operator deviation and extremal eigenvalue ratios."""
    bit_logits = np.asarray(bit_logits, dtype=float)
    if bit_logits.ndim != 2 or bit_logits.shape[1] < 1:
        raise ValueError("bit_logits must be a nonempty two-dimensional array")
    centered = bit_logits - np.mean(bit_logits, axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(centered), 1)
    target = float(np.trace(cov) / cov.shape[0])
    if target <= 0.0:
        return 0.0, 1.0, 1.0
    vals = np.linalg.eigvalsh(cov)
    rel_min = float(vals[0] / target)
    rel_max = float(vals[-1] / target)
    rel_op = max(abs(rel_min - 1.0), abs(rel_max - 1.0))
    return rel_op, rel_min, rel_max


def _column_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError("left and right must have the same shape")
    left_centered = left - np.mean(left, axis=0, keepdims=True)
    right_centered = right - np.mean(right, axis=0, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=0)
    left_norm = np.sum(left_centered * left_centered, axis=0)
    right_norm = np.sum(right_centered * right_centered, axis=0)
    denom = np.sqrt(np.maximum(left_norm * right_norm, 1e-300))
    return numerator / denom


def gaussian_row_pair_parameters(
    left: np.ndarray,
    right: np.ndarray,
    *,
    transform: np.ndarray,
    bit_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact one-row Gaussian law for transformed bit logits.

    If a basis row is ``g @ transform / sqrt(d)`` with ``g`` standard normal,
    then the bit logits of ``left`` and ``right`` are centered Gaussian with
    these standard deviations and correlation.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    transform = np.asarray(transform, dtype=float)
    if left.shape != right.shape:
        raise ValueError("left and right must have the same shape")
    if left.ndim != 2 or transform.shape != (left.shape[1], left.shape[1]):
        raise ValueError("transform must be square with dimension matching points")
    if bit_scale <= 0:
        raise ValueError("bit_scale must be positive")

    left_features = left @ transform.T
    right_features = right @ transform.T
    scale = bit_scale / math.sqrt(left.shape[1])
    left_norm = np.linalg.norm(left_features, axis=1)
    right_norm = np.linalg.norm(right_features, axis=1)
    sigma_left = scale * left_norm
    sigma_right = scale * right_norm
    denom = np.maximum(left_norm * right_norm, 1e-300)
    corr = np.sum(left_features * right_features, axis=1) / denom
    return sigma_left, sigma_right, corr


def _log_affinity(up: np.ndarray, uq: np.ndarray, alpha: float) -> np.ndarray:
    tilt = alpha * up + (1.0 - alpha) * uq
    return np.sum(
        log_cosh(tilt) - alpha * log_cosh(up) - (1.0 - alpha) * log_cosh(uq),
        axis=1,
    )


def exact_tilted_good_mass(
    *,
    up: np.ndarray,
    uq: np.ndarray,
    alpha: float,
    labels: np.ndarray,
    thresholds: np.ndarray,
    margin: float,
    chunk_size: int = 128,
) -> np.ndarray:
    """Return exact tilted good mass when ``labels`` enumerates all signs."""
    if up.shape != uq.shape:
        raise ValueError("up and uq must have the same shape")
    if labels.ndim != 2 or labels.shape[1] != up.shape[1]:
        raise ValueError("labels must have one column per bit")
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape != (labels.shape[0],):
        raise ValueError("thresholds must have one entry per label")
    if margin < 0:
        raise ValueError("margin must be nonnegative")

    out = np.empty(up.shape[0], dtype=float)
    for start in range(0, up.shape[0], chunk_size):
        stop = min(start + chunk_size, up.shape[0])
        up_chunk = up[start:stop]
        tilt = alpha * up_chunk + (1.0 - alpha) * uq[start:stop]
        logits = tilt @ labels.T
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=1, keepdims=True)
        scores = _score_matrix(up_chunk, labels)
        good = scores >= thresholds[None, :] + margin
        out[start:stop] = np.sum(np.where(good, weights, 0.0), axis=1)
    return out


def _q(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def _finite_rate_summary(
    *,
    sigma: float,
    corr: float,
    c: float,
    level_slack: float,
    quadrature: int,
    theta_grid: int,
) -> dict[str, float]:
    corr = min(0.999999, max(1e-6, corr))
    sigma = max(1e-6, sigma)
    return gaussian_rate_gap_summary(
        sigma=sigma,
        corr=corr,
        c=c,
        level_slack=level_slack,
        quadrature=quadrature,
        theta_grid=theta_grid,
    )


def run_trial(
    *,
    n: int,
    d: int,
    c: float,
    queries: int,
    panel_kind: str,
    bit_sigma: float,
    scale: float | None,
    seed: int,
    b_count: int,
    ref_samples: int,
    threshold_label_samples: int,
    tilted_label_samples: int,
    ideal_samples: int,
    margin: float,
    near_correlation: float | None = None,
    rate_level_slack: float = 0.1,
    rate_quadrature: int = 96,
    rate_theta_grid: int = 2000,
) -> dict[str, object]:
    if panel_kind not in {"product_sign", "whitened_product_sign"}:
        raise ValueError("panel_kind must be product_sign or whitened_product_sign")
    if not is_power_of_two(b_count):
        raise ValueError("b_count must be a power of two")
    if bit_sigma <= 0:
        raise ValueError("bit_sigma must be positive")
    if scale is not None and scale <= 0:
        raise ValueError("scale must be positive")
    if ref_samples < 1:
        raise ValueError("ref_samples must be positive")
    if ideal_samples < 1:
        raise ValueError("ideal_samples must be positive")
    if margin < 0:
        raise ValueError("margin must be nonnegative")

    rng = np.random.default_rng(seed)
    data, query_points, near_indices, radius = synthetic_near_pairs(
        n, d, c, queries, seed, near_correlation=near_correlation)
    eta = n ** (-1.0 / (2.0 * c * c))
    alpha = 1.0 / math.log(n)
    actual_near_corr = 1.0 - 0.5 * radius * radius

    r_bits = int(math.log2(b_count))
    actual_scale = bit_sigma * math.sqrt(r_bits) if scale is None else scale
    bit_scale = actual_scale / math.sqrt(r_bits)
    transform = (
        _data_whitening(data)
        if panel_kind == "whitened_product_sign"
        else np.eye(d)
    )
    basis = make_product_sign_basis(
        d,
        b_count,
        seed=seed + 1,
        data=data,
        whitened=(panel_kind == "whitened_product_sign"),
    )

    data_bits = (data @ basis.T) * bit_scale
    query_bits = (query_points @ basis.T) * bit_scale
    near_bits = data_bits[near_indices]
    ref_bits = sphere_points(ref_samples, d, seed + 10_000) @ basis.T * bit_scale

    labels, enumerated = _label_sample(
        b_count=b_count,
        r_bits=r_bits,
        label_samples=threshold_label_samples,
        rng=rng,
    )
    threshold_label_count = labels.shape[0]
    ref_thresholds = upper_tail_thresholds(_score_matrix(ref_bits, labels), eta)
    node_thresholds = upper_tail_thresholds(_score_matrix(data_bits, labels), eta)
    threshold_abs_diff = np.abs(ref_thresholds - node_thresholds)

    bit_means = np.mean(ref_bits, axis=0)
    bit_vars = np.var(ref_bits, axis=0)
    bit_sigma_fit = math.sqrt(float(np.mean(bit_vars)))
    ref_cov_rel_op, ref_cov_lambda_min_rel, ref_cov_lambda_max_rel = (
        covariance_spectral_summary(ref_bits))
    offdiag_corr = _abs_offdiag_correlations(ref_bits)
    pair_corr = _column_correlations(near_bits, query_bits)
    metric_sigma_near, metric_sigma_query, metric_pair_corr = gaussian_row_pair_parameters(
        data[near_indices],
        query_points,
        transform=transform,
        bit_scale=bit_scale,
    )

    ideal_q_eta, _near_ceiling = score_quantile(
        r_bits=r_bits,
        sigma=bit_sigma_fit,
        eta=eta,
        samples=ideal_samples,
        margin=margin,
        rng=rng,
    )

    certificate_threshold = float(np.max(ref_thresholds))
    tilted_mean, tilted_variance = tilted_score_mean_variance(near_bits, query_bits, alpha)
    mean_gap = tilted_mean - (certificate_threshold + margin)
    cantelli_good = np.zeros_like(mean_gap)
    positive = mean_gap > 0.0
    cantelli_good[positive] = (mean_gap[positive] * mean_gap[positive]) / (
        tilted_variance[positive] + mean_gap[positive] * mean_gap[positive])
    sampled_good = tilted_good_mass_samples(
        up=near_bits,
        uq=query_bits,
        alpha=alpha,
        threshold=certificate_threshold + margin,
        label_samples=tilted_label_samples,
        rng=rng,
    )

    if enumerated:
        exact_good = exact_tilted_good_mass(
            up=near_bits,
            uq=query_bits,
            alpha=alpha,
            labels=labels,
            thresholds=ref_thresholds,
            margin=margin,
        )
    else:
        exact_good = np.full_like(sampled_good, float("nan"))

    affinity = np.exp(_log_affinity(near_bits, query_bits, alpha))
    margin_factor = 1.0 - math.exp(-alpha * margin)
    sampled_bound = margin_factor * affinity * sampled_good
    exact_bound = margin_factor * affinity * exact_good
    cantelli_bound = margin_factor * affinity * cantelli_good

    rate = _finite_rate_summary(
        sigma=bit_sigma_fit,
        corr=float(np.mean(pair_corr)),
        c=c,
        level_slack=rate_level_slack,
        quadrature=rate_quadrature,
        theta_grid=rate_theta_grid,
    )
    threshold_ref_q95 = _q(ref_thresholds, 0.95)
    actual_score_level_q95 = threshold_ref_q95 / r_bits

    return {
        "n": n,
        "d": d,
        "c": c,
        "queries": queries,
        "near_corr": actual_near_corr,
        "panel": panel_kind,
        "scale": actual_scale,
        "bit_sigma_target": bit_sigma,
        "seed": seed,
        "B": b_count,
        "r_bits": r_bits,
        "eta": eta,
        "alpha": alpha,
        "margin": margin,
        "ref_samples": ref_samples,
        "threshold_labels": threshold_label_count,
        "tilted_label_samples": tilted_label_samples,
        "enumerated_thresholds": float(enumerated),
        "bit_mean_abs_max": float(np.max(np.abs(bit_means))),
        "bit_var_mean": float(np.mean(bit_vars)),
        "bit_var_q05": _q(bit_vars, 0.05),
        "bit_var_q95": _q(bit_vars, 0.95),
        "bit_sigma_fit": bit_sigma_fit,
        "ref_cov_rel_op": ref_cov_rel_op,
        "ref_cov_lambda_min_rel": ref_cov_lambda_min_rel,
        "ref_cov_lambda_max_rel": ref_cov_lambda_max_rel,
        "cross_corr_abs_q95": _q(offdiag_corr, 0.95),
        "cross_corr_abs_max": float(np.max(offdiag_corr)),
        "metric_sigma_near_mean": float(np.mean(metric_sigma_near)),
        "metric_sigma_query_mean": float(np.mean(metric_sigma_query)),
        "metric_pair_corr_mean": float(np.mean(metric_pair_corr)),
        "metric_pair_corr_q05": _q(metric_pair_corr, 0.05),
        "metric_pair_corr_q95": _q(metric_pair_corr, 0.95),
        "pair_corr_mean": float(np.mean(pair_corr)),
        "pair_corr_q05": _q(pair_corr, 0.05),
        "pair_corr_q95": _q(pair_corr, 0.95),
        "ideal_q_eta": ideal_q_eta,
        "threshold_ref_median": float(np.median(ref_thresholds)),
        "threshold_ref_q95": threshold_ref_q95,
        "threshold_ref_max": certificate_threshold,
        "threshold_node_q95": _q(node_thresholds, 0.95),
        "threshold_ref_node_abs_q50": float(np.median(threshold_abs_diff)),
        "threshold_ref_node_abs_q90": _q(threshold_abs_diff, 0.90),
        "threshold_q95_minus_ideal": threshold_ref_q95 - ideal_q_eta,
        "threshold_max_minus_ideal": certificate_threshold - ideal_q_eta,
        "tilted_mean_q05": _q(tilted_mean, 0.05),
        "tilted_variance_q95": _q(tilted_variance, 0.95),
        "mean_gap_q01": _q(mean_gap, 0.01),
        "mean_gap_q05": _q(mean_gap, 0.05),
        "cantelli_good_q01": _q(cantelli_good, 0.01),
        "cantelli_good_q05": _q(cantelli_good, 0.05),
        "sampled_good_mass_q01": _q(sampled_good, 0.01),
        "sampled_good_mass_q05": _q(sampled_good, 0.05),
        "exact_good_mass_q01": _q(exact_good, 0.01),
        "exact_good_mass_q05": _q(exact_good, 0.05),
        "affinity_q05": _q(affinity, 0.05),
        "sampled_bound_over_alpha_q05": _q(sampled_bound / alpha, 0.05),
        "exact_bound_over_alpha_q05": _q(exact_bound / alpha, 0.05),
        "cantelli_bound_over_alpha_q05": _q(cantelli_bound / alpha, 0.05),
        "ideal_score_level": rate["score_level"],
        "actual_score_level_q95": actual_score_level_q95,
        "score_level_gap_q95": rate["score_level"] - actual_score_level_q95,
        "gaussian_rate": rate["chernoff_rate"],
        "min_log_b_exponent": rate["min_log_b_exponent"],
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
        description="Stress ideal-to-whitened product-sign transfer.")
    parser.add_argument("--n", default="10000,30000")
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--panel", default="whitened_product_sign")
    parser.add_argument("--B", type=int, default=16)
    parser.add_argument("--bit-sigma", default="3.5")
    parser.add_argument("--scale", type=float, default=0.0,
                        help="panel scale; default bit_sigma*sqrt(log2(B))")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--ref-samples", type=int, default=100000)
    parser.add_argument("--threshold-label-samples", type=int, default=4096)
    parser.add_argument("--tilted-label-samples", type=int, default=512)
    parser.add_argument("--ideal-samples", type=int, default=200000)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--near-corr", type=float, default=0.95)
    parser.add_argument("--rate-level-slack", type=float, default=0.1)
    parser.add_argument("--rate-quadrature", type=int, default=96)
    parser.add_argument("--rate-theta-grid", type=int, default=2000)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    rows = []
    scale = None if args.scale == 0.0 else args.scale
    for n in parse_csv_list(args.n, int):
        for bit_sigma in parse_csv_list(args.bit_sigma, float):
            for seed in parse_csv_list(args.seeds, int):
                rows.append(run_trial(
                    n=n,
                    d=args.d,
                    c=args.c,
                    queries=args.queries,
                    panel_kind=args.panel,
                    bit_sigma=bit_sigma,
                    scale=scale,
                    seed=seed,
                    b_count=args.B,
                    ref_samples=args.ref_samples,
                    threshold_label_samples=args.threshold_label_samples,
                    tilted_label_samples=args.tilted_label_samples,
                    ideal_samples=args.ideal_samples,
                    margin=args.margin,
                    near_correlation=args.near_corr,
                    rate_level_slack=args.rate_level_slack,
                    rate_quadrature=args.rate_quadrature,
                    rate_theta_grid=args.rate_theta_grid,
                ))
    write_rows(rows, args.csv)


if __name__ == "__main__":
    main()
