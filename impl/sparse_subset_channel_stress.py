"""Stress an ordered-statistics sparse-subset product channel.

Product-sign labels use every bit coordinate.  The sparse-subset channel uses
labels (A,s), where A is a width-w coordinate subset and s is a sign pattern on
A.  Query top-one decoding is ordered-statistics decoding: choose the w largest
absolute query bit LLRs and their signs.  This tests whether reliability
selection can beat the sparse random-code product-sign saddlepoint barrier.

For logits u, the normalized centered label score is

    phi_{A,s}(u) = sum_{i in A} s_i u_i - log e_w(2 cosh u_1,...,2 cosh u_r)
                  + log(2^w * binom(r,w)).

When w=r this is exactly the product-sign score
sum_i s_i u_i - sum_i log cosh(u_i).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys

import numpy as np


FIELDNAMES = [
    "r_bits",
    "width",
    "active_fraction",
    "labels",
    "label_log_count",
    "label_rate",
    "tail_prob",
    "expected_tail_samples",
    "sigma",
    "corr",
    "far_samples",
    "near_samples",
    "theta_grid",
    "seed",
    "far_mean_level",
    "far_std_level",
    "far_empirical_max_level",
    "empirical_threshold_level",
    "empirical_threshold_tail",
    "saddle_threshold_level",
    "saddle_threshold_rate",
    "near_score_level_q05",
    "near_score_level_median",
    "near_score_level_mean",
    "near_score_level_std",
    "near_mean_gap",
    "near_median_gap",
    "near_normal_hit",
    "near_saddle_hit",
    "near_empirical_hit",
    "query_selected_abs_level_q05",
    "query_selected_abs_level_median",
    "full_abs_level_median",
]


def parse_csv_list(text: str, cast=str) -> list:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    if not out:
        raise ValueError("list must contain at least one item")
    return out


def label_log_count(r_bits: int, width: int) -> float:
    if r_bits <= 0:
        raise ValueError("r_bits must be positive")
    if width <= 0 or width > r_bits:
        raise ValueError("width must lie in [1, r_bits]")
    return (
        math.lgamma(r_bits + 1)
        - math.lgamma(width + 1)
        - math.lgamma(r_bits - width + 1)
        + width * math.log(2.0)
    )


def log_elementary_symmetric(log_weights: np.ndarray, width: int) -> float:
    log_weights = np.asarray(log_weights, dtype=float)
    if width <= 0 or width > len(log_weights):
        raise ValueError("width must lie in [1, len(log_weights)]")
    dp = np.full(width + 1, -np.inf)
    dp[0] = 0.0
    for value in log_weights:
        upper = min(width, len(log_weights))
        for k in range(upper, 0, -1):
            dp[k] = np.logaddexp(dp[k], dp[k - 1] + value)
    return float(dp[width])


def log_elementary_symmetric_batch(log_weights: np.ndarray, width: int) -> np.ndarray:
    log_weights = np.asarray(log_weights, dtype=float)
    if log_weights.ndim != 2:
        raise ValueError("log_weights must be a two-dimensional array")
    rows, r_bits = log_weights.shape
    if width <= 0 or width > r_bits:
        raise ValueError("width must lie in [1, r_bits]")
    dp = np.full((rows, width + 1), -np.inf)
    dp[:, 0] = 0.0
    for i in range(r_bits):
        upper = min(width, i + 1)
        for k in range(upper, 0, -1):
            dp[:, k] = np.logaddexp(dp[:, k], dp[:, k - 1] + log_weights[:, i])
    return dp[:, width]


def _log_2cosh(logits: np.ndarray) -> np.ndarray:
    return np.logaddexp(logits, -logits)


def sparse_subset_scores(
    logits: np.ndarray,
    indices: np.ndarray,
    signs: np.ndarray,
) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    indices = np.asarray(indices, dtype=int)
    signs = np.asarray(signs, dtype=float)
    if logits.ndim != 2:
        raise ValueError("logits must be two-dimensional")
    if indices.shape != signs.shape:
        raise ValueError("indices and signs must have the same shape")
    if indices.ndim != 2 or indices.shape[0] != logits.shape[0]:
        raise ValueError("indices must have one row per logits row")
    rows, r_bits = logits.shape
    width = indices.shape[1]
    if width <= 0 or width > r_bits:
        raise ValueError("invalid width")
    gathered = np.take_along_axis(logits, indices, axis=1)
    signed_sum = np.sum(signs * gathered, axis=1)
    log_z = log_elementary_symmetric_batch(_log_2cosh(logits), width)
    return signed_sum - log_z + label_log_count(r_bits, width)


def canonical_label_scores(logits: np.ndarray, width: int) -> np.ndarray:
    rows, _r_bits = logits.shape
    indices = np.tile(np.arange(width), (rows, 1))
    signs = np.ones((rows, width))
    return sparse_subset_scores(logits, indices, signs)


def top_query_label(query_logits: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    query_logits = np.asarray(query_logits, dtype=float)
    if query_logits.ndim != 2:
        raise ValueError("query_logits must be two-dimensional")
    if width <= 0 or width > query_logits.shape[1]:
        raise ValueError("invalid width")
    chosen = np.argpartition(-np.abs(query_logits), width - 1, axis=1)[:, :width]
    chosen_abs = np.take_along_axis(np.abs(query_logits), chosen, axis=1)
    order = np.argsort(-chosen_abs, axis=1)
    indices = np.take_along_axis(chosen, order, axis=1)
    gathered = np.take_along_axis(query_logits, indices, axis=1)
    signs = np.where(gathered >= 0.0, 1.0, -1.0)
    return indices, signs


def empirical_rate(
    samples: np.ndarray,
    *,
    level: float,
    blocklength: int,
    theta_max: float = 80.0,
    theta_grid: int = 1000,
) -> float:
    theta, log_mgf = empirical_log_mgf_grid(
        samples,
        theta_max=theta_max,
        theta_grid=theta_grid,
    )
    return empirical_rate_from_grid(
        samples,
        level=level,
        blocklength=blocklength,
        theta=theta,
        log_mgf=log_mgf,
    )


def empirical_log_mgf_grid(
    samples: np.ndarray,
    *,
    theta_max: float = 80.0,
    theta_grid: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    if theta_grid < 2:
        raise ValueError("theta_grid must be at least 2")
    theta = np.linspace(0.0, theta_max, theta_grid)
    values = theta[:, None] * samples[None, :]
    shifted = values - np.max(values, axis=1, keepdims=True)
    log_mgf = np.max(values, axis=1) + np.log(np.mean(np.exp(shifted), axis=1))
    return theta, log_mgf


def empirical_rate_from_grid(
    samples: np.ndarray,
    *,
    level: float,
    blocklength: int,
    theta: np.ndarray,
    log_mgf: np.ndarray,
) -> float:
    samples = np.asarray(samples, dtype=float)
    if blocklength <= 0:
        raise ValueError("blocklength must be positive")
    if level <= float(np.mean(samples)) / blocklength:
        return 0.0
    rates = theta * (blocklength * level) - log_mgf
    return max(0.0, float(np.max(rates) / blocklength))


def score_level_for_empirical_rate(
    samples: np.ndarray,
    *,
    target_rate: float,
    blocklength: int,
    theta_max: float = 80.0,
    theta_grid: int = 1000,
) -> tuple[float, float]:
    if target_rate < 0.0:
        raise ValueError("target_rate must be nonnegative")
    mean_level = float(np.mean(samples)) / blocklength
    theta, log_mgf = empirical_log_mgf_grid(
        samples,
        theta_max=theta_max,
        theta_grid=theta_grid,
    )
    high = label_log_count(blocklength, blocklength) / blocklength
    high = max(high, float(np.max(samples)) / blocklength)
    low = mean_level
    for _ in range(45):
        mid = 0.5 * (low + high)
        rate = empirical_rate_from_grid(
            samples,
            level=mid,
            blocklength=blocklength,
            theta=theta,
            log_mgf=log_mgf,
        )
        if rate < target_rate:
            low = mid
        else:
            high = mid
    level = 0.5 * (low + high)
    rate = empirical_rate_from_grid(
        samples,
        level=level,
        blocklength=blocklength,
        theta=theta,
        log_mgf=log_mgf,
    )
    return float(level), float(rate)


def normal_upper_tail(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def run_trial(
    *,
    r_bits: int,
    width: int,
    sigma: float,
    corr: float,
    far_samples: int,
    near_samples: int,
    theta_grid: int,
    seed: int,
) -> dict[str, float | int]:
    if not (0.0 < corr < 1.0):
        raise ValueError("corr must lie in (0, 1)")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng(seed)
    log_labels = label_log_count(r_bits, width)
    label_rate = log_labels / r_bits
    tail_prob = math.exp(-log_labels)

    far_logits = sigma * rng.standard_normal((far_samples, r_bits))
    far_scores = canonical_label_scores(far_logits, width)
    expected_tail_samples = far_samples * tail_prob
    empirical_threshold_level = float("nan")
    empirical_threshold_tail = float("nan")
    if expected_tail_samples >= 5.0:
        empirical_threshold = float(
            np.quantile(far_scores, 1.0 - tail_prob, method="higher")
        )
        empirical_threshold_level = empirical_threshold / r_bits
        empirical_threshold_tail = float(np.mean(far_scores >= empirical_threshold))
    threshold_level, threshold_rate = score_level_for_empirical_rate(
        far_scores,
        target_rate=label_rate,
        blocklength=r_bits,
        theta_grid=theta_grid,
    )

    query_logits = sigma * rng.standard_normal((near_samples, r_bits))
    near_logits = (
        corr * query_logits
        + sigma * math.sqrt(1.0 - corr * corr) * rng.standard_normal((near_samples, r_bits))
    )
    indices, signs = top_query_label(query_logits, width)
    near_scores = sparse_subset_scores(near_logits, indices, signs)
    near_levels = near_scores / r_bits
    query_selected_abs = np.sum(
        np.take_along_axis(np.abs(query_logits), indices, axis=1),
        axis=1,
    ) / r_bits
    full_abs = np.sum(np.abs(query_logits), axis=1) / r_bits

    mean_level = float(np.mean(near_levels))
    std_total = float(np.std(near_scores))
    if std_total <= 0:
        normal_hit = 1.0 if mean_level >= threshold_level else 0.0
    else:
        z = (r_bits * threshold_level - float(np.mean(near_scores))) / std_total
        normal_hit = normal_upper_tail(z)

    return {
        "r_bits": int(r_bits),
        "width": int(width),
        "active_fraction": float(width / r_bits),
        "labels": int(round(math.exp(log_labels))) if log_labels < 60 else float("inf"),
        "label_log_count": float(log_labels),
        "label_rate": float(label_rate),
        "tail_prob": float(tail_prob),
        "expected_tail_samples": float(expected_tail_samples),
        "sigma": float(sigma),
        "corr": float(corr),
        "far_samples": int(far_samples),
        "near_samples": int(near_samples),
        "theta_grid": int(theta_grid),
        "seed": int(seed),
        "far_mean_level": float(np.mean(far_scores) / r_bits),
        "far_std_level": float(np.std(far_scores) / r_bits),
        "far_empirical_max_level": float(np.max(far_scores) / r_bits),
        "empirical_threshold_level": float(empirical_threshold_level),
        "empirical_threshold_tail": float(empirical_threshold_tail),
        "saddle_threshold_level": float(threshold_level),
        "saddle_threshold_rate": float(threshold_rate),
        "near_score_level_q05": float(np.quantile(near_levels, 0.05)),
        "near_score_level_median": float(np.quantile(near_levels, 0.5)),
        "near_score_level_mean": mean_level,
        "near_score_level_std": float(np.std(near_levels)),
        "near_mean_gap": float(mean_level - threshold_level),
        "near_median_gap": float(np.quantile(near_levels, 0.5) - threshold_level),
        "near_normal_hit": float(normal_hit),
        "near_saddle_hit": float(np.mean(near_levels >= threshold_level)),
        "near_empirical_hit": float(
            np.mean(near_levels >= empirical_threshold_level)
            if empirical_threshold_level == empirical_threshold_level
            else float("nan")
        ),
        "query_selected_abs_level_q05": float(np.quantile(query_selected_abs, 0.05)),
        "query_selected_abs_level_median": float(np.quantile(query_selected_abs, 0.5)),
        "full_abs_level_median": float(np.quantile(full_abs, 0.5)),
    }


def run_sweep(args: argparse.Namespace) -> list[dict[str, float | int]]:
    rows = []
    for shape in parse_csv_list(args.shapes):
        parts = shape.split(":")
        if len(parts) != 2:
            raise ValueError("shape specs must have form r_bits:width")
        r_bits = int(parts[0])
        width = int(parts[1])
        for sigma in args.sigmas:
            for corr in args.corrs:
                for seed in args.seeds:
                    row = run_trial(
                        r_bits=r_bits,
                        width=width,
                        sigma=sigma,
                        corr=corr,
                        far_samples=args.far_samples,
                        near_samples=args.near_samples,
                        theta_grid=args.theta_grid,
                        seed=seed,
                    )
                    rows.append(row)
                    print(
                        " ".join(
                            [
                                f"shape={r_bits}:{width}",
                                f"sigma={sigma:g}",
                                f"corr={corr:g}",
                                f"seed={seed}",
                                f"rate={row['label_rate']:.3f}",
                                f"thr={row['saddle_threshold_level']:.3f}",
                                f"gap={row['near_mean_gap']:.3f}",
                                f"hit={row['near_saddle_hit']:.3f}",
                                f"qabs={row['query_selected_abs_level_median']:.3f}",
                            ]
                        ),
                        file=sys.stderr,
                    )
    return rows


def write_csv(path: str, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in FIELDNAMES})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="16:1,16:2,32:2,32:4")
    parser.add_argument("--sigmas", type=lambda text: parse_csv_list(text, float), default=[2.0, 3.5])
    parser.add_argument("--corrs", type=lambda text: parse_csv_list(text, float), default=[0.95])
    parser.add_argument("--far-samples", type=int, default=30_000)
    parser.add_argument("--near-samples", type=int, default=10_000)
    parser.add_argument("--theta-grid", type=int, default=600)
    parser.add_argument(
        "--seeds",
        type=lambda text: parse_csv_list(text, int),
        default=[0, 1, 2],
    )
    parser.add_argument("--csv", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = run_sweep(args)
    if args.csv:
        write_csv(args.csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
