"""Stress additive score-max candidate mechanisms.

The smoothed-score endpoint reduces ANN to returning a point whose score

    A_Y(x) = <Y, x> - t ||x||^2 / 2,    Y ~ N(t q, t I)

is within a constant fraction of the planted near point's score gap.  This file
keeps that normal form fixed.  It first verifies the oracle score gap, then
tests whether a concrete candidate generator captures a high-score valid point.

The only non-oracle generator here is the existing smooth cumulative-score
filter from ``smooth_filter.py``.  It is intentionally treated as a diagnostic:
failure is a barrier to that particular implementation route, not to the
smoothed-score theorem itself.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Callable, Iterable

import numpy as np

from smooth_filter import SmoothFilter


FIELDNAMES = [
    "n",
    "d",
    "c",
    "rho",
    "near_corr",
    "far_corr_max",
    "t_scale",
    "smooth_filter_theta",
    "smooth_filter_k",
    "smooth_filter_B",
    "smooth_filter_storage",
    "query_trials",
    "score_gap_fail_rate",
    "p_score_rank_q50",
    "p_score_rank_q90",
    "p_score_rank_max",
    "oracle_top1_valid_rate",
    "smooth_filter_valid_rate",
    "smooth_filter_p_rate",
    "smooth_filter_scoremax_rate",
    "smooth_filter_additive_success_rate",
    "smooth_filter_candidate_mean",
    "smooth_filter_candidate_q90",
    "smooth_filter_candidate_over_nrho",
    "smooth_filter_best_loss_q50",
    "smooth_filter_best_loss_q90",
    "smooth_filter_best_loss_over_margin_q90",
    "seed",
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


def rho_for_c(c: float) -> float:
    if c <= 1.0:
        raise ValueError("c must be greater than one")
    return 1.0 / (2.0 * c * c - 1.0)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("zero vector")
    return v / norm


def _orthogonal_unit(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        w = rng.standard_normal(len(q))
        w = w - float(w @ q) * q
        norm = np.linalg.norm(w)
        if norm > 1e-12:
            return w / norm
    raise RuntimeError("failed to sample orthogonal direction")


def _sphere_with_corr_at_most(
    count: int,
    d: int,
    q: np.ndarray,
    corr_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    batch = max(1024, count)
    while sum(len(x) for x in rows) < count:
        x = rng.standard_normal((batch, d))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        keep = x @ q <= corr_max
        if np.any(keep):
            rows.append(x[keep])
    return np.vstack(rows)[:count]


def promised_spherical_instance(
    *,
    n_far: int,
    d: int,
    c: float,
    far_corr_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (X, q, p, p_index) in the critical spherical normalization."""
    if n_far < 1:
        raise ValueError("n_far must be positive")
    if d < 2:
        raise ValueError("d must be at least two")
    near_corr = 1.0 - 1.0 / (c * c)
    if far_corr_max >= 0.0:
        # Far points at correlation zero have exactly distance c*r in the
        # critical normalization; use a strict negative cap for invalid points.
        far_corr_max = -1e-6
    q = _unit(rng.standard_normal(d))
    w = _orthogonal_unit(q, rng)
    p = near_corr * q + math.sqrt(1.0 - near_corr * near_corr) * w
    far = _sphere_with_corr_at_most(n_far, d, q, far_corr_max, rng)
    x = np.vstack([far, p[None, :]])
    return x, q, p, n_far


def smoothed_scores(
    x: np.ndarray,
    q: np.ndarray,
    *,
    c: float,
    t_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return scores, perturbed center y, t, and the score margin."""
    if t_scale <= 0.0:
        raise ValueError("t_scale must be positive")
    m = len(x)
    r2 = 2.0 / (c * c)
    t = t_scale * math.log(max(m, 3)) / r2
    y = q + rng.standard_normal(len(q)) / math.sqrt(t)
    scores = t * (x @ y) - 0.5 * t * np.sum(x * x, axis=1)
    delta_c = (c * c - 1.0) / 4.0
    margin = delta_c * t * r2 / 2.0
    return scores, y, t, margin


def _candidate_best_loss(scores: np.ndarray, candidates: set[int]) -> float:
    if not candidates:
        return float("inf")
    best = float(np.max(scores))
    cand_scores = scores[np.fromiter(candidates, dtype=int)]
    return best - float(np.max(cand_scores))


def _quantile_with_inf(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    if np.any(np.isnan(values)):
        raise ValueError("nan in diagnostic values")
    if len(values) == 0:
        raise ValueError("empty diagnostic values")
    ordered = np.sort(values)
    index = int(math.ceil(q * len(ordered))) - 1
    index = min(max(index, 0), len(ordered) - 1)
    return float(ordered[index])


def run_trial(
    *,
    n: int,
    d: int,
    c: float,
    far_corr_max: float,
    t_scale: float,
    smooth_filter_theta: float,
    query_trials: int,
    seed: int,
) -> dict[str, float | int]:
    if n < 2:
        raise ValueError("n must be at least two")
    if query_trials < 1:
        raise ValueError("query_trials must be positive")
    rng = np.random.default_rng(seed)
    rho = rho_for_c(c)
    x, q, _p, p_index = promised_spherical_instance(
        n_far=n - 1,
        d=d,
        c=c,
        far_corr_max=far_corr_max,
        rng=rng,
    )
    near_corr = 1.0 - 1.0 / (c * c)
    sf = SmoothFilter(
        d=d,
        a=near_corr,
        theta=smooth_filter_theta,
        seed=seed + 17,
    ).build(x)

    p_ranks = []
    score_gap_failures = 0
    oracle_valid = 0
    sf_valid = 0
    sf_p = 0
    sf_scoremax = 0
    sf_additive = 0
    sf_counts = []
    sf_losses = []

    for _ in range(query_trials):
        scores, y, _t, margin = smoothed_scores(
            x, q, c=c, t_scale=t_scale, rng=rng)
        p_score = float(scores[p_index])
        p_rank = 1 + int(np.sum(scores > p_score))
        p_ranks.append(p_rank)
        if np.max(scores[:p_index]) > p_score - margin:
            score_gap_failures += 1

        top = int(np.argmax(scores))
        oracle_valid += int(top == p_index)

        candidates = sf.query_candidates(y)
        sf_counts.append(len(candidates))
        loss = _candidate_best_loss(scores, candidates)
        sf_losses.append(loss)
        sf_valid += int(p_index in candidates or any(i == p_index for i in candidates))
        sf_p += int(p_index in candidates)
        sf_scoremax += int(top in candidates)
        sf_additive += int(loss <= margin)

    p_ranks_arr = np.asarray(p_ranks, dtype=float)
    counts = np.asarray(sf_counts, dtype=float)
    losses = np.asarray(sf_losses, dtype=float)
    loss_ratios = losses / margin
    n_rho = n ** rho
    storage = sum(len(bucket) for bucket in sf.buckets.values())
    return {
        "n": n,
        "d": d,
        "c": c,
        "rho": rho,
        "near_corr": near_corr,
        "far_corr_max": far_corr_max,
        "t_scale": t_scale,
        "smooth_filter_theta": smooth_filter_theta,
        "smooth_filter_k": sf.k,
        "smooth_filter_B": sf.B,
        "smooth_filter_storage": storage,
        "query_trials": query_trials,
        "score_gap_fail_rate": score_gap_failures / query_trials,
        "p_score_rank_q50": float(np.quantile(p_ranks_arr, 0.5)),
        "p_score_rank_q90": float(np.quantile(p_ranks_arr, 0.9)),
        "p_score_rank_max": float(np.max(p_ranks_arr)),
        "oracle_top1_valid_rate": oracle_valid / query_trials,
        "smooth_filter_valid_rate": sf_valid / query_trials,
        "smooth_filter_p_rate": sf_p / query_trials,
        "smooth_filter_scoremax_rate": sf_scoremax / query_trials,
        "smooth_filter_additive_success_rate": sf_additive / query_trials,
        "smooth_filter_candidate_mean": float(np.mean(counts)),
        "smooth_filter_candidate_q90": float(np.quantile(counts, 0.9)),
        "smooth_filter_candidate_over_nrho": float(np.mean(counts) / n_rho),
        "smooth_filter_best_loss_q50": _quantile_with_inf(losses, 0.5),
        "smooth_filter_best_loss_q90": _quantile_with_inf(losses, 0.9),
        "smooth_filter_best_loss_over_margin_q90": _quantile_with_inf(
            loss_ratios, 0.9),
        "seed": seed,
    }


def _rows(args: argparse.Namespace) -> Iterable[dict[str, float | int]]:
    for n in parse_csv_list(args.n, int):
        for d in parse_csv_list(args.d, int):
            for t_scale in parse_csv_list(args.t_scales, float):
                for theta in parse_csv_list(args.smooth_filter_thetas, float):
                    for seed in parse_csv_list(args.seeds, int):
                        yield run_trial(
                            n=n,
                            d=d,
                            c=args.c,
                            far_corr_max=args.far_corr_max,
                            t_scale=t_scale,
                            smooth_filter_theta=theta,
                            query_trials=args.query_trials,
                            seed=seed,
                        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default="1000,3000")
    parser.add_argument("--d", default="32")
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--far-corr-max", type=float, default=-0.02)
    parser.add_argument("--t-scales", default="12,24,48")
    parser.add_argument("--smooth-filter-thetas", default="0.6,0.8,1.0")
    parser.add_argument("--query-trials", type=int, default=80)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--csv")
    args = parser.parse_args()

    rows = list(_rows(args))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    else:
        print(",".join(FIELDNAMES))
        for row in rows:
            print(",".join(str(row[name]) for name in FIELDNAMES))


if __name__ == "__main__":
    main()
