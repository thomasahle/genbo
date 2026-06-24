"""Sweep the finite-channel Conjecture 9.1 diagnostic.

This runner evaluates the balanced-softmax channel from
finite_channel_diagnostic.py over several data sizes, panel types, and seeds.
It emits one CSV row per trial, with emphasis on lower quantiles of
g(q)+H_eta(p,q), the quantity that should stay inverse-polylogarithmic or
constant if the finite restricted-channel conjecture is true.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Callable

import numpy as np

from finite_channel_diagnostic import (
    PANEL_KINDS,
    balance_softmax_offsets,
    diagnose_channel,
    effective_support,
    make_panel,
    softmax_channel,
    softmax_hessian_condition,
    stable_softmax,
    synthetic_near_pairs,
)


FIELDNAMES = [
    "n",
    "d",
    "c",
    "queries",
    "near_corr",
    "r",
    "panel",
    "seed",
    "B",
    "eta",
    "alpha",
    "pi_relerr",
    "guard_mean",
    "h_mean",
    "h_q00",
    "h_q01",
    "h_q05",
    "h_q10",
    "score_mean",
    "score_q00",
    "score_q01",
    "score_q05",
    "score_q10",
    "support_q01",
    "support_q05",
    "support_median",
    "hessian_cond_median",
    "hessian_cond_q90",
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


def default_b_count(n: int, c: float, b_mult: float) -> int:
    eta = n ** (-1.0 / (2.0 * c * c))
    return max(1, int(math.ceil(b_mult / eta)))


def _q(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def _finite_or_inf(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("inf")
    return _q(finite, q)


def run_trial(
    *,
    n: int,
    d: int,
    c: float,
    queries: int,
    panel_kind: str,
    seed: int,
    b_count: int,
    balance_iters: int,
    near_correlation: float | None = None,
) -> dict[str, object]:
    data, query_points, near_indices, r = synthetic_near_pairs(
        n, d, c, queries, seed, near_correlation=near_correlation)
    eta = n ** (-1.0 / (2.0 * c * c))
    actual_near_corr = 1.0 - 0.5 * r * r

    panel = make_panel(d, b_count, kind=panel_kind, seed=seed + 1, data=data)
    scores = data @ panel.T
    offsets = balance_softmax_offsets(scores, max_iter=balance_iters)
    k_data = stable_softmax(scores + offsets)
    k_query = softmax_channel(query_points, panel, offsets)
    result = diagnose_channel(data, query_points, near_indices, k_data, k_query, c=c, r=r, eta=eta)

    z = result.alpha * data[near_indices] + (1.0 - result.alpha) * query_points
    k_z = softmax_channel(z, panel, offsets)
    support = effective_support(k_z)
    cond = softmax_hessian_condition(panel, k_z)
    target = np.full(b_count, 1.0 / b_count)

    return {
        "n": n,
        "d": d,
        "c": c,
        "queries": queries,
        "near_corr": actual_near_corr,
        "r": r,
        "panel": panel_kind,
        "seed": seed,
        "B": b_count,
        "eta": result.eta,
        "alpha": result.alpha,
        "pi_relerr": float(np.max(np.abs(result.pi - target) / target)),
        "guard_mean": float(result.guard_density.mean()),
        "h_mean": float(result.h_eta.mean()),
        "h_q00": _q(result.h_eta, 0.0),
        "h_q01": _q(result.h_eta, 0.01),
        "h_q05": _q(result.h_eta, 0.05),
        "h_q10": _q(result.h_eta, 0.10),
        "score_mean": float(result.score.mean()),
        "score_q00": _q(result.score, 0.0),
        "score_q01": _q(result.score, 0.01),
        "score_q05": _q(result.score, 0.05),
        "score_q10": _q(result.score, 0.10),
        "support_q01": _q(support, 0.01),
        "support_q05": _q(support, 0.05),
        "support_median": float(np.median(support)),
        "hessian_cond_median": _finite_or_inf(cond, 0.5),
        "hessian_cond_q90": _finite_or_inf(cond, 0.9),
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
    parser = argparse.ArgumentParser(description="Sweep balanced-softmax finite-channel diagnostics.")
    parser.add_argument("--n", default="300,1000,3000", help="comma-separated data sizes")
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--panels", default="gaussian,whitened_gaussian,cross_polytope,pca,landmark")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--near-corr", type=float, default=None,
                        help="near-pair sphere correlation; default is 1-1/c^2")
    parser.add_argument("--B", type=int, default=0, help="fixed outcome count; overrides --B-mult")
    parser.add_argument("--B-mult", type=float, default=1.0, help="multiply ceil(n^(1/(2c^2))) by this factor")
    parser.add_argument("--balance-iters", type=int, default=200)
    parser.add_argument("--csv", default=None, help="optional output CSV path")
    args = parser.parse_args()

    ns = parse_csv_list(args.n, int)
    panels = parse_csv_list(args.panels, str)
    seeds = parse_csv_list(args.seeds, int)
    unknown = sorted(set(panels) - set(PANEL_KINDS))
    if unknown:
        raise ValueError(f"unknown panel kind(s): {', '.join(unknown)}")

    rows = []
    for n in ns:
        b_count = args.B or default_b_count(n, args.c, args.B_mult)
        for panel in panels:
            for seed in seeds:
                rows.append(run_trial(
                    n=n,
                    d=args.d,
                    c=args.c,
                    queries=args.queries,
                    panel_kind=panel,
                    seed=seed,
                    b_count=b_count,
                    balance_iters=args.balance_iters,
                    near_correlation=args.near_corr,
                ))
    write_rows(rows, args.csv)


if __name__ == "__main__":
    main()
