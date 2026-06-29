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
    "scale",
    "seed",
    "B",
    "eta",
    "alpha",
    "margin_delta",
    "pi_relerr",
    "guard_mean",
    "affinity_mean",
    "affinity_q05",
    "soft_surplus_mean",
    "soft_surplus_q05",
    "tilted_top_mass_q05",
    "tilted_top_mass_median",
    "top_contribution_fraction_q05",
    "top_contribution_fraction_median",
    "top2_contribution_fraction_q05",
    "top2_contribution_fraction_median",
    "top4_contribution_fraction_q05",
    "top4_contribution_fraction_median",
    "top8_contribution_fraction_q05",
    "top8_contribution_fraction_median",
    "query_top_contribution_fraction_q05",
    "query_top_contribution_fraction_median",
    "query_top2_contribution_fraction_q05",
    "query_top2_contribution_fraction_median",
    "query_top4_contribution_fraction_q05",
    "query_top4_contribution_fraction_median",
    "query_top8_contribution_fraction_q05",
    "query_top8_contribution_fraction_median",
    "query_tilt_perturbation_q90",
    "query_top4_gap_q05",
    "query_top4_stability_margin_q05",
    "query_top4_stable_fraction",
    "query_top8_gap_q05",
    "query_top8_stability_margin_q05",
    "query_top8_stable_fraction",
    "margin_mass_mean",
    "margin_mass_q05",
    "margin_bound_mean",
    "margin_bound_q05",
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
    "route_depth",
    "leader_eps",
    "leader_boost_overhead_q50",
    "leader_boost_overhead_q90",
    "leader_boost_overhead_q99",
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


def _upper_quantile(values: np.ndarray, q: float) -> float:
    if not (0.0 <= q <= 1.0):
        raise ValueError("quantile must lie in [0, 1]")
    if len(values) == 0:
        return float("nan")
    ordered = np.sort(values)
    index = int(math.ceil(q * (len(ordered) - 1)))
    return float(ordered[index])


def leader_boost_overhead(h_values: np.ndarray, route_depth: int) -> tuple[float, np.ndarray]:
    if route_depth < 1:
        raise ValueError("route_depth must be positive")
    leader_eps = (route_depth + 1) ** -2
    log_boost = math.log(1.0 / leader_eps)
    overhead = np.divide(
        log_boost,
        h_values,
        out=np.full_like(h_values, float("inf"), dtype=float),
        where=h_values > 0,
    )
    return leader_eps, overhead


def run_trial(
    *,
    n: int,
    d: int,
    c: float,
    queries: int,
    panel_kind: str,
    scale: float,
    seed: int,
    b_count: int,
    balance_iters: int,
    near_correlation: float | None = None,
    route_depth: int | None = None,
) -> dict[str, object]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    data, query_points, near_indices, r = synthetic_near_pairs(
        n, d, c, queries, seed, near_correlation=near_correlation)
    eta = n ** (-1.0 / (2.0 * c * c))
    actual_near_corr = 1.0 - 0.5 * r * r

    panel = scale * make_panel(d, b_count, kind=panel_kind, seed=seed + 1, data=data)
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
    if route_depth is None:
        route_depth = max(1, int(math.ceil(math.log(max(n, 3)))))
    leader_eps, boost_overhead = leader_boost_overhead(result.h_eta, route_depth)

    return {
        "n": n,
        "d": d,
        "c": c,
        "queries": queries,
        "near_corr": actual_near_corr,
        "r": r,
        "panel": panel_kind,
        "scale": scale,
        "seed": seed,
        "B": b_count,
        "eta": result.eta,
        "alpha": result.alpha,
        "margin_delta": result.margin_delta,
        "pi_relerr": float(np.max(np.abs(result.pi - target) / target)),
        "guard_mean": float(result.guard_density.mean()),
        "affinity_mean": float(result.affinity.mean()),
        "affinity_q05": _q(result.affinity, 0.05),
        "soft_surplus_mean": float(result.soft_surplus.mean()),
        "soft_surplus_q05": _q(result.soft_surplus, 0.05),
        "tilted_top_mass_q05": _q(result.tilted_top_mass, 0.05),
        "tilted_top_mass_median": float(np.median(result.tilted_top_mass)),
        "top_contribution_fraction_q05": _q(result.top_contribution_fraction, 0.05),
        "top_contribution_fraction_median": float(np.median(result.top_contribution_fraction)),
        "top2_contribution_fraction_q05": _q(result.top2_contribution_fraction, 0.05),
        "top2_contribution_fraction_median": float(np.median(result.top2_contribution_fraction)),
        "top4_contribution_fraction_q05": _q(result.top4_contribution_fraction, 0.05),
        "top4_contribution_fraction_median": float(np.median(result.top4_contribution_fraction)),
        "top8_contribution_fraction_q05": _q(result.top8_contribution_fraction, 0.05),
        "top8_contribution_fraction_median": float(np.median(result.top8_contribution_fraction)),
        "query_top_contribution_fraction_q05": _q(result.query_top_contribution_fraction, 0.05),
        "query_top_contribution_fraction_median": float(
            np.median(result.query_top_contribution_fraction)),
        "query_top2_contribution_fraction_q05": _q(result.query_top2_contribution_fraction, 0.05),
        "query_top2_contribution_fraction_median": float(
            np.median(result.query_top2_contribution_fraction)),
        "query_top4_contribution_fraction_q05": _q(result.query_top4_contribution_fraction, 0.05),
        "query_top4_contribution_fraction_median": float(
            np.median(result.query_top4_contribution_fraction)),
        "query_top8_contribution_fraction_q05": _q(result.query_top8_contribution_fraction, 0.05),
        "query_top8_contribution_fraction_median": float(
            np.median(result.query_top8_contribution_fraction)),
        "query_tilt_perturbation_q90": _q(result.query_tilt_perturbation, 0.90),
        "query_top4_gap_q05": _finite_or_inf(result.query_top4_gap, 0.05),
        "query_top4_stability_margin_q05": _finite_or_inf(
            result.query_top4_stability_margin, 0.05),
        "query_top4_stable_fraction": float(np.mean(result.query_top4_stability_margin > 0)),
        "query_top8_gap_q05": _finite_or_inf(result.query_top8_gap, 0.05),
        "query_top8_stability_margin_q05": _finite_or_inf(
            result.query_top8_stability_margin, 0.05),
        "query_top8_stable_fraction": float(np.mean(result.query_top8_stability_margin > 0)),
        "margin_mass_mean": float(result.margin_mass.mean()),
        "margin_mass_q05": _q(result.margin_mass, 0.05),
        "margin_bound_mean": float(result.margin_bound.mean()),
        "margin_bound_q05": _q(result.margin_bound, 0.05),
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
        "route_depth": route_depth,
        "leader_eps": leader_eps,
        "leader_boost_overhead_q50": _upper_quantile(boost_overhead, 0.5),
        "leader_boost_overhead_q90": _upper_quantile(boost_overhead, 0.9),
        "leader_boost_overhead_q99": _upper_quantile(boost_overhead, 0.99),
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
    parser.add_argument("--scales", default="1", help="comma-separated inverse-temperature scales")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--near-corr", type=float, default=None,
                        help="near-pair sphere correlation; default is 1-1/c^2")
    parser.add_argument("--B", type=int, default=0, help="fixed outcome count; overrides --B-mult")
    parser.add_argument("--B-mult", type=float, default=1.0, help="multiply ceil(n^(1/(2c^2))) by this factor")
    parser.add_argument("--balance-iters", type=int, default=200)
    parser.add_argument("--route-depth", type=int, default=0,
                        help="depth H for boosted-leader epsilon=(H+1)^-2; default is ceil(log n)")
    parser.add_argument("--csv", default=None, help="optional output CSV path")
    args = parser.parse_args()

    ns = parse_csv_list(args.n, int)
    panels = parse_csv_list(args.panels, str)
    scales = parse_csv_list(args.scales, float)
    seeds = parse_csv_list(args.seeds, int)
    if any(scale <= 0 for scale in scales):
        raise ValueError("all scales must be positive")
    unknown = sorted(set(panels) - set(PANEL_KINDS))
    if unknown:
        raise ValueError(f"unknown panel kind(s): {', '.join(unknown)}")

    rows = []
    for n in ns:
        b_count = args.B or default_b_count(n, args.c, args.B_mult)
        for panel in panels:
            for scale in scales:
                for seed in seeds:
                    rows.append(run_trial(
                        n=n,
                        d=args.d,
                        c=args.c,
                        queries=args.queries,
                        panel_kind=panel,
                        scale=scale,
                        seed=seed,
                        b_count=b_count,
                        balance_iters=args.balance_iters,
                        near_correlation=args.near_corr,
                        route_depth=args.route_depth or None,
                    ))
    write_rows(rows, args.csv)


if __name__ == "__main__":
    main()
