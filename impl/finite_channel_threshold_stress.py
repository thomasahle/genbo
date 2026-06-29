"""Compare empirical and held-out finite-channel posterior thresholds.

The main finite-channel sweep uses the node's data to estimate the top-eta
posterior thresholds tau_j.  This diagnostic keeps the same retained panel and
near pairs, but recomputes pi_j and tau_j on a larger independent spherical
reference sample.  If the product-sign lower tail is stable under this
replacement, the remaining theorem is distributional rather than a finite-node
threshold artifact.
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
    h_eta_for_pairs,
    h_eta_for_pairs_with_thresholds,
    is_product_sign_kind,
    make_panel,
    posterior_margin_certificate,
    posterior_thresholds,
    softmax_channel,
    stable_softmax,
    synthetic_near_pairs,
    tilted_surplus_statistics,
)


FIELDNAMES = [
    "n",
    "d",
    "c",
    "queries",
    "near_corr",
    "panel",
    "scale",
    "seed",
    "B",
    "eta",
    "alpha",
    "ref_samples",
    "pi_emp_relerr",
    "pi_ref_relerr",
    "pi_ref_emp_logerr_max",
    "tau_ref_emp_logerr_q50",
    "tau_ref_emp_logerr_q90",
    "emp_h_q01",
    "emp_h_q05",
    "ref_h_q01",
    "ref_h_q05",
    "emp_soft_surplus_q05",
    "ref_soft_surplus_q05",
    "emp_top4_value_q05",
    "ref_top4_value_q05",
    "emp_margin025_mass_q05",
    "emp_margin025_bound_q05",
    "ref_margin025_mass_q05",
    "ref_margin025_bound_q05",
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


def _q(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def _threshold_stats(
    *,
    k_data: np.ndarray,
    k_query: np.ndarray,
    near_indices: np.ndarray,
    pi: np.ndarray,
    tau: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_eta = h_eta_for_pairs_with_thresholds(
        k_data, k_query, near_indices, pi, tau, alpha=alpha)
    (
        affinity,
        soft_surplus,
        _tilted_top_mass,
        _top_contribution_fraction,
        topk_contribution,
        _query_topk_contribution,
    ) = tilted_surplus_statistics(k_data, k_query, near_indices, pi, tau, alpha=alpha)
    if not np.allclose(h_eta, affinity * soft_surplus):
        raise AssertionError("held-out soft surplus decomposition mismatch")
    _affinity2, margin025_mass, margin025_bound = posterior_margin_certificate(
        k_data, k_query, near_indices, pi, tau, alpha=alpha, margin_delta=0.25 / alpha)
    top4_value = h_eta * topk_contribution[4]
    return h_eta, soft_surplus, top4_value, margin025_mass, margin025_bound


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
    ref_samples: int,
    balance_iters: int,
    near_correlation: float | None = None,
) -> dict[str, object]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if ref_samples < 1:
        raise ValueError("ref_samples must be positive")

    data, query_points, near_indices, r = synthetic_near_pairs(
        n, d, c, queries, seed, near_correlation=near_correlation)
    eta = n ** (-1.0 / (2.0 * c * c))
    alpha = 1.0 / max(math.log(max(n, 3)), 1.0)
    actual_near_corr = 1.0 - 0.5 * r * r

    panel = scale * make_panel(d, b_count, kind=panel_kind, seed=seed + 1, data=data)
    scores = data @ panel.T
    if is_product_sign_kind(panel_kind):
        offsets = np.zeros(b_count)
    else:
        offsets = balance_softmax_offsets(scores, max_iter=balance_iters)
    k_data = stable_softmax(scores + offsets)
    k_query = softmax_channel(query_points, panel, offsets)

    emp_h, emp_pi, emp_tau = h_eta_for_pairs(
        k_data, k_query, near_indices, eta=eta, alpha=alpha)
    ref_data = sphere_points(ref_samples, d, seed + 10_000)
    k_ref = softmax_channel(ref_data, panel, offsets)
    ref_pi, _ref_ratios, ref_tau = posterior_thresholds(k_ref, eta)
    ref_h = h_eta_for_pairs_with_thresholds(
        k_data, k_query, near_indices, ref_pi, ref_tau, alpha=alpha)

    emp_h2, emp_soft, emp_top4, emp_m025, emp_b025 = _threshold_stats(
        k_data=k_data, k_query=k_query, near_indices=near_indices,
        pi=emp_pi, tau=emp_tau, alpha=alpha)
    ref_h2, ref_soft, ref_top4, ref_m025, ref_b025 = _threshold_stats(
        k_data=k_data, k_query=k_query, near_indices=near_indices,
        pi=ref_pi, tau=ref_tau, alpha=alpha)
    if not np.allclose(emp_h, emp_h2) or not np.allclose(ref_h, ref_h2):
        raise AssertionError("threshold h_eta mismatch")

    target = np.full(b_count, 1.0 / b_count)
    tau_logerr = np.abs(np.log(np.maximum(ref_tau, 1e-300) / np.maximum(emp_tau, 1e-300)))
    pi_logerr = np.abs(np.log(np.maximum(ref_pi, 1e-300) / np.maximum(emp_pi, 1e-300)))

    return {
        "n": n,
        "d": d,
        "c": c,
        "queries": queries,
        "near_corr": actual_near_corr,
        "panel": panel_kind,
        "scale": scale,
        "seed": seed,
        "B": b_count,
        "eta": eta,
        "alpha": alpha,
        "ref_samples": ref_samples,
        "pi_emp_relerr": float(np.max(np.abs(emp_pi - target) / target)),
        "pi_ref_relerr": float(np.max(np.abs(ref_pi - target) / target)),
        "pi_ref_emp_logerr_max": float(np.max(pi_logerr)),
        "tau_ref_emp_logerr_q50": _q(tau_logerr, 0.50),
        "tau_ref_emp_logerr_q90": _q(tau_logerr, 0.90),
        "emp_h_q01": _q(emp_h, 0.01),
        "emp_h_q05": _q(emp_h, 0.05),
        "ref_h_q01": _q(ref_h, 0.01),
        "ref_h_q05": _q(ref_h, 0.05),
        "emp_soft_surplus_q05": _q(emp_soft, 0.05),
        "ref_soft_surplus_q05": _q(ref_soft, 0.05),
        "emp_top4_value_q05": _q(emp_top4, 0.05),
        "ref_top4_value_q05": _q(ref_top4, 0.05),
        "emp_margin025_mass_q05": _q(emp_m025, 0.05),
        "emp_margin025_bound_q05": _q(emp_b025, 0.05),
        "ref_margin025_mass_q05": _q(ref_m025, 0.05),
        "ref_margin025_bound_q05": _q(ref_b025, 0.05),
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
        description="Compare empirical and held-out finite-channel thresholds.")
    parser.add_argument("--n", default="10000,30000")
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--queries", type=int, default=160)
    parser.add_argument("--panel", default="whitened_product_sign")
    parser.add_argument("--scales", default="8,12")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--B", type=int, default=16)
    parser.add_argument("--ref-samples", type=int, default=100000)
    parser.add_argument("--balance-iters", type=int, default=200)
    parser.add_argument("--near-corr", type=float, default=0.95)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    if args.panel not in PANEL_KINDS:
        raise ValueError(f"unknown panel kind: {args.panel}")
    rows = []
    for n in parse_csv_list(args.n, int):
        for scale in parse_csv_list(args.scales, float):
            for seed in parse_csv_list(args.seeds, int):
                rows.append(run_trial(
                    n=n,
                    d=args.d,
                    c=args.c,
                    queries=args.queries,
                    panel_kind=args.panel,
                    scale=scale,
                    seed=seed,
                    b_count=args.B,
                    ref_samples=args.ref_samples,
                    balance_iters=args.balance_iters,
                    near_correlation=args.near_corr,
                ))
    write_rows(rows, args.csv)


if __name__ == "__main__":
    main()
