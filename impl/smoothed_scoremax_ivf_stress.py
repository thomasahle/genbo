"""Stress centroid/IVF routing for the smoothed score-max primitive.

For unit-norm data, maximizing

    A_Y(x) = <Y, x> - t ||x||^2 / 2

is the same as maximizing the inner product with the perturbed center
``y = Y/t``.  This diagnostic tests the natural IVF route: assign points to
data-dependent centroid buckets, route ``y`` to a few best centroids, and exact
rerank by the smoothed score inside the reported buckets.

This is a candidate implementation test for ``asm:smoothed-score-max``.  It is
not a new proof form; failure only rules out this simple centroid-route path.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Callable, Iterable

import numpy as np

from smoothed_scoremax_stress import (
    _candidate_best_loss,
    _quantile_with_inf,
    promised_spherical_instance,
    rho_for_c,
    smoothed_scores,
)


FIELDNAMES = [
    "n",
    "d",
    "c",
    "rho",
    "near_corr",
    "far_corr_max",
    "t_scale",
    "router",
    "centers",
    "center_exponent",
    "center_mult",
    "lloyd_iters",
    "probes",
    "bucket_mean",
    "bucket_q90",
    "query_trials",
    "score_gap_fail_rate",
    "oracle_top1_valid_rate",
    "ivf_p_bucket_rank",
    "ivf_p_rate",
    "ivf_scoremax_rate",
    "ivf_additive_success_rate",
    "ivf_candidate_mean",
    "ivf_candidate_q90",
    "ivf_candidate_over_nrho",
    "ivf_best_loss_q50",
    "ivf_best_loss_q90",
    "ivf_best_loss_over_margin_q90",
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


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class CentroidRouter:
    """Small dependency-free cosine IVF router."""

    def __init__(
        self,
        *,
        centers: int,
        mode: str,
        lloyd_iters: int,
        seed: int,
    ) -> None:
        if centers < 1:
            raise ValueError("centers must be positive")
        if mode not in {"sample", "lloyd", "kmeanspp"}:
            raise ValueError("mode must be 'sample', 'lloyd', or 'kmeanspp'")
        if lloyd_iters < 0:
            raise ValueError("lloyd_iters must be nonnegative")
        self.centers = centers
        self.mode = mode
        self.lloyd_iters = lloyd_iters
        self.seed = seed

    def build(self, x: np.ndarray) -> "CentroidRouter":
        self.x = np.asarray(x, dtype=float)
        n, d = self.x.shape
        k = min(self.centers, n)
        rng = np.random.default_rng(self.seed)
        if self.mode == "kmeanspp":
            init = self._kmeanspp_indices(k, rng)
        else:
            init = rng.choice(n, size=k, replace=False)
        centers = self.x[init].copy()
        centers = _normalize_rows(centers)

        if self.mode in {"lloyd", "kmeanspp"}:
            for _ in range(self.lloyd_iters):
                labels = self._assign_to(centers)
                new_centers = centers.copy()
                for j in range(k):
                    idx = np.flatnonzero(labels == j)
                    if len(idx):
                        new_centers[j] = self.x[idx].mean(axis=0)
                centers = _normalize_rows(new_centers)

        self.center_vecs = centers
        self.labels = self._assign_to(centers)
        self.buckets = [np.flatnonzero(self.labels == j) for j in range(k)]
        self.bucket_sizes = np.array([len(b) for b in self.buckets], dtype=float)
        return self

    def _assign_to(self, centers: np.ndarray) -> np.ndarray:
        return np.argmax(self.x @ centers.T, axis=1)

    def _kmeanspp_indices(self, k: int, rng: np.random.Generator) -> np.ndarray:
        n = len(self.x)
        chosen = np.empty(k, dtype=int)
        chosen[0] = int(rng.integers(n))
        # For unit vectors, squared Euclidean distance to a center is
        # 2 - 2<x,c>.  Keep the best squared distance so far.
        best_d2 = np.maximum(0.0, 2.0 - 2.0 * (self.x @ self.x[chosen[0]]))
        for j in range(1, k):
            total = float(np.sum(best_d2))
            if total <= 1e-12:
                remaining = np.setdiff1d(np.arange(n), chosen[:j], assume_unique=False)
                chosen[j:] = rng.choice(remaining, size=k - j, replace=False)
                break
            idx = int(rng.choice(n, p=best_d2 / total))
            chosen[j] = idx
            new_d2 = np.maximum(0.0, 2.0 - 2.0 * (self.x @ self.x[idx]))
            best_d2 = np.minimum(best_d2, new_d2)
            best_d2[chosen[: j + 1]] = 0.0
        return chosen

    def bucket_rank(self, point_index: int, y: np.ndarray) -> int:
        label = int(self.labels[point_index])
        scores = self.center_vecs @ y
        return 1 + int(np.sum(scores > scores[label]))

    def query_candidates(self, y: np.ndarray, probes: int) -> set[int]:
        if probes < 1:
            raise ValueError("probes must be positive")
        scores = self.center_vecs @ y
        p = min(probes, len(scores))
        top = np.argpartition(-scores, p - 1)[:p]
        parts = [self.buckets[int(j)] for j in top if len(self.buckets[int(j)])]
        if not parts:
            return set()
        return set(int(i) for i in np.concatenate(parts))


def run_trial(
    *,
    n: int,
    d: int,
    c: float,
    far_corr_max: float,
    t_scale: float,
    router: str,
    center_exponent: float,
    center_mult: float,
    lloyd_iters: int,
    probes: int,
    query_trials: int,
    seed: int,
) -> dict[str, float | int | str]:
    if n < 2:
        raise ValueError("n must be at least two")
    if d < 2:
        raise ValueError("d must be at least two")
    if center_exponent <= 0.0 or center_mult <= 0.0:
        raise ValueError("center parameters must be positive")
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
    center_count = max(1, min(n, int(math.ceil(center_mult * n ** center_exponent))))
    ivf = CentroidRouter(
        centers=center_count,
        mode=router,
        lloyd_iters=lloyd_iters,
        seed=seed + 31,
    ).build(x)

    score_gap_failures = 0
    oracle_valid = 0
    ivf_p = 0
    ivf_scoremax = 0
    ivf_additive = 0
    counts = []
    losses = []
    bucket_ranks = []

    for _ in range(query_trials):
        scores, y, _t, margin = smoothed_scores(
            x, q, c=c, t_scale=t_scale, rng=rng)
        p_score = float(scores[p_index])
        if np.max(scores[:p_index]) > p_score - margin:
            score_gap_failures += 1
        top = int(np.argmax(scores))
        oracle_valid += int(top == p_index)

        bucket_ranks.append(ivf.bucket_rank(p_index, y))
        candidates = ivf.query_candidates(y, probes=probes)
        counts.append(len(candidates))
        loss = _candidate_best_loss(scores, candidates)
        losses.append(loss)
        ivf_p += int(p_index in candidates)
        ivf_scoremax += int(top in candidates)
        ivf_additive += int(loss <= margin)

    counts_arr = np.asarray(counts, dtype=float)
    losses_arr = np.asarray(losses, dtype=float)
    n_rho = n ** rho
    near_corr = 1.0 - 1.0 / (c * c)
    return {
        "n": n,
        "d": d,
        "c": c,
        "rho": rho,
        "near_corr": near_corr,
        "far_corr_max": far_corr_max,
        "t_scale": t_scale,
        "router": router,
        "centers": center_count,
        "center_exponent": center_exponent,
        "center_mult": center_mult,
        "lloyd_iters": lloyd_iters,
        "probes": probes,
        "bucket_mean": float(np.mean(ivf.bucket_sizes)),
        "bucket_q90": float(np.quantile(ivf.bucket_sizes, 0.9)),
        "query_trials": query_trials,
        "score_gap_fail_rate": score_gap_failures / query_trials,
        "oracle_top1_valid_rate": oracle_valid / query_trials,
        "ivf_p_bucket_rank": float(np.median(bucket_ranks)),
        "ivf_p_rate": ivf_p / query_trials,
        "ivf_scoremax_rate": ivf_scoremax / query_trials,
        "ivf_additive_success_rate": ivf_additive / query_trials,
        "ivf_candidate_mean": float(np.mean(counts_arr)),
        "ivf_candidate_q90": float(np.quantile(counts_arr, 0.9)),
        "ivf_candidate_over_nrho": float(np.mean(counts_arr) / n_rho),
        "ivf_best_loss_q50": _quantile_with_inf(losses_arr, 0.5),
        "ivf_best_loss_q90": _quantile_with_inf(losses_arr, 0.9),
        "ivf_best_loss_over_margin_q90": _quantile_with_inf(losses_arr / margin, 0.9),
        "seed": seed,
    }


def _rows(args: argparse.Namespace) -> Iterable[dict[str, float | int | str]]:
    rho = rho_for_c(args.c)
    default_exp = 1.0 - rho
    for n in parse_csv_list(args.n, int):
        for d in parse_csv_list(args.d, int):
            for router in parse_csv_list(args.routers, str):
                for center_mult in parse_csv_list(args.center_mults, float):
                    for probes in parse_csv_list(args.probes, int):
                        for seed in parse_csv_list(args.seeds, int):
                            yield run_trial(
                                n=n,
                                d=d,
                                c=args.c,
                                far_corr_max=args.far_corr_max,
                                t_scale=args.t_scale,
                                router=router,
                                center_exponent=args.center_exponent or default_exp,
                                center_mult=center_mult,
                                lloyd_iters=args.lloyd_iters,
                                probes=probes,
                                query_trials=args.query_trials,
                                seed=seed,
                            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default="1000,3000")
    parser.add_argument("--d", default="24")
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--far-corr-max", type=float, default=-0.05)
    parser.add_argument("--t-scale", type=float, default=64.0)
    parser.add_argument("--routers", default="sample,lloyd,kmeanspp")
    parser.add_argument("--center-exponent", type=float, default=0.0)
    parser.add_argument("--center-mults", default="1.0")
    parser.add_argument("--lloyd-iters", type=int, default=4)
    parser.add_argument("--probes", default="1,2,4,8")
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
