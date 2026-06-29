"""Diagnostics for the finite restricted-channel conjecture.

Given a finite stochastic channel K_x(j), this module computes the restricted
top-eta negative moment

    H_eta(p,q) = sum_j pi_j^alpha K_q(j)^(1-alpha)
                 (R_j(p)^alpha - tau_j^alpha)_+

where pi_j is the empirical output mass, R_j(x)=K_x(j)/pi_j, and tau_j is the
top-eta posterior threshold for outcome j.  The quantity to track is
g(q)+H_eta(p,q), especially its lower quantiles over near pairs.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np


_EPS = 1e-300
PANEL_KINDS = ("gaussian", "whitened_gaussian", "cross_polytope", "pca", "landmark")


@dataclass
class DiagnosticResult:
    h_eta: np.ndarray
    guard_density: np.ndarray
    score: np.ndarray
    affinity: np.ndarray
    soft_surplus: np.ndarray
    tilted_top_mass: np.ndarray
    top_contribution_fraction: np.ndarray
    top2_contribution_fraction: np.ndarray
    top4_contribution_fraction: np.ndarray
    top8_contribution_fraction: np.ndarray
    margin_mass: np.ndarray
    margin_bound: np.ndarray
    pi: np.ndarray
    tau: np.ndarray
    eta: float
    alpha: float
    margin_delta: float

    def summary(self) -> dict[str, float]:
        qs = [0.0, 0.01, 0.05, 0.1, 0.5]
        out: dict[str, float] = {
            "eta": float(self.eta),
            "alpha": float(self.alpha),
            "pi_min": float(self.pi.min()),
            "pi_max": float(self.pi.max()),
            "tau_min": float(self.tau.min()),
            "tau_max": float(self.tau.max()),
            "h_mean": float(self.h_eta.mean()),
            "guard_mean": float(self.guard_density.mean()),
            "score_mean": float(self.score.mean()),
            "affinity_mean": float(self.affinity.mean()),
            "soft_surplus_mean": float(self.soft_surplus.mean()),
            "tilted_top_mass_mean": float(self.tilted_top_mass.mean()),
            "top_contribution_fraction_mean": float(self.top_contribution_fraction.mean()),
            "top2_contribution_fraction_mean": float(self.top2_contribution_fraction.mean()),
            "top4_contribution_fraction_mean": float(self.top4_contribution_fraction.mean()),
            "top8_contribution_fraction_mean": float(self.top8_contribution_fraction.mean()),
            "margin_delta": float(self.margin_delta),
            "margin_mass_mean": float(self.margin_mass.mean()),
            "margin_bound_mean": float(self.margin_bound.mean()),
        }
        for q in qs:
            out[f"h_q{q:g}"] = float(np.quantile(self.h_eta, q))
            out[f"score_q{q:g}"] = float(np.quantile(self.score, q))
            out[f"affinity_q{q:g}"] = float(np.quantile(self.affinity, q))
            out[f"soft_surplus_q{q:g}"] = float(np.quantile(self.soft_surplus, q))
            out[f"tilted_top_mass_q{q:g}"] = float(np.quantile(self.tilted_top_mass, q))
            out[f"top_contribution_fraction_q{q:g}"] = float(
                np.quantile(self.top_contribution_fraction, q))
            out[f"top2_contribution_fraction_q{q:g}"] = float(
                np.quantile(self.top2_contribution_fraction, q))
            out[f"top4_contribution_fraction_q{q:g}"] = float(
                np.quantile(self.top4_contribution_fraction, q))
            out[f"top8_contribution_fraction_q{q:g}"] = float(
                np.quantile(self.top8_contribution_fraction, q))
            out[f"margin_mass_q{q:g}"] = float(np.quantile(self.margin_mass, q))
            out[f"margin_bound_q{q:g}"] = float(np.quantile(self.margin_bound, q))
        return out


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def balance_softmax_offsets(
    scores: np.ndarray,
    *,
    target: np.ndarray | None = None,
    max_iter: int = 200,
    damping: float = 1.0,
    tol: float = 1e-4,
) -> np.ndarray:
    """Find offsets b so softmax(scores+b) has approximately target column mass."""
    m, b_count = scores.shape
    if target is None:
        target = np.full(b_count, 1.0 / b_count)
    target = np.asarray(target, dtype=float)
    target = target / target.sum()

    offsets = np.zeros(b_count)
    for _ in range(max_iter):
        probs = stable_softmax(scores + offsets)
        pi = probs.mean(axis=0)
        err = np.max(np.abs(pi - target) / np.maximum(target, _EPS))
        if err <= tol:
            break
        offsets += damping * (np.log(np.maximum(target, _EPS)) - np.log(np.maximum(pi, _EPS)))
        offsets -= offsets.mean()
    return offsets


def make_panel(
    d: int,
    b_count: int,
    *,
    kind: str = "gaussian",
    seed: int = 0,
    data: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        return rng.standard_normal((b_count, d)) / math.sqrt(d)
    if kind == "whitened_gaussian":
        if data is None:
            raise ValueError("whitened_gaussian requires data")
        centered = data - data.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(len(centered), 1)
        vals, vecs = np.linalg.eigh(cov + 1e-6 * np.eye(d))
        whitening = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        return (rng.standard_normal((b_count, d)) @ whitening) / math.sqrt(d)
    if kind == "cross_polytope":
        panel = np.zeros((b_count, d))
        coords = np.arange(b_count) % d
        signs = np.where((np.arange(b_count) // d) % 2 == 0, 1.0, -1.0)
        panel[np.arange(b_count), coords] = signs
        return panel
    if kind == "pca":
        if data is None:
            raise ValueError("pca requires data")
        centered = data - data.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(len(centered), 1)
        vals, vecs = np.linalg.eigh(cov)
        dirs = vecs[:, np.argsort(vals)[::-1]].T
        coords = np.arange(b_count) % d
        signs = np.where((np.arange(b_count) // d) % 2 == 0, 1.0, -1.0)
        return signs[:, None] * dirs[coords]
    if kind == "landmark":
        if data is None:
            raise ValueError("landmark requires data")
        centered = data - data.mean(axis=0, keepdims=True)
        if len(centered) == 0:
            raise ValueError("landmark requires nonempty data")
        idx = rng.integers(0, len(centered), size=b_count)
        panel = centered[idx].copy()
        norm = np.linalg.norm(panel, axis=1, keepdims=True)
        bad = norm[:, 0] <= 1e-12
        if np.any(bad):
            panel[bad] = rng.standard_normal((int(np.sum(bad)), d))
            norm = np.linalg.norm(panel, axis=1, keepdims=True)
        return panel / np.maximum(norm, 1e-12)
    raise ValueError(f"unknown panel kind: {kind}")


def softmax_channel(data: np.ndarray, panel: np.ndarray, offsets: np.ndarray | None = None) -> np.ndarray:
    logits = data @ panel.T
    if offsets is not None:
        logits = logits + offsets
    return stable_softmax(logits)


def posterior_thresholds(k_data: np.ndarray, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0 < eta <= 1):
        raise ValueError("eta must lie in (0, 1]")
    m = k_data.shape[0]
    pi = np.maximum(k_data.mean(axis=0), _EPS)
    ratios = k_data / pi
    top_k = max(1, int(math.ceil(eta * m)))
    kth_index = m - top_k
    tau = np.partition(ratios, kth_index, axis=0)[kth_index]
    return pi, ratios, tau


def h_eta_for_pairs(
    k_data: np.ndarray,
    k_query: np.ndarray,
    near_indices: np.ndarray,
    *,
    eta: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pi, ratios, tau = posterior_thresholds(k_data, eta)
    r_near = ratios[np.asarray(near_indices, dtype=int)]
    positive = np.maximum(np.power(np.maximum(r_near, _EPS), alpha) - tau[None, :] ** alpha, 0.0)
    weights = (pi[None, :] ** alpha) * (np.maximum(k_query, _EPS) ** (1.0 - alpha))
    return np.sum(weights * positive, axis=1), pi, tau


def posterior_margin_certificate(
    k_data: np.ndarray,
    k_query: np.ndarray,
    near_indices: np.ndarray,
    pi: np.ndarray,
    tau: np.ndarray,
    *,
    alpha: float,
    margin_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return affinity, tilted good mass, and the rank-margin lower bound."""
    if margin_delta < 0:
        raise ValueError("margin_delta must be nonnegative")
    near = np.asarray(near_indices, dtype=int)
    k_near = np.maximum(k_data[near], _EPS)
    k_query = np.maximum(k_query, _EPS)
    affinity_terms = (k_near ** alpha) * (k_query ** (1.0 - alpha))
    affinity = np.sum(affinity_terms, axis=1)

    ratios = k_near / pi[None, :]
    good = ratios >= math.exp(margin_delta) * tau[None, :]
    good_affinity = np.sum(np.where(good, affinity_terms, 0.0), axis=1)
    margin_mass = np.divide(good_affinity, affinity, out=np.zeros_like(affinity), where=affinity > 0)
    margin_bound = (1.0 - math.exp(-alpha * margin_delta)) * good_affinity
    return affinity, margin_mass, margin_bound


def tilted_surplus_statistics(
    k_data: np.ndarray,
    k_query: np.ndarray,
    near_indices: np.ndarray,
    pi: np.ndarray,
    tau: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Return affinity, average surplus, and top-k contribution statistics.

    The identity

        H_eta = A_beta E_W[(1 - (tau_j/R_j(p))^alpha)_+]

    separates two possible ways the finite-channel diagnostic can be large:
    either the tilted law is spread over many mildly positive outcomes, or the
    top tilted outcome already carries most of the positive surplus.  The latter
    is the regime relevant to a single-leader reporter.
    """
    near = np.asarray(near_indices, dtype=int)
    k_near = np.maximum(k_data[near], _EPS)
    k_query = np.maximum(k_query, _EPS)
    affinity_terms = (k_near ** alpha) * (k_query ** (1.0 - alpha))
    affinity = np.sum(affinity_terms, axis=1)

    ratios = k_near / pi[None, :]
    surplus = np.maximum(1.0 - (tau[None, :] / np.maximum(ratios, _EPS)) ** alpha, 0.0)
    contributions = affinity_terms * surplus
    h_eta = np.sum(contributions, axis=1)
    soft_surplus = np.divide(h_eta, affinity, out=np.zeros_like(h_eta), where=affinity > 0)

    order = np.argsort(-affinity_terms, axis=1)
    rows = np.arange(len(order))
    top = order[:, 0]
    tilted_top_mass = np.divide(
        affinity_terms[rows, top],
        affinity,
        out=np.zeros_like(affinity),
        where=affinity > 0,
    )
    top_contribution_fraction = np.divide(
        contributions[rows, top],
        h_eta,
        out=np.zeros_like(h_eta),
        where=h_eta > 0,
    )
    topk_contribution = {}
    for k in (2, 4, 8):
        kk = min(k, affinity_terms.shape[1])
        chosen = order[:, :kk]
        cumulative = np.sum(np.take_along_axis(contributions, chosen, axis=1), axis=1)
        topk_contribution[k] = np.divide(
            cumulative,
            h_eta,
            out=np.zeros_like(h_eta),
            where=h_eta > 0,
        )
    return affinity, soft_surplus, tilted_top_mass, top_contribution_fraction, topk_contribution


def guard_density(data: np.ndarray, queries: np.ndarray, *, c: float, r: float) -> np.ndarray:
    threshold = c * r
    out = np.empty(len(queries), dtype=float)
    for i, q in enumerate(queries):
        out[i] = np.mean(np.linalg.norm(data - q, axis=1) <= threshold)
    return out


def diagnose_channel(
    data: np.ndarray,
    queries: np.ndarray,
    near_indices: np.ndarray,
    k_data: np.ndarray,
    k_query: np.ndarray,
    *,
    c: float,
    r: float,
    eta: float | None = None,
    alpha: float | None = None,
    margin_delta: float | None = None,
) -> DiagnosticResult:
    m = len(data)
    if eta is None:
        eta = m ** (-1.0 / (2.0 * c * c))
    if alpha is None:
        alpha = 1.0 / max(math.log(max(m, 3)), 1.0)
    h_eta, pi, tau = h_eta_for_pairs(k_data, k_query, near_indices, eta=eta, alpha=alpha)
    if margin_delta is None:
        margin_delta = 1.0 / alpha
    (
        affinity2,
        soft_surplus,
        tilted_top_mass,
        top_contribution_fraction,
        topk_contribution,
    ) = tilted_surplus_statistics(
        k_data, k_query, near_indices, pi, tau, alpha=alpha)
    affinity, margin_mass, margin_bound = posterior_margin_certificate(
        k_data, k_query, near_indices, pi, tau, alpha=alpha, margin_delta=margin_delta)
    if not np.allclose(affinity, affinity2):
        raise AssertionError("affinity decomposition mismatch")
    guard = guard_density(data, queries, c=c, r=r)
    return DiagnosticResult(h_eta=h_eta, guard_density=guard, score=guard + h_eta,
                            affinity=affinity, soft_surplus=soft_surplus,
                            tilted_top_mass=tilted_top_mass,
                            top_contribution_fraction=top_contribution_fraction,
                            top2_contribution_fraction=topk_contribution[2],
                            top4_contribution_fraction=topk_contribution[4],
                            top8_contribution_fraction=topk_contribution[8],
                            margin_mass=margin_mass, margin_bound=margin_bound, pi=pi, tau=tau,
                            eta=eta, alpha=alpha, margin_delta=margin_delta)


def effective_support(k: np.ndarray) -> np.ndarray:
    return 1.0 / np.sum(k * k, axis=1)


def softmax_hessian_condition(panel: np.ndarray, probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    out = np.empty(len(probs), dtype=float)
    for i, p in enumerate(probs):
        mean = p @ panel
        centered = panel - mean
        cov = centered.T @ (centered * p[:, None])
        vals = np.linalg.eigvalsh(cov)
        vals = vals[vals > eps]
        out[i] = np.inf if len(vals) == 0 else float(vals[-1] / vals[0])
    return out


def _sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def synthetic_near_pairs(
    n: int,
    d: int,
    c: float,
    n_queries: int,
    seed: int,
    *,
    near_correlation: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    data = _sphere(n, d, rng)
    near_indices = rng.integers(0, n, size=n_queries)
    a = 1.0 - 1.0 / (c * c) if near_correlation is None else near_correlation
    if not (-1.0 < a < 1.0):
        raise ValueError("near_correlation must lie in (-1, 1)")
    r = math.sqrt(2.0 - 2.0 * a)
    queries = []
    for idx in near_indices:
        p = data[idx]
        w = rng.standard_normal(d)
        w -= (w @ p) * p
        w /= np.linalg.norm(w)
        queries.append(a * p + math.sqrt(1.0 - a * a) * w)
    return data, np.asarray(queries), near_indices, r


_synthetic_near_pairs = synthetic_near_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure g(q)+H_eta(p,q) for a balanced softmax channel.")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--B", type=int, default=0, help="channel outcomes; default is ceil(m^(1/(2c^2)))")
    parser.add_argument("--panel", choices=PANEL_KINDS, default="gaussian")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply panel scores by this inverse-temperature scale")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--balance-iters", type=int, default=200)
    parser.add_argument("--near-corr", type=float, default=None,
                        help="near-pair sphere correlation; default is 1-1/c^2")
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error("--scale must be positive")

    data, queries, near_indices, r = synthetic_near_pairs(
        args.n, args.d, args.c, args.queries, args.seed, near_correlation=args.near_corr)
    eta = args.n ** (-1.0 / (2.0 * args.c * args.c))
    b_count = args.B or int(math.ceil(1.0 / eta))

    panel = args.scale * make_panel(args.d, b_count, kind=args.panel, seed=args.seed + 1, data=data)
    scores = data @ panel.T
    offsets = balance_softmax_offsets(scores, max_iter=args.balance_iters)
    k_data = stable_softmax(scores + offsets)
    k_query = softmax_channel(queries, panel, offsets)
    result = diagnose_channel(data, queries, near_indices, k_data, k_query, c=args.c, r=r, eta=eta)

    k_z = softmax_channel((result.alpha * data[near_indices] + (1.0 - result.alpha) * queries), panel, offsets)
    support = effective_support(k_z)
    cond = softmax_hessian_condition(panel, k_z)

    summary = result.summary()
    summary["B"] = float(b_count)
    summary["scale"] = float(args.scale)
    summary["effective_support_q05"] = float(np.quantile(support, 0.05))
    summary["effective_support_median"] = float(np.median(support))
    summary["hessian_cond_median"] = float(np.median(cond[np.isfinite(cond)])) if np.any(np.isfinite(cond)) else float("inf")
    for key in sorted(summary):
        print(f"{key:>24s}: {summary[key]:.6g}")


if __name__ == "__main__":
    main()
