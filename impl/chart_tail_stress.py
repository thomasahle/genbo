"""Stress test the chart-incidence tilted-overlap target.

The frozen target is the weighted positive-overlap tail in paper.tex around
cor:positive-overlap-chart-shadow-tail.  This script samples balanced q-ary
code panels, draws Gaussian symbol scores, forms the internal-gap prices

    omega_j = exp(lambda * (M - W_j)),

and compares the full weighted pair tail with the same statistic after clipping
omega to a polylog-sized cap.  A large full-price diagonal share is evidence for
the singleton/localized-residual barrier rather than for another certificate
reformulation.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterable

import numpy as np


FIELDNAMES = [
    "seed",
    "n_words",
    "blocks",
    "alphabet",
    "lambda",
    "eta",
    "price_cap",
    "leader",
    "score_max",
    "score_min",
    "gap_max",
    "omega_max",
    "omega_sum",
    "omega_eff_count",
    "unweighted_pair_mgf",
    "unweighted_offdiag_pair_mgf",
    "unweighted_diag_share",
    "full_pair_mgf",
    "full_offdiag_pair_mgf",
    "clipped_pair_mgf",
    "clipped_offdiag_pair_mgf",
    "full_overlap_energy",
    "clipped_overlap_energy",
    "full_diag_share",
    "clipped_diag_share",
    "full_leader_overlap_excess",
    "clipped_leader_overlap_excess",
    "uniform_singleton_cost",
    "clipped_singleton_cost",
]


def make_balanced_code(n_words: int, blocks: int, alphabet: int, seed: int) -> np.ndarray:
    """Return an n_words x blocks q-ary array balanced in every coordinate."""
    if n_words <= 0:
        raise ValueError("n_words must be positive")
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    if alphabet <= 1:
        raise ValueError("alphabet must be at least 2")

    rng = np.random.default_rng(seed)
    code = np.empty((n_words, blocks), dtype=np.int16)
    base = np.arange(n_words, dtype=np.int64) % alphabet
    for i in range(blocks):
        code[:, i] = rng.permutation(base)
    return code


def centered_symbol_scores(blocks: int, alphabet: int, seed: int) -> np.ndarray:
    """Independent Gaussian symbol scores, centered inside each coordinate."""
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((blocks, alphabet))
    return scores - scores.mean(axis=1, keepdims=True)


def codeword_scores(code: np.ndarray, symbol_scores: np.ndarray) -> np.ndarray:
    blocks = code.shape[1]
    if symbol_scores.shape[0] != blocks:
        raise ValueError("symbol_scores block count does not match code")
    rows = np.arange(blocks)[:, None]
    return symbol_scores[rows, code.T].sum(axis=0)


def centered_overlap_matrix(code: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Centered pair overlap A_{j,k} - V/q for optional coordinate weights."""
    n_words, blocks = code.shape
    alphabet = int(code.max()) + 1
    if weights is None:
        weights = np.ones(blocks)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (blocks,):
        raise ValueError("weights must have one entry per block")

    overlap = np.zeros((n_words, n_words), dtype=float)
    for i, weight in enumerate(weights):
        overlap += weight * (code[:, i, None] == code[None, :, i])
    return overlap - weights.sum() / alphabet


def _pair_stats(centered_overlap: np.ndarray, omega: np.ndarray, eta: float) -> tuple[float, float, float]:
    positive = centered_overlap >= 0.0
    pair_weight = omega[:, None] * omega[None, :]
    raw_mgf = float(np.sum(pair_weight * positive * np.exp(eta * centered_overlap)))
    raw_energy = float(np.sum(pair_weight * np.maximum(centered_overlap, 0.0)))
    diag_mgf = float(np.sum((omega * omega) * np.exp(eta * np.diag(centered_overlap))))
    diag_share = diag_mgf / raw_mgf if raw_mgf > 0 else 0.0
    norm = float(np.sum(omega) ** 2)
    return raw_mgf / norm, raw_energy / norm, diag_share


def _leader_overlap_excess(code: np.ndarray, omega: np.ndarray, leader: int, alphabet: int) -> float:
    total = float(np.sum(omega))
    if total <= 0:
        return 0.0
    loads = []
    for i in range(code.shape[1]):
        same = code[:, i] == code[leader, i]
        loads.append(alphabet * float(np.sum(omega[same])) / total - 1.0)
    return float(max(0.0, max(loads)))


def run_trial(
    *,
    n_words: int,
    blocks: int,
    alphabet: int,
    seed: int,
    lam: float | None = None,
    eta: float | None = None,
    price_cap: float | None = None,
) -> dict[str, float | int]:
    if lam is None:
        lam = math.sqrt(2.0 * math.log(n_words) / blocks)
    if eta is None:
        variance = blocks * (1.0 / alphabet) * (1.0 - 1.0 / alphabet)
        eta = 1.0 / math.sqrt(max(variance, 1e-12))
    if price_cap is None:
        price_cap = math.log(max(n_words, 3)) ** 3
    if lam < 0:
        raise ValueError("lambda must be nonnegative")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if price_cap <= 0:
        raise ValueError("price_cap must be positive")

    code = make_balanced_code(n_words, blocks, alphabet, seed)
    scores = codeword_scores(code, centered_symbol_scores(blocks, alphabet, seed + 1009))
    leader = int(np.argmax(scores))
    score_max = float(scores[leader])
    score_min = float(np.min(scores))
    gaps = score_max - scores
    omega = np.exp(np.minimum(lam * gaps, 700.0))
    clipped = np.minimum(omega, price_cap)
    centered_overlap = centered_overlap_matrix(code)

    unweighted_pair_mgf, _, unweighted_diag_share = _pair_stats(
        centered_overlap, np.ones(n_words), eta)
    full_pair_mgf, full_overlap_energy, full_diag_share = _pair_stats(centered_overlap, omega, eta)
    clipped_pair_mgf, clipped_overlap_energy, clipped_diag_share = _pair_stats(
        centered_overlap, clipped, eta)

    omega_sum = float(np.sum(omega))
    omega_eff_count = omega_sum * omega_sum / float(np.sum(omega * omega))
    uniform_singleton_cost = (alphabet - 1.0) * blocks * float(np.max(omega))
    clipped_singleton_cost = (alphabet - 1.0) * blocks * float(np.max(clipped))

    return {
        "seed": seed,
        "n_words": n_words,
        "blocks": blocks,
        "alphabet": alphabet,
        "lambda": float(lam),
        "eta": float(eta),
        "price_cap": float(price_cap),
        "leader": leader,
        "score_max": score_max,
        "score_min": score_min,
        "gap_max": float(np.max(gaps)),
        "omega_max": float(np.max(omega)),
        "omega_sum": omega_sum,
        "omega_eff_count": float(omega_eff_count),
        "unweighted_pair_mgf": float(unweighted_pair_mgf),
        "unweighted_offdiag_pair_mgf": float(unweighted_pair_mgf * (1.0 - unweighted_diag_share)),
        "unweighted_diag_share": float(unweighted_diag_share),
        "full_pair_mgf": float(full_pair_mgf),
        "full_offdiag_pair_mgf": float(full_pair_mgf * (1.0 - full_diag_share)),
        "clipped_pair_mgf": float(clipped_pair_mgf),
        "clipped_offdiag_pair_mgf": float(clipped_pair_mgf * (1.0 - clipped_diag_share)),
        "full_overlap_energy": float(full_overlap_energy),
        "clipped_overlap_energy": float(clipped_overlap_energy),
        "full_diag_share": float(full_diag_share),
        "clipped_diag_share": float(clipped_diag_share),
        "full_leader_overlap_excess": _leader_overlap_excess(code, omega, leader, alphabet),
        "clipped_leader_overlap_excess": _leader_overlap_excess(code, clipped, leader, alphabet),
        "uniform_singleton_cost": float(uniform_singleton_cost),
        "clipped_singleton_cost": float(clipped_singleton_cost),
    }


def summarize(rows: Iterable[dict[str, float | int]]) -> dict[str, float]:
    rows = list(rows)
    out: dict[str, float] = {}
    for key in FIELDNAMES:
        if key in {"seed", "leader"}:
            continue
        values = np.array([float(row[key]) for row in rows], dtype=float)
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_max"] = float(np.max(values))
    return out


def write_rows(rows: list[dict[str, float | int]], output: str | None) -> None:
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
    parser = argparse.ArgumentParser(description="Stress test chart-incidence tilted-overlap tails.")
    parser.add_argument("--n-words", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--alphabet", type=int, default=8)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda", dest="lam", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--price-cap", type=float, default=None)
    parser.add_argument("--cap-power", type=float, default=3.0)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    price_cap = args.price_cap
    if price_cap is None:
        price_cap = math.log(max(args.n_words, 3)) ** args.cap_power

    rows = [
        run_trial(
            n_words=args.n_words,
            blocks=args.blocks,
            alphabet=args.alphabet,
            seed=args.seed + t,
            lam=args.lam,
            eta=args.eta,
            price_cap=price_cap,
        )
        for t in range(args.trials)
    ]
    if args.summary:
        for key, value in summarize(rows).items():
            print(f"{key},{value}")
    else:
        write_rows(rows, args.csv)


if __name__ == "__main__":
    main()
