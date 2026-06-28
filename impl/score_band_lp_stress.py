"""Tiny finite score-band LP verifier.

This is a direct numerical stress test for the frozen condition (LNN2) in
paper.tex, restricted to the product-simplex relaxation and small panels.  It
fixes a block-minimum chart and a block-maximum chart, then solves the resulting
ordinary LP with a small dependency-free simplex routine.

The script is intentionally for toy instances.  It is meant to distinguish
actual score-band violations from coefficient-tail surrogates before more
theory is added.
"""

from __future__ import annotations

import argparse
import itertools
import math

import numpy as np

from chart_tail_stress import (
    centered_symbol_scores,
    codeword_scores,
    make_balanced_code,
)


class LPUnbounded(RuntimeError):
    pass


def _simplex_max_nonnegative(
    objective: np.ndarray,
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    tol: float = 1e-9,
    max_iter: int = 10000,
) -> tuple[float, np.ndarray]:
    """Maximize objective @ x subject to lhs @ x <= rhs, x >= 0.

    The origin must be feasible.  This is enough for the chart LPs below after
    splitting free variables into positive and negative parts.
    """
    objective = np.asarray(objective, dtype=float)
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if lhs.ndim != 2:
        raise ValueError("lhs must be a matrix")
    if objective.shape != (lhs.shape[1],):
        raise ValueError("objective dimension does not match lhs")
    if rhs.shape != (lhs.shape[0],):
        raise ValueError("rhs dimension does not match lhs")
    if np.any(rhs < -tol):
        raise ValueError("the origin must be feasible")

    rows, cols = lhs.shape
    tableau = np.zeros((rows + 1, cols + rows + 1), dtype=float)
    tableau[:rows, :cols] = lhs
    tableau[:rows, cols:cols + rows] = np.eye(rows)
    tableau[:rows, -1] = np.maximum(rhs, 0.0)
    tableau[-1, :cols] = -objective
    basis = list(range(cols, cols + rows))

    for _ in range(max_iter):
        entering_candidates = np.flatnonzero(tableau[-1, :-1] < -tol)
        if len(entering_candidates) == 0:
            solution = np.zeros(cols + rows, dtype=float)
            solution[basis] = tableau[:rows, -1]
            return float(tableau[-1, -1]), solution[:cols]

        entering = int(entering_candidates[0])
        column = tableau[:rows, entering]
        positive = np.flatnonzero(column > tol)
        if len(positive) == 0:
            raise LPUnbounded("LP is unbounded")

        ratios = tableau[positive, -1] / column[positive]
        min_ratio = float(np.min(ratios))
        tied = positive[np.flatnonzero(ratios <= min_ratio + tol)]
        leaving_row = int(tied[np.argmin([basis[i] for i in tied])])

        pivot = tableau[leaving_row, entering]
        tableau[leaving_row, :] /= pivot
        for row in range(rows + 1):
            if row == leaving_row:
                continue
            factor = tableau[row, entering]
            if abs(factor) > tol:
                tableau[row, :] -= factor * tableau[leaving_row, :]
        basis[leaving_row] = entering

    raise RuntimeError("simplex iteration limit exceeded")


def solve_free_lp_max(
    objective: np.ndarray,
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    tol: float = 1e-9,
) -> tuple[float, np.ndarray]:
    """Maximize a free-variable LP by splitting each variable into two parts."""
    objective = np.asarray(objective, dtype=float)
    lhs = np.asarray(lhs, dtype=float)
    split_objective = np.concatenate([objective, -objective])
    split_lhs = np.concatenate([lhs, -lhs], axis=1)
    value, split_solution = _simplex_max_nonnegative(
        split_objective,
        split_lhs,
        rhs,
        tol=tol,
    )
    n_vars = len(objective)
    solution = split_solution[:n_vars] - split_solution[n_vars:]
    return value, solution


def code_indicator_matrix(code: np.ndarray, alphabet: int) -> np.ndarray:
    n_words, blocks = code.shape
    matrix = np.zeros((blocks * alphabet, n_words), dtype=float)
    for j, word in enumerate(code):
        for i, symbol in enumerate(word):
            matrix[i * alphabet + int(symbol), j] = 1.0
    return matrix


def span_basis(matrix: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular_values > tol))
    return u[:, :rank]


def chart_score_band_gap(
    code: np.ndarray,
    omega: np.ndarray,
    min_chart: tuple[int, ...],
    max_chart: tuple[int, ...],
    *,
    tol: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Solve one fixed min/max chart LP for the product-simplex score-band gap."""
    code = np.asarray(code, dtype=int)
    omega = np.asarray(omega, dtype=float)
    n_words, blocks = code.shape
    alphabet = int(np.max(code)) + 1
    if omega.shape != (n_words,):
        raise ValueError("omega must have one entry per codeword")
    if len(min_chart) != blocks or len(max_chart) != blocks:
        raise ValueError("charts must have one symbol per block")
    if np.any(omega < -tol):
        raise ValueError("omega must be nonnegative")

    indicators = code_indicator_matrix(code, alphabet)
    basis = span_basis(indicators)
    dim = basis.shape[1]
    raw_scores = indicators.T @ basis

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for i in range(blocks):
        min_idx = i * alphabet + min_chart[i]
        max_idx = i * alphabet + max_chart[i]
        for a in range(alphabet):
            idx = i * alphabet + a

            # Y_{i,a} >= Y_{i,min_chart_i}
            row = np.zeros(dim + 1, dtype=float)
            row[:dim] = -(basis[idx] - basis[min_idx])
            rows.append(row)
            rhs.append(0.0)

            # Y_{i,max_chart_i} >= Y_{i,a}
            row = np.zeros(dim + 1, dtype=float)
            row[:dim] = basis[idx] - basis[max_idx]
            rows.append(row)
            rhs.append(0.0)

    for j in range(n_words):
        # <Y, c_j> <= m
        row = np.zeros(dim + 1, dtype=float)
        row[:dim] = raw_scores[j]
        row[dim] = -1.0
        rows.append(row)
        rhs.append(0.0)

        # m - <Y, c_j> <= omega_j
        row = np.zeros(dim + 1, dtype=float)
        row[:dim] = -raw_scores[j]
        row[dim] = 1.0
        rows.append(row)
        rhs.append(float(omega[j]))

    objective = np.zeros(dim + 1, dtype=float)
    for i, symbol in enumerate(max_chart):
        objective[:dim] += basis[i * alphabet + symbol]
    objective[dim] = -1.0

    value, solution = solve_free_lp_max(objective, np.vstack(rows), np.array(rhs), tol=tol)
    return max(0.0, value), solution


def product_simplex_score_band_gap(
    code: np.ndarray,
    omega: np.ndarray,
    *,
    tol: float = 1e-8,
) -> dict[str, object]:
    code = np.asarray(code, dtype=int)
    blocks = code.shape[1]
    alphabet = int(np.max(code)) + 1
    best_value = -math.inf
    best_min_chart: tuple[int, ...] | None = None
    best_max_chart: tuple[int, ...] | None = None
    best_solution: np.ndarray | None = None

    charts = list(itertools.product(range(alphabet), repeat=blocks))
    for min_chart in charts:
        for max_chart in charts:
            value, solution = chart_score_band_gap(
                code,
                omega,
                min_chart,
                max_chart,
                tol=tol,
            )
            if value > best_value:
                best_value = value
                best_min_chart = min_chart
                best_max_chart = max_chart
                best_solution = solution

    return {
        "gap": float(best_value),
        "min_chart": best_min_chart,
        "max_chart": best_max_chart,
        "solution": best_solution,
    }


def run_trial(
    *,
    n_words: int,
    blocks: int,
    alphabet: int,
    seed: int,
    xi: float,
    lam: float | None = None,
) -> dict[str, float | int | tuple[int, ...]]:
    if lam is None:
        lam = math.sqrt(2.0 * math.log(n_words) / blocks)
    code = make_balanced_code(n_words, blocks, alphabet, seed)
    symbol_scores = centered_symbol_scores(blocks, alphabet, seed + 1009)
    scores = codeword_scores(code, symbol_scores)
    gaps = float(np.max(scores)) - scores
    omega = np.exp(np.minimum(lam * gaps, 700.0)) + xi
    lp = product_simplex_score_band_gap(code, omega)
    gap = float(lp["gap"])
    return {
        "seed": seed,
        "n_words": n_words,
        "blocks": blocks,
        "alphabet": alphabet,
        "lambda": float(lam),
        "xi": float(xi),
        "gap": gap,
        "gap_over_xi": gap / xi if xi > 0 else math.inf,
        "omega_max": float(np.max(omega)),
        "score_gap_max": float(np.max(gaps)),
        "min_chart": lp["min_chart"],
        "max_chart": lp["max_chart"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny product-simplex score-band LP stress test.")
    parser.add_argument("--n-words", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--alphabet", type=int, default=2)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--xi", type=float, default=None)
    parser.add_argument("--cap-power", type=float, default=3.0)
    parser.add_argument("--lambda", dest="lam", type=float, default=None)
    args = parser.parse_args()
    xi = args.xi
    if xi is None:
        xi = math.log(max(args.n_words, 3)) ** args.cap_power

    rows = [
        run_trial(
            n_words=args.n_words,
            blocks=args.blocks,
            alphabet=args.alphabet,
            seed=args.seed + trial,
            xi=xi,
            lam=args.lam,
        )
        for trial in range(args.trials)
    ]
    keys = ["gap", "gap_over_xi", "omega_max", "score_gap_max"]
    for key in keys:
        values = np.array([float(row[key]) for row in rows])
        print(f"{key}_mean,{float(np.mean(values))}")
        print(f"{key}_max,{float(np.max(values))}")
    worst = max(rows, key=lambda row: float(row["gap_over_xi"]))
    print(f"worst_seed,{worst['seed']}")
    print(f"worst_min_chart,{worst['min_chart']}")
    print(f"worst_max_chart,{worst['max_chart']}")


if __name__ == "__main__":
    main()
