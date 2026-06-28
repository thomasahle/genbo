"""Tiny finite score-band LP verifier.

This is a direct numerical stress test for the frozen condition (LNN2) in
paper.tex, restricted to tiny relaxations.  For the product-simplex relaxation it
fixes a block-minimum chart and a block-maximum chart, then solves the resulting
ordinary LP with a small dependency-free simplex routine.  It can also fix one
candidate source word and test whether that integral pseudoword already violates
the same score-band condition.  Finally, it can sample vertices of a toy
local-marginal relaxation using a small equality-form phase-I simplex solver and
evaluate the same fixed-source score-band gap there.

The script is intentionally for toy instances.  It is meant to distinguish
actual score-band violations from coefficient-tail surrogates before more
theory is added.
"""

from __future__ import annotations

import argparse
import itertools
import math
from math import comb

import numpy as np

from chart_tail_stress import (
    centered_symbol_scores,
    codeword_scores,
    make_balanced_code,
)


class LPUnbounded(RuntimeError):
    pass


class LPInfeasible(RuntimeError):
    pass


def _pivot_tableau(
    tableau: np.ndarray,
    basis: list[int],
    leaving_row: int,
    entering: int,
    *,
    tol: float,
) -> None:
    pivot = tableau[leaving_row, entering]
    if abs(pivot) <= tol:
        raise RuntimeError("attempted to pivot on a zero entry")
    tableau[leaving_row, :] /= pivot
    for row in range(tableau.shape[0]):
        if row == leaving_row:
            continue
        factor = tableau[row, entering]
        if abs(factor) > tol:
            tableau[row, :] -= factor * tableau[leaving_row, :]
    basis[leaving_row] = entering


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

        _pivot_tableau(tableau, basis, leaving_row, entering, tol=tol)

    raise RuntimeError("simplex iteration limit exceeded")


def _simplex_max_with_basis(
    objective: np.ndarray,
    lhs: np.ndarray,
    rhs: np.ndarray,
    basis: list[int],
    *,
    tol: float = 1e-9,
    max_iter: int = 10000,
) -> tuple[float, np.ndarray, list[int], np.ndarray]:
    """Maximize objective @ x over lhs @ x = rhs, x >= 0 from a feasible basis."""
    objective = np.asarray(objective, dtype=float)
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if lhs.ndim != 2:
        raise ValueError("lhs must be a matrix")
    rows, cols = lhs.shape
    if objective.shape != (cols,):
        raise ValueError("objective dimension does not match lhs")
    if rhs.shape != (rows,):
        raise ValueError("rhs dimension does not match lhs")
    if len(basis) != rows:
        raise ValueError("basis must contain one variable per equality")

    tableau = np.zeros((rows + 1, cols + 1), dtype=float)
    tableau[:rows, :cols] = lhs
    tableau[:rows, -1] = rhs
    tableau[-1, :cols] = -objective
    basis = list(basis)
    for row, basic in enumerate(basis):
        coefficient = objective[basic]
        if abs(coefficient) > tol:
            tableau[-1, :] += coefficient * tableau[row, :]

    for _ in range(max_iter):
        entering_candidates = np.flatnonzero(tableau[-1, :-1] < -tol)
        if len(entering_candidates) == 0:
            solution = np.zeros(cols, dtype=float)
            for row, basic in enumerate(basis):
                solution[basic] = max(0.0, tableau[row, -1])
            return float(tableau[-1, -1]), solution, basis, tableau

        entering = int(entering_candidates[0])
        column = tableau[:rows, entering]
        positive = np.flatnonzero(column > tol)
        if len(positive) == 0:
            raise LPUnbounded("LP is unbounded")

        ratios = tableau[positive, -1] / column[positive]
        min_ratio = float(np.min(ratios))
        tied = positive[np.flatnonzero(ratios <= min_ratio + tol)]
        leaving_row = int(tied[np.argmin([basis[i] for i in tied])])
        _pivot_tableau(tableau, basis, leaving_row, entering, tol=tol)

    raise RuntimeError("simplex iteration limit exceeded")


def _independent_equalities(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[int] = []
    rank = 0
    for row in range(lhs.shape[0]):
        if np.linalg.norm(lhs[row]) <= tol:
            if abs(rhs[row]) > tol:
                raise LPInfeasible("inconsistent zero equality")
            continue
        trial = lhs[selected + [row]]
        trial_rank = int(np.linalg.matrix_rank(trial, tol=tol))
        if trial_rank > rank:
            selected.append(row)
            rank = trial_rank
    if not selected:
        return np.zeros((0, lhs.shape[1]), dtype=float), np.zeros(0, dtype=float)
    return lhs[selected].copy(), rhs[selected].copy()


def solve_equality_lp_max(
    objective: np.ndarray,
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    tol: float = 1e-9,
) -> tuple[float, np.ndarray]:
    """Maximize objective @ x subject to lhs @ x = rhs and x >= 0.

    This is a tiny phase-I simplex for local-marginal diagnostics.  It is not a
    production LP solver, but it handles the nonzero right-hand sides that arise
    from check-normalization constraints.
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

    original_lhs = lhs.copy()
    original_rhs = rhs.copy()
    lhs, rhs = _independent_equalities(lhs, rhs, tol=tol)
    rows, cols = lhs.shape
    if rows == 0:
        if np.any(objective > tol):
            raise LPUnbounded("LP is unbounded")
        return 0.0, np.zeros(cols, dtype=float)

    for row in range(rows):
        if rhs[row] < -tol:
            lhs[row, :] *= -1.0
            rhs[row] *= -1.0
        elif rhs[row] < 0.0:
            rhs[row] = 0.0

    phase_lhs = np.concatenate([lhs, np.eye(rows)], axis=1)
    phase_objective = np.concatenate([np.zeros(cols), -np.ones(rows)])
    phase_basis = list(range(cols, cols + rows))
    phase_value, _, phase_basis, tableau = _simplex_max_with_basis(
        phase_objective,
        phase_lhs,
        rhs,
        phase_basis,
        tol=tol,
    )
    if phase_value < -tol:
        raise LPInfeasible("equality LP is infeasible")

    row = 0
    while row < len(phase_basis):
        if phase_basis[row] < cols:
            row += 1
            continue
        candidates = [
            col for col in range(cols)
            if col not in phase_basis and abs(tableau[row, col]) > tol
        ]
        if candidates:
            _pivot_tableau(tableau, phase_basis, row, candidates[0], tol=tol)
            row += 1
            continue
        if abs(tableau[row, -1]) > tol:
            raise LPInfeasible("artificial variable remained positive")
        tableau = np.delete(tableau, row, axis=0)
        phase_basis.pop(row)

    active_rows = len(phase_basis)
    canonical_lhs = tableau[:active_rows, :cols]
    canonical_rhs = tableau[:active_rows, -1]
    value, solution, _, _ = _simplex_max_with_basis(
        objective,
        canonical_lhs,
        canonical_rhs,
        phase_basis,
        tol=tol,
    )
    if np.linalg.norm(original_lhs @ solution - original_rhs, ord=np.inf) > 1e-6:
        raise LPInfeasible("solution violates redundant equalities")
    return value, solution


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


def word_indicator(word: tuple[int, ...] | list[int] | np.ndarray, alphabet: int) -> np.ndarray:
    word = tuple(int(symbol) for symbol in word)
    vector = np.zeros(len(word) * alphabet, dtype=float)
    for i, symbol in enumerate(word):
        vector[i * alphabet + symbol] = 1.0
    return vector


def span_basis(matrix: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular_values > tol))
    return u[:, :rank]


def _score_band_constraints(
    code: np.ndarray,
    omega: np.ndarray,
    min_chart: tuple[int, ...],
    *,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    code = np.asarray(code, dtype=int)
    omega = np.asarray(omega, dtype=float)
    n_words, blocks = code.shape
    alphabet = int(np.max(code)) + 1
    if omega.shape != (n_words,):
        raise ValueError("omega must have one entry per codeword")
    if len(min_chart) != blocks:
        raise ValueError("min_chart must have one symbol per block")
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
        for a in range(alphabet):
            idx = i * alphabet + a
            row = np.zeros(dim + 1, dtype=float)
            row[:dim] = -(basis[idx] - basis[min_idx])
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

    return basis, raw_scores, np.vstack(rows), np.array(rhs)


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

    basis, _, base_lhs, base_rhs = _score_band_constraints(
        code,
        omega,
        min_chart,
        tol=tol,
    )
    dim = basis.shape[1]
    rows = [row.copy() for row in base_lhs]
    rhs = [float(value) for value in base_rhs]
    for i in range(blocks):
        max_idx = i * alphabet + max_chart[i]
        for a in range(alphabet):
            idx = i * alphabet + a
            # Y_{i,max_chart_i} >= Y_{i,a}
            row = np.zeros(dim + 1, dtype=float)
            row[:dim] = basis[idx] - basis[max_idx]
            rows.append(row)
            rhs.append(0.0)

    objective = np.zeros(dim + 1, dtype=float)
    for i, symbol in enumerate(max_chart):
        objective[:dim] += basis[i * alphabet + symbol]
    objective[dim] = -1.0

    value, solution = solve_free_lp_max(objective, np.vstack(rows), np.array(rhs), tol=tol)
    return max(0.0, value), solution


def source_chart_score_band_gap(
    code: np.ndarray,
    omega: np.ndarray,
    source: np.ndarray,
    min_chart: tuple[int, ...],
    *,
    tol: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Solve one min-chart LP for a fixed source point z."""
    code = np.asarray(code, dtype=int)
    omega = np.asarray(omega, dtype=float)
    n_words, blocks = code.shape
    alphabet = int(np.max(code)) + 1
    source = np.asarray(source, dtype=float)
    if source.shape != (blocks * alphabet,):
        raise ValueError("source must be a flattened block-symbol vector")

    basis, _, lhs, rhs = _score_band_constraints(
        code,
        omega,
        min_chart,
        tol=tol,
    )
    objective = np.zeros(basis.shape[1] + 1, dtype=float)
    objective[:basis.shape[1]] = source @ basis
    objective[-1] = -1.0
    value, solution = solve_free_lp_max(objective, lhs, rhs, tol=tol)
    return max(0.0, value), solution


def source_score_band_gap(
    code: np.ndarray,
    omega: np.ndarray,
    *,
    source: np.ndarray | None = None,
    source_word: tuple[int, ...] | list[int] | np.ndarray | None = None,
    tol: float = 1e-8,
) -> dict[str, object]:
    code = np.asarray(code, dtype=int)
    blocks = code.shape[1]
    alphabet = int(np.max(code)) + 1
    if (source is None) == (source_word is None):
        raise ValueError("provide exactly one of source or source_word")
    if source_word is not None:
        source = word_indicator(source_word, alphabet)
    assert source is not None

    best_value = -math.inf
    best_min_chart: tuple[int, ...] | None = None
    best_solution: np.ndarray | None = None
    for min_chart in itertools.product(range(alphabet), repeat=blocks):
        value, solution = source_chart_score_band_gap(
            code,
            omega,
            source,
            min_chart,
            tol=tol,
        )
        if value > best_value:
            best_value = value
            best_min_chart = min_chart
            best_solution = solution
    return {
        "gap": float(best_value),
        "min_chart": best_min_chart,
        "solution": best_solution,
    }


def convex_hull_dominance_deficit(
    code: np.ndarray,
    *,
    source: np.ndarray | None = None,
    source_word: tuple[int, ...] | list[int] | np.ndarray | None = None,
    tol: float = 1e-8,
) -> float:
    """Minimum residual fraction after dominating a source by the true hull.

    The returned value is the smallest eps for which
    source = (1 - eps) h + eps z, where h is in the true codeword convex hull
    and z is in the product simplex.
    """
    result = true_hull_dominance_decomposition(
        code,
        source=source,
        source_word=source_word,
        tol=tol,
    )
    return float(result["deficit"])


def true_hull_dominance_decomposition(
    code: np.ndarray,
    *,
    source: np.ndarray | None = None,
    source_word: tuple[int, ...] | list[int] | np.ndarray | None = None,
    tol: float = 1e-8,
) -> dict[str, object]:
    """Decompose a source into true-code dominated mass plus residual mass."""
    code = np.asarray(code, dtype=int)
    blocks = code.shape[1]
    alphabet = int(np.max(code)) + 1
    if (source is None) == (source_word is None):
        raise ValueError("provide exactly one of source or source_word")
    if source_word is not None:
        source = word_indicator(source_word, alphabet)
    assert source is not None
    source = np.asarray(source, dtype=float)
    if source.shape != (blocks * alphabet,):
        raise ValueError("source must be a flattened block-symbol vector")

    indicators = code_indicator_matrix(code, alphabet)
    objective = np.ones(indicators.shape[1], dtype=float)
    value, coefficients = _simplex_max_nonnegative(objective, indicators, source, tol=tol)
    dominated_source = indicators @ coefficients
    dominated_source[np.abs(dominated_source) <= tol] = 0.0
    mass = max(0.0, min(1.0, float(value)))
    deficit = max(0.0, 1.0 - mass)
    if deficit <= tol:
        residual_source = np.zeros_like(source)
    else:
        residual_source = (source - dominated_source) / deficit
        residual_source[np.abs(residual_source) <= tol] = 0.0
        if np.min(residual_source) < -1e-6:
            raise RuntimeError("dominance residual has negative coordinates")
        residual_source = np.maximum(residual_source, 0.0)
    return {
        "deficit": float(deficit),
        "dominated_mass": float(mass),
        "coefficients": coefficients,
        "dominated_source": dominated_source,
        "residual_source": residual_source,
    }


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


def _all_words(blocks: int, alphabet: int) -> list[tuple[int, ...]]:
    return list(itertools.product(range(alphabet), repeat=blocks))


def local_check_scopes(blocks: int, size: int) -> list[tuple[int, ...]]:
    if size <= 0 or size > blocks:
        raise ValueError("check size must be between 1 and the block count")
    return list(itertools.combinations(range(blocks), size))


def _locally_consistent_words(
    code: np.ndarray,
    checks: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    code = np.asarray(code, dtype=int)
    blocks = code.shape[1]
    alphabet = int(np.max(code)) + 1
    projection_sets = {
        tuple(check): {tuple(int(word[i]) for i in check) for word in code}
        for check in checks
    }
    out = []
    for word in _all_words(blocks, alphabet):
        if all(tuple(word[i] for i in check) in allowed
               for check, allowed in projection_sets.items()):
            out.append(word)
    return out


def _source_key(source: np.ndarray, *, tol: float) -> tuple[int, ...]:
    return tuple(int(round(float(value) / tol)) for value in source)


def _local_projection_assignments(
    code: np.ndarray,
    checks: list[tuple[int, ...]],
) -> list[list[tuple[int, ...]]]:
    return [
        sorted({tuple(int(word[i]) for i in check) for word in code})
        for check in checks
    ]


def _local_marginal_system(
    code: np.ndarray,
    checks: list[tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray, int]:
    code = np.asarray(code, dtype=int)
    blocks = code.shape[1]
    alphabet = int(np.max(code)) + 1
    if not checks:
        raise ValueError("at least one local check is required")
    for check in checks:
        if not check:
            raise ValueError("checks must be nonempty")
        if len(set(check)) != len(check):
            raise ValueError("checks cannot repeat a coordinate")
        if min(check) < 0 or max(check) >= blocks:
            raise ValueError("check coordinate is out of range")

    assignments_by_check = _local_projection_assignments(code, checks)
    local_count = sum(len(assignments) for assignments in assignments_by_check)
    z_offset = local_count
    n_vars = local_count + blocks * alphabet
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    offset = 0
    check_offsets = []
    for assignments in assignments_by_check:
        check_offsets.append(offset)
        row = np.zeros(n_vars, dtype=float)
        row[offset:offset + len(assignments)] = 1.0
        rows.append(row)
        rhs.append(1.0)
        offset += len(assignments)

    covered = set()
    for check_id, check in enumerate(checks):
        assignments = assignments_by_check[check_id]
        base = check_offsets[check_id]
        for local_pos, coord in enumerate(check):
            covered.add(coord)
            for symbol in range(alphabet):
                row = np.zeros(n_vars, dtype=float)
                row[z_offset + coord * alphabet + symbol] = 1.0
                for assignment_id, assignment in enumerate(assignments):
                    if assignment[local_pos] == symbol:
                        row[base + assignment_id] -= 1.0
                rows.append(row)
                rhs.append(0.0)

    for coord in range(blocks):
        if coord in covered:
            continue
        row = np.zeros(n_vars, dtype=float)
        row[z_offset + coord * alphabet:z_offset + (coord + 1) * alphabet] = 1.0
        rows.append(row)
        rhs.append(1.0)

    return np.vstack(rows), np.array(rhs), z_offset


def _local_only_marginal_system(
    code: np.ndarray,
    checks: list[tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return A y = b for local check marginals and source = S y.

    The larger equality system used by the LP optimizer carries explicit shared
    unary variables.  For exact vertex enumeration those variables only inflate
    the basis count, so this reduced system keeps local assignment marginals and
    encodes unary consistency directly between checks.
    """
    code = np.asarray(code, dtype=int)
    blocks = code.shape[1]
    alphabet = int(np.max(code)) + 1
    assignments_by_check = _local_projection_assignments(code, checks)

    offsets = []
    n_vars = 0
    for assignments in assignments_by_check:
        offsets.append(n_vars)
        n_vars += len(assignments)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for offset, assignments in zip(offsets, assignments_by_check):
        row = np.zeros(n_vars, dtype=float)
        row[offset:offset + len(assignments)] = 1.0
        rows.append(row)
        rhs.append(1.0)

    refs: dict[int, tuple[int, int]] = {}
    for check_id, check in enumerate(checks):
        base = offsets[check_id]
        assignments = assignments_by_check[check_id]
        for local_pos, coord in enumerate(check):
            if coord not in refs:
                refs[coord] = (check_id, local_pos)
                continue
            ref_check_id, ref_pos = refs[coord]
            ref_base = offsets[ref_check_id]
            ref_assignments = assignments_by_check[ref_check_id]
            for symbol in range(alphabet):
                row = np.zeros(n_vars, dtype=float)
                for assignment_id, assignment in enumerate(assignments):
                    if assignment[local_pos] == symbol:
                        row[base + assignment_id] += 1.0
                for assignment_id, assignment in enumerate(ref_assignments):
                    if assignment[ref_pos] == symbol:
                        row[ref_base + assignment_id] -= 1.0
                rows.append(row)
                rhs.append(0.0)

    if set(refs) != set(range(blocks)):
        missing = sorted(set(range(blocks)) - set(refs))
        raise ValueError(f"checks do not cover coordinates {missing}")

    source_matrix = np.zeros((blocks * alphabet, n_vars), dtype=float)
    for coord in range(blocks):
        ref_check_id, ref_pos = refs[coord]
        ref_base = offsets[ref_check_id]
        ref_assignments = assignments_by_check[ref_check_id]
        for symbol in range(alphabet):
            row = coord * alphabet + symbol
            for assignment_id, assignment in enumerate(ref_assignments):
                if assignment[ref_pos] == symbol:
                    source_matrix[row, ref_base + assignment_id] = 1.0

    return np.vstack(rows), np.array(rhs), source_matrix


def _local_overlap_marginal_system(
    code: np.ndarray,
    checks: list[tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local-check marginals with full pairwise-overlap consistency."""
    code = np.asarray(code, dtype=int)
    blocks = code.shape[1]
    alphabet = int(np.max(code)) + 1
    assignments_by_check = _local_projection_assignments(code, checks)

    offsets = []
    n_vars = 0
    for assignments in assignments_by_check:
        offsets.append(n_vars)
        n_vars += len(assignments)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for offset, assignments in zip(offsets, assignments_by_check):
        row = np.zeros(n_vars, dtype=float)
        row[offset:offset + len(assignments)] = 1.0
        rows.append(row)
        rhs.append(1.0)

    for left_id, right_id in itertools.combinations(range(len(checks)), 2):
        left_check = checks[left_id]
        right_check = checks[right_id]
        overlap = tuple(coord for coord in left_check if coord in right_check)
        if not overlap:
            continue
        left_positions = [left_check.index(coord) for coord in overlap]
        right_positions = [right_check.index(coord) for coord in overlap]
        left_assignments = assignments_by_check[left_id]
        right_assignments = assignments_by_check[right_id]
        patterns = sorted(
            {tuple(assignment[pos] for pos in left_positions)
             for assignment in left_assignments}
            |
            {tuple(assignment[pos] for pos in right_positions)
             for assignment in right_assignments}
        )
        for pattern in patterns:
            row = np.zeros(n_vars, dtype=float)
            for assignment_id, assignment in enumerate(left_assignments):
                if tuple(assignment[pos] for pos in left_positions) == pattern:
                    row[offsets[left_id] + assignment_id] += 1.0
            for assignment_id, assignment in enumerate(right_assignments):
                if tuple(assignment[pos] for pos in right_positions) == pattern:
                    row[offsets[right_id] + assignment_id] -= 1.0
            rows.append(row)
            rhs.append(0.0)

    refs: dict[int, tuple[int, int]] = {}
    for check_id, check in enumerate(checks):
        for local_pos, coord in enumerate(check):
            refs.setdefault(coord, (check_id, local_pos))
    if set(refs) != set(range(blocks)):
        missing = sorted(set(range(blocks)) - set(refs))
        raise ValueError(f"checks do not cover coordinates {missing}")

    source_matrix = np.zeros((blocks * alphabet, n_vars), dtype=float)
    for coord, (check_id, local_pos) in refs.items():
        base = offsets[check_id]
        assignments = assignments_by_check[check_id]
        for symbol in range(alphabet):
            row = coord * alphabet + symbol
            for assignment_id, assignment in enumerate(assignments):
                if assignment[local_pos] == symbol:
                    source_matrix[row, base + assignment_id] = 1.0

    return np.vstack(rows), np.array(rhs), source_matrix


def local_overlap_source_feasible(
    code: np.ndarray,
    checks: list[tuple[int, ...]],
    source: np.ndarray,
    *,
    tol: float = 1e-8,
) -> dict[str, object]:
    """Check whether a unary source extends to overlap-consistent local marginals."""
    code = np.asarray(code, dtype=int)
    blocks = code.shape[1]
    alphabet = int(np.max(code)) + 1
    source = np.asarray(source, dtype=float)
    if source.shape != (blocks * alphabet,):
        raise ValueError("source must be a flattened block-symbol vector")

    lhs, rhs, source_matrix = _local_overlap_marginal_system(code, checks)
    source_lhs = []
    source_rhs = []
    for idx, value in enumerate(source):
        source_lhs.append(source_matrix[idx].copy())
        source_rhs.append(float(value))
    augmented_lhs = np.vstack([lhs, np.vstack(source_lhs)])
    augmented_rhs = np.concatenate([rhs, np.array(source_rhs)])
    try:
        _, solution = solve_equality_lp_max(
            np.zeros(augmented_lhs.shape[1], dtype=float),
            augmented_lhs,
            augmented_rhs,
            tol=tol,
        )
    except LPInfeasible:
        return {
            "feasible": False,
            "row_count": int(lhs.shape[0]),
            "variable_count": int(lhs.shape[1]),
            "rank": int(np.linalg.matrix_rank(lhs, tol=tol)),
            "solution": None,
        }
    return {
        "feasible": True,
        "row_count": int(lhs.shape[0]),
        "variable_count": int(lhs.shape[1]),
        "rank": int(np.linalg.matrix_rank(lhs, tol=tol)),
        "solution": solution,
    }


def local_overlap_optimize_source(
    code: np.ndarray,
    checks: list[tuple[int, ...]],
    objective_on_source: np.ndarray,
    *,
    tol: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Optimize a unary objective over overlap-consistent local marginals."""
    code = np.asarray(code, dtype=int)
    blocks = code.shape[1]
    alphabet = int(np.max(code)) + 1
    objective_on_source = np.asarray(objective_on_source, dtype=float)
    if objective_on_source.shape != (blocks * alphabet,):
        raise ValueError("source objective must have one entry per block-symbol")

    lhs, rhs, source_matrix = _local_overlap_marginal_system(code, checks)
    value, solution = solve_equality_lp_max(
        source_matrix.T @ objective_on_source,
        lhs,
        rhs,
        tol=tol,
    )
    source = source_matrix @ solution
    source[np.abs(source) <= tol] = 0.0
    return value, source


def _enumerate_equality_polytope_vertices(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    max_bases: int,
    tol: float,
) -> tuple[list[np.ndarray], int, int]:
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    lhs, rhs = _independent_equalities(lhs, rhs, tol=tol)
    rank, n_vars = lhs.shape
    basis_count = comb(n_vars, rank)
    if basis_count > max_bases:
        raise ValueError(
            f"exact vertex enumeration needs {basis_count} bases, "
            f"above max_bases={max_bases}",
        )
    vertices: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    feasible_bases = 0
    for basis in itertools.combinations(range(n_vars), rank):
        basis_matrix = lhs[:, basis]
        try:
            values = np.linalg.solve(basis_matrix, rhs)
        except np.linalg.LinAlgError:
            continue
        if np.any(values < -tol):
            continue
        point = np.zeros(n_vars, dtype=float)
        point[list(basis)] = np.maximum(values, 0.0)
        residual = lhs @ point - rhs
        if np.linalg.norm(residual, ord=np.inf) > 1e-7:
            continue
        feasible_bases += 1
        key = _source_key(point, tol=tol)
        if key in seen:
            continue
        seen.add(key)
        vertices.append(point)
    return vertices, basis_count, feasible_bases


def _assignment_reduced_indicator(
    assignment: tuple[int, ...],
    alphabet: int,
) -> np.ndarray:
    vector = np.zeros(len(assignment) * (alphabet - 1), dtype=float)
    for pos, symbol in enumerate(assignment):
        if symbol < alphabet - 1:
            vector[pos * (alphabet - 1) + symbol] = 1.0
    return vector


def _reduced_source_to_full(
    reduced: np.ndarray,
    blocks: int,
    alphabet: int,
) -> np.ndarray:
    matrix = np.zeros((blocks, alphabet), dtype=float)
    if alphabet > 1:
        head = np.asarray(reduced, dtype=float).reshape(blocks, alphabet - 1)
        matrix[:, :alphabet - 1] = head
        matrix[:, alphabet - 1] = 1.0 - np.sum(head, axis=1)
    else:
        matrix[:, 0] = 1.0
    return matrix.reshape(-1)


def _point_facets(points: np.ndarray, *, tol: float) -> list[tuple[np.ndarray, float]]:
    """Enumerate halfspaces for a tiny point hull, including affine hull sides."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a matrix")
    n_points, dim = points.shape
    if dim == 0:
        return []
    if n_points == 0:
        raise ValueError("cannot build facets of an empty hull")
    base = points[0]
    centered = points - base
    if n_points == 1:
        affine_rank = 0
        affine_basis = np.zeros((dim, 0), dtype=float)
        nullspace = np.eye(dim, dtype=float)
    else:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=True)
        affine_rank = int(np.sum(singular_values > tol))
        affine_basis = vt[:affine_rank].T
        nullspace = vt[affine_rank:].T

    halfspaces: list[tuple[np.ndarray, float]] = []

    for null_idx in range(nullspace.shape[1]):
        normal = nullspace[:, null_idx].copy()
        normal[np.abs(normal) <= tol] = 0.0
        offset = float(normal @ base)
        halfspaces.append((normal.copy(), offset))
        halfspaces.append((-normal.copy(), -offset))

    if affine_rank == 0:
        facets = []
    else:
        projected = centered @ affine_basis
        facets: list[tuple[np.ndarray, float]] = []
        for support in itertools.combinations(range(n_points), affine_rank):
            selected = projected[list(support)]
            if affine_rank == 1:
                normal = np.array([1.0])
            else:
                differences = selected[1:] - selected[0]
                if np.linalg.matrix_rank(differences, tol=tol) < affine_rank - 1:
                    continue
                _, _, vt = np.linalg.svd(differences, full_matrices=True)
                normal = vt[-1]
            norm = float(np.linalg.norm(normal))
            if norm <= tol:
                continue
            normal = normal / norm
            offset = float(normal @ selected[0])
            values = projected @ normal - offset
            if np.all(values <= tol):
                pass
            elif np.all(values >= -tol):
                normal = -normal
                offset = -offset
            else:
                continue
            lifted = affine_basis @ normal
            lifted[np.abs(lifted) <= tol] = 0.0
            halfspaces.append((lifted.copy(), float(offset + lifted @ base)))

    deduped: list[tuple[np.ndarray, float]] = []
    seen: set[tuple[int, ...]] = set()
    for normal, offset in halfspaces:
        norm = float(np.linalg.norm(normal))
        if norm <= tol:
            continue
        if affine_rank == dim:
            scale = norm
        else:
            scale = 1.0
        normal = normal / scale
        offset = float(offset / scale)
        normal[np.abs(normal) <= tol] = 0.0
        key = tuple(int(round(float(value) / tol)) for value in np.r_[normal, offset])
        if key in seen:
            continue
        seen.add(key)
        deduped.append((normal.copy(), offset))
    return deduped


def _dedupe_halfspaces(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove duplicate halfspaces, keeping the tightest right-hand side."""
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if lhs.ndim != 2:
        raise ValueError("lhs must be a matrix")
    if rhs.shape != (lhs.shape[0],):
        raise ValueError("rhs must have one entry per halfspace")

    kept: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
    for row, bound in zip(lhs, rhs):
        norm = float(np.linalg.norm(row))
        if norm <= tol:
            if bound < -tol:
                raise LPInfeasible("inconsistent zero halfspace")
            continue
        normal = row / norm
        offset = float(bound / norm)
        normal[np.abs(normal) <= tol] = 0.0
        if abs(offset) <= tol:
            offset = 0.0
        key = tuple(int(round(float(value) / tol)) for value in normal)
        previous = kept.get(key)
        if previous is None or offset < previous[1]:
            kept[key] = (normal.copy(), offset)

    if not kept:
        return np.zeros((0, lhs.shape[1]), dtype=float), np.zeros(0, dtype=float)
    rows = []
    bounds = []
    for normal, offset in kept.values():
        rows.append(normal)
        bounds.append(offset)
    return np.vstack(rows), np.array(bounds)


def _projected_local_marginal_vertex_sources(
    code: np.ndarray,
    checks: list[tuple[int, ...]],
    *,
    max_bases: int,
    tol: float,
) -> dict[str, object]:
    """Enumerate vertices of the projected unary local-marginal polytope."""
    code = np.asarray(code, dtype=int)
    blocks = code.shape[1]
    alphabet = int(np.max(code)) + 1
    dim = blocks * (alphabet - 1)
    if dim == 0:
        source = np.ones(blocks, dtype=float)
        return {
            "sources": [source],
            "source_count": 1,
            "vertex_count": 1,
            "basis_count": 1,
            "feasible_basis_count": 1,
        }

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for coord in range(blocks):
        block_slice = slice(coord * (alphabet - 1), (coord + 1) * (alphabet - 1))
        for symbol in range(alphabet - 1):
            row = np.zeros(dim, dtype=float)
            row[coord * (alphabet - 1) + symbol] = -1.0
            rows.append(row)
            rhs.append(0.0)
        row = np.zeros(dim, dtype=float)
        row[block_slice] = 1.0
        rows.append(row)
        rhs.append(1.0)

    for check, assignments in zip(checks, _local_projection_assignments(code, checks)):
        local_points = np.array([
            _assignment_reduced_indicator(assignment, alphabet)
            for assignment in assignments
        ])
        facets = _point_facets(local_points, tol=tol)
        for normal, offset in facets:
            row = np.zeros(dim, dtype=float)
            for local_pos, coord in enumerate(check):
                local_base = local_pos * (alphabet - 1)
                global_base = coord * (alphabet - 1)
                row[global_base:global_base + alphabet - 1] += normal[
                    local_base:local_base + alphabet - 1
                ]
            rows.append(row)
            rhs.append(offset)

    raw_halfspace_count = len(rows)
    lhs, bounds = _dedupe_halfspaces(np.vstack(rows), np.array(rhs), tol=tol)
    basis_count = comb(lhs.shape[0], dim)
    if basis_count > max_bases:
        raise ValueError(
            f"projected vertex enumeration needs {basis_count} active sets, "
            f"above max_bases={max_bases}",
        )

    sources: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    feasible_bases = 0
    for active in itertools.combinations(range(lhs.shape[0]), dim):
        active_lhs = lhs[list(active)]
        if np.linalg.matrix_rank(active_lhs, tol=tol) < dim:
            continue
        try:
            reduced = np.linalg.solve(active_lhs, bounds[list(active)])
        except np.linalg.LinAlgError:
            continue
        if np.any(lhs @ reduced - bounds > 1e-7):
            continue
        source = _reduced_source_to_full(reduced, blocks, alphabet)
        if np.min(source) < -1e-7:
            continue
        source[np.abs(source) <= tol] = 0.0
        source[np.abs(source - 1.0) <= tol] = 1.0
        feasible_bases += 1
        key = _source_key(source, tol=tol)
        if key in seen:
            continue
        seen.add(key)
        sources.append(source)

    return {
        "sources": sources,
        "source_count": len(sources),
        "vertex_count": len(sources),
        "basis_count": basis_count,
        "feasible_basis_count": feasible_bases,
        "halfspace_count": int(lhs.shape[0]),
        "raw_halfspace_count": int(raw_halfspace_count),
    }


def local_marginal_vertex_sources(
    code: np.ndarray,
    checks: list[tuple[int, ...]],
    *,
    max_bases: int = 1_000_000,
    tol: float = 1e-8,
) -> dict[str, object]:
    """Enumerate projected local-marginal vertices for tiny panels."""
    try:
        return _projected_local_marginal_vertex_sources(
            code,
            checks,
            max_bases=max_bases,
            tol=tol,
        )
    except ValueError as exc:
        if "above max_bases" in str(exc):
            raise
        pass

    lhs, rhs, source_matrix = _local_only_marginal_system(code, checks)
    vertices, basis_count, feasible_bases = _enumerate_equality_polytope_vertices(
        lhs,
        rhs,
        max_bases=max_bases,
        tol=tol,
    )
    sources: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for vertex in vertices:
        source = source_matrix @ vertex
        source[np.abs(source) <= tol] = 0.0
        key = _source_key(source, tol=tol)
        if key in seen:
            continue
        seen.add(key)
        sources.append(source)
    return {
        "sources": sources,
        "source_count": len(sources),
        "vertex_count": len(vertices),
        "basis_count": basis_count,
        "feasible_basis_count": feasible_bases,
    }


def _local_marginal_optimize_source(
    code: np.ndarray,
    checks: list[tuple[int, ...]],
    objective_on_source: np.ndarray,
    *,
    tol: float = 1e-8,
) -> tuple[float, np.ndarray]:
    lhs, rhs, z_offset = _local_marginal_system(code, checks)
    objective_on_source = np.asarray(objective_on_source, dtype=float)
    blocks = code.shape[1]
    alphabet = int(np.max(code)) + 1
    if objective_on_source.shape != (blocks * alphabet,):
        raise ValueError("source objective must have one entry per block-symbol")
    objective = np.zeros(lhs.shape[1], dtype=float)
    objective[z_offset:z_offset + blocks * alphabet] = objective_on_source
    value, solution = solve_equality_lp_max(objective, lhs, rhs, tol=tol)
    source = solution[z_offset:z_offset + blocks * alphabet]
    source[np.abs(source) <= tol] = 0.0
    return value, source


def local_marginal_source_candidates(
    code: np.ndarray,
    checks: list[tuple[int, ...]],
    *,
    random_objectives: int = 16,
    seed: int = 0,
    max_integral_words: int = 100000,
    tol: float = 1e-8,
) -> list[np.ndarray]:
    """Return toy local-marginal source candidates.

    Integral locally consistent words are included when the ambient cube is
    small enough.  Additional candidates come from optimizing random unary
    objectives over the local-marginal equality LP.
    """
    code = np.asarray(code, dtype=int)
    blocks = code.shape[1]
    alphabet = int(np.max(code)) + 1
    sources: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()

    def add_source(source: np.ndarray) -> None:
        source = np.asarray(source, dtype=float)
        key = _source_key(source, tol=tol)
        if key in seen:
            return
        seen.add(key)
        sources.append(source.copy())

    ambient_size = alphabet ** blocks
    if ambient_size <= max_integral_words:
        for word in _locally_consistent_words(code, checks):
            add_source(word_indicator(word, alphabet))

    for coord in range(blocks):
        for symbol in range(alphabet):
            objective = np.zeros(blocks * alphabet, dtype=float)
            objective[coord * alphabet + symbol] = 1.0
            _, source = _local_marginal_optimize_source(
                code,
                checks,
                objective,
                tol=tol,
            )
            add_source(source)

    rng = np.random.default_rng(seed)
    for _ in range(random_objectives):
        objective = rng.standard_normal(blocks * alphabet)
        _, source = _local_marginal_optimize_source(
            code,
            checks,
            objective,
            tol=tol,
        )
        add_source(source)

    return sources


def _score_table_from_solution(code: np.ndarray, solution: np.ndarray) -> np.ndarray:
    alphabet = int(np.max(code)) + 1
    indicators = code_indicator_matrix(code, alphabet)
    basis = span_basis(indicators)
    coeffs = np.asarray(solution, dtype=float)[:basis.shape[1]]
    return basis @ coeffs


def local_marginal_score_band_gap(
    code: np.ndarray,
    omega: np.ndarray,
    checks: list[tuple[int, ...]],
    *,
    random_objectives: int = 16,
    seed: int = 0,
    closure_rounds: int = 2,
    max_integral_words: int = 100000,
    tol: float = 1e-8,
) -> dict[str, object]:
    """Stress the score-band gap on sampled vertices of a local-marginal LP."""
    code = np.asarray(code, dtype=int)
    sources = local_marginal_source_candidates(
        code,
        checks,
        random_objectives=random_objectives,
        seed=seed,
        max_integral_words=max_integral_words,
        tol=tol,
    )
    queue = list(sources)
    seen = {_source_key(source, tol=tol) for source in sources}
    evaluated: set[tuple[int, ...]] = set()
    best_gap = -math.inf
    best_source: np.ndarray | None = None
    best_result: dict[str, object] | None = None

    for _ in range(closure_rounds + 1):
        current = queue
        queue = []
        if not current:
            break
        for source in current:
            key = _source_key(source, tol=tol)
            if key in evaluated:
                continue
            evaluated.add(key)
            result = source_score_band_gap(code, omega, source=source, tol=tol)
            gap = float(result["gap"])
            if gap > best_gap:
                best_gap = gap
                best_source = source.copy()
                best_result = result

            solution = result.get("solution")
            if solution is None:
                continue
            score_table = _score_table_from_solution(code, np.asarray(solution))
            try:
                _, next_source = _local_marginal_optimize_source(
                    code,
                    checks,
                    score_table,
                    tol=tol,
                )
            except (LPInfeasible, LPUnbounded):
                continue
            next_key = _source_key(next_source, tol=tol)
            if next_key not in seen:
                seen.add(next_key)
                sources.append(next_source.copy())
                queue.append(next_source.copy())

    if best_result is None:
        best_gap = 0.0
    return {
        "gap": float(best_gap),
        "source": best_source,
        "source_count": len(sources),
        "evaluated_count": len(evaluated),
        "source_result": best_result,
    }


def local_marginal_score_band_gap_exact(
    code: np.ndarray,
    omega: np.ndarray,
    checks: list[tuple[int, ...]],
    *,
    max_bases: int = 1_000_000,
    tol: float = 1e-8,
) -> dict[str, object]:
    """Exactly maximize the fixed-source score-band gap over tiny local marginals.

    The fixed-source gap is convex as a function of the source vector because it
    is a supremum of linear score-table objectives.  Therefore its maximum over
    a local-marginal polytope is attained at a projected vertex.  Enumerating all
    local vertices is only feasible for tiny panels, but it gives an exact check
    where the sampled diagnostic would otherwise be ambiguous.
    """
    vertex_data = local_marginal_vertex_sources(
        code,
        checks,
        max_bases=max_bases,
        tol=tol,
    )
    sources = vertex_data["sources"]
    best_gap = -math.inf
    best_source: np.ndarray | None = None
    best_result: dict[str, object] | None = None
    for source in sources:
        result = source_score_band_gap(code, omega, source=source, tol=tol)
        gap = float(result["gap"])
        if gap > best_gap:
            best_gap = gap
            best_source = np.asarray(source).copy()
            best_result = result

    if best_result is None:
        best_gap = 0.0
    return {
        "gap": float(best_gap),
        "source": best_source,
        "source_count": int(vertex_data["source_count"]),
        "vertex_count": int(vertex_data["vertex_count"]),
        "basis_count": int(vertex_data["basis_count"]),
        "feasible_basis_count": int(vertex_data["feasible_basis_count"]),
        "source_result": best_result,
    }


def local_marginal_dominance_screen(
    code: np.ndarray,
    omega: np.ndarray,
    checks: list[tuple[int, ...]],
    *,
    max_bases: int = 1_000_000,
    tol: float = 1e-8,
) -> dict[str, object]:
    """Bound local-marginal score-band gaps by true-hull dominance deficits."""
    vertex_data = local_marginal_vertex_sources(
        code,
        checks,
        max_bases=max_bases,
        tol=tol,
    )
    product = product_simplex_score_band_gap(code, omega, tol=tol)
    product_gap = float(product["gap"])
    max_deficit = 0.0
    worst_source: np.ndarray | None = None
    for source in vertex_data["sources"]:
        deficit = convex_hull_dominance_deficit(code, source=source, tol=tol)
        if deficit > max_deficit:
            max_deficit = deficit
            worst_source = np.asarray(source).copy()
    return {
        "max_deficit": float(max_deficit),
        "gap_bound": float(max_deficit * product_gap),
        "product_gap": product_gap,
        "source": worst_source,
        "source_count": int(vertex_data["source_count"]),
        "vertex_count": int(vertex_data["vertex_count"]),
        "basis_count": int(vertex_data["basis_count"]),
        "feasible_basis_count": int(vertex_data["feasible_basis_count"]),
    }


def local_marginal_residual_dominance_screen(
    code: np.ndarray,
    omega: np.ndarray,
    checks: list[tuple[int, ...]],
    *,
    max_bases: int = 1_000_000,
    tol: float = 1e-8,
) -> dict[str, object]:
    """Bound local vertices using their actual true-hull residual sources."""
    vertex_data = local_marginal_vertex_sources(
        code,
        checks,
        max_bases=max_bases,
        tol=tol,
    )
    best_bound = 0.0
    max_deficit = 0.0
    best_source: np.ndarray | None = None
    best_residual: np.ndarray | None = None
    best_residual_gap = 0.0
    best_residual_result: dict[str, object] | None = None
    for source in vertex_data["sources"]:
        decomp = true_hull_dominance_decomposition(code, source=source, tol=tol)
        deficit = float(decomp["deficit"])
        max_deficit = max(max_deficit, deficit)
        if deficit <= tol:
            bound = 0.0
            residual_gap = 0.0
            residual_result = None
            residual = np.asarray(decomp["residual_source"], dtype=float)
        else:
            residual = np.asarray(decomp["residual_source"], dtype=float)
            residual_result = source_score_band_gap(
                code,
                omega,
                source=residual,
                tol=tol,
            )
            residual_gap = float(residual_result["gap"])
            bound = deficit * residual_gap
        if bound > best_bound:
            best_bound = bound
            best_source = np.asarray(source).copy()
            best_residual = residual.copy()
            best_residual_gap = residual_gap
            best_residual_result = residual_result
    return {
        "max_deficit": float(max_deficit),
        "gap_bound": float(best_bound),
        "residual_gap": float(best_residual_gap),
        "source": best_source,
        "residual_source": best_residual,
        "residual_result": best_residual_result,
        "source_count": int(vertex_data["source_count"]),
        "vertex_count": int(vertex_data["vertex_count"]),
        "basis_count": int(vertex_data["basis_count"]),
        "feasible_basis_count": int(vertex_data["feasible_basis_count"]),
    }


def local_marginal_combined_screen(
    code: np.ndarray,
    omega: np.ndarray,
    checks: list[tuple[int, ...]],
    *,
    max_bases: int = 1_000_000,
    tol: float = 1e-8,
) -> dict[str, object]:
    """Run exact and dominance local-vertex screens from one enumeration."""
    vertex_data = local_marginal_vertex_sources(
        code,
        checks,
        max_bases=max_bases,
        tol=tol,
    )
    product = product_simplex_score_band_gap(code, omega, tol=tol)
    product_gap = float(product["gap"])

    best_exact_gap = -math.inf
    best_exact_source: np.ndarray | None = None
    best_exact_result: dict[str, object] | None = None
    max_deficit = 0.0
    worst_dominance_source: np.ndarray | None = None
    best_residual_bound = 0.0
    best_residual_gap = 0.0
    best_residual_source: np.ndarray | None = None
    best_residual_parent: np.ndarray | None = None
    best_residual_result: dict[str, object] | None = None

    for source in vertex_data["sources"]:
        source = np.asarray(source, dtype=float)
        exact_result = source_score_band_gap(code, omega, source=source, tol=tol)
        exact_gap = float(exact_result["gap"])
        if exact_gap > best_exact_gap:
            best_exact_gap = exact_gap
            best_exact_source = source.copy()
            best_exact_result = exact_result

        decomp = true_hull_dominance_decomposition(code, source=source, tol=tol)
        deficit = float(decomp["deficit"])
        if deficit > max_deficit:
            max_deficit = deficit
            worst_dominance_source = source.copy()
        if deficit <= tol:
            residual_gap = 0.0
            residual_bound = 0.0
            residual = np.asarray(decomp["residual_source"], dtype=float)
            residual_result = None
        else:
            residual = np.asarray(decomp["residual_source"], dtype=float)
            residual_result = source_score_band_gap(
                code,
                omega,
                source=residual,
                tol=tol,
            )
            residual_gap = float(residual_result["gap"])
            residual_bound = deficit * residual_gap
        if residual_bound > best_residual_bound:
            best_residual_bound = residual_bound
            best_residual_gap = residual_gap
            best_residual_source = residual.copy()
            best_residual_parent = source.copy()
            best_residual_result = residual_result

    if best_exact_result is None:
        best_exact_gap = 0.0
    return {
        "exact_gap": float(best_exact_gap),
        "exact_source": best_exact_source,
        "exact_result": best_exact_result,
        "dominance_max_deficit": float(max_deficit),
        "dominance_gap_bound": float(max_deficit * product_gap),
        "dominance_product_gap": product_gap,
        "dominance_source": worst_dominance_source,
        "residual_dominance_gap_bound": float(best_residual_bound),
        "residual_dominance_residual_gap": float(best_residual_gap),
        "residual_dominance_source": best_residual_parent,
        "residual_dominance_residual_source": best_residual_source,
        "residual_dominance_residual_result": best_residual_result,
        "source_count": int(vertex_data["source_count"]),
        "vertex_count": int(vertex_data["vertex_count"]),
        "basis_count": int(vertex_data["basis_count"]),
        "feasible_basis_count": int(vertex_data["feasible_basis_count"]),
    }


def integral_local_pseudoword_gaps(
    code: np.ndarray,
    omega: np.ndarray,
    checks: list[tuple[int, ...]],
    *,
    tol: float = 1e-8,
) -> dict[str, object]:
    """Find the worst integral word allowed by local projections but not by the code."""
    code = np.asarray(code, dtype=int)
    code_set = {tuple(int(symbol) for symbol in word) for word in code}
    candidates = [
        word for word in _locally_consistent_words(code, checks)
        if word not in code_set
    ]
    best_gap = -math.inf
    best_word: tuple[int, ...] | None = None
    best_result: dict[str, object] | None = None
    for word in candidates:
        result = source_score_band_gap(code, omega, source_word=word, tol=tol)
        gap = float(result["gap"])
        if gap > best_gap:
            best_gap = gap
            best_word = word
            best_result = result
    if best_result is None:
        best_gap = 0.0
    return {
        "count": len(candidates),
        "gap": float(best_gap),
        "word": best_word,
        "source_result": best_result,
    }


def run_local_integral_trial(
    *,
    n_words: int,
    blocks: int,
    alphabet: int,
    seed: int,
    xi: float,
    checks: list[tuple[int, ...]],
    lam: float | None = None,
) -> dict[str, float | int | tuple[int, ...] | None]:
    if lam is None:
        lam = math.sqrt(2.0 * math.log(n_words) / blocks)
    code = make_balanced_code(n_words, blocks, alphabet, seed)
    symbol_scores = centered_symbol_scores(blocks, alphabet, seed + 1009)
    scores = codeword_scores(code, symbol_scores)
    gaps = float(np.max(scores)) - scores
    omega = np.exp(np.minimum(lam * gaps, 700.0)) + xi
    local = integral_local_pseudoword_gaps(code, omega, checks)
    gap = float(local["gap"])
    return {
        "seed": seed,
        "n_words": n_words,
        "blocks": blocks,
        "alphabet": alphabet,
        "lambda": float(lam),
        "xi": float(xi),
        "local_count": int(local["count"]),
        "local_gap": gap,
        "local_gap_over_xi": gap / xi if xi > 0 else math.inf,
        "local_word": local["word"],
    }


def run_local_marginal_trial(
    *,
    n_words: int,
    blocks: int,
    alphabet: int,
    seed: int,
    xi: float,
    checks: list[tuple[int, ...]],
    lam: float | None = None,
    random_objectives: int = 16,
    closure_rounds: int = 2,
) -> dict[str, float | int | tuple[int, ...] | None]:
    if lam is None:
        lam = math.sqrt(2.0 * math.log(n_words) / blocks)
    code = make_balanced_code(n_words, blocks, alphabet, seed)
    symbol_scores = centered_symbol_scores(blocks, alphabet, seed + 1009)
    scores = codeword_scores(code, symbol_scores)
    gaps = float(np.max(scores)) - scores
    omega = np.exp(np.minimum(lam * gaps, 700.0)) + xi
    local = local_marginal_score_band_gap(
        code,
        omega,
        checks,
        random_objectives=random_objectives,
        seed=seed + 2003,
        closure_rounds=closure_rounds,
    )
    gap = float(local["gap"])
    source = local["source"]
    if source is None:
        integrality_defect = 0.0
    else:
        source_matrix = np.asarray(source).reshape(blocks, alphabet)
        integrality_defect = float(np.sum(1.0 - np.max(source_matrix, axis=1)))
    source_result = local["source_result"]
    min_chart = None
    if isinstance(source_result, dict):
        min_chart = source_result.get("min_chart")
    return {
        "seed": seed,
        "n_words": n_words,
        "blocks": blocks,
        "alphabet": alphabet,
        "lambda": float(lam),
        "xi": float(xi),
        "local_marginal_gap": gap,
        "local_marginal_gap_over_xi": gap / xi if xi > 0 else math.inf,
        "local_marginal_source_count": int(local["source_count"]),
        "local_marginal_evaluated_count": int(local["evaluated_count"]),
        "local_marginal_integrality_defect": integrality_defect,
        "local_marginal_min_chart": min_chart,
    }


def run_local_marginal_exact_trial(
    *,
    n_words: int,
    blocks: int,
    alphabet: int,
    seed: int,
    xi: float,
    checks: list[tuple[int, ...]],
    lam: float | None = None,
    max_bases: int = 1_000_000,
) -> dict[str, float | int | tuple[int, ...] | None]:
    if lam is None:
        lam = math.sqrt(2.0 * math.log(n_words) / blocks)
    code = make_balanced_code(n_words, blocks, alphabet, seed)
    symbol_scores = centered_symbol_scores(blocks, alphabet, seed + 1009)
    scores = codeword_scores(code, symbol_scores)
    gaps = float(np.max(scores)) - scores
    omega = np.exp(np.minimum(lam * gaps, 700.0)) + xi
    local = local_marginal_score_band_gap_exact(
        code,
        omega,
        checks,
        max_bases=max_bases,
    )
    gap = float(local["gap"])
    source = local["source"]
    if source is None:
        integrality_defect = 0.0
    else:
        source_matrix = np.asarray(source).reshape(blocks, alphabet)
        integrality_defect = float(np.sum(1.0 - np.max(source_matrix, axis=1)))
    source_result = local["source_result"]
    min_chart = None
    if isinstance(source_result, dict):
        min_chart = source_result.get("min_chart")
    return {
        "seed": seed,
        "n_words": n_words,
        "blocks": blocks,
        "alphabet": alphabet,
        "lambda": float(lam),
        "xi": float(xi),
        "local_marginal_exact_gap": gap,
        "local_marginal_exact_gap_over_xi": gap / xi if xi > 0 else math.inf,
        "local_marginal_exact_source_count": int(local["source_count"]),
        "local_marginal_exact_vertex_count": int(local["vertex_count"]),
        "local_marginal_exact_basis_count": int(local["basis_count"]),
        "local_marginal_exact_feasible_basis_count": int(local["feasible_basis_count"]),
        "local_marginal_exact_integrality_defect": integrality_defect,
        "local_marginal_exact_min_chart": min_chart,
    }


def run_local_dominance_trial(
    *,
    n_words: int,
    blocks: int,
    alphabet: int,
    seed: int,
    xi: float,
    checks: list[tuple[int, ...]],
    lam: float | None = None,
    max_bases: int = 1_000_000,
) -> dict[str, float | int]:
    if lam is None:
        lam = math.sqrt(2.0 * math.log(n_words) / blocks)
    code = make_balanced_code(n_words, blocks, alphabet, seed)
    symbol_scores = centered_symbol_scores(blocks, alphabet, seed + 1009)
    scores = codeword_scores(code, symbol_scores)
    gaps = float(np.max(scores)) - scores
    omega = np.exp(np.minimum(lam * gaps, 700.0)) + xi
    screen = local_marginal_dominance_screen(
        code,
        omega,
        checks,
        max_bases=max_bases,
    )
    gap_bound = float(screen["gap_bound"])
    return {
        "seed": seed,
        "n_words": n_words,
        "blocks": blocks,
        "alphabet": alphabet,
        "lambda": float(lam),
        "xi": float(xi),
        "dominance_max_deficit": float(screen["max_deficit"]),
        "dominance_gap_bound": gap_bound,
        "dominance_gap_bound_over_xi": gap_bound / xi if xi > 0 else math.inf,
        "dominance_product_gap": float(screen["product_gap"]),
        "dominance_product_gap_over_xi": float(screen["product_gap"]) / xi if xi > 0 else math.inf,
        "dominance_source_count": int(screen["source_count"]),
        "dominance_vertex_count": int(screen["vertex_count"]),
        "dominance_basis_count": int(screen["basis_count"]),
    }


def run_local_residual_dominance_trial(
    *,
    n_words: int,
    blocks: int,
    alphabet: int,
    seed: int,
    xi: float,
    checks: list[tuple[int, ...]],
    lam: float | None = None,
    max_bases: int = 1_000_000,
) -> dict[str, float | int]:
    if lam is None:
        lam = math.sqrt(2.0 * math.log(n_words) / blocks)
    code = make_balanced_code(n_words, blocks, alphabet, seed)
    symbol_scores = centered_symbol_scores(blocks, alphabet, seed + 1009)
    scores = codeword_scores(code, symbol_scores)
    gaps = float(np.max(scores)) - scores
    omega = np.exp(np.minimum(lam * gaps, 700.0)) + xi
    screen = local_marginal_residual_dominance_screen(
        code,
        omega,
        checks,
        max_bases=max_bases,
    )
    gap_bound = float(screen["gap_bound"])
    residual_gap = float(screen["residual_gap"])
    return {
        "seed": seed,
        "n_words": n_words,
        "blocks": blocks,
        "alphabet": alphabet,
        "lambda": float(lam),
        "xi": float(xi),
        "residual_dominance_max_deficit": float(screen["max_deficit"]),
        "residual_dominance_gap_bound": gap_bound,
        "residual_dominance_gap_bound_over_xi": gap_bound / xi if xi > 0 else math.inf,
        "residual_dominance_residual_gap": residual_gap,
        "residual_dominance_residual_gap_over_xi": residual_gap / xi if xi > 0 else math.inf,
        "residual_dominance_source_count": int(screen["source_count"]),
        "residual_dominance_vertex_count": int(screen["vertex_count"]),
        "residual_dominance_basis_count": int(screen["basis_count"]),
    }


def run_local_combined_screen_trial(
    *,
    n_words: int,
    blocks: int,
    alphabet: int,
    seed: int,
    xi: float,
    checks: list[tuple[int, ...]],
    lam: float | None = None,
    max_bases: int = 1_000_000,
) -> dict[str, float | int | tuple[int, ...] | None]:
    if lam is None:
        lam = math.sqrt(2.0 * math.log(n_words) / blocks)
    code = make_balanced_code(n_words, blocks, alphabet, seed)
    symbol_scores = centered_symbol_scores(blocks, alphabet, seed + 1009)
    scores = codeword_scores(code, symbol_scores)
    gaps = float(np.max(scores)) - scores
    omega = np.exp(np.minimum(lam * gaps, 700.0)) + xi
    screen = local_marginal_combined_screen(
        code,
        omega,
        checks,
        max_bases=max_bases,
    )
    exact_gap = float(screen["exact_gap"])
    exact_source = screen["exact_source"]
    if exact_source is None:
        integrality_defect = 0.0
    else:
        source_matrix = np.asarray(exact_source).reshape(blocks, alphabet)
        integrality_defect = float(np.sum(1.0 - np.max(source_matrix, axis=1)))
    exact_result = screen["exact_result"]
    min_chart = None
    if isinstance(exact_result, dict):
        min_chart = exact_result.get("min_chart")
    dominance_gap_bound = float(screen["dominance_gap_bound"])
    dominance_product_gap = float(screen["dominance_product_gap"])
    residual_bound = float(screen["residual_dominance_gap_bound"])
    residual_gap = float(screen["residual_dominance_residual_gap"])
    return {
        "seed": seed,
        "n_words": n_words,
        "blocks": blocks,
        "alphabet": alphabet,
        "lambda": float(lam),
        "xi": float(xi),
        "local_marginal_exact_gap": exact_gap,
        "local_marginal_exact_gap_over_xi": exact_gap / xi if xi > 0 else math.inf,
        "local_marginal_exact_source_count": int(screen["source_count"]),
        "local_marginal_exact_vertex_count": int(screen["vertex_count"]),
        "local_marginal_exact_basis_count": int(screen["basis_count"]),
        "local_marginal_exact_feasible_basis_count": int(screen["feasible_basis_count"]),
        "local_marginal_exact_integrality_defect": integrality_defect,
        "local_marginal_exact_min_chart": min_chart,
        "dominance_max_deficit": float(screen["dominance_max_deficit"]),
        "dominance_gap_bound": dominance_gap_bound,
        "dominance_gap_bound_over_xi": dominance_gap_bound / xi if xi > 0 else math.inf,
        "dominance_product_gap": dominance_product_gap,
        "dominance_product_gap_over_xi": dominance_product_gap / xi if xi > 0 else math.inf,
        "dominance_source_count": int(screen["source_count"]),
        "dominance_vertex_count": int(screen["vertex_count"]),
        "dominance_basis_count": int(screen["basis_count"]),
        "residual_dominance_max_deficit": float(screen["dominance_max_deficit"]),
        "residual_dominance_gap_bound": residual_bound,
        "residual_dominance_gap_bound_over_xi": (
            residual_bound / xi if xi > 0 else math.inf
        ),
        "residual_dominance_residual_gap": residual_gap,
        "residual_dominance_residual_gap_over_xi": (
            residual_gap / xi if xi > 0 else math.inf
        ),
        "residual_dominance_source_count": int(screen["source_count"]),
        "residual_dominance_vertex_count": int(screen["vertex_count"]),
        "residual_dominance_basis_count": int(screen["basis_count"]),
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
    parser.add_argument("--local-check-size", type=int, default=None)
    parser.add_argument("--local-marginal", action="store_true")
    parser.add_argument("--local-marginal-exact", action="store_true")
    parser.add_argument("--local-dominance-screen", action="store_true")
    parser.add_argument("--local-residual-dominance-screen", action="store_true")
    parser.add_argument("--local-marginal-max-bases", type=int, default=1_000_000)
    parser.add_argument("--local-marginal-objectives", type=int, default=16)
    parser.add_argument("--local-marginal-closure-rounds", type=int, default=2)
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

    if args.local_check_size is not None:
        checks = local_check_scopes(args.blocks, args.local_check_size)
        local_rows = [
            run_local_integral_trial(
                n_words=args.n_words,
                blocks=args.blocks,
                alphabet=args.alphabet,
                seed=args.seed + trial,
                xi=xi,
                checks=checks,
                lam=args.lam,
            )
            for trial in range(args.trials)
        ]
        for key in ["local_count", "local_gap", "local_gap_over_xi"]:
            values = np.array([float(row[key]) for row in local_rows])
            print(f"{key}_mean,{float(np.mean(values))}")
            print(f"{key}_max,{float(np.max(values))}")
        worst_local = max(local_rows, key=lambda row: float(row["local_gap_over_xi"]))
        print(f"worst_local_seed,{worst_local['seed']}")
        print(f"worst_local_word,{worst_local['local_word']}")

        if args.local_marginal:
            marginal_rows = [
                run_local_marginal_trial(
                    n_words=args.n_words,
                    blocks=args.blocks,
                    alphabet=args.alphabet,
                    seed=args.seed + trial,
                    xi=xi,
                    checks=checks,
                    lam=args.lam,
                    random_objectives=args.local_marginal_objectives,
                    closure_rounds=args.local_marginal_closure_rounds,
                )
                for trial in range(args.trials)
            ]
            for key in [
                "local_marginal_gap",
                "local_marginal_gap_over_xi",
                "local_marginal_source_count",
                "local_marginal_evaluated_count",
                "local_marginal_integrality_defect",
            ]:
                values = np.array([float(row[key]) for row in marginal_rows])
                print(f"{key}_mean,{float(np.mean(values))}")
                print(f"{key}_max,{float(np.max(values))}")
            worst_marginal = max(
                marginal_rows,
                key=lambda row: float(row["local_marginal_gap_over_xi"]),
            )
            print(f"worst_local_marginal_seed,{worst_marginal['seed']}")
            print(f"worst_local_marginal_min_chart,{worst_marginal['local_marginal_min_chart']}")
            clean_marginal_rows = [
                row for row, local_row in zip(marginal_rows, local_rows)
                if int(local_row["local_count"]) == 0
            ]
            print(f"local_marginal_clean_count,{len(clean_marginal_rows)}")
            if clean_marginal_rows:
                clean_values = np.array([
                    float(row["local_marginal_gap_over_xi"])
                    for row in clean_marginal_rows
                ])
                print(f"local_marginal_clean_gap_over_xi_mean,{float(np.mean(clean_values))}")
                print(f"local_marginal_clean_gap_over_xi_max,{float(np.max(clean_values))}")
                clean_over_budget = int(np.sum(clean_values > 1.0 + 1e-8))
                print(f"local_marginal_clean_over_budget_count,{clean_over_budget}")
                worst_clean = max(
                    clean_marginal_rows,
                    key=lambda row: float(row["local_marginal_gap_over_xi"]),
                )
                print(f"worst_local_marginal_clean_seed,{worst_clean['seed']}")

        if (
            args.local_marginal_exact
            or args.local_dominance_screen
            or args.local_residual_dominance_screen
        ):
            combined_rows = [
                run_local_combined_screen_trial(
                    n_words=args.n_words,
                    blocks=args.blocks,
                    alphabet=args.alphabet,
                    seed=args.seed + trial,
                    xi=xi,
                    checks=checks,
                    lam=args.lam,
                    max_bases=args.local_marginal_max_bases,
                )
                for trial in range(args.trials)
            ]

        if args.local_marginal_exact:
            for key in [
                "local_marginal_exact_gap",
                "local_marginal_exact_gap_over_xi",
                "local_marginal_exact_source_count",
                "local_marginal_exact_vertex_count",
                "local_marginal_exact_basis_count",
                "local_marginal_exact_feasible_basis_count",
                "local_marginal_exact_integrality_defect",
            ]:
                values = np.array([float(row[key]) for row in combined_rows])
                print(f"{key}_mean,{float(np.mean(values))}")
                print(f"{key}_max,{float(np.max(values))}")
            worst_exact = max(
                combined_rows,
                key=lambda row: float(row["local_marginal_exact_gap_over_xi"]),
            )
            print(f"worst_local_marginal_exact_seed,{worst_exact['seed']}")
            print(f"worst_local_marginal_exact_min_chart,{worst_exact['local_marginal_exact_min_chart']}")
            clean_exact_rows = [
                row for row, local_row in zip(combined_rows, local_rows)
                if int(local_row["local_count"]) == 0
            ]
            print(f"local_marginal_exact_clean_count,{len(clean_exact_rows)}")
            if clean_exact_rows:
                clean_values = np.array([
                    float(row["local_marginal_exact_gap_over_xi"])
                    for row in clean_exact_rows
                ])
                print(f"local_marginal_exact_clean_gap_over_xi_mean,{float(np.mean(clean_values))}")
                print(f"local_marginal_exact_clean_gap_over_xi_max,{float(np.max(clean_values))}")
                clean_over_budget = int(np.sum(clean_values > 1.0 + 1e-8))
                print(f"local_marginal_exact_clean_over_budget_count,{clean_over_budget}")
                worst_clean = max(
                    clean_exact_rows,
                    key=lambda row: float(row["local_marginal_exact_gap_over_xi"]),
                )
                print(f"worst_local_marginal_exact_clean_seed,{worst_clean['seed']}")

        if args.local_dominance_screen:
            for key in [
                "dominance_max_deficit",
                "dominance_gap_bound",
                "dominance_gap_bound_over_xi",
                "dominance_product_gap",
                "dominance_product_gap_over_xi",
                "dominance_source_count",
                "dominance_vertex_count",
                "dominance_basis_count",
            ]:
                values = np.array([float(row[key]) for row in combined_rows])
                print(f"{key}_mean,{float(np.mean(values))}")
                print(f"{key}_max,{float(np.max(values))}")
            worst_dominance = max(
                combined_rows,
                key=lambda row: float(row["dominance_gap_bound_over_xi"]),
            )
            print(f"worst_dominance_seed,{worst_dominance['seed']}")
            clean_dominance_rows = [
                row for row, local_row in zip(combined_rows, local_rows)
                if int(local_row["local_count"]) == 0
            ]
            print(f"dominance_clean_count,{len(clean_dominance_rows)}")
            if clean_dominance_rows:
                clean_values = np.array([
                    float(row["dominance_gap_bound_over_xi"])
                    for row in clean_dominance_rows
                ])
                print(f"dominance_clean_gap_bound_over_xi_mean,{float(np.mean(clean_values))}")
                print(f"dominance_clean_gap_bound_over_xi_max,{float(np.max(clean_values))}")
                clean_over_budget = int(np.sum(clean_values > 1.0 + 1e-8))
                print(f"dominance_clean_over_budget_count,{clean_over_budget}")
                worst_clean = max(
                    clean_dominance_rows,
                    key=lambda row: float(row["dominance_gap_bound_over_xi"]),
                )
                print(f"worst_dominance_clean_seed,{worst_clean['seed']}")

        if args.local_residual_dominance_screen:
            for key in [
                "residual_dominance_max_deficit",
                "residual_dominance_gap_bound",
                "residual_dominance_gap_bound_over_xi",
                "residual_dominance_residual_gap",
                "residual_dominance_residual_gap_over_xi",
                "residual_dominance_source_count",
                "residual_dominance_vertex_count",
                "residual_dominance_basis_count",
            ]:
                values = np.array([float(row[key]) for row in combined_rows])
                print(f"{key}_mean,{float(np.mean(values))}")
                print(f"{key}_max,{float(np.max(values))}")
            worst_residual = max(
                combined_rows,
                key=lambda row: float(row["residual_dominance_gap_bound_over_xi"]),
            )
            print(f"worst_residual_dominance_seed,{worst_residual['seed']}")
            clean_residual_rows = [
                row for row, local_row in zip(combined_rows, local_rows)
                if int(local_row["local_count"]) == 0
            ]
            print(f"residual_dominance_clean_count,{len(clean_residual_rows)}")
            if clean_residual_rows:
                clean_values = np.array([
                    float(row["residual_dominance_gap_bound_over_xi"])
                    for row in clean_residual_rows
                ])
                print(f"residual_dominance_clean_gap_bound_over_xi_mean,{float(np.mean(clean_values))}")
                print(f"residual_dominance_clean_gap_bound_over_xi_max,{float(np.max(clean_values))}")
                clean_over_budget = int(np.sum(clean_values > 1.0 + 1e-8))
                print(f"residual_dominance_clean_over_budget_count,{clean_over_budget}")
                worst_clean = max(
                    clean_residual_rows,
                    key=lambda row: float(row["residual_dominance_gap_bound_over_xi"]),
                )
                print(f"worst_residual_dominance_clean_seed,{worst_clean['seed']}")
    elif (
        args.local_marginal
        or args.local_marginal_exact
        or args.local_dominance_screen
        or args.local_residual_dominance_screen
    ):
        parser.error(
            "--local-marginal, --local-marginal-exact, and "
            "--local-dominance-screen/--local-residual-dominance-screen "
            "require --local-check-size",
        )


if __name__ == "__main__":
    main()
