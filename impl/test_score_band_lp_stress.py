import numpy as np

from score_band_lp_stress import (
    _simplex_max_nonnegative,
    convex_hull_dominance_deficit,
    integral_local_pseudoword_gaps,
    local_check_scopes,
    local_marginal_dominance_screen,
    local_marginal_residual_dominance_screen,
    local_marginal_score_band_gap_exact,
    local_marginal_score_band_gap,
    local_marginal_vertex_sources,
    product_simplex_score_band_gap,
    solve_equality_lp_max,
    source_score_band_gap,
    true_hull_dominance_decomposition,
)


def test_simplex_solves_small_lp():
    objective = np.array([1.0, 1.0])
    lhs = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
    ])
    rhs = np.array([4.0, 6.0])
    value, solution = _simplex_max_nonnegative(objective, lhs, rhs)

    assert np.isclose(value, 10.0 / 3.0)
    assert np.allclose(solution, [8.0 / 3.0, 2.0 / 3.0])


def test_equality_lp_solves_nonzero_rhs_simplex():
    objective = np.array([2.0, 1.0])
    lhs = np.array([[1.0, 1.0]])
    rhs = np.array([1.0])
    value, solution = solve_equality_lp_max(objective, lhs, rhs)

    assert np.isclose(value, 2.0)
    assert np.allclose(solution, [1.0, 0.0])


def test_full_binary_cube_has_zero_product_simplex_gap():
    code = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    omega = np.ones(len(code))
    result = product_simplex_score_band_gap(code, omega)

    assert result["gap"] <= 1e-8


def test_missing_binary_corner_exposes_score_band_gap():
    code = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
    ])
    omega = np.ones(len(code))
    result = product_simplex_score_band_gap(code, omega)

    assert np.isclose(result["gap"], 1.0)
    assert result["max_chart"] == (1, 1)


def test_fixed_true_codeword_has_zero_gap():
    code = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
    ])
    omega = np.ones(len(code))
    result = source_score_band_gap(code, omega, source_word=(0, 1))

    assert result["gap"] <= 1e-8


def test_fixed_missing_corner_matches_product_gap():
    code = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
    ])
    omega = np.ones(len(code))
    result = source_score_band_gap(code, omega, source_word=(1, 1))

    assert np.isclose(result["gap"], 1.0)


def test_convex_hull_dominance_deficit_matches_simple_cases():
    code = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
    ])
    assert convex_hull_dominance_deficit(code, source_word=(0, 1)) <= 1e-8
    assert np.isclose(convex_hull_dominance_deficit(code, source_word=(1, 1)), 1.0)


def test_true_hull_dominance_decomposition_reconstructs_source():
    code = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
    ])
    source = np.array([0.25, 0.75, 0.25, 0.75])
    result = true_hull_dominance_decomposition(code, source=source)

    assert np.isclose(result["deficit"], 0.5)
    assert np.allclose(
        source,
        result["dominated_source"] + result["deficit"] * result["residual_source"],
    )


def test_pairwise_local_parity_pseudoword_is_detected():
    code = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])
    omega = np.ones(len(code))
    result = integral_local_pseudoword_gaps(
        code,
        omega,
        checks=[(0, 1), (0, 2), (1, 2)],
    )

    assert result["count"] == 4
    assert result["gap"] > 0.0
    assert result["word"] in {(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)}


def test_local_marginal_full_check_collapses_to_true_hull():
    code = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
    ])
    omega = np.ones(len(code))
    result = local_marginal_score_band_gap(
        code,
        omega,
        checks=[(0, 1)],
        random_objectives=4,
        seed=0,
    )

    assert result["gap"] <= 1e-8
    assert result["source_count"] > 0


def test_exact_local_marginal_full_check_collapses_to_true_hull():
    code = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
    ])
    omega = np.ones(len(code))
    result = local_marginal_score_band_gap_exact(
        code,
        omega,
        checks=[(0, 1)],
        max_bases=100,
    )

    assert result["gap"] <= 1e-8
    assert result["source_count"] == len(code)
    assert result["basis_count"] <= 100


def test_local_marginal_pairwise_parity_sees_integral_gap():
    code = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])
    omega = np.ones(len(code))
    result = local_marginal_score_band_gap(
        code,
        omega,
        checks=[(0, 1), (0, 2), (1, 2)],
        random_objectives=4,
        seed=0,
    )

    assert result["gap"] >= 0.5 - 1e-8
    assert result["source_count"] >= 8


def test_exact_local_marginal_pairwise_parity_matches_integral_gap():
    code = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])
    omega = np.ones(len(code))
    sources = local_marginal_vertex_sources(
        code,
        checks=[(0, 1), (0, 2), (1, 2)],
        max_bases=1000,
    )
    result = local_marginal_score_band_gap_exact(
        code,
        omega,
        checks=[(0, 1), (0, 2), (1, 2)],
        max_bases=1000,
    )

    assert sources["source_count"] == 9
    assert np.isclose(result["gap"], 0.5)
    assert result["source_count"] == 9


def test_local_marginal_dominance_screen_bounds_exact_gap():
    code = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])
    omega = np.ones(len(code))
    checks = [(0, 1), (0, 2), (1, 2)]
    exact = local_marginal_score_band_gap_exact(
        code,
        omega,
        checks=checks,
        max_bases=1000,
    )
    screen = local_marginal_dominance_screen(
        code,
        omega,
        checks=checks,
        max_bases=1000,
    )

    assert screen["max_deficit"] <= 1.0
    assert exact["gap"] <= screen["gap_bound"] + 1e-8


def test_local_marginal_residual_dominance_screen_bounds_exact_gap():
    code = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])
    omega = np.ones(len(code))
    checks = [(0, 1), (0, 2), (1, 2)]
    exact = local_marginal_score_band_gap_exact(
        code,
        omega,
        checks=checks,
        max_bases=1000,
    )
    global_screen = local_marginal_dominance_screen(
        code,
        omega,
        checks=checks,
        max_bases=1000,
    )
    residual_screen = local_marginal_residual_dominance_screen(
        code,
        omega,
        checks=checks,
        max_bases=1000,
    )

    assert exact["gap"] <= residual_screen["gap_bound"] + 1e-8
    assert residual_screen["gap_bound"] <= global_screen["gap_bound"] + 1e-8


def test_local_check_scopes_enumerates_subsets():
    assert local_check_scopes(3, 2) == [(0, 1), (0, 2), (1, 2)]


if __name__ == "__main__":
    test_simplex_solves_small_lp()
    test_equality_lp_solves_nonzero_rhs_simplex()
    test_full_binary_cube_has_zero_product_simplex_gap()
    test_missing_binary_corner_exposes_score_band_gap()
    test_fixed_true_codeword_has_zero_gap()
    test_fixed_missing_corner_matches_product_gap()
    test_convex_hull_dominance_deficit_matches_simple_cases()
    test_true_hull_dominance_decomposition_reconstructs_source()
    test_pairwise_local_parity_pseudoword_is_detected()
    test_local_marginal_full_check_collapses_to_true_hull()
    test_exact_local_marginal_full_check_collapses_to_true_hull()
    test_local_marginal_pairwise_parity_sees_integral_gap()
    test_exact_local_marginal_pairwise_parity_matches_integral_gap()
    test_local_marginal_dominance_screen_bounds_exact_gap()
    test_local_marginal_residual_dominance_screen_bounds_exact_gap()
    test_local_check_scopes_enumerates_subsets()
    print("score_band_lp_stress tests passed")
