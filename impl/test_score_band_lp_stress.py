import itertools

import numpy as np

from chart_tail_stress import (
    centered_symbol_scores,
    codeword_scores,
    make_balanced_code,
)
from score_band_lp_stress import (
    _simplex_max_nonnegative,
    convex_hull_dominance_deficit,
    integral_local_pseudoword_gaps,
    local_check_scopes,
    local_marginal_combined_screen,
    local_marginal_dominance_screen,
    local_marginal_residual_dominance_screen,
    local_marginal_score_band_gap_exact,
    local_marginal_score_band_gap,
    local_marginal_vertex_sources,
    local_overlap_score_band_gap,
    local_overlap_source_candidates,
    local_overlap_source_feasible,
    local_overlap_vertex_sources,
    product_simplex_score_band_gap,
    run_local_combined_screen_trial,
    run_local_integral_trial,
    solve_equality_lp_max,
    source_integrality_defect,
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


def test_exact_local_marginal_handles_lower_dimensional_check_hull():
    code = np.array([
        [0, 0],
        [1, 1],
    ])
    omega = np.ones(len(code))
    sources = local_marginal_vertex_sources(
        code,
        checks=[(0, 1)],
        max_bases=100,
    )
    result = local_marginal_score_band_gap_exact(
        code,
        omega,
        checks=[(0, 1)],
        max_bases=100,
    )

    assert sources["source_count"] == 2
    assert sources["basis_count"] <= 100
    assert result["gap"] <= 1e-8


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

    assert sources["source_count"] == 8
    assert np.isclose(result["gap"], 0.5)
    assert result["source_count"] == 8


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


def test_local_marginal_combined_screen_matches_separate_screens():
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
    combined = local_marginal_combined_screen(
        code,
        omega,
        checks=checks,
        max_bases=1000,
    )

    assert np.isclose(combined["exact_gap"], exact["gap"])
    assert np.isclose(combined["dominance_gap_bound"], global_screen["gap_bound"])
    assert np.isclose(
        combined["residual_dominance_gap_bound"],
        residual_screen["gap_bound"],
    )
    assert combined["source_count"] == exact["source_count"]


def test_clean_panel_can_fail_residual_dominance_certificate():
    n_words = 8
    xi = np.log(n_words) ** 3
    checks = local_check_scopes(4, 3)
    local = run_local_integral_trial(
        n_words=n_words,
        blocks=4,
        alphabet=2,
        seed=3,
        xi=xi,
        checks=checks,
    )
    combined = run_local_combined_screen_trial(
        n_words=n_words,
        blocks=4,
        alphabet=2,
        seed=3,
        xi=xi,
        checks=checks,
        max_bases=1_000_000,
    )

    assert local["local_count"] == 0
    assert combined["local_marginal_exact_gap_over_xi"] < 1.0
    assert combined["residual_dominance_gap_bound_over_xi"] > 1.0


def test_b5_exact_projected_source_finds_clean_fractional_obstruction():
    n_words = 10
    blocks = 5
    alphabet = 2
    seed = 3
    xi = np.log(n_words) ** 3
    checks = local_check_scopes(blocks, 4)
    code = make_balanced_code(n_words, blocks, alphabet, seed)
    symbol_scores = centered_symbol_scores(blocks, alphabet, seed + 1009)
    scores = codeword_scores(code, symbol_scores)
    lam = np.sqrt(2.0 * np.log(n_words) / blocks)
    omega = np.exp(np.minimum(lam * (float(np.max(scores)) - scores), 700.0)) + xi
    local = integral_local_pseudoword_gaps(code, omega, checks=checks)
    exact = local_marginal_score_band_gap_exact(
        code,
        omega,
        checks=checks,
        max_bases=1_000_000,
    )

    assert local["count"] == 0
    assert exact["basis_count"] < 100_000
    assert exact["source_count"] == 16
    assert exact["gap"] > 10.0 * xi
    overlap = local_overlap_source_feasible(code, checks, exact["source"])
    assert not overlap["feasible"]
    overlap_gap = local_overlap_score_band_gap(
        code,
        omega,
        checks,
        random_objectives=16,
        seed=0,
        closure_rounds=4,
    )
    assert overlap_gap["gap"] <= xi


def test_ternary_overlap_projection_can_have_fractional_source():
    code = np.array([
        [0, 1, 0, 1],
        [1, 2, 1, 2],
        [0, 2, 2, 2],
        [0, 1, 0, 0],
        [2, 2, 1, 0],
        [0, 0, 2, 1],
        [1, 0, 1, 1],
        [0, 0, 0, 0],
        [1, 2, 1, 2],
        [2, 0, 0, 1],
        [2, 1, 2, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 2],
        [1, 0, 1, 2],
        [1, 2, 2, 0],
        [0, 1, 2, 0],
        [1, 2, 2, 0],
        [2, 1, 1, 2],
        [2, 1, 0, 1],
        [2, 0, 0, 1],
        [2, 0, 0, 0],
        [0, 2, 2, 2],
        [0, 0, 1, 0],
        [2, 2, 2, 2],
    ])
    source = np.array([
        [1.0 / 3.0, 0.0, 2.0 / 3.0],
        [2.0 / 3.0, 0.0, 1.0 / 3.0],
        [1.0 / 3.0, 0.0, 2.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0, 0.0],
    ]).reshape(-1)
    result = local_overlap_source_feasible(
        code,
        local_check_scopes(4, 3),
        source,
    )

    assert result["feasible"]
    assert source_integrality_defect(source, blocks=4, alphabet=3) > 1.0


def test_binary_overlap_projection_samples_integral_parity_sources():
    blocks = 5
    code = np.array([
        word for word in itertools.product([0, 1], repeat=blocks)
        if sum(word) % 2 == 0
    ])
    sources = local_overlap_source_candidates(
        code,
        local_check_scopes(blocks, blocks - 1),
        random_objectives=32,
        seed=0,
    )
    defects = [
        source_integrality_defect(source, blocks=blocks, alphabet=2)
        for source in sources
    ]
    code_set = {tuple(word) for word in code}
    sampled_words = {
        tuple(int(np.argmax(source.reshape(blocks, 2)[coord])) for coord in range(blocks))
        for source in sources
    }

    assert max(defects) <= 1e-8
    assert any(word not in code_set for word in sampled_words)


def test_binary_overlap_projection_can_have_fractional_source():
    code = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ])
    source = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ]).reshape(-1)
    result = local_overlap_source_feasible(
        code,
        local_check_scopes(3, 2),
        source,
    )

    assert result["feasible"]
    assert source_integrality_defect(source, blocks=3, alphabet=2) == 1.5


def test_binary_overlap_vertex_enumerator_finds_fractional_triangle_source():
    code = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ])
    vertices = local_overlap_vertex_sources(
        code,
        local_check_scopes(3, 2),
        random_objectives=0,
    )
    defects = [
        source_integrality_defect(source, blocks=3, alphabet=2)
        for source in vertices["sources"]
    ]
    all_half = np.full(6, 0.5)

    assert vertices["certified"]
    assert any(np.allclose(source, all_half) for source in vertices["sources"])
    assert np.isclose(max(defects), 1.5)


def test_binary_overlap_vertex_enumerator_certifies_integral_parity_projection():
    blocks = 3
    code = np.array([
        word for word in itertools.product([0, 1], repeat=blocks)
        if sum(word) % 2 == 0
    ])
    vertices = local_overlap_vertex_sources(
        code,
        local_check_scopes(blocks, blocks - 1),
        random_objectives=0,
    )
    defects = [
        source_integrality_defect(source, blocks=blocks, alphabet=2)
        for source in vertices["sources"]
    ]
    code_set = {tuple(word) for word in code}
    vertex_words = {
        tuple(int(np.argmax(source.reshape(blocks, 2)[coord])) for coord in range(blocks))
        for source in vertices["sources"]
    }

    assert vertices["certified"]
    assert vertices["source_count"] == 2 ** blocks
    assert max(defects) <= 1e-8
    assert any(word not in code_set for word in vertex_words)


def test_leave_one_out_overlap_collapses_for_two_deletion_injective_code():
    blocks = 5
    code = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 1, 1],
    ])
    vertices = local_overlap_vertex_sources(
        code,
        local_check_scopes(blocks, blocks - 1),
        random_objectives=0,
    )
    defects = [
        source_integrality_defect(source, blocks=blocks, alphabet=2)
        for source in vertices["sources"]
    ]
    code_set = {tuple(word) for word in code}
    vertex_words = {
        tuple(int(np.argmax(source.reshape(blocks, 2)[coord])) for coord in range(blocks))
        for source in vertices["sources"]
    }

    assert vertices["certified"]
    assert vertices["source_count"] == len(code)
    assert max(defects) <= 1e-8
    assert vertex_words == code_set


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
    test_exact_local_marginal_handles_lower_dimensional_check_hull()
    test_local_marginal_pairwise_parity_sees_integral_gap()
    test_exact_local_marginal_pairwise_parity_matches_integral_gap()
    test_local_marginal_dominance_screen_bounds_exact_gap()
    test_local_marginal_residual_dominance_screen_bounds_exact_gap()
    test_local_marginal_combined_screen_matches_separate_screens()
    test_clean_panel_can_fail_residual_dominance_certificate()
    test_b5_exact_projected_source_finds_clean_fractional_obstruction()
    test_ternary_overlap_projection_can_have_fractional_source()
    test_binary_overlap_projection_samples_integral_parity_sources()
    test_binary_overlap_projection_can_have_fractional_source()
    test_binary_overlap_vertex_enumerator_finds_fractional_triangle_source()
    test_binary_overlap_vertex_enumerator_certifies_integral_parity_projection()
    test_leave_one_out_overlap_collapses_for_two_deletion_injective_code()
    test_local_check_scopes_enumerates_subsets()
    print("score_band_lp_stress tests passed")
