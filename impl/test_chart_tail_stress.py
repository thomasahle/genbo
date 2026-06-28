import numpy as np

from chart_tail_stress import (
    allowed_tail_mass,
    centered_overlap_matrix,
    codeword_scores,
    coefficient_decay_budget,
    make_balanced_code,
    min_decay_ratio_for_budget,
    posterior_residual_metrics,
    run_trial,
    top_anchor_cover_metrics,
)


def test_balanced_code_has_nearly_equal_coordinate_counts():
    code = make_balanced_code(n_words=22, blocks=7, alphabet=5, seed=0)
    assert code.shape == (22, 7)
    for i in range(code.shape[1]):
        counts = np.bincount(code[:, i], minlength=5)
        assert counts.max() - counts.min() <= 1


def test_centered_overlap_matches_direct_formula():
    code = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ])
    overlap = centered_overlap_matrix(code)
    direct = np.array([
        [3, 2, 0],
        [2, 3, 1],
        [0, 1, 3],
    ], dtype=float) - 3 / 2
    assert np.allclose(overlap, direct)
    assert np.allclose(overlap, overlap.T)


def test_codeword_scores_are_additive_over_blocks():
    code = np.array([
        [0, 1, 0],
        [1, 1, 0],
    ])
    symbol_scores = np.array([
        [1.0, 2.0],
        [3.0, 5.0],
        [7.0, 11.0],
    ])
    assert np.allclose(codeword_scores(code, symbol_scores), [13.0, 14.0])


def test_posterior_residual_metrics_match_direct_costs():
    code = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
    ])
    prices = np.array([6.0, 3.0, 1.0])
    metrics = posterior_residual_metrics(code, prices, alphabet=2)

    posterior = np.array([
        [0.9, 0.1],
        [0.6, 0.4],
    ])
    offmass = np.array([
        (1 - 0.9) + (1 - 0.6),
        (1 - 0.9) + (1 - 0.4),
        (1 - 0.1) + (1 - 0.4),
    ])
    assert np.isclose(metrics["singleton_cost"], np.max(prices * offmass))
    assert np.isclose(metrics["max_offmass"], np.max(offmass))
    assert np.isclose(metrics["avg_offmass"], np.dot(prices, offmass) / prices.sum())

    plurality_minus_collision = np.max(posterior, axis=1).sum() - np.sum(posterior * posterior)
    assert np.isclose(metrics["self_selector_mismatch"], prices.sum() * plurality_minus_collision)


def test_top_anchor_cover_metrics_removes_large_singleton():
    code = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
    ])
    prices = np.array([100.0, 2.0, 1.0])
    no_anchor = posterior_residual_metrics(code, prices, alphabet=2)
    assert no_anchor["singleton_cost"] > 2.0

    metrics = top_anchor_cover_metrics(code, prices, alphabet=2, budget=2.0)
    assert metrics["count"] == 1.0
    assert metrics["remaining_cost"] <= 2.0
    assert np.isclose(metrics["weight_fraction"], 100.0 / 103.0)
    assert metrics["remaining_max_price"] == 2.0


def test_allowed_tail_mass_matches_boltzmann_necessity():
    assert np.isclose(allowed_tail_mass(10.0, 2.0, 2.0), 10.0 * np.exp(-4.0))
    assert allowed_tail_mass(10.0, 2.0, 0.1) == 1.0
    try:
        allowed_tail_mass(-1.0, 1.0, 1.0)
    except ValueError as exc:
        assert "budget" in str(exc)
    else:
        raise AssertionError("negative budget was accepted")


def test_coefficient_decay_budget_has_lambda_threshold():
    gaps = np.array([0.0, 1.0, 2.0])
    lam = 1.0
    uniform_budget = coefficient_decay_budget(gaps, lam, 0.0)
    lambda_budget = coefficient_decay_budget(gaps, lam, lam)
    steep_budget = coefficient_decay_budget(gaps, lam, 2.0 * lam)

    assert np.isclose(uniform_budget, np.mean(np.exp(gaps)))
    assert np.isclose(lambda_budget, 3.0 / np.sum(np.exp(-gaps)))
    assert uniform_budget > lambda_budget > steep_budget >= 1.0
    assert np.isclose(min_decay_ratio_for_budget(gaps, lam, lambda_budget), 1.0)
    assert min_decay_ratio_for_budget(gaps, lam, 0.5) == np.inf


def test_run_trial_is_deterministic_and_clips_the_singleton_cost():
    row1 = run_trial(n_words=40, blocks=10, alphabet=4, seed=3, price_cap=5.0)
    row2 = run_trial(n_words=40, blocks=10, alphabet=4, seed=3, price_cap=5.0)
    assert row1 == row2
    assert row1["omega_max"] >= row1["price_cap"]
    assert row1["clipped_singleton_cost"] <= row1["uniform_singleton_cost"]
    assert 0.0 <= row1["full_diag_share"] <= 1.0
    assert 0.0 <= row1["clipped_diag_share"] <= 1.0
    assert row1["near_leader_count"] + row1["far_tail_count"] == row1["n_words"]
    assert row1["clipped_gap_cutoff"] > 0.0
    assert 0.0 <= row1["far_tail_weight_fraction"] <= 1.0
    assert 0.0 <= row1["coeff_mass_allow_q50"] <= 1.0
    assert row1["coeff_mass_allow_max"] <= row1["coeff_mass_allow_q90"]
    assert row1["coeff_mass_allow_q90"] <= row1["coeff_mass_allow_q50"]
    assert row1["coeff_decay_budget_uniform"] >= row1["coeff_decay_budget_half_lambda"]
    assert row1["coeff_decay_budget_half_lambda"] >= row1["coeff_decay_budget_lambda"]
    assert row1["coeff_decay_budget_lambda"] >= row1["coeff_decay_budget_3half_lambda"]
    assert row1["coeff_decay_budget_3half_lambda"] >= 1.0
    assert row1["coeff_decay_min_ratio_for_cap"] >= 0.0


if __name__ == "__main__":
    test_balanced_code_has_nearly_equal_coordinate_counts()
    test_centered_overlap_matches_direct_formula()
    test_codeword_scores_are_additive_over_blocks()
    test_posterior_residual_metrics_match_direct_costs()
    test_top_anchor_cover_metrics_removes_large_singleton()
    test_allowed_tail_mass_matches_boltzmann_necessity()
    test_coefficient_decay_budget_has_lambda_threshold()
    test_run_trial_is_deterministic_and_clips_the_singleton_cost()
    print("chart_tail_stress tests passed")
