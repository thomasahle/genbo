import math

import numpy as np

from product_sign_margin_stress import (
    FIELDNAMES,
    bit_score,
    ceiling_gap_lower_tail,
    fixed_score_rate,
    gaussian_rate_gap_summary,
    log_cosh,
    product_sign_mode_deficit,
    run_trial,
    tilted_score_mean_variance,
)


def test_bit_score_has_log_two_ceiling():
    z = np.array([-20.0, -1.0, 0.0, 1.0, 20.0])
    scores = bit_score(z)
    assert np.all(scores <= math.log(2.0) + 1e-12)
    assert scores[-1] > math.log(2.0) - 1e-8
    assert np.allclose(log_cosh(z), log_cosh(-z))


def test_ceiling_gap_lower_tail_is_valid_sufficient_event():
    prob = ceiling_gap_lower_tail(r_bits=4, sigma=2.0, gap=1.0)
    rng = np.random.default_rng(0)
    z = 2.0 * rng.standard_normal((20_000, 4))
    scores = bit_score(z).sum(axis=1)
    empirical = np.mean(scores >= 4 * math.log(2.0) - 1.0)
    assert 0.0 < prob < empirical < 1.0


def test_tilted_score_mean_variance_matches_enumeration():
    up = np.array([[0.4, -1.2]])
    uq = np.array([[0.7, -0.3]])
    alpha = 0.2
    mean, variance = tilted_score_mean_variance(up, uq, alpha)

    tilt = alpha * up[0] + (1.0 - alpha) * uq[0]
    probs = []
    scores = []
    for s0 in (-1.0, 1.0):
        for s1 in (-1.0, 1.0):
            signs = np.array([s0, s1])
            weight = np.exp(np.dot(signs, tilt))
            probs.append(weight)
            scores.append(np.sum(signs * up[0] - log_cosh(up[0])))
    probs = np.array(probs) / np.sum(probs)
    scores = np.array(scores)
    assert np.allclose(mean, np.sum(probs * scores))
    assert np.allclose(variance, np.sum(probs * (scores - mean[0]) ** 2))


def test_gaussian_rate_gap_summary_has_positive_gap_and_rate():
    summary = gaussian_rate_gap_summary(
        sigma=2.0,
        corr=0.75,
        c=2.0,
        level_slack=0.1,
        quadrature=48,
        theta_grid=800,
    )
    assert summary["tilted_mean_limit"] > summary["fixed_score_mean"]
    assert summary["score_level"] < summary["tilted_mean_limit"]
    assert summary["chernoff_rate"] > 0.0
    assert summary["min_log_b_exponent"] < float("inf")
    assert summary["mode_deficit"] > 0.0
    assert summary["min_topk_mass_exponent"] > 0.0

    mean_rate = fixed_score_rate(
        summary["fixed_score_mean"],
        sigma=2.0,
        quadrature=48,
        theta_grid=200,
    )
    assert mean_rate < 1e-8


def test_product_sign_mode_deficit_matches_monte_carlo_mode_mass():
    sigma = 1.5
    estimate = product_sign_mode_deficit(sigma, quadrature=64)
    rng = np.random.default_rng(4)
    z = sigma * rng.standard_normal(100_000)
    empirical = np.mean(np.log1p(np.exp(-2.0 * np.abs(z))))
    assert abs(estimate - empirical) < 0.01


def test_run_trial_produces_complete_row():
    row = run_trial(
        m=1_000,
        c=2.0,
        r_bits=4,
        sigma=2.0,
        corr=0.9,
        margin=0.5,
        quantile_samples=4_000,
        pair_trials=64,
        label_samples=96,
        seed=1,
    )
    assert set(row) == set(FIELDNAMES)
    assert row["r"] == 4
    assert row["m"] == 1_000
    assert row["ceiling_gap"] >= 0.0
    assert row["good_mass_q01"] <= row["good_mass_q05"] + 1e-12
    assert row["good_mass_q05"] <= row["good_mass_median"] + 1e-12
    assert row["bound_q05"] <= row["affinity_q05"] + 1e-12
    assert row["bound_over_alpha_q05"] >= 0.0
    assert row["mean_gap_q01"] <= row["mean_gap_q05"] + 1e-12
    assert row["cantelli_good_q05"] <= row["good_mass_q05"] + 1e-12
    assert row["cantelli_good_q01"] <= row["cantelli_good_q05"] + 1e-12
    assert row["cantelli_bound_over_alpha_q05"] <= row["bound_over_alpha_q05"] + 1e-12
    for value in row.values():
        assert value == value


if __name__ == "__main__":
    test_bit_score_has_log_two_ceiling()
    test_ceiling_gap_lower_tail_is_valid_sufficient_event()
    test_tilted_score_mean_variance_matches_enumeration()
    test_gaussian_rate_gap_summary_has_positive_gap_and_rate()
    test_product_sign_mode_deficit_matches_monte_carlo_mode_mass()
    test_run_trial_produces_complete_row()
    print("product_sign_margin_stress tests passed")
