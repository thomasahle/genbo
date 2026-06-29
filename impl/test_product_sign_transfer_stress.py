import math

import numpy as np

from finite_channel_diagnostic import product_sign_labels
from product_sign_transfer_stress import (
    FIELDNAMES,
    coupled_sphere_gaussian_points,
    covariance_spectral_summary,
    exact_tilted_good_mass,
    gaussian_row_pair_parameters,
    run_trial,
    upper_tail_thresholds,
)


def test_upper_tail_thresholds_use_top_eta_order_statistic():
    scores = np.array([
        [0.0, 10.0],
        [1.0, 9.0],
        [2.0, 8.0],
        [3.0, 7.0],
        [4.0, 6.0],
    ])
    thresholds = upper_tail_thresholds(scores, eta=0.4)
    assert np.allclose(thresholds, [3.0, 9.0])


def test_exact_tilted_good_mass_matches_manual_enumeration():
    labels = product_sign_labels(4)
    up = np.array([[0.4, -0.8]])
    uq = np.array([[0.1, -0.2]])
    alpha = 0.25
    thresholds = np.array([-2.0, -0.5, -0.1, 0.25])
    margin = 0.1

    mass = exact_tilted_good_mass(
        up=up,
        uq=uq,
        alpha=alpha,
        labels=labels,
        thresholds=thresholds,
        margin=margin,
    )

    tilt = alpha * up[0] + (1.0 - alpha) * uq[0]
    weights = np.exp(labels @ tilt)
    weights /= np.sum(weights)
    scores = labels @ up[0] - np.sum(np.logaddexp(up[0], -up[0]) - math.log(2.0))
    manual = np.sum(weights * (scores >= thresholds + margin))
    assert np.allclose(mass, [manual])


def test_covariance_spectral_summary_detects_scalar_and_anisotropic_cases():
    scalar = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ])
    rel_op, rel_min, rel_max = covariance_spectral_summary(scalar)
    assert rel_op < 1e-12
    assert abs(rel_min - 1.0) < 1e-12
    assert abs(rel_max - 1.0) < 1e-12

    stretched = scalar * np.array([2.0, 1.0])
    rel_op2, rel_min2, rel_max2 = covariance_spectral_summary(stretched)
    assert rel_op2 > 0.5
    assert rel_min2 < 1.0 < rel_max2


def test_coupled_sphere_gaussian_points_share_radial_direction():
    sphere, gaussian_shell, rel_shell = coupled_sphere_gaussian_points(32, 5, seed=7)
    assert sphere.shape == gaussian_shell.shape == (32, 5)
    assert rel_shell.shape == (32,)
    assert np.allclose(np.linalg.norm(sphere, axis=1), 1.0)
    radial_ratio = np.linalg.norm(gaussian_shell, axis=1)
    assert np.allclose(rel_shell, np.abs(radial_ratio - 1.0))
    dots = np.sum(sphere * gaussian_shell, axis=1)
    assert np.all(dots > 0.0)


def test_gaussian_row_pair_parameters_match_monte_carlo_rows():
    rng = np.random.default_rng(3)
    left = np.array([[0.6, -0.2, 0.4]])
    right = np.array([[0.1, 0.7, -0.3]])
    transform = np.array([
        [1.2, 0.1, -0.2],
        [0.0, 0.8, 0.3],
        [0.4, -0.1, 1.1],
    ])
    bit_scale = 1.7
    sigma_left, sigma_right, corr = gaussian_row_pair_parameters(
        left, right, transform=transform, bit_scale=bit_scale)

    rows = rng.standard_normal((60_000, 3)) @ transform / math.sqrt(3.0)
    left_logits = bit_scale * (rows @ left[0])
    right_logits = bit_scale * (rows @ right[0])
    empirical_cov = np.cov(np.stack([left_logits, right_logits]), bias=True)
    assert np.allclose(math.sqrt(empirical_cov[0, 0]), sigma_left[0], rtol=0.03)
    assert np.allclose(math.sqrt(empirical_cov[1, 1]), sigma_right[0], rtol=0.03)
    empirical_corr = empirical_cov[0, 1] / math.sqrt(empirical_cov[0, 0] * empirical_cov[1, 1])
    assert np.allclose(empirical_corr, corr[0], atol=0.03)


def test_transfer_trial_produces_complete_row():
    row = run_trial(
        n=80,
        d=6,
        c=2.0,
        queries=6,
        panel_kind="whitened_product_sign",
        bit_sigma=1.4,
        scale=None,
        seed=0,
        b_count=8,
        ref_samples=120,
        threshold_label_samples=8,
        tilted_label_samples=96,
        ideal_samples=1000,
        margin=0.5,
        near_correlation=0.9,
        rate_level_slack=0.05,
        rate_quadrature=32,
        rate_theta_grid=200,
    )
    assert set(row) == set(FIELDNAMES)
    assert row["B"] == 8
    assert row["r_bits"] == 3
    assert row["threshold_labels"] == 8
    assert row["enumerated_thresholds"] == 1.0
    assert row["bit_sigma_fit"] > 0.0
    assert row["bit_var_q05"] <= row["bit_var_q95"] + 1e-12
    assert row["ref_cov_rel_op"] >= 0.0
    assert row["ref_cov_lambda_min_rel"] <= 1.0 <= row["ref_cov_lambda_max_rel"]
    assert row["shell_rel_dev_q95"] <= row["shell_rel_dev_max"] + 1e-12
    assert row["metric_sigma_near_mean"] > 0.0
    assert row["metric_sigma_query_mean"] > 0.0
    assert row["metric_pair_corr_q05"] <= row["metric_pair_corr_mean"] + 1e-12
    assert row["pair_corr_q05"] <= row["pair_corr_mean"] + 1e-12
    assert row["threshold_gaussian_ref_q95"] == row["threshold_gaussian_ref_q95"]
    assert row["threshold_sphere_minus_gaussian_q95"] <= (
        row["threshold_sphere_minus_gaussian_max"] + 1e-12)
    assert row["threshold_ref_node_abs_q50"] <= row["threshold_ref_node_abs_q90"] + 1e-12
    assert row["exact_good_mass_q05"] >= 0.0
    assert row["sampled_good_mass_q05"] >= 0.0
    for key, value in row.items():
        if key == "panel":
            continue
        assert value == value
        assert math.isfinite(value)


if __name__ == "__main__":
    test_upper_tail_thresholds_use_top_eta_order_statistic()
    test_exact_tilted_good_mass_matches_manual_enumeration()
    test_covariance_spectral_summary_detects_scalar_and_anisotropic_cases()
    test_coupled_sphere_gaussian_points_share_radial_direction()
    test_gaussian_row_pair_parameters_match_monte_carlo_rows()
    test_transfer_trial_produces_complete_row()
    print("product_sign_transfer_stress tests passed")
