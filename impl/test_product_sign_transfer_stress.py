import math

import numpy as np

from finite_channel_diagnostic import product_sign_labels
from product_sign_transfer_stress import (
    FIELDNAMES,
    exact_tilted_good_mass,
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
    assert row["pair_corr_q05"] <= row["pair_corr_mean"] + 1e-12
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
    test_transfer_trial_produces_complete_row()
    print("product_sign_transfer_stress tests passed")
