import math

import numpy as np

from smoothed_scoremax_stress import (
    FIELDNAMES,
    promised_spherical_instance,
    rho_for_c,
    run_trial,
    smoothed_scores,
)


def test_promised_instance_has_critical_distances():
    rng = np.random.default_rng(0)
    x, q, p, p_index = promised_spherical_instance(
        n_far=200,
        d=16,
        c=2.0,
        far_corr_max=-0.05,
        rng=rng,
    )
    assert p_index == 200
    assert np.all(x[:p_index] @ q <= -0.05 + 1e-12)
    assert abs(float(p @ q) - 0.75) < 1e-12
    r = math.sqrt(2.0) / 2.0
    assert np.linalg.norm(p - q) <= r + 1e-12
    assert np.all(np.linalg.norm(x[:p_index] - q, axis=1) > 2.0 * r)


def test_smoothed_scores_scale_and_margin_are_consistent():
    rng = np.random.default_rng(1)
    x, q, _p, p_index = promised_spherical_instance(
        n_far=100,
        d=12,
        c=2.0,
        far_corr_max=-0.1,
        rng=rng,
    )
    scores, y, t, margin = smoothed_scores(x, q, c=2.0, t_scale=64.0, rng=rng)
    assert scores.shape == (101,)
    assert y.shape == q.shape
    assert t > 0.0
    assert margin > 0.0
    assert int(np.argmax(scores)) == p_index


def test_run_trial_reports_complete_scoremax_diagnostics():
    row = run_trial(
        n=400,
        d=16,
        c=2.0,
        far_corr_max=-0.05,
        t_scale=64.0,
        smooth_filter_theta=0.7,
        query_trials=12,
        seed=2,
    )
    assert set(row) == set(FIELDNAMES)
    assert row["rho"] == rho_for_c(2.0)
    assert row["score_gap_fail_rate"] <= 1.0
    assert row["oracle_top1_valid_rate"] >= 0.0
    assert row["smooth_filter_candidate_mean"] >= 0.0
    assert row["smooth_filter_candidate_q90"] >= row["smooth_filter_candidate_mean"] / 12.0
    assert row["smooth_filter_best_loss_over_margin_q90"] >= 0.0
    for value in row.values():
        assert value == value


if __name__ == "__main__":
    test_promised_instance_has_critical_distances()
    test_smoothed_scores_scale_and_margin_are_consistent()
    test_run_trial_reports_complete_scoremax_diagnostics()
    print("smoothed_scoremax_stress tests passed")
