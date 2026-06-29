import math

import numpy as np

from smoothed_gibbs_stress import (
    FIELDNAMES,
    plateau_bound,
    rho_for_c,
    sample_plateau_trial,
    topk_gibbs_mass_from_scores,
)


def test_topk_gibbs_mass_matches_flat_plateau():
    scores = np.array([0.0] * 10 + [-100.0] * 90)
    mass = topk_gibbs_mass_from_scores(scores, 3)
    assert abs(mass - 0.3) < 1e-10
    assert topk_gibbs_mass_from_scores(scores, 0) == 0.0
    assert topk_gibbs_mass_from_scores(scores, len(scores)) == 1.0


def test_plateau_bound_exposes_asymptotic_gap():
    c = 2.0
    rho = rho_for_c(c)
    row = plateau_bound(
        m=10**24,
        c=c,
        valid_exponent=0.5,
        list_log_power=2.0,
        capture_log_power=2.0,
    )
    assert set(row) == set(FIELDNAMES)
    assert math.isclose(row["rho"], rho)
    assert row["mass_power_gap"] < 0.0
    assert row["guard_power_gap"] < 0.0
    assert row["guard_density"] < row["guard_threshold"]
    assert row["topk_mass_upper"] < row["capture_threshold"]
    assert row["capture_fails_finite"]


def test_sample_plateau_trial_keeps_invalid_mass_negligible():
    row = sample_plateau_trial(
        m=20_000,
        c=2.0,
        valid_exponent=0.7,
        list_log_power=0.0,
        capture_log_power=1.0,
        seed=3,
        t_constant=48.0,
        invalid_radius=3.0,
    )
    assert row["guard_density"] < row["guard_threshold"]
    assert row["invalid_max_score"] < 0.0
    assert row["invalid_mass"] < 1e-12
    assert row["sampled_topk_mass"] <= row["topk_mass_upper"] * (1.0 + 1e-9)
    assert row["sampled_topk_mass"] < row["capture_threshold"]


if __name__ == "__main__":
    test_topk_gibbs_mass_matches_flat_plateau()
    test_plateau_bound_exposes_asymptotic_gap()
    test_sample_plateau_trial_keeps_invalid_mass_negligible()
    print("smoothed_gibbs_stress tests passed")
