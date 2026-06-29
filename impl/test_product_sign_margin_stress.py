import math

import numpy as np

from product_sign_margin_stress import (
    FIELDNAMES,
    bit_score,
    ceiling_gap_lower_tail,
    log_cosh,
    run_trial,
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
    for value in row.values():
        assert value == value


if __name__ == "__main__":
    test_bit_score_has_log_two_ceiling()
    test_ceiling_gap_lower_tail_is_valid_sufficient_event()
    test_run_trial_produces_complete_row()
    print("product_sign_margin_stress tests passed")
