import itertools
import math

import numpy as np

from product_sign_margin_stress import bit_score
from sparse_subset_channel_stress import (
    FIELDNAMES,
    empirical_rate,
    label_log_count,
    log_elementary_symmetric,
    log_elementary_symmetric_batch,
    run_trial,
    sparse_subset_scores,
)


def test_log_elementary_symmetric_matches_bruteforce():
    log_weights = np.log(np.array([1.2, 0.7, 2.0, 3.5]))
    for width in range(1, 4):
        got = log_elementary_symmetric(log_weights, width)
        terms = []
        for combo in itertools.combinations(range(len(log_weights)), width):
            terms.append(float(np.sum(log_weights[list(combo)])))
        expected = np.logaddexp.reduce(terms)
        assert abs(got - expected) < 1e-12


def test_batch_log_elementary_symmetric_matches_scalar():
    rng = np.random.default_rng(3)
    log_weights = rng.normal(size=(7, 5))
    batch = log_elementary_symmetric_batch(log_weights, width=3)
    scalar = np.array([log_elementary_symmetric(row, 3) for row in log_weights])
    assert np.allclose(batch, scalar)


def test_full_width_sparse_subset_is_product_sign_score():
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(11, 4))
    signs = np.ones((len(logits), 4))
    indices = np.tile(np.arange(4), (len(logits), 1))
    scores = sparse_subset_scores(logits, indices, signs)
    expected = np.sum(bit_score(logits), axis=1)
    assert np.allclose(scores, expected)


def test_empirical_rate_is_zero_below_mean_and_positive_above():
    samples = np.array([-1.0, 0.0, 1.0, 2.0])
    assert empirical_rate(samples, level=-0.5, blocklength=2, theta_grid=40) == 0.0
    assert empirical_rate(samples, level=0.7, blocklength=2, theta_grid=80) > 0.0


def test_run_trial_produces_complete_row_and_positive_gap_fields():
    row = run_trial(
        r_bits=12,
        width=2,
        sigma=2.0,
        corr=0.9,
        far_samples=2_000,
        near_samples=800,
        theta_grid=120,
        seed=9,
    )
    assert set(row) == set(FIELDNAMES)
    assert row["r_bits"] == 12
    assert row["width"] == 2
    assert row["labels"] == math.comb(12, 2) * 4
    assert abs(row["label_log_count"] - label_log_count(12, 2)) < 1e-12
    assert row["tail_prob"] == math.exp(-row["label_log_count"])
    assert row["expected_tail_samples"] == row["far_samples"] * row["tail_prob"]
    assert row["saddle_threshold_level"] <= row["label_rate"] + 1e-12
    assert row["near_score_level_q05"] <= row["near_score_level_median"]
    assert 0.0 <= row["near_saddle_hit"] <= 1.0
    assert 0.0 <= row["near_normal_hit"] <= 1.0
    if row["empirical_threshold_level"] == row["empirical_threshold_level"]:
        assert 0.0 <= row["near_empirical_hit"] <= 1.0
    for value in row.values():
        if isinstance(value, float):
            assert value == value


if __name__ == "__main__":
    test_log_elementary_symmetric_matches_bruteforce()
    test_batch_log_elementary_symmetric_matches_scalar()
    test_full_width_sparse_subset_is_product_sign_score()
    test_empirical_rate_is_zero_below_mean_and_positive_above()
    test_run_trial_produces_complete_row_and_positive_gap_fields()
    print("sparse_subset_channel_stress tests passed")
