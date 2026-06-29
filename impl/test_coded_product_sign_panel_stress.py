import math

import numpy as np

from coded_product_sign_panel_stress import (
    code_from_spec,
    empirical_far_threshold,
    enumerate_binary_linear_code,
    gf2_rank,
    reed_muller_generator,
    run_code_trial,
    summarize_code,
)


def test_reed_muller_generator_has_expected_dimensions_and_distance():
    gen = reed_muller_generator(m=4, degree=2)
    assert gen.shape == (11, 16)
    assert gf2_rank(gen) == 11

    signs = enumerate_binary_linear_code(gen)
    assert signs.shape == (2048, 16)
    assert set(np.unique(signs)) == {-1, 1}

    weights = np.sum(signs == -1, axis=1)
    nonzero_weights = weights[weights > 0]
    assert int(np.min(nonzero_weights)) == 4


def test_code_spec_builds_rm_full_and_random_controls():
    rm = code_from_spec("rm:3:1", seed=0)
    full = code_from_spec("full:4", seed=0)
    random = code_from_spec("random:8:4:7", seed=0)

    assert rm.name == "rm(1,3)"
    assert rm.signs.shape == (16, 8)
    assert full.signs.shape == (16, 4)
    assert random.signs.shape == (16, 8)
    assert gf2_rank(random.generator) == 4


def test_empirical_far_threshold_matches_requested_tail():
    threshold, empirical_tail = empirical_far_threshold(
        r_bits=8,
        sigma=1.2,
        tail_prob=0.05,
        samples=30_000,
        seed=11,
    )
    assert threshold < 8 * math.log(2.0)
    assert 0.04 <= empirical_tail <= 0.06


def test_run_code_trial_detects_query_list_and_oracle_hits():
    code = code_from_spec("rm:3:1", seed=0)
    row = run_code_trial(
        code,
        local_m=512,
        c=2.0,
        sigma=2.0,
        corr=0.9,
        top_l=4,
        threshold_tail=0.02,
        threshold_samples=20_000,
        query_trials=80,
        seed=5,
    )

    assert row["code"] == "rm(1,3)"
    assert row["r_bits"] == 8
    assert row["dimension"] == 4
    assert row["labels"] == 16
    assert 0.0 <= row["query_top_l_mass_median"] <= 1.0
    assert 0.0 <= row["tilted_top_l_mass_median"] <= 1.0
    assert 0.0 <= row["near_hit_rate"] <= row["oracle_near_hit_rate"] <= 1.0
    assert row["far_load_over_budget"] > 0.0
    assert row["storage_per_point"] > 0.0
    for value in row.values():
        if isinstance(value, float):
            assert value == value


def test_summarize_code_compares_near_target_label_exponent():
    code = code_from_spec("rm:4:2", seed=0)
    summary = summarize_code(code, local_m=8192, c=2.0, top_l=8)
    assert summary["dimension"] == 11
    assert summary["labels"] == 2048
    assert abs(summary["label_exponent"] - 11 / 13) < 1e-12
    assert abs(summary["target_label_exponent"] - 6 / 7) < 1e-12


if __name__ == "__main__":
    test_reed_muller_generator_has_expected_dimensions_and_distance()
    test_code_spec_builds_rm_full_and_random_controls()
    test_empirical_far_threshold_matches_requested_tail()
    test_run_code_trial_detects_query_list_and_oracle_hits()
    test_summarize_code_compares_near_target_label_exponent()
    print("coded_product_sign_panel_stress tests passed")
