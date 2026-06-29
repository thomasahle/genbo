import math

import numpy as np

from coded_product_sign_panel_stress import (
    code_from_spec,
    empirical_far_threshold,
    enumerate_binary_linear_code,
    gf2_rank,
    normal_cdf,
    normal_ppf,
    random_code_query_levels,
    random_code_saddlepoint_summary,
    random_code_saddlepoint_summary_for_shape,
    reed_muller_generator,
    run_code_trial,
    shifted_score_moments,
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
    iid = code_from_spec("iid:8:4:7", seed=0)

    assert rm.name == "rm(1,3)"
    assert rm.signs.shape == (16, 8)
    assert full.signs.shape == (16, 4)
    assert random.signs.shape == (16, 8)
    assert iid.signs.shape == (16, 8)
    assert gf2_rank(random.generator) == 4
    assert iid.family == "iid"


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
    assert row["query_top1_score_level_q05"] <= row["query_top1_score_level_median"]
    assert row["reported_score_level_median"] <= row["oracle_score_level_median"] + 1e-12
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


def test_normal_ppf_inverts_cdf():
    for p in (0.001, 0.1, 0.5, 0.9, 0.999):
        z = normal_ppf(p)
        assert abs(normal_cdf(z) - p) < 1e-12


def test_shifted_score_moments_match_monte_carlo():
    mean = 1.3
    sigma = 2.0
    quad_mean, quad_var = shifted_score_moments(
        mean=mean,
        sigma=sigma,
        quadrature=80,
    )
    rng = np.random.default_rng(17)
    z = mean + sigma * rng.standard_normal(200_000)
    scores = z - np.logaddexp(z, -z) + math.log(2.0)
    assert abs(quad_mean - float(np.mean(scores))) < 0.01
    assert abs(quad_var - float(np.var(scores))) < 0.02


def test_random_code_query_levels_use_finite_extreme_correction():
    median, asymptotic = random_code_query_levels(
        r_bits=32,
        dimension=16,
        sigma=3.5,
    )
    assert 0.0 < median < asymptotic


def test_random_code_saddlepoint_improves_with_correlation():
    code = code_from_spec("rm:5:2", seed=0)
    low = random_code_saddlepoint_summary(
        code,
        local_m=2**19,
        c=2.0,
        sigma=3.5,
        corr=0.95,
        top_l=1,
        quadrature=48,
        theta_grid=600,
    )
    high = random_code_saddlepoint_summary(
        code,
        local_m=2**19,
        c=2.0,
        sigma=3.5,
        corr=0.99,
        top_l=1,
        quadrature=48,
        theta_grid=600,
    )
    assert low["dimension"] == 16
    assert low["r_bits"] == 32
    assert abs(low["label_exponent"] - 16 / 19) < 1e-12
    assert low["far_threshold_level"] < math.log(2.0)
    assert 0.0 <= low["median_normal_hit"] <= 1.0
    assert high["median_mean_gap"] > low["median_mean_gap"]
    assert high["median_normal_hit"] > low["median_normal_hit"]


def test_shape_saddlepoint_does_not_materialize_codebook():
    row = random_code_saddlepoint_summary_for_shape(
        name="shape(32,64)",
        family="shape",
        r_bits=64,
        dimension=32,
        labels=1 << 32,
        local_m=2**37,
        c=2.0,
        sigma=3.5,
        corr=0.95,
        top_l=1,
        quadrature=40,
        theta_grid=400,
    )
    assert row["r_bits"] == 64
    assert row["dimension"] == 32
    assert row["labels"] == 1 << 32
    assert 0.0 <= row["median_normal_hit"] <= 1.0
    assert row["median_decay_rate"] >= 0.0


if __name__ == "__main__":
    test_reed_muller_generator_has_expected_dimensions_and_distance()
    test_code_spec_builds_rm_full_and_random_controls()
    test_empirical_far_threshold_matches_requested_tail()
    test_run_code_trial_detects_query_list_and_oracle_hits()
    test_summarize_code_compares_near_target_label_exponent()
    test_normal_ppf_inverts_cdf()
    test_shifted_score_moments_match_monte_carlo()
    test_random_code_query_levels_use_finite_extreme_correction()
    test_random_code_saddlepoint_improves_with_correlation()
    test_shape_saddlepoint_does_not_materialize_codebook()
    print("coded_product_sign_panel_stress tests passed")
