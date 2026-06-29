from smoothed_scoremax_centroid_decoder_stress import (
    ExactCenterDecoder,
    FIELDNAMES,
    LSHCenterDecoder,
    PivotGraphCenterDecoder,
    RPCenterDecoder,
    run_trial,
)
from smoothed_scoremax_ivf_stress import _normalize_rows
from smoothed_scoremax_stress import rho_for_c

import numpy as np


def test_exact_center_decoder_returns_true_order():
    centers = _normalize_rows(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.8, 0.6, 0.0],
            ]
        )
    )
    ids, stats = ExactCenterDecoder(centers).query_top(
        np.array([1.0, 0.1, 0.0]), top=2)
    assert list(ids) == [0, 2]
    assert stats.work == len(centers)
    assert stats.leaf_candidates == len(centers)


def test_rp_center_decoder_full_beam_matches_exact():
    rng = np.random.default_rng(0)
    centers = _normalize_rows(rng.standard_normal((32, 6)))
    query = centers[7] + 0.01 * rng.standard_normal(6)
    exact, _stats = ExactCenterDecoder(centers).query_top(query, top=3)
    decoded, stats = RPCenterDecoder(
        centers,
        trees=1,
        fanout=4,
        leaf_size=1,
        beam=64,
        seed=1,
    ).query_top(query, top=3)
    assert list(decoded) == list(exact)
    assert stats.leaf_candidates == len(centers)


def test_lsh_center_decoder_exhaustive_one_bit_matches_exact():
    rng = np.random.default_rng(1)
    centers = _normalize_rows(rng.standard_normal((24, 5)))
    query = centers[3] + 0.02 * rng.standard_normal(5)
    exact, _stats = ExactCenterDecoder(centers).query_top(query, top=4)
    decoded, stats = LSHCenterDecoder(
        centers,
        tables=1,
        bits=1,
        hash_probes=2,
        seed=2,
    ).query_top(query, top=4)
    assert list(decoded) == list(exact)
    assert stats.leaf_candidates == len(centers)


def test_pivot_graph_decoder_exhaustive_matches_exact():
    rng = np.random.default_rng(2)
    centers = _normalize_rows(rng.standard_normal((30, 5)))
    query = centers[11] + 0.01 * rng.standard_normal(5)
    exact, _stats = ExactCenterDecoder(centers).query_top(query, top=5)
    decoded, stats = PivotGraphCenterDecoder(
        centers,
        degree=4,
        pivots=len(centers),
        entries=len(centers),
        ef=len(centers),
        seed=3,
    ).query_top(query, top=5)
    assert list(decoded) == list(exact)
    assert stats.leaf_candidates == len(centers)


def test_centroid_decoder_run_trial_reports_complete_row():
    row = run_trial(
        n=180,
        d=12,
        c=2.0,
        far_corr_max=-0.05,
        t_scale=64.0,
        router="lloyd",
        center_exponent=1.0 - rho_for_c(2.0),
        center_mult=1.0,
        lloyd_iters=2,
        decoder="rptree",
        center_trees=2,
        center_fanout=4,
        center_leaf_size=4,
        center_beam=2,
        center_graph_degree=8,
        center_graph_pivots=12,
        center_graph_entries=4,
        center_graph_ef=24,
        probes=2,
        query_trials=8,
        seed=5,
    )
    assert set(row) == set(FIELDNAMES)
    assert row["centers"] > 0
    assert row["center_work_mean"] > 0.0
    assert row["center_leaf_candidate_mean"] > 0.0
    assert 0.0 <= row["center_top1_rate"] <= 1.0
    assert 0.0 <= row["decoder_scoremax_rate"] <= 1.0
    assert row["decoder_candidate_mean"] >= 0.0
    for value in row.values():
        if isinstance(value, float):
            assert value == value


if __name__ == "__main__":
    test_exact_center_decoder_returns_true_order()
    test_rp_center_decoder_full_beam_matches_exact()
    test_lsh_center_decoder_exhaustive_one_bit_matches_exact()
    test_pivot_graph_decoder_exhaustive_matches_exact()
    test_centroid_decoder_run_trial_reports_complete_row()
    print("smoothed_scoremax_centroid_decoder_stress tests passed")
