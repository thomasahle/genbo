from smoothed_scoremax_ivf_stress import (
    CentroidRouter,
    FIELDNAMES,
    run_trial,
)
from smoothed_scoremax_stress import promised_spherical_instance, rho_for_c

import numpy as np


def test_centroid_router_finds_own_center_bucket():
    rng = np.random.default_rng(0)
    x, _q, _p, p_index = promised_spherical_instance(
        n_far=20,
        d=8,
        c=2.0,
        far_corr_max=-0.05,
        rng=rng,
    )
    router = CentroidRouter(centers=len(x), mode="sample", lloyd_iters=0, seed=1)
    router.build(x)
    candidates = router.query_candidates(x[p_index], probes=len(x))
    assert p_index in candidates
    assert router.bucket_rank(p_index, x[p_index]) >= 1


def test_ivf_run_trial_reports_complete_row():
    row = run_trial(
        n=160,
        d=12,
        c=2.0,
        far_corr_max=-0.05,
        t_scale=64.0,
        router="lloyd",
        center_exponent=1.0 - rho_for_c(2.0),
        center_mult=1.0,
        lloyd_iters=2,
        probes=4,
        query_trials=10,
        seed=3,
    )
    assert set(row) == set(FIELDNAMES)
    assert row["centers"] > 0
    assert row["bucket_mean"] > 0.0
    assert 0.0 <= row["ivf_scoremax_rate"] <= 1.0
    assert row["ivf_candidate_mean"] >= 0.0
    assert row["ivf_best_loss_over_margin_q90"] >= 0.0
    for value in row.values():
        if isinstance(value, float):
            assert value == value


if __name__ == "__main__":
    test_centroid_router_finds_own_center_bucket()
    test_ivf_run_trial_reports_complete_row()
    print("smoothed_scoremax_ivf_stress tests passed")
