import math

from finite_channel_threshold_stress import FIELDNAMES, run_trial


def test_threshold_stress_produces_complete_row():
    row = run_trial(
        n=60,
        d=5,
        c=2.0,
        queries=4,
        panel_kind="whitened_product_sign",
        scale=2.0,
        seed=0,
        b_count=8,
        ref_samples=80,
        balance_iters=5,
        near_correlation=0.9,
    )
    assert set(row) == set(FIELDNAMES)
    assert row["B"] == 8
    assert row["ref_samples"] == 80
    assert abs(row["near_corr"] - 0.9) < 1e-12
    assert row["tau_ref_emp_logerr_q50"] <= row["tau_ref_emp_logerr_q90"] + 1e-12
    for key, value in row.items():
        if key == "panel":
            continue
        assert value == value
        assert math.isfinite(value)


def test_threshold_stress_rejects_empty_reference():
    try:
        run_trial(
            n=20,
            d=4,
            c=2.0,
            queries=2,
            panel_kind="product_sign",
            scale=2.0,
            seed=0,
            b_count=4,
            ref_samples=0,
            balance_iters=1,
        )
    except ValueError as exc:
        assert "ref_samples" in str(exc)
    else:
        raise AssertionError("empty reference sample was accepted")


if __name__ == "__main__":
    test_threshold_stress_produces_complete_row()
    test_threshold_stress_rejects_empty_reference()
    print("finite_channel_threshold_stress tests passed")
