import math

from finite_channel_sweep import FIELDNAMES, default_b_count, parse_csv_list, run_trial


def test_parse_csv_list_and_default_b_count():
    assert parse_csv_list("10, 20,30", int) == [10, 20, 30]
    n = 256
    c = 2.0
    assert default_b_count(n, c, 1.0) == math.ceil(n ** (1.0 / (2.0 * c * c)))
    assert default_b_count(n, c, 3.0) == math.ceil(3.0 * n ** (1.0 / (2.0 * c * c)))


def test_run_trial_produces_complete_finite_row():
    row = run_trial(
        n=40,
        d=5,
        c=2.0,
        queries=4,
        panel_kind="gaussian",
        scale=2.0,
        seed=0,
        b_count=3,
        balance_iters=10,
        near_correlation=0.9,
    )
    assert set(row) == set(FIELDNAMES)
    assert row["B"] == 3
    assert row["panel"] == "gaussian"
    assert row["scale"] == 2.0
    assert abs(row["near_corr"] - 0.9) < 1e-12
    for key in FIELDNAMES:
        if key == "panel":
            continue
        assert row[key] == row[key]


def test_run_trial_rejects_nonpositive_scale():
    try:
        run_trial(
            n=20,
            d=4,
            c=2.0,
            queries=2,
            panel_kind="gaussian",
            scale=0.0,
            seed=0,
            b_count=2,
            balance_iters=1,
        )
    except ValueError as exc:
        assert "scale" in str(exc)
    else:
        raise AssertionError("nonpositive scale was accepted")


if __name__ == "__main__":
    test_parse_csv_list_and_default_b_count()
    test_run_trial_produces_complete_finite_row()
    test_run_trial_rejects_nonpositive_scale()
    print("finite_channel_sweep tests passed")
