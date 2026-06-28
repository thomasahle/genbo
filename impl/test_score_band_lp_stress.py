import numpy as np

from score_band_lp_stress import (
    _simplex_max_nonnegative,
    product_simplex_score_band_gap,
)


def test_simplex_solves_small_lp():
    objective = np.array([1.0, 1.0])
    lhs = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
    ])
    rhs = np.array([4.0, 6.0])
    value, solution = _simplex_max_nonnegative(objective, lhs, rhs)

    assert np.isclose(value, 10.0 / 3.0)
    assert np.allclose(solution, [8.0 / 3.0, 2.0 / 3.0])


def test_full_binary_cube_has_zero_product_simplex_gap():
    code = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    omega = np.ones(len(code))
    result = product_simplex_score_band_gap(code, omega)

    assert result["gap"] <= 1e-8


def test_missing_binary_corner_exposes_score_band_gap():
    code = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
    ])
    omega = np.ones(len(code))
    result = product_simplex_score_band_gap(code, omega)

    assert np.isclose(result["gap"], 1.0)
    assert result["max_chart"] == (1, 1)


if __name__ == "__main__":
    test_simplex_solves_small_lp()
    test_full_binary_cube_has_zero_product_simplex_gap()
    test_missing_binary_corner_exposes_score_band_gap()
    print("score_band_lp_stress tests passed")
