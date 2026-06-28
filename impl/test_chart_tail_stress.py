import numpy as np

from chart_tail_stress import (
    centered_overlap_matrix,
    codeword_scores,
    make_balanced_code,
    run_trial,
)


def test_balanced_code_has_nearly_equal_coordinate_counts():
    code = make_balanced_code(n_words=22, blocks=7, alphabet=5, seed=0)
    assert code.shape == (22, 7)
    for i in range(code.shape[1]):
        counts = np.bincount(code[:, i], minlength=5)
        assert counts.max() - counts.min() <= 1


def test_centered_overlap_matches_direct_formula():
    code = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ])
    overlap = centered_overlap_matrix(code)
    direct = np.array([
        [3, 2, 0],
        [2, 3, 1],
        [0, 1, 3],
    ], dtype=float) - 3 / 2
    assert np.allclose(overlap, direct)
    assert np.allclose(overlap, overlap.T)


def test_codeword_scores_are_additive_over_blocks():
    code = np.array([
        [0, 1, 0],
        [1, 1, 0],
    ])
    symbol_scores = np.array([
        [1.0, 2.0],
        [3.0, 5.0],
        [7.0, 11.0],
    ])
    assert np.allclose(codeword_scores(code, symbol_scores), [13.0, 14.0])


def test_run_trial_is_deterministic_and_clips_the_singleton_cost():
    row1 = run_trial(n_words=40, blocks=10, alphabet=4, seed=3, price_cap=5.0)
    row2 = run_trial(n_words=40, blocks=10, alphabet=4, seed=3, price_cap=5.0)
    assert row1 == row2
    assert row1["omega_max"] >= row1["price_cap"]
    assert row1["clipped_singleton_cost"] <= row1["uniform_singleton_cost"]
    assert 0.0 <= row1["full_diag_share"] <= 1.0
    assert 0.0 <= row1["clipped_diag_share"] <= 1.0


if __name__ == "__main__":
    test_balanced_code_has_nearly_equal_coordinate_counts()
    test_centered_overlap_matches_direct_formula()
    test_codeword_scores_are_additive_over_blocks()
    test_run_trial_is_deterministic_and_clips_the_singleton_cost()
    print("chart_tail_stress tests passed")
