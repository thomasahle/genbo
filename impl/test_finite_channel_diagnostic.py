import numpy as np

from finite_channel_diagnostic import (
    balance_softmax_offsets,
    diagnose_channel,
    h_eta_for_pairs,
    posterior_thresholds,
    stable_softmax,
)


def test_h_eta_matches_direct_formula():
    k_data = np.array([
        [0.80, 0.20],
        [0.50, 0.50],
        [0.20, 0.80],
        [0.10, 0.90],
    ])
    k_query = np.array([
        [0.70, 0.30],
        [0.25, 0.75],
    ])
    near = np.array([0, 2])
    eta = 0.5
    alpha = 0.25

    h, pi, tau = h_eta_for_pairs(k_data, k_query, near, eta=eta, alpha=alpha)
    pi2, ratios, tau2 = posterior_thresholds(k_data, eta)
    assert np.allclose(pi, pi2)
    assert np.allclose(tau, tau2)

    expected = []
    for row, p_idx in enumerate(near):
        total = 0.0
        for j in range(k_data.shape[1]):
            r_p = ratios[p_idx, j]
            total += (pi[j] ** alpha) * (k_query[row, j] ** (1 - alpha)) * max(
                r_p ** alpha - tau[j] ** alpha, 0.0
            )
        expected.append(total)
    assert np.allclose(h, expected)


def test_diagnostic_adds_guard_density():
    data = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 2.0],
    ])
    queries = np.array([
        [0.0, 0.1],
        [3.0, 3.0],
    ])
    near = np.array([0, 1])
    k_data = np.full((3, 2), 0.5)
    k_query = np.full((2, 2), 0.5)

    result = diagnose_channel(data, queries, near, k_data, k_query, c=2.0, r=0.2, eta=1 / 3, alpha=0.5)
    assert np.allclose(result.guard_density, [1 / 3, 0.0])
    assert np.allclose(result.score, result.guard_density + result.h_eta)


def test_softmax_balancing_reduces_column_mass_error():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(80, 5))
    before = stable_softmax(scores).mean(axis=0)
    offsets = balance_softmax_offsets(scores, max_iter=100)
    after = stable_softmax(scores + offsets).mean(axis=0)
    target = np.full(5, 0.2)
    assert np.max(np.abs(after - target)) < np.max(np.abs(before - target))
    assert np.max(np.abs(after - target)) < 2e-3


if __name__ == "__main__":
    test_h_eta_matches_direct_formula()
    test_diagnostic_adds_guard_density()
    test_softmax_balancing_reduces_column_mass_error()
    print("finite_channel_diagnostic tests passed")
