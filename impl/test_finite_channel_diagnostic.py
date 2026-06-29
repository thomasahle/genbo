import numpy as np

from finite_channel_diagnostic import (
    PANEL_KINDS,
    balance_softmax_offsets,
    diagnose_channel,
    h_eta_for_pairs,
    make_panel,
    posterior_margin_certificate,
    posterior_thresholds,
    stable_softmax,
    synthetic_near_pairs,
    tilted_surplus_statistics,
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

    affinity, margin_mass, margin_bound = posterior_margin_certificate(
        k_data, k_query, near, pi, tau, alpha=alpha, margin_delta=0.1)
    assert np.all(affinity > 0)
    assert np.all((0 <= margin_mass) & (margin_mass <= 1))
    assert np.all(margin_bound <= h + 1e-12)

    affinity2, soft_surplus, tilted_top_mass, top_fraction, topk = tilted_surplus_statistics(
        k_data, k_query, near, pi, tau, alpha=alpha)
    assert np.allclose(affinity2, affinity)
    assert np.allclose(h, affinity * soft_surplus)
    assert np.all((0 <= soft_surplus) & (soft_surplus <= 1))
    assert np.all((0 <= tilted_top_mass) & (tilted_top_mass <= 1))
    assert np.all((0 <= top_fraction) & (top_fraction <= 1))
    assert np.allclose(topk[2][h > 0], 1.0)
    assert np.all(topk[2] >= top_fraction - 1e-12)


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
    assert np.all(result.margin_bound <= result.h_eta + 1e-12)
    assert np.allclose(result.h_eta, result.affinity * result.soft_surplus)
    assert np.all(result.top2_contribution_fraction >= result.top_contribution_fraction - 1e-12)
    assert np.all(result.top4_contribution_fraction >= result.top2_contribution_fraction - 1e-12)
    assert np.all(result.top8_contribution_fraction >= result.top4_contribution_fraction - 1e-12)


def test_softmax_balancing_reduces_column_mass_error():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(80, 5))
    before = stable_softmax(scores).mean(axis=0)
    offsets = balance_softmax_offsets(scores, max_iter=100)
    after = stable_softmax(scores + offsets).mean(axis=0)
    target = np.full(5, 0.2)
    assert np.max(np.abs(after - target)) < np.max(np.abs(before - target))
    assert np.max(np.abs(after - target)) < 2e-3


def test_panel_variants_have_expected_shape_and_unit_adaptive_rows():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(30, 4))
    for kind in PANEL_KINDS:
        panel = make_panel(4, 6, kind=kind, seed=2, data=data)
        assert panel.shape == (6, 4)
        assert np.all(np.isfinite(panel))
        if kind in {"cross_polytope", "pca", "landmark"}:
            assert np.allclose(np.linalg.norm(panel, axis=1), 1.0)


def test_synthetic_near_pairs_honors_requested_correlation():
    data, queries, near, r = synthetic_near_pairs(20, 6, 2.0, 5, 3, near_correlation=0.95)
    dots = np.sum(data[near] * queries, axis=1)
    assert np.allclose(dots, 0.95)
    assert np.isclose(r, np.sqrt(0.1))


if __name__ == "__main__":
    test_h_eta_matches_direct_formula()
    test_diagnostic_adds_guard_density()
    test_softmax_balancing_reduces_column_mass_error()
    test_panel_variants_have_expected_shape_and_unit_adaptive_rows()
    test_synthetic_near_pairs_honors_requested_correlation()
    print("finite_channel_diagnostic tests passed")
