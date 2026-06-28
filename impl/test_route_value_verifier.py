import math

import numpy as np

from route_value_verifier import (
    all_route_cut_masses,
    least_superharmonic_majorant,
    prefix_masses_from_leaf_law,
    saturated_values,
    superharmonic_route_law,
    waterfilled_allocation,
)


def test_saturated_value_matches_explicit_min_cut():
    children = {
        "r": [("a", 0.4), ("b", 0.7)],
        "a": [("a0", 0.5), ("a1", 0.2)],
        "b": [("b0", 0.1)],
    }

    values = saturated_values(children, root="r")
    cut_masses = all_route_cut_masses(children, root="r")

    assert np.isclose(values["a"], 0.7)
    assert np.isclose(values["b"], 0.1)
    assert np.isclose(values["r"], 0.35)
    assert np.isclose(min(cut_masses), values["r"])


def test_superharmonic_majorant_is_exact_antichain_obstruction():
    children = {
        "r": [("a", 1.0), ("b", 1.0)],
        "a": [("a0", 1.0), ("a1", 1.0)],
    }
    charge = {
        "r": 0.1,
        "a": 0.2,
        "b": 0.25,
        "a0": 0.4,
        "a1": 0.3,
    }

    phi = least_superharmonic_majorant(children, charge, root="r")

    assert np.isclose(phi["a"], 0.7)
    assert np.isclose(phi["r"], 0.95)


def test_superharmonic_route_law_dominates_prefix_demands():
    children = {
        "r": [("a", 1.0), ("b", 1.0)],
        "a": [("a0", 1.0), ("a1", 1.0)],
    }
    phi = {
        "r": 0.8,
        "a": 0.3,
        "b": 0.2,
        "a0": 0.1,
        "a1": 0.1,
    }

    leaf_law = superharmonic_route_law(children, phi, root="r")
    prefix = prefix_masses_from_leaf_law(children, leaf_law, root="r")

    assert np.isclose(sum(leaf_law.values()), 1.0)
    for node, demand in phi.items():
        assert prefix[node] + 1e-12 >= demand


def test_waterfilled_allocation_has_zero_defect_when_charge_is_affordable():
    occupation = np.array([0.4, 0.3, 0.0, 0.2])
    charge = np.array([0.2, 0.1, 7.0, 0.3])

    allocation, defect = waterfilled_allocation(occupation, charge)

    assert np.isclose(defect, 0.0)
    assert np.allclose(allocation[[0, 1, 3]], charge[[0, 1, 3]])
    assert allocation[2] == 0.0


def test_waterfilled_allocation_matches_scalar_threshold_formula():
    occupation = np.array([0.5, 0.25, 0.25])
    charge = np.array([2.0, 1.0, 0.25])

    allocation, defect = waterfilled_allocation(occupation, charge)
    expected_t = 1.0
    expected = np.minimum(charge, expected_t * occupation)
    expected_defect = np.sum(occupation * np.maximum(0.0, np.log(charge / expected)))

    assert np.allclose(allocation, expected)
    assert np.isclose(defect, expected_defect)
    assert np.isclose(allocation.sum(), 1.0)


def test_transition_density_bound_controls_waterfilled_defect():
    occupation = np.array([0.4, 0.35, 0.25])
    depth_bound = occupation.sum()
    C = 1.7
    transition_aware_charge = (C / depth_bound) * occupation

    _, defect = waterfilled_allocation(occupation, transition_aware_charge)

    assert defect <= depth_bound * math.log(C) + 1e-12


if __name__ == "__main__":
    test_saturated_value_matches_explicit_min_cut()
    test_superharmonic_majorant_is_exact_antichain_obstruction()
    test_superharmonic_route_law_dominates_prefix_demands()
    test_waterfilled_allocation_has_zero_defect_when_charge_is_affordable()
    test_waterfilled_allocation_matches_scalar_threshold_formula()
    test_transition_density_bound_controls_waterfilled_defect()
    print("route_value_verifier tests passed")
