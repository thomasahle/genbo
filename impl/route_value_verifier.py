"""Finite route-value verifier for the budget-thinned aggregate target.

The routines here implement the finite objects used in paper.tex:

* saturated value and its route-cut min-cut formula;
* least superharmonic majorants for antichain charge certificates;
* the top-down route law induced by a superharmonic potential; and
* water-filled activation allocations.

The implementation is intentionally small and exact enough for toy stress
tests.  It is not a fast large-instance optimizer.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Hashable, Mapping, Sequence

import numpy as np

Node = Hashable
ChildList = Mapping[Node, Sequence[tuple[Node, float]]]


def _postorder(children: ChildList, root: Node) -> list[Node]:
    order: list[Node] = []
    visiting: set[Node] = set()
    visited: set[Node] = set()

    def visit(node: Node) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ValueError("route graph must be a tree or DAG without cycles")
        visiting.add(node)
        for child, _ in children.get(node, ()):
            visit(child)
        visiting.remove(node)
        visited.add(node)
        order.append(node)

    visit(root)
    return order


def _validate_probabilities(children: ChildList) -> None:
    for node, out in children.items():
        for child, gamma in out:
            if gamma < 0.0 or gamma > 1.0:
                raise ValueError(f"edge {node!r}->{child!r} has probability {gamma}")


def saturated_values(children: ChildList, *, root: Node) -> dict[Node, float]:
    """Return the saturated Bellman value U at every reachable node."""
    _validate_probabilities(children)
    values: dict[Node, float] = {}
    for node in _postorder(children, root):
        out = children.get(node, ())
        if len(out) == 0:
            values[node] = 1.0
        else:
            values[node] = min(1.0, sum(gamma * values[child] for child, gamma in out))
    return values


def harmonic_values(children: ChildList, *, root: Node) -> dict[Node, float]:
    """Return the harmonic prefix-energy value W at every reachable node."""
    _validate_probabilities(children)
    values: dict[Node, float] = {}
    for node in _postorder(children, root):
        out = children.get(node, ())
        if len(out) == 0:
            values[node] = 1.0
        else:
            conductance = sum(gamma * values[child] for child, gamma in out)
            values[node] = conductance / (1.0 + conductance)
    return values


def prefix_success_masses(children: ChildList, *, root: Node) -> dict[Node, float]:
    """Return A(u), the product of edge success probabilities to each prefix."""
    _validate_probabilities(children)
    masses: dict[Node, float] = {root: 1.0}
    stack = [root]
    while stack:
        node = stack.pop()
        for child, gamma in children.get(node, ()):
            if child in masses:
                raise ValueError("prefix masses require a tree, not a shared DAG")
            masses[child] = masses[node] * gamma
            stack.append(child)
    return masses


def all_route_cut_masses(children: ChildList, *, root: Node) -> list[float]:
    """Enumerate all route-cut masses for a small finite tree.

    This is exponential and intended only for regression tests or tiny
    diagnostics.
    """
    masses = prefix_success_masses(children, root=root)

    def subtree_cut_masses(node: Node) -> list[float]:
        out = children.get(node, ())
        if len(out) == 0:
            return [masses[node]]
        child_options = [subtree_cut_masses(child) for child, _ in out]
        below = [
            sum(choice)
            for choice in itertools.product(*child_options)
        ]
        return [masses[node], *below]

    return subtree_cut_masses(root)


def least_superharmonic_majorant(
    children: ChildList,
    charge: Mapping[Node, float],
    *,
    root: Node,
) -> dict[Node, float]:
    """Compute the least potential Phi with charge <= Phi and sum child Phi <= Phi."""
    if any(value < 0.0 for value in charge.values()):
        raise ValueError("charges must be nonnegative")
    phi: dict[Node, float] = {}
    for node in _postorder(children, root):
        child_sum = sum(phi[child] for child, _ in children.get(node, ()))
        phi[node] = max(float(charge.get(node, 0.0)), child_sum)
    return phi


def superharmonic_route_law(
    children: ChildList,
    phi: Mapping[Node, float],
    *,
    root: Node,
    tol: float = 1e-12,
) -> dict[Node, float]:
    """Construct a leaf law whose prefix masses dominate a superharmonic Phi."""
    if phi.get(root, 0.0) > 1.0 + tol:
        raise ValueError("root potential must be at most one")
    if any(value < -tol for value in phi.values()):
        raise ValueError("potential must be nonnegative")

    leaf_law: dict[Node, float] = {}
    stack: list[tuple[Node, float]] = [(root, 1.0)]
    while stack:
        node, mass = stack.pop()
        out = list(children.get(node, ()))
        if len(out) == 0:
            leaf_law[node] = leaf_law.get(node, 0.0) + mass
            continue

        mandatory = [max(0.0, float(phi.get(child, 0.0))) for child, _ in out]
        mandatory_sum = sum(mandatory)
        if mandatory_sum > float(phi.get(node, 0.0)) + tol:
            raise ValueError("potential is not superharmonic")
        if mandatory_sum > mass + tol:
            raise ValueError("incoming route mass is too small for child demands")

        child_masses = mandatory[:]
        child_masses[0] += max(0.0, mass - mandatory_sum)
        for (child, _), child_mass in zip(out, child_masses):
            stack.append((child, child_mass))

    return leaf_law


def prefix_masses_from_leaf_law(
    children: ChildList,
    leaf_law: Mapping[Node, float],
    *,
    root: Node,
) -> dict[Node, float]:
    """Return prefix probabilities induced by a leaf distribution."""
    prefix: dict[Node, float] = {}

    def suffix_mass(node: Node) -> float:
        out = children.get(node, ())
        if len(out) == 0:
            total = float(leaf_law.get(node, 0.0))
        else:
            total = sum(suffix_mass(child) for child, _ in out)
        prefix[node] = total
        return total

    suffix_mass(root)
    return prefix


def waterfilled_allocation(
    occupation: Sequence[float],
    charge: Sequence[float],
    *,
    tol: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Return the optimal allocation and defect for sum ell log_+(a/y)."""
    ell = np.asarray(occupation, dtype=float)
    a = np.asarray(charge, dtype=float)
    if ell.shape != a.shape:
        raise ValueError("occupation and charge must have the same shape")
    if np.any(ell < -tol) or np.any(a < -tol):
        raise ValueError("occupation and charge must be nonnegative")

    ell = np.maximum(ell, 0.0)
    a = np.maximum(a, 0.0)
    active = (ell > tol) & (a > tol)
    allocation = np.zeros_like(a)
    if not np.any(active):
        return allocation, 0.0

    active_charge_sum = float(np.sum(a[active]))
    if active_charge_sum <= 1.0 + tol:
        allocation[active] = a[active]
        return allocation, 0.0

    lo = 0.0
    hi = max(float(np.max(a[active] / ell[active])), 1.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        mass = float(np.sum(np.minimum(a[active], mid * ell[active])))
        if mass < 1.0:
            lo = mid
        else:
            hi = mid
    t = hi
    allocation[active] = np.minimum(a[active], t * ell[active])

    ratio = a[active] / allocation[active]
    defect = float(np.sum(ell[active] * np.maximum(0.0, np.log(ratio))))
    return allocation, defect


def _demo() -> None:
    children = {
        "r": [("a", 0.4), ("b", 0.7)],
        "a": [("a0", 0.5), ("a1", 0.2)],
        "b": [("b0", 0.1)],
    }
    values = saturated_values(children, root="r")
    cut_masses = all_route_cut_masses(children, root="r")
    print(f"saturated root value: {values['r']:.6g}")
    print(f"minimum route-cut mass: {min(cut_masses):.6g}")

    charge = {"r": 0.1, "a": 0.2, "b": 0.25, "a0": 0.4, "a1": 0.3}
    phi = least_superharmonic_majorant(children, charge, root="r")
    print(f"least superharmonic root charge: {phi['r']:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run a tiny deterministic diagnostic")
    args = parser.parse_args()
    if args.demo:
        _demo()


if __name__ == "__main__":
    main()
