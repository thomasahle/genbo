#!/usr/bin/env python3
"""Extract transferable fixed-round tuning laws from the DEEP sweeps.

The report deliberately separates exact accounting identities from empirical
relationships.  The former transfer by construction; the latter are checked on
the independently built 1M prefix before being proposed as calibration rules.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/thomas-ahle/genbo")
EXPERIMENTS = ROOT / "sbann-rs/experiments"


def load_10m() -> list[dict[str, Any]]:
    """Load and deduplicate the overlapping 500-query screen families."""
    unique: dict[tuple[int, int, int], dict[str, Any]] = {}
    pattern = str(EXPERIMENTS / "deep_fixed_walk*_screen.json")
    for name in sorted(glob.glob(pattern)):
        for row in json.loads(Path(name).read_text())["samples"]:
            if row["mode"] == "round":
                unique[row["rounds"], row["frontier"], row["walk_l"]] = row
    return list(unique.values())


def load_1m(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["samples"]


def geometric_fit(
    rows: list[dict[str, Any]], frontier: int = 12, width: int = 20
) -> dict[str, float]:
    """Fit recall(R) = ceiling - amplitude * contraction**(R-1).

    A grid over contraction leaves only a two-column linear least-squares fit,
    avoiding a scipy dependency in the reproducibility path.
    """
    by_round = {
        row["rounds"]: row["recall"]
        for row in rows
        if row["frontier"] == frontier and row["walk_l"] == width
    }
    rounds = np.array(sorted(by_round), dtype=np.float64)
    recall = np.array([by_round[int(rounds_i)] for rounds_i in rounds])
    best: tuple[float, float, float, float] | None = None
    for contraction in np.linspace(0.001, 0.999, 999):
        design = np.column_stack(
            [np.ones_like(rounds), -(contraction ** (rounds - 1))]
        )
        ceiling, amplitude = np.linalg.lstsq(design, recall, rcond=None)[0]
        if not (recall.max() <= ceiling <= 1.0 and amplitude >= 0.0):
            continue
        prediction = ceiling - amplitude * contraction ** (rounds - 1)
        rmse = float(np.sqrt(np.mean((prediction - recall) ** 2)))
        candidate = (rmse, float(ceiling), float(amplitude), float(contraction))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("no feasible geometric fit")
    rmse, ceiling, amplitude, contraction = best
    return {
        "frontier": frontier,
        "rerank_width": width,
        "round_min": int(rounds.min()),
        "round_max": int(rounds.max()),
        "ceiling": ceiling,
        "amplitude": amplitude,
        "contraction": contraction,
        "rmse": rmse,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entry_counts = {
        row["hops_per_query"] - (row["rounds"] - 1) * row["frontier"]
        for row in rows
    }
    if len(entry_counts) != 1:
        raise AssertionError(f"expansion identity failed: E={sorted(entry_counts)}")
    entries = entry_counts.pop()
    overlap = [
        (row["evals_per_query"] - entries)
        / (32.0 * row["hops_per_query"])
        for row in rows
        if row["rounds"] >= 3
    ]

    by_policy: dict[tuple[int, int], dict[int, float]] = {}
    for row in rows:
        by_policy.setdefault((row["rounds"], row["frontier"]), {})[
            row["walk_l"]
        ] = row["recall"]
    width_gain: dict[str, dict[str, float | int]] = {}
    for width in (12, 14, 20, 25):
        fractions = []
        for recalls in by_policy.values():
            if not {10, width, 32}.issubset(recalls):
                continue
            total = recalls[32] - recalls[10]
            if total > 0:
                fractions.append((recalls[width] - recalls[10]) / total)
        width_gain[str(width)] = {
            "median_fraction_of_W10_to_W32_gain": statistics.median(fractions),
            "policies": len(fractions),
        }

    return {
        "policies": len(by_policy),
        "samples": len(rows),
        "entries": entries,
        "expansion_identity": f"H = {entries} + (R - 1) B",
        "unique_neighbor_fraction_rounds_ge_3": {
            "median": statistics.median(overlap),
            "min": min(overlap),
            "max": max(overlap),
        },
        "rerank_width_gain": width_gain,
        "geometric_round_fit_B12_W20": geometric_fit(rows),
    }


def same_policy_transfer(
    one_million: list[dict[str, Any]], ten_million: list[dict[str, Any]]
) -> dict[str, Any]:
    def keyed(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict]:
        return {
            (row["rounds"], row["frontier"], row["walk_l"]): row
            for row in rows
        }

    small, large = keyed(one_million), keyed(ten_million)
    common = sorted(small.keys() & large.keys())
    recall_delta = [small[key]["recall"] - large[key]["recall"] for key in common]
    eval_ratio = [
        small[key]["evals_per_query"] / large[key]["evals_per_query"]
        for key in common
    ]
    return {
        "common_policies": len(common),
        "median_recall_1m_minus_10m": statistics.median(recall_delta),
        "recall_delta_range": [min(recall_delta), max(recall_delta)],
        "median_evals_1m_over_10m": statistics.median(eval_ratio),
        "interpretation": (
            "work accounting transfers closely, but recall does not; estimate the "
            "round contraction on a query pilot instead of scaling R from n alone"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deep1m",
        type=Path,
        default=EXPERIMENTS / "deep1m_fixed_walk_transfer.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=EXPERIMENTS / "tuning_laws_results.json",
    )
    args = parser.parse_args()

    ten_million = load_10m()
    one_million = load_1m(args.deep1m)
    report = {
        "datasets": {
            "DEEP-1M": summarize(one_million),
            "DEEP-10M": summarize(ten_million),
        },
        "same_policy_scale_transfer": same_policy_transfer(
            one_million, ten_million
        ),
        "proposed_calibration": {
            "frontier": (
                "start at B ~= 1.5 E; bracket only E and 2 E if the target lies "
                "between two round settings"
            ),
            "rounds": (
                "sweep the one-dimensional R ladder and fit "
                "recall(R)=r_inf-A*delta^(R-1)"
            ),
            "rerank_width": (
                "start at W=2k; test 1.4k when latency matters and raise W only "
                "if a score-containment pilot says the final ranker is binding"
            ),
            "score_work": "N_score ~= E + k_edge * nu * [E + (R - 1) B]",
            "path_choice": (
                "compare predicted scattered score bytes against sequential scan "
                "bytes; do not use the walk merely because it wins at low d"
            ),
        },
    }
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
