#!/usr/bin/env python3
"""Materialize coherent paper CSVs from the uniform all-dataset campaign.

Selection is deliberately two-level:
1. Keep only the latest accepted run for each compatible artifact family.
2. Pick the better of its mirrored forward/reverse occurrences per semantic
   plan, then take the Pareto envelope across families within a dataset.

This prevents the old failure mode where individual points from different
machine windows were stitched into one apparently jagged method curve.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rerun_all_frontiers import latest_accepted_samples, pareto


ROOT = Path("/home/thomas-ahle/genbo")
REPORT = ROOT / "sbann-rs/experiments/uniform_frontier_results.json"
PAPERDATA = ROOT / "paperdata"
MANIFEST = ROOT / "sbann-rs/experiments/uniform_frontier_manifest.csv"
SUMMARY = ROOT / "sbann-rs/experiments/uniform_frontier_summary.md"


def mirrored_best(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in samples:
        key = (sample["group"], sample["label"])
        old = best.get(key)
        if old is None or sample["qps"] > old["qps"]:
            best[key] = sample
    return list(best.values())


def write_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("recall", "qps"))
        for row in rows:
            writer.writerow((f"{row['recall']:.4f}", f"{row['qps']:.0f}"))


def main() -> None:
    report = json.loads(REPORT.read_text())
    coherent = latest_accepted_samples(report)
    plans = mirrored_best(coherent)
    datasets = sorted({sample["dataset"] for sample in plans})
    frontiers = {
        dataset: pareto([row for row in plans if row["dataset"] == dataset])
        for dataset in datasets
    }

    for dataset, rows in frontiers.items():
        write_curve(PAPERDATA / f"{dataset}_ours.csv", rows)

    # DEEP's two mechanisms are intentionally not joined in the plot.
    for family in ("walk", "cascade"):
        rows = pareto(
            [
                row
                for row in plans
                if row["dataset"] == "deep10m" and row["family"] == family
            ]
        )
        write_curve(PAPERDATA / f"deep10m_{family}_ours.csv", rows)

    run_by_index = {
        index: run for index, run in enumerate(report.get("runs", []))
    }
    fields = [
        "dataset",
        "family",
        "group",
        "label",
        "recall",
        "qps",
        "direction",
        "p",
        "floor",
        "k",
        "beam",
        "edges",
        "hops",
        "bestfirst",
        "walk_l",
        "rounds",
        "frontier",
        "run_index",
        "load1",
        "run_mirror_max",
        "run_finished",
    ]
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(
            plans,
            key=lambda item: (
                item["dataset"],
                item["family"],
                item["recall"],
                -item["qps"],
            ),
        ):
            run = run_by_index[row["run_index"]]
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fields},
                    "run_mirror_max": run.get("mirror", {}).get(
                        "max_qps_spread", ""
                    ),
                    "run_finished": run.get("finished", ""),
                }
            )

    lines = [
        "# Uniform all-dataset frontier campaign",
        "",
        "Full official query sets, one thread, best of five repetitions. Each "
        "artifact family receives one highest-work full-query warmup, then an "
        "exact forward/reverse plan in one loaded process. A run is accepted "
        "only when recall matches in both directions and every mirrored QPS "
        "pair differs by at most 5%. Curves use the latest accepted run per "
        "family; no points are stitched across windows within a family.",
        "",
        "## Materialized envelopes",
        "",
        "| dataset | points | recall range | families |",
        "|---|---:|---:|---|",
    ]
    for dataset, rows in frontiers.items():
        families = ", ".join(sorted({row["family"] for row in rows}))
        lines.append(
            f"| {dataset} | {len(rows)} | "
            f"{rows[0]['recall']:.4f}--{rows[-1]['recall']:.4f} | "
            f"{families} |"
        )
    lines.extend(
        [
            "",
            "DEEP's fixed-round walk and cascade are emitted as separate CSVs "
            "and are not visually connected. Its legacy 0.9987 point is absent: "
            "the current corrected-fp16/layout stack plateaus at reproduced "
            "recall 0.9980 for p=768--1024.",
            "",
        ]
    )
    SUMMARY.write_text("\n".join(lines))

    report["protocol"].update(
        {
            "warmup": "highest-work full official query pass before measured arms",
            "acceptance": "recall-identical mirrors and max QPS spread <=5%",
            "selection": "latest accepted run per family; best mirrored occurrence per plan; dataset Pareto",
        }
    )
    report["frontiers"] = frontiers
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
