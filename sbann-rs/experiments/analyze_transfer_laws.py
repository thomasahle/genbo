#!/usr/bin/env python3
"""Consolidate the cross-dataset tuning-law confirmations.

This is deliberately descriptive rather than an automatic plot editor: it
records every duplicated ABBA arm, the robust median, and the benchmark-style
best.  Plot promotion remains an explicit review decision.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/thomas-ahle/genbo")
DEFAULT_INPUTS = (
    "law_transfer_confirm.json",
    "law_transfer_cohere10m_confirm.json",
    "law_transfer_t2i10m_confirm2.json",
    "law_transfer_webvid_confirm.json",
    "law_transfer_webvid_buyback.json",
    "law_transfer_t2i100m_confirm.json",
    "law_transfer_t2i100m_frontier.json",
    "law_transfer_wiki35m_confirm.json",
)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="checkpoint JSON files; defaults to the confirmation artifacts",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "sbann-rs/experiments/law_transfer_results.json",
    )
    args = parser.parse_args()
    base = ROOT / "sbann-rs/experiments"
    inputs = args.inputs or [base / name for name in DEFAULT_INPUTS]
    inputs = [path for path in inputs if path.is_file()]

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    runs: list[dict[str, Any]] = []
    for path in inputs:
        report = json.loads(path.read_text())
        offset = len(runs)
        for run in report.get("runs", []):
            runs.append({**run, "source": str(path.relative_to(ROOT))})
        for sample in report.get("samples", []):
            run = report["runs"][sample["run_index"]]
            key = (
                sample["dataset"],
                sample["label"],
                sample["p"],
                run["floor"],
                run["hops"],
                run["beam"],
                run["edges"],
                run["graph"],
                sample["cascade_k"],
            )
            grouped[key].append(
                {
                    **sample,
                    "run_index": sample["run_index"] + offset,
                    "source": str(path.relative_to(ROOT)),
                }
            )

    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        dataset, label, p, floor, hops, beam, edges, graph, cascade_k = key
        recalls = [row["recall"] for row in rows]
        qps = [row["qps"] for row in rows]
        summaries.append(
            {
                "dataset": dataset,
                "label": label,
                "p": p,
                "floor": floor,
                "hops": hops,
                "beam": beam,
                "edges": edges,
                "graph": graph,
                "cascade_k": cascade_k,
                "observations": len(rows),
                "recall": summarize(recalls),
                "qps": summarize(qps),
                "arms": [
                    {
                        "recall": row["recall"],
                        "qps": row["qps"],
                        "load1": row["load1"],
                        "source": row["source"],
                    }
                    for row in rows
                ],
            }
        )

    result = {
        "protocol": (
            "same-process mirrored KLIST arms; each arm reports best-of-SBANN_REPS; "
            "median is the robust comparison, max is the paper's warm-best convention"
        ),
        "inputs": [str(path.relative_to(ROOT)) for path in inputs],
        "runs": runs,
        "summaries": summaries,
    }
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)

    for row in summaries:
        recall = row["recall"]
        qps = row["qps"]
        print(
            f"{row['dataset']:10s} {row['label']:18s} K={row['cascade_k']:3d} "
            f"r={recall['median']:.4f} "
            f"qps={qps['median']:.0f} [{qps['min']:.0f},{qps['max']:.0f}] "
            f"graph={Path(row['graph']).name if row['graph'] else '-'}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
