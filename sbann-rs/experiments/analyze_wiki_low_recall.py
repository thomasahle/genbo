#!/usr/bin/env python3
"""Select Wikipedia low-recall confirmation candidates from the one-load screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


ROOT = Path("/home/thomas-ahle/genbo")
DATA = Path("/home/thomas-ahle/big-ann-data")
DEFAULT_INPUT = ROOT / "sbann-rs/experiments/wiki_low_recall_law_screen.json"
DEFAULT_OUTPUT = ROOT / "sbann-rs/experiments/wiki_low_recall_law_analysis.json"
DEFAULT_LOG = DATA / "wiki_low_recall_law_screen.log"
OURS = ROOT / "paperdata/wiki35m_ours.csv"
ROAR = ROOT / "paperdata/wiki35m_roar_full.csv"
TARGETS = (0.9326, 0.94, 0.9492, 0.9580, 0.9630)
RESULT_RE = re.compile(
    r"p=\s*(?P<p>\d+)\s+t=\s*(?P<tm>\d+).*?"
    r"K=(?P<width>\d+):\s+recall@10=(?P<recall>[0-9.]+)\s+"
    r"QPS=(?P<qps>[0-9.]+).*?"
    r"\[M=(?P<beam>\d+)\s+ke=(?P<edges>\d+)\s+tf=(?P<floor>\d+)\s+"
    r"pk=(?P<portal_keep>\d+)\]"
)


def read_curve(path: Path) -> list[tuple[float, float]]:
    with path.open() as handle:
        return [
            (float(row["recall"]), float(row["qps"]))
            for row in csv.DictReader(
                line for line in handle if not line.lstrip().startswith("#")
            )
        ]


def log_interpolate(curve: list[tuple[float, float]], recall: float) -> float:
    curve = sorted(curve)
    if recall <= curve[0][0]:
        return curve[0][1]
    if recall >= curve[-1][0]:
        return curve[-1][1]
    for (r0, q0), (r1, q1) in zip(curve, curve[1:]):
        if r0 <= recall <= r1:
            weight = (recall - r0) / (r1 - r0)
            return math.exp(math.log(q0) * (1.0 - weight) + math.log(q1) * weight)
    raise AssertionError("unreachable")


def dominates(a: dict, b: dict) -> bool:
    return (
        a["recall"] >= b["recall"]
        and a["qps"] >= b["qps"]
        and (a["recall"] > b["recall"] or a["qps"] > b["qps"])
    )


def config(sample: dict) -> dict:
    result = {
        key: sample[key]
        for key in ("beam", "edges", "floor", "p", "width", "portal_keep")
    }
    return result


def recover_samples(log: Path) -> list[dict]:
    samples = []
    for line in log.read_text(errors="replace").splitlines():
        match = RESULT_RE.search(line)
        if not match:
            continue
        samples.append(
            {
                key: (
                    float(value)
                    if key in {"recall", "qps"}
                    else int(value or 0)
                )
                for key, value in match.groupdict().items()
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    samples = payload.get("samples", [])
    if not samples:
        samples = recover_samples(args.log)
        if not samples:
            raise SystemExit(f"no samples in {args.input} or {args.log}")
        payload["samples"] = samples
        payload["recovered_from_log"] = str(args.log)
        args.input.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    frontier = [
        sample
        for sample in samples
        if not any(dominates(other, sample) for other in samples if other is not sample)
    ]
    frontier.sort(key=lambda sample: (sample["recall"], -sample["qps"]))

    ours = read_curve(OURS)
    roar = read_curve(ROAR)
    anchors = []
    confirmation_configs: list[dict] = []
    for target in TARGETS:
        eligible = [
            sample
            for sample in samples
            if target <= sample["recall"] <= target + 0.012
        ]
        eligible.sort(key=lambda sample: (-sample["qps"], sample["recall"]))
        chosen = eligible[: args.top]
        for sample in chosen[:2]:
            candidate = config(sample)
            if candidate not in confirmation_configs:
                confirmation_configs.append(candidate)
        anchors.append(
            {
                "target_recall": target,
                "current_qps": log_interpolate(ours, target),
                "roar_qps": log_interpolate(roar, target),
                "candidates": chosen,
            }
        )

    result = {
        "source": str(args.input),
        "sample_count": len(samples),
        "pareto_count": len(frontier),
        "frontier": frontier,
        "anchors": anchors,
        "confirmation_configs": confirmation_configs,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"{len(samples)} samples -> {len(frontier)} Pareto points; "
        f"{len(confirmation_configs)} confirmation configurations"
    )
    for anchor in anchors:
        best = anchor["candidates"][:1]
        if best:
            row = best[0]
            print(
                f"r>={anchor['target_recall']:.4f}: "
                f"{row['recall']:.4f}@{row['qps']:.0f} "
                f"M{row['beam']} ke{row['edges']} p{row['p']} "
                f"tf{row['floor']} W{row['width']} "
                f"(current~{anchor['current_qps']:.0f}, "
                f"Roar~{anchor['roar_qps']:.0f})"
            )


if __name__ == "__main__":
    main()
