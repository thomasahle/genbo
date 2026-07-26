#!/usr/bin/env python3
"""Checkpointed DEEP sweep for bounded graph-refinement policies.

The adaptive reference uses the existing DiskANN-style best-first termination.
The capped arm adds a per-query expansion ceiling, while the round arm expands a
fixed-width synchronous frontier for a fixed number of layers. The DEEP-1M mode
is a scale-transfer check: it uses the same query set and graph degree, but a
separately built index, portal sidecars, and exact-L2 ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from abba_bench import choose_core


ROOT = Path("/home/thomas-ahle/genbo")
DATA = Path("/home/thomas-ahle/big-ann-data/deep10m")
DATA_1M = Path("/home/thomas-ahle/big-ann-data/deep1m")
RESULT_RE = re.compile(
    r"RW L=\s*(?P<l>\d+)\s+e=portal:\s+"
    r"recall@10=(?P<recall>[0-9.]+)\s+"
    r"QPS=(?P<qps>[0-9.]+).*?"
    r"evals/q=(?P<evals>\d+)\s+hops/q=(?P<hops>\d+)"
)


def ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("deep10m", "deep1m"), default="deep10m")
    parser.add_argument("--modes", default="adaptive,cap,round")
    parser.add_argument("--caps", default="8,16,24,32,48,64,96")
    parser.add_argument("--rounds", default="1,2,3,4")
    parser.add_argument("--frontiers", default="8,16,32,64")
    parser.add_argument("--walk-l", default="25,34,44,53,66,76,88")
    parser.add_argument("--nq", type=int, default=500)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--core", type=int)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--out",
        type=Path,
    )
    args = parser.parse_args()

    if args.core is None:
        core, siblings, busy = choose_core()
    else:
        core, siblings, busy = args.core, [args.core], -1
    print(f"core={core} siblings={siblings} sampled_busy_ticks={busy}", flush=True)

    modes = {item.strip() for item in args.modes.split(",") if item.strip()}
    unknown = modes - {"adaptive", "cap", "round"}
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    policies: list[dict[str, int | str]] = []
    if "adaptive" in modes:
        policies.append({"mode": "adaptive", "max_hops": 0, "rounds": 0, "frontier": 0})
    if "cap" in modes:
        policies.extend(
            {"mode": "cap", "max_hops": cap, "rounds": 0, "frontier": 0}
            for cap in ints(args.caps)
        )
    if "round" in modes:
        policies.extend(
            {"mode": "round", "max_hops": 0, "rounds": rounds, "frontier": frontier}
            for rounds in ints(args.rounds)
            for frontier in ints(args.frontiers)
        )

    if args.dataset == "deep10m":
        dataset_name = "DEEP-10M"
        common_env = {
            "SBANN_INDEX_LOAD": str(DATA / "eng_deep10m_kf65536.idx"),
            "SBANN_CENTROID_GRAPH": str(DATA / "deep10m_centroid_vamana16.u32"),
            "SBANN_CENTROID_LANDMARKS": str(DATA / "deep10m_centroid_landmarks64.u32"),
            "SBANN_CENTROID_GRAPH_K": "16",
            "SBANN_CENTROID_GRAPH_EF": "8",
            "SBANN_PORTAL_FILE": str(DATA / "deep10m_portals16.side"),
            "SBANN_PORTAL_SQ4_FILE": str(DATA / "deep10m_portals16.sq4p64"),
            "SBANN_FBASE": str(DATA / "base.10M.fbin"),
            "SBANN_FQUERY": str(DATA / "query2k.fbin"),
            "SBANN_GRAPH_FILE": str(DATA / "graph_layout_gate.cellpair.graph.u32"),
            "SBANN_GRAPH_BASE": str(DATA / "graph_layout_gate.cellpair.aligned64.i8bin"),
            "SBANN_GRAPH_BASE_OFFSET": "64",
            "SBANN_GRAPH_RANK": str(DATA / "graph_layout_gate.cellpair.u32"),
        }
        base = DATA / "base.10M.i8bin"
        query = DATA / "query2k.i8bin"
        ground_truth = DATA / "deep10m_gt.ibin"
        cells = "65536"
    else:
        dataset_name = "DEEP-1M"
        common_env = {
            "SBANN_INDEX_LOAD": str(DATA_1M / "eng_deep1m_kf8192.idx"),
            "SBANN_PORTAL_FILE": str(DATA_1M / "deep1m_portals16.side"),
            "SBANN_PORTAL_SQ4_FILE": str(DATA_1M / "deep1m_portals16.sq4p64"),
            "SBANN_FBASE": str(DATA_1M / "base.1M.fbin"),
            "SBANN_FQUERY": str(DATA / "query2k.fbin"),
            "SBANN_GRAPH_FILE": str(DATA_1M / "vamana_R32_a1.2.u32"),
        }
        base = DATA_1M / "base.1M.i8bin"
        query = DATA / "query2k.i8bin"
        ground_truth = DATA_1M / "deep1m_gt.ibin"
        cells = "8192"

    if args.out is None:
        filename = (
            "deep_fixed_walk_screen.json"
            if args.dataset == "deep10m"
            else "deep1m_fixed_walk_transfer.json"
        )
        args.out = ROOT / "sbann-rs/experiments" / filename

    if args.out.exists():
        report = json.loads(args.out.read_text())
        if report.get("dataset") != dataset_name:
            raise ValueError(
                f"{args.out} contains {report.get('dataset')}, requested {dataset_name}"
            )
    else:
        report = {"dataset": dataset_name, "samples": [], "runs": []}
    completed = {
        (
            run["mode"],
            run["max_hops"],
            run["rounds"],
            run["frontier"],
            run["walk_l"],
            run["nq"],
            run["reps"],
        )
        for run in report["runs"]
        if run.get("returncode") == 0
    }
    walk_l = args.walk_l
    common_env.update({
        "OMP_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "SBANN_ROAR_PORTAL_CELLS": "8",
        "SBANN_ROAR_PORTAL_BUCKETS": "1",
        "SBANN_ROAR_PORTAL_ROWS": "1",
        "SBANN_FLOAT_RERANK": "1",
        "SBANN_GRAPH_M": "32",
        "SBANN_GRAPH_KEDGE": "32",
        "SBANN_IP": "1",
        "SBANN_RESIDENT_I8": "1",
        "SBANN_ROARMODE": walk_l,
        "SBANN_NQ": str(args.nq),
        "SBANN_REPS": str(args.reps),
    })
    argv = [
        "taskset",
        "-c",
        str(core),
        str(ROOT / "sbann-rs/target/release/sbann"),
        "run",
        str(base),
        str(query),
        str(ground_truth),
        "hierkn",
        "apq4",
        "2",
        cells,
        "8",
    ]

    started = time.time()
    new_runs = 0
    for index, policy in enumerate(policies, 1):
        key = (
            policy["mode"],
            policy["max_hops"],
            policy["rounds"],
            policy["frontier"],
            walk_l,
            args.nq,
            args.reps,
        )
        if key in completed:
            print(f"[{index}/{len(policies)}] skip {policy}", flush=True)
            continue
        env = os.environ.copy()
        env.update(common_env)
        env["SBANN_ROAR_MAX_HOPS"] = str(policy["max_hops"])
        env["SBANN_ROAR_ROUNDS"] = str(policy["rounds"])
        env["SBANN_ROAR_FRONTIER"] = str(policy["frontier"])
        load = os.getloadavg()
        t0 = time.monotonic()
        proc = subprocess.run(
            argv,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
        wall = time.monotonic() - t0
        matches = list(RESULT_RE.finditer(proc.stdout))
        run = {
            **policy,
            "walk_l": walk_l,
            "nq": args.nq,
            "reps": args.reps,
            "core": core,
            "load1": load[0],
            "load5": load[1],
            "load15": load[2],
            "wall_s": wall,
            "returncode": proc.returncode,
            "matches": len(matches),
            "output_tail": proc.stdout[-3000:],
        }
        report["runs"].append(run)
        new_runs += 1
        if proc.returncode != 0 or not matches:
            write_report(args.out, report)
            raise RuntimeError(
                f"{policy} failed rc={proc.returncode}, matches={len(matches)}\n"
                f"{proc.stdout[-5000:]}"
            )
        samples = [
            {
                **policy,
                "walk_l": int(match.group("l")),
                "recall": float(match.group("recall")),
                "qps": float(match.group("qps")),
                "evals_per_query": int(match.group("evals")),
                "hops_per_query": int(match.group("hops")),
                "nq": args.nq,
                "reps": args.reps,
                "load1": load[0],
            }
            for match in matches
        ]
        report["samples"].extend(samples)
        write_report(args.out, report)
        best = max(samples, key=lambda row: row["recall"])
        print(
            f"[{index}/{len(policies)}] {policy} wall={wall:.1f}s "
            f"top={best['recall']:.4f}@{best['qps']:.0f} "
            f"evals={best['evals_per_query']} hops={best['hops_per_query']}",
            flush=True,
        )
    if new_runs:
        report["elapsed_seconds"] = time.time() - started
        write_report(args.out, report)


if __name__ == "__main__":
    main()
