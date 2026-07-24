#!/usr/bin/env python3
"""Screen genbo's low-recall DEEP-10M graph policy on the physical layout.

Each process fixes (hops, beam, edge count, survivor floor) and sweeps probes plus
cascade width inside one loaded index. Results are checkpointed after every process,
so a long grid can be resumed without repeating completed configurations.
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


DATA = Path("/home/thomas-ahle/big-ann-data/deep10m")
ROOT = Path("/home/thomas-ahle/genbo")
RESULT_RE = re.compile(
    r"p=\s*(?P<p>\d+).*?K=(?P<k>\d+):\s+"
    r"recall@10=(?P<recall>[0-9.]+)\s+QPS=(?P<qps>[0-9.]+)"
)


def ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (-row["recall"], -row["qps"]))
    frontier: list[dict[str, Any]] = []
    best_qps = -1.0
    for row in ordered:
        if row["qps"] > best_qps:
            frontier.append(row)
            best_qps = row["qps"]
    return sorted(frontier, key=lambda row: row["recall"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hops", default="1,2,3")
    parser.add_argument("--beams", default="4,8,12,16,24")
    parser.add_argument("--edges", default="8,16,24,32")
    parser.add_argument("--floors", default="320")
    parser.add_argument("--cascade-widths", default="32")
    parser.add_argument("--probes", default="2,4,6,8,12,16,24,32")
    parser.add_argument("--nq", type=int, default=500)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--core", type=int)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--graph",
        type=Path,
        default=DATA / "graph_layout_gate.cellpair.graph.u32",
        help="use unsorted relabeled adjacency when screening edge prefixes",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA / "deep_low_recall_sweep.json",
    )
    args = parser.parse_args()
    graph_path = str(args.graph.resolve())

    if args.core is None:
        core, siblings, busy = choose_core()
    else:
        core, siblings, busy = args.core, [args.core], -1
    print(f"core={core} siblings={siblings} sampled_busy_ticks={busy}", flush=True)

    report: dict[str, Any]
    if args.out.exists():
        report = json.loads(args.out.read_text())
    else:
        report = {"samples": [], "runs": []}
    completed = {
        (
            run["hops"],
            run["beam"],
            run["edges"],
            run["floor"],
            tuple(run["cascade_widths"]),
            tuple(run["probes"]),
            run["nq"],
            run.get("reps", 1),
            run.get(
                "graph",
                str((DATA / "graph_layout_gate.cellpair.graph.u32").resolve()),
            ),
        )
        for run in report.get("runs", [])
        if run.get("returncode") == 0
    }

    probes = ints(args.probes)
    cascade_widths = ints(args.cascade_widths)
    grid = [
        (hops, beam, edges, floor)
        for hops in ints(args.hops)
        for beam in ints(args.beams)
        for edges in ints(args.edges)
        for floor in ints(args.floors)
    ]
    started = time.time()
    for index, (hops, beam, edges, floor) in enumerate(grid, 1):
        key = (
            hops,
            beam,
            edges,
            floor,
            tuple(cascade_widths),
            tuple(probes),
            args.nq,
            args.reps,
            graph_path,
        )
        if key in completed:
            print(
                f"[{index}/{len(grid)}] skip h{hops} M{beam} e{edges} f{floor}",
                flush=True,
            )
            continue
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "RAYON_NUM_THREADS": "1",
                "SBANN_FLOAT_RERANK": "1",
                "SBANN_FBASE": str(DATA / "base.10M.fbin"),
                "SBANN_FQUERY": str(DATA / "query2k.fbin"),
                "SBANN_GRAPH_BASE": str(
                    DATA / "graph_layout_gate.cellpair.aligned64.i8bin"
                ),
                "SBANN_GRAPH_BASE_OFFSET": "64",
                "SBANN_GRAPH_BESTFIRST": "1",
                "SBANN_GRAPH_FILE": graph_path,
                "SBANN_GRAPH_HOPS": str(hops),
                "SBANN_GRAPH_KEDGE": str(edges),
                "SBANN_GRAPH_M": str(beam),
                "SBANN_GRAPH_RANK": str(
                    DATA / "graph_layout_gate.cellpair.u32"
                ),
                "SBANN_INDEX_LOAD": str(DATA / "eng_deep10m_kf65536.idx"),
                "SBANN_KLIST": ",".join(map(str, cascade_widths)),
                "SBANN_NQ": str(args.nq),
                "SBANN_PLIST": ",".join(map(str, probes)),
                "SBANN_REPS": str(args.reps),
                "SBANN_RESIDENT_I8": "1",
                "SBANN_TFLOOR": str(floor),
            }
        )
        argv = [
            "taskset",
            "-c",
            str(core),
            str(ROOT / "sbann-rs/target/release/sbann"),
            "run",
            str(DATA / "base.10M.i8bin"),
            str(DATA / "query2k.i8bin"),
            str(DATA / "deep10m_gt.ibin"),
            "hierkn",
            "apq4",
            "2",
            "65536",
            "8",
        ]
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
            "hops": hops,
            "beam": beam,
            "edges": edges,
            "floor": floor,
            "cascade_widths": cascade_widths,
            "probes": probes,
            "nq": args.nq,
            "reps": args.reps,
            "graph": graph_path,
            "core": core,
            "load1": load[0],
            "load5": load[1],
            "load15": load[2],
            "wall_s": wall,
            "returncode": proc.returncode,
            "matches": len(matches),
            "output_tail": proc.stdout[-2000:],
        }
        report.setdefault("runs", []).append(run)
        if proc.returncode != 0 or not matches:
            write_report(args.out, report)
            raise RuntimeError(
                f"h{hops} M{beam} e{edges} f{floor} failed "
                f"rc={proc.returncode}, matches={len(matches)}\n"
                f"{proc.stdout[-4000:]}"
            )
        for match in matches:
            report.setdefault("samples", []).append(
                {
                    "hops": hops,
                    "beam": beam,
                    "edges": edges,
                    "floor": floor,
                    "p": int(match.group("p")),
                    "cascade_k": int(match.group("k")),
                    "recall": float(match.group("recall")),
                    "qps": float(match.group("qps")),
                    "core": core,
                    "load1": load[0],
                    "run_index": len(report["runs"]) - 1,
                }
            )
        report["frontier"] = pareto(report["samples"])
        report["elapsed_s"] = time.time() - started
        write_report(args.out, report)
        useful = [
            row
            for row in report["samples"][-len(matches) :]
            if 0.88 <= row["recall"] <= 0.93
        ]
        best = max(useful, key=lambda row: row["qps"], default=None)
        tag = (
            f" best-band={best['recall']:.4f}@{best['qps']:.0f}"
            if best
            else ""
        )
        print(
            f"[{index}/{len(grid)}] h{hops} M{beam} e{edges} f{floor} "
            f"{len(matches)} points {wall:.1f}s{tag}",
            flush=True,
        )

    print(json.dumps({"frontier": report["frontier"]}, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
