#!/usr/bin/env python3
"""Checkpointed screens for the post-retune DEEP experiments."""

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
RESULT_RE = re.compile(r"recall@10=([0-9.]+)\s+QPS=([0-9]+)")
POLICIES = {
    "r0903": {"p": 15, "m": 8, "e": 32, "floor": 192, "k": 24},
    "r0926": {"p": 20, "m": 16, "e": 24, "floor": 192, "k": 16},
    "r0943": {"p": 24, "m": 12, "e": 32, "floor": 320, "k": 32},
    "r0960": {"p": 28, "m": 24, "e": 32, "floor": 256, "k": 48},
    "r0972": {"p": 48, "m": 24, "e": 32, "floor": 384, "k": 48},
}


def comma_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def feature_env(feature: str, value: str) -> dict[str, str]:
    if feature == "fp16-refine":
        return {
            "SBANN_RERANK_F16": "1",
            "SBANN_RERANK_F16_REFINE": value,
        }
    if feature == "beam0":
        return {"SBANN_BEAM0": value}
    if feature == "batch-chunk":
        return {"SBANN_BATCHSCAN": "1", "SBANN_BATCH_CHUNK": value}
    raise ValueError(feature)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "feature",
        choices=(
            "fp16-refine",
            "beam0",
            "batch-chunk",
        ),
    )
    parser.add_argument("--values", required=True)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--nq", type=int, default=500)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--core", type=int)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--result-dir",
        type=Path,
        help="also dump each run's nq x 10 final ids for per-query gates",
    )
    parser.add_argument(
        "--confidence-dir",
        type=Path,
        help="also dump diagnostic confidence features for each query",
    )
    args = parser.parse_args()

    requested = comma_list(args.policies)
    unknown = set(requested) - set(POLICIES)
    if unknown:
        raise ValueError(f"unknown policies: {sorted(unknown)}")
    values = comma_list(args.values)
    if args.core is None:
        core, siblings, busy = choose_core()
    else:
        core, siblings, busy = args.core, [args.core], -1
    print(f"core={core} siblings={siblings} sampled_busy_ticks={busy}", flush=True)

    if args.out.exists():
        report = json.loads(args.out.read_text())
    else:
        report = {"feature": args.feature, "runs": []}
    completed = {
        (run["policy"], run["value"], run["nq"], run["reps"])
        for run in report["runs"]
        if run.get("returncode") == 0
    }

    for policy_name in requested:
        policy = POLICIES[policy_name]
        for value in values:
            key = (policy_name, value, args.nq, args.reps)
            if key in completed:
                print(f"skip {policy_name} {args.feature}={value}", flush=True)
                continue
            env = os.environ.copy()
            env.update(
                {
                    "OMP_NUM_THREADS": "1",
                    "RAYON_NUM_THREADS": "1",
                    "SBANN_CASCADE_K": str(policy["k"]),
                    "SBANN_FLOAT_RERANK": "1",
                    "SBANN_FBASE": str(DATA / "base.10M.fbin"),
                    "SBANN_FQUERY": str(DATA / "query2k.fbin"),
                    "SBANN_GRAPH_BASE": str(
                        DATA / "graph_layout_gate.cellpair.aligned64.i8bin"
                    ),
                    "SBANN_GRAPH_BASE_OFFSET": "64",
                    "SBANN_GRAPH_BESTFIRST": "1",
                    "SBANN_GRAPH_FILE": str(
                        DATA / "graph_layout_gate.cellpair.graph.u32"
                    ),
                    "SBANN_GRAPH_HOPS": "2",
                    "SBANN_GRAPH_KEDGE": str(policy["e"]),
                    "SBANN_GRAPH_M": str(policy["m"]),
                    "SBANN_GRAPH_RANK": str(
                        DATA / "graph_layout_gate.cellpair.u32"
                    ),
                    "SBANN_INDEX_LOAD": str(DATA / "eng_deep10m_kf65536.idx"),
                    "SBANN_NQ": str(args.nq),
                    "SBANN_PLIST": str(policy["p"]),
                    "SBANN_REPS": str(args.reps),
                    "SBANN_RESIDENT_I8": "1",
                    "SBANN_TFLOOR": str(policy["floor"]),
                    **feature_env(args.feature, value),
                }
            )
            result_path = None
            if args.result_dir is not None:
                args.result_dir.mkdir(parents=True, exist_ok=True)
                result_path = (
                    args.result_dir
                    / f"{policy_name}.{args.feature}.{value}.u32"
                )
                env["SBANN_RESULT_DUMP"] = str(result_path)
            confidence_path = None
            if args.confidence_dir is not None:
                args.confidence_dir.mkdir(parents=True, exist_ok=True)
                confidence_path = (
                    args.confidence_dir
                    / f"{policy_name}.{args.feature}.{value}.csv"
                )
                env["SBANN_DUMP_CONFIDENCE"] = str(confidence_path)
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
            matches = list(RESULT_RE.finditer(proc.stdout))
            run = {
                "policy": policy_name,
                "policy_config": policy,
                "feature": args.feature,
                "value": value,
                "nq": args.nq,
                "reps": args.reps,
                "core": core,
                "load": os.getloadavg(),
                "wall_s": time.monotonic() - t0,
                "returncode": proc.returncode,
                "output_tail": proc.stdout[-2000:],
                "result_dump": str(result_path) if result_path is not None else None,
                "confidence_dump": (
                    str(confidence_path) if confidence_path is not None else None
                ),
            }
            if matches:
                run["recall"] = float(matches[-1].group(1))
                run["qps"] = float(matches[-1].group(2))
            report["runs"].append(run)
            write_report(args.out, report)
            if proc.returncode != 0 or not matches:
                raise RuntimeError(proc.stdout[-4000:])
            print(
                f"{policy_name} {args.feature}={value}: "
                f"{run['recall']:.4f}@{run['qps']:.0f} "
                f"wall={run['wall_s']:.1f}s",
                flush=True,
            )

    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
