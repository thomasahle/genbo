#!/usr/bin/env python3
"""One-load Wikipedia-35M low-recall policy sweep.

The resident Wikipedia stack is too large to reload once per policy.  This
runner uses SBANN_{M,KEDGE,TFLOOR,K}LIST to screen the existing cascade in one
process, preserving the exact graph/index/scoring stack used by the paper.
It waits for the older queued Wikipedia confirmations before claiming memory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


DATA = Path("/home/thomas-ahle/big-ann-data")
WIKI = DATA / "wiki35m"
ROOT = Path("/home/thomas-ahle/genbo")
DEFAULT_BINARY = DATA / "cargo-wiki-law/release/sbann"
DEFAULT_LOG = DATA / "wiki_low_recall_law_screen.log"
DEFAULT_JSON = ROOT / "sbann-rs/experiments/wiki_low_recall_law_screen.json"
WAIT_MARKERS = (
    (DATA / "wiki_m64_confirm.log", "WIKI_M64_CONFIRM_DONE"),
    (DATA / "wiki_portal_seed_probe.log", "WIKI_PORTAL_SEED_PROBE_DONE"),
)

RESULT_RE = re.compile(
    r"p=\s*(?P<p>\d+)\s+t=\s*(?P<tm>\d+).*?"
    r"K=(?P<width>\d+):\s+recall@10=(?P<recall>[0-9.]+)\s+"
    r"QPS=(?P<qps>[0-9.]+).*?"
    r"\[M=(?P<beam>\d+)\s+ke=(?P<edges>\d+)\s+tf=(?P<floor>\d+)\s+"
    r"pk=(?P<portal_keep>\d+)(?:\s+ca=(?P<cell_add>\d+))?\]"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load1() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def available_gib() -> float:
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.strip().split()[0])
    return fields["MemAvailable"] / 1024 / 1024


def marker_present(path: Path, marker: str) -> bool:
    try:
        return marker in path.read_text(errors="replace")
    except FileNotFoundError:
        return False


def wait_for_resources(log, skip_wait: bool) -> None:
    if not skip_wait:
        for path, marker in WAIT_MARKERS:
            while not marker_present(path, marker):
                message = f"[{now()}] waiting for {marker} in {path}"
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
                time.sleep(60)
    stable = 0
    while stable < 3:
        good = load1() < 18.0 and available_gib() >= 150.0
        stable = stable + 1 if good else 0
        if stable < 3:
            message = (
                f"[{now()}] resource gate load={load1():.2f} "
                f"available={available_gib():.1f}GiB stable={stable}/3"
            )
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()
            time.sleep(60)


def write_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def integer_list(spec: str) -> list[int]:
    values = [int(value) for value in spec.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected a non-empty list of positive integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    # Wikipedia's query file is ordered: its first 200 queries are materially
    # easier than the full benchmark. Default to all 1,000 so a quick prefix
    # cannot silently promote an inflated-recall operating point.
    parser.add_argument("--nq", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--core", type=int, default=43)
    parser.add_argument("--beam", type=integer_list, default=integer_list("24,32,48,64"))
    parser.add_argument("--edges", type=integer_list, default=integer_list("24,32,48,64"))
    parser.add_argument("--floor", type=integer_list, default=integer_list("500,700,900,1100"))
    parser.add_argument("--probes", type=integer_list, default=integer_list("6,8,10,12,14,16"))
    parser.add_argument("--width", type=integer_list, default=integer_list("16,20,24,32"))
    parser.add_argument("--portal-keep", type=integer_list, default=integer_list("1"))
    parser.add_argument("--portal-seed", action="store_true")
    parser.add_argument("--completion-marker-file", type=Path)
    parser.add_argument("--completion-marker")
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "wiki35m",
        "purpose": "low-recall law-guided cascade screen",
        "started": now(),
        "binary": str(args.binary),
        "nq": args.nq,
        "reps": args.reps,
        "grid": {
            "beam": args.beam,
            "edges": args.edges,
            "floor": args.floor,
            "probes": args.probes,
            "width": args.width,
            "portal_keep": args.portal_keep,
        },
        "samples": [],
    }
    write_checkpoint(args.output, payload)

    with args.log.open("a", buffering=1) as log:
        wait_for_resources(log, args.skip_wait)
        env = os.environ.copy()
        env.update(
            {
                "SBANN_ROUTE_FP16": "1",
                "SBANN_INDEX_LOAD": str(
                    WIKI / "eng_wiki35m_fp16_65536_dpb4_a2.idx"
                ),
                "SBANN_IP": "1",
                "SBANN_KLIST": ",".join(map(str, args.width)),
                "SBANN_SQ4_FILE": str(WIKI / "sq4.side"),
                "SBANN_SQ4_NAV": "1",
                "SBANN_SQ4_INT8K": "256",
                "SBANN_RESIDENT_I8": "1",
                "SBANN_RERANK_F16": "1",
                "SBANN_RERANK_F16_FILE": str(WIKI / "base_f16.side"),
                "SBANN_GRAPH_FILE": str(WIKI / "wiki35m_nd_k64s16.u32"),
                "SBANN_GRAPH_HOPS": "1",
                "SBANN_GRAPH_BESTFIRST": "1",
                "SBANN_MLIST": ",".join(map(str, args.beam)),
                "SBANN_KEDGELIST": ",".join(map(str, args.edges)),
                "SBANN_TFLOORLIST": ",".join(map(str, args.floor)),
                "SBANN_PLIST": ",".join(map(str, args.probes)),
                "SBANN_NQ": str(args.nq),
                "SBANN_REPS": str(args.reps),
                "RAYON_NUM_THREADS": "1",
                "SBANN_FLOAT_RERANK": "1",
                "SBANN_FBASE": str(WIKI / "base.fbin"),
                "SBANN_FQUERY": str(WIKI / "query.fbin"),
            }
        )
        if args.portal_seed:
            env.update(
                {
                    "SBANN_PORTAL_FILE": str(WIKI / "wiki35m_portals16.side"),
                    "SBANN_PORTAL_SQ4_FILE": str(
                        WIKI / "wiki35m_portals16.sq4p64"
                    ),
                    "SBANN_PORTAL_KEEPLIST": ",".join(
                        map(str, args.portal_keep)
                    ),
                }
            )
        command = [
            "nice",
            "-n",
            "5",
            "taskset",
            "-c",
            str(args.core),
            str(args.binary),
            "run",
            str(WIKI / "base.i8bin"),
            str(WIKI / "query.i8bin"),
            str(WIKI / "wiki35m_gt_clean.ibin"),
            "hierkn",
            "apq4",
            "2",
            "65536",
            "8",
        ]
        header = (
            f"=== WIKI LOW-RECALL LAW SCREEN start {now()} "
            f"load={load1():.2f} available={available_gib():.1f}GiB ==="
        )
        print(header, flush=True)
        log.write(header + "\n")
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            match = RESULT_RE.search(line)
            if match:
                sample = {
                    key: (
                        float(value)
                        if key in {"recall", "qps"}
                        else int(value or 0)
                    )
                    for key, value in match.groupdict().items()
                }
                sample["load1"] = load1()
                payload["samples"].append(sample)
                if len(payload["samples"]) % 32 == 0:
                    write_checkpoint(args.output, payload)
        returncode = process.wait()
        payload["finished"] = now()
        payload["returncode"] = returncode
        payload["final_load1"] = load1()
        payload["final_available_gib"] = available_gib()
        write_checkpoint(args.output, payload)
        footer = (
            f"=== WIKI LOW-RECALL LAW SCREEN done {now()} "
            f"returncode={returncode} samples={len(payload['samples'])} ==="
        )
        print(footer, flush=True)
        log.write(footer + "\n")
        if returncode == 0 and args.completion_marker_file and args.completion_marker:
            with args.completion_marker_file.open("a") as marker_log:
                marker_log.write(
                    f"{args.completion_marker} (one-load replacement) {now()}\n"
                )
        return returncode


if __name__ == "__main__":
    raise SystemExit(main())
