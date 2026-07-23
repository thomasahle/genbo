#!/usr/bin/env python3
"""Paired ABBA benchmark runner for sbann experiments.

The runner deliberately keeps policy outside the engine: each arm is an argv/env
object in JSON, never a shell string.  It selects the least-busy physical core
(counting both SMT siblings), alternates ABBA/BAAB to cancel drift, captures
recall/QPS plus rusage faults/context switches, and writes every raw sample.

Config schema:
{
  "cwd": "/path",
  "common_env": {"RAYON_NUM_THREADS": "1"},
  "a": {"name": "baseline", "env": {...}, "argv": ["binary", ...]},
  "b": {"name": "candidate", "env": {...}, "argv": ["binary", ...]}
}
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import resource
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


RESULT_RE = re.compile(r"recall@10=([0-9.]+)\s+QPS=([0-9]+)")


def apply_variant(cfg: dict[str, Any], name: str | None) -> dict[str, Any]:
    if name is None:
        return cfg
    variants = {variant["name"]: variant for variant in cfg.get("variants", [])}
    if name not in variants:
        choices = ", ".join(sorted(variants)) or "(none)"
        raise ValueError(f"unknown variant {name!r}; choices: {choices}")
    out = copy.deepcopy(cfg)
    variant = variants[name]
    for key in ("a", "b"):
        patch = variant.get(key, {})
        arm = out[key]
        if "name" in patch:
            arm["name"] = patch["name"]
        if "result_regex" in patch:
            arm["result_regex"] = patch["result_regex"]
        arm.setdefault("env", {}).update(patch.get("env", {}))
        replacements = {
            str(old): str(new) for old, new in patch.get("argv_replace", {}).items()
        }
        arm["argv"] = [replacements.get(str(arg), arg) for arg in arm["argv"]]
    return out


def cpu_busy_ticks() -> dict[int, int]:
    out: dict[int, int] = {}
    for line in Path("/proc/stat").read_text().splitlines():
        m = re.match(r"cpu(\d+)\s+(.+)", line)
        if not m:
            continue
        fields = [int(x) for x in m.group(2).split()]
        # Linux fields: user nice system idle iowait irq softirq steal.
        # Idle and iowait must not make an unused core look busy.
        out[int(m.group(1))] = sum(fields[i] for i in (0, 1, 2, 5, 6, 7))
    return out


def sibling_groups() -> list[list[int]]:
    groups: set[tuple[int, ...]] = set()
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology/thread_siblings_list"):
        cpus: list[int] = []
        for part in path.read_text().strip().split(","):
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                cpus.extend(range(lo, hi + 1))
            else:
                cpus.append(int(part))
        groups.add(tuple(sorted(cpus)))
    return [list(x) for x in sorted(groups)]


def choose_core(sample_seconds: float = 0.5) -> tuple[int, list[int], int]:
    before = cpu_busy_ticks()
    time.sleep(sample_seconds)
    after = cpu_busy_ticks()
    scored = []
    for siblings in sibling_groups():
        # Avoid CPU0 because kernel housekeeping is commonly pinned there.
        if 0 in siblings:
            continue
        busy = sum(after.get(c, 0) - before.get(c, 0) for c in siblings)
        scored.append((busy, siblings[0], siblings))
    if not scored:
        raise RuntimeError("no physical CPU sibling groups found")
    busy, core, siblings = min(scored)
    return core, siblings, busy


def rusage_snapshot() -> resource.struct_rusage:
    return resource.getrusage(resource.RUSAGE_CHILDREN)


def rusage_delta(a: resource.struct_rusage, b: resource.struct_rusage) -> dict[str, float]:
    return {
        "user_s": b.ru_utime - a.ru_utime,
        "sys_s": b.ru_stime - a.ru_stime,
        "minor_faults": b.ru_minflt - a.ru_minflt,
        "major_faults": b.ru_majflt - a.ru_majflt,
        "voluntary_ctx": b.ru_nvcsw - a.ru_nvcsw,
        "involuntary_ctx": b.ru_nivcsw - a.ru_nivcsw,
        "maxrss_kb": b.ru_maxrss,
    }


def run_arm(
    cfg: dict[str, Any],
    common_env: dict[str, str],
    cwd: str,
    core: int,
    timeout: float,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update({k: str(v) for k, v in common_env.items()})
    env.update({k: str(v) for k, v in cfg.get("env", {}).items()})
    argv = ["taskset", "-c", str(core), *[str(x) for x in cfg["argv"]]]
    load0 = os.getloadavg()
    ru0 = rusage_snapshot()
    t0 = time.monotonic()
    proc = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    wall = time.monotonic() - t0
    ru1 = rusage_snapshot()
    result_re = re.compile(cfg.get("result_regex", RESULT_RE.pattern), re.MULTILINE)
    matches = list(result_re.finditer(proc.stdout))
    if proc.returncode != 0 or not matches:
        raise RuntimeError(
            f"{cfg['name']} failed rc={proc.returncode}, matches={len(matches)}\n"
            f"{proc.stdout[-4000:]}"
        )
    match = matches[-1]
    if {"recall", "qps"} <= match.groupdict().keys():
        recall, qps = match.group("recall"), match.group("qps")
    else:
        recall, qps = match.group(1), match.group(2)
    return {
        "arm": cfg["name"],
        "recall": float(recall),
        "qps": float(qps),
        "wall_s": wall,
        "load1": load0[0],
        "load5": load0[1],
        "load15": load0[2],
        "returncode": proc.returncode,
        **rusage_delta(ru0, ru1),
        "output_tail": proc.stdout[-2000:],
    }


def summarize(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in names:
        arm = [r for r in rows if r["arm"] == name]
        qps = [r["qps"] for r in arm]
        recall = [r["recall"] for r in arm]
        out[name] = {
            "n": len(arm),
            "qps_best": max(qps),
            "qps_median": statistics.median(qps),
            "qps_mean": statistics.fmean(qps),
            "qps_stdev": statistics.stdev(qps) if len(qps) > 1 else 0.0,
            "recall_min": min(recall),
            "recall_max": max(recall),
            "major_faults_median": statistics.median(r["major_faults"] for r in arm),
            "minor_faults_median": statistics.median(r["minor_faults"] for r in arm),
            "involuntary_ctx_median": statistics.median(r["involuntary_ctx"] for r in arm),
        }
    a, b = names
    out["ratio"] = {
        "median_b_over_a": out[b]["qps_median"] / out[a]["qps_median"],
        "best_b_over_a": out[b]["qps_best"] / out[a]["qps_best"],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--rounds", type=int, default=2, help="four measured legs per round")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--core", type=int)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--variant", help="named arm override from config's variants list")
    args = ap.parse_args()

    cfg = apply_variant(json.loads(args.config.read_text()), args.variant)
    arms = {"a": cfg["a"], "b": cfg["b"]}
    if args.core is None:
        core, siblings, busy = choose_core()
    else:
        core, siblings, busy = args.core, [args.core], -1
    out_path = args.out or args.config.with_suffix(".results.json")
    print(f"core={core} siblings={siblings} sampled_busy_ticks={busy}", flush=True)

    if not args.no_warmup:
        for key in ("a", "b"):
            row = run_arm(arms[key], cfg.get("common_env", {}), cfg["cwd"], core, args.timeout)
            print(f"warmup {row['arm']}: recall={row['recall']:.4f} qps={row['qps']}", flush=True)

    rows: list[dict[str, Any]] = []
    for rnd in range(args.rounds):
        order = ("a", "b", "b", "a") if rnd % 2 == 0 else ("b", "a", "a", "b")
        for leg, key in enumerate(order):
            row = run_arm(arms[key], cfg.get("common_env", {}), cfg["cwd"], core, args.timeout)
            row["round"] = rnd
            row["leg"] = leg
            rows.append(row)
            print(
                f"r{rnd}l{leg} {row['arm']}: recall={row['recall']:.4f} "
                f"qps={row['qps']} faults={row['major_faults']}/{row['minor_faults']} "
                f"ctx={row['involuntary_ctx']}",
                flush=True,
            )
            out_path.write_text(json.dumps({"samples": rows}, indent=2) + "\n")

    names = [arms["a"]["name"], arms["b"]["name"]]
    report = {
        "config": str(args.config),
        "core": core,
        "siblings": siblings,
        "samples": rows,
        "summary": summarize(rows, names),
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
