#!/usr/bin/env python3
"""Transfer genbo's fixed-work tuning laws to the other paper datasets.

The expensive index/base setup is amortized by sweeping survivor widths with
SBANN_KLIST inside one process.  Each named point is an existing paper operating
point; only the law-predicted local alternatives are varied.  Results are
checkpointed after every process so large datasets can be resumed safely.
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
DATA = Path("/home/thomas-ahle/big-ann-data")
BIN = ROOT / "sbann-rs/target/release/sbann"
RESULT_RE = re.compile(
    r"p=\s*(?P<p>\d+)\s+t=\s*(?P<t>\d+)\s+K=(?P<k>\d+):\s+"
    r"recall@10=(?P<recall>[0-9.]+)\s+QPS=(?P<qps>[0-9.]+)"
)


def point(
    label: str,
    p: int,
    floor: int,
    *,
    hops: int | None = None,
    beam: int | None = None,
    graph: str | None = None,
    edges: int | None = None,
    widths: tuple[int, ...] = (14, 16, 20, 32),
    probes: tuple[int, ...] | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "p": p,
        "probes": list(probes or (p,)),
        "floor": floor,
        "hops": hops,
        "beam": beam,
        "graph": graph,
        "edges": edges,
        "widths": list(widths),
        "extra": extra or {},
    }


def dataset_registry() -> dict[str, dict[str, Any]]:
    cohere = DATA / "cohere"
    streaming = DATA / "streaming"
    webvid = Path("/home/thomas-ahle/RoarGraph/data/clip-webvid-2.5M")
    return {
        "t2i1m": {
            "base": DATA / "base1M.i8bin",
            "query": DATA / "query100K_s1m.i8bin",
            "gt": DATA / "t2i1m_gt.ibin",
            "router": "hierk",
            "compressor": "apq4",
            "a0": 3,
            "kf": 16384,
            "tmul": 8,
            "env": {
                "SBANN_INDEX_LOAD": str(DATA / "eng_scale_1m_kf16384.idx"),
                "SBANN_FLOAT_RERANK": "1",
                "SBANN_FBASE": str(DATA / "base1M.fbin"),
                "SBANN_FQUERY": str(DATA / "query.public.100K.fbin"),
                "SBANN_IP": "1",
                "SBANN_ROUTE_GAMMA": "0.5",
                "SBANN_BATCHSCAN": "0",
            },
            "points": [
                point(
                    "r90",
                    4,
                    600,
                    hops=2,
                    beam=48,
                    graph=str(DATA / "hyb_t2i1m_k32.u32"),
                    edges=32,
                ),
                point(
                    "r93",
                    8,
                    800,
                    hops=2,
                    beam=48,
                    graph=str(DATA / "hyb_t2i1m_k32.u32"),
                    edges=32,
                ),
                point(
                    "r95",
                    16,
                    1000,
                    hops=2,
                    beam=48,
                    graph=str(DATA / "hyb_t2i1m_k32.u32"),
                    edges=32,
                ),
                point(
                    "r98",
                    40,
                    1500,
                    hops=3,
                    beam=48,
                    graph=str(DATA / "hyb_t2i1m_k32.u32"),
                    edges=32,
                ),
                point(
                    "r99",
                    128,
                    3000,
                    hops=3,
                    beam=48,
                    graph=str(DATA / "hyb_t2i1m_k32.u32"),
                    edges=32,
                ),
            ],
        },
        "cohere1m": {
            "base": cohere / "base1m.i8bin",
            "query": cohere / "query.i8bin",
            "gt": cohere / "cohere1m_gt.ibin",
            "router": "hierkn",
            "compressor": "apq4",
            "a0": 3,
            "kf": 16384,
            "tmul": 8,
            "env": {
                "SBANN_INDEX_LOAD": str(cohere / "eng_cohere1m_kf16384_dpb4.idx"),
                "SBANN_FLOAT_RERANK": "1",
                "SBANN_FBASE": str(cohere / "base1m.fbin"),
                "SBANN_FQUERY": str(cohere / "query.fbin"),
                "SBANN_IP": "1",
                "SBANN_NOMU": "1",
            },
            "points": [
                point("r97", 96, 3000, widths=(14, 16, 20, 32, 64)),
                point("r98", 128, 4000, widths=(14, 16, 20, 32, 64)),
                point("r99", 192, 6000, widths=(14, 16, 20, 32, 64)),
                point("r995", 384, 12000, widths=(14, 16, 20, 32, 64)),
            ],
        },
        "t2i10m": {
            "base": DATA / "base10M.i8bin",
            "query": DATA / "query100K.i8bin",
            "gt": DATA / "t2i10m_gt.ibin",
            "router": "hierk",
            "compressor": "apq4",
            "a0": 3,
            "kf": 65536,
            "tmul": 8,
            "env": {
                "SBANN_INDEX_LOAD": str(
                    DATA / "eng_t2i10m_kf65536_c4096_b128_a3_em2.idx"
                ),
                "SBANN_FLOAT_RERANK": "1",
                "SBANN_FBASE": str(DATA / "base10M.fbin"),
                "SBANN_FQUERY": str(DATA / "query.public.100K.fbin"),
                "SBANN_IP": "1",
                "SBANN_ROUTE_GAMMA": "0.5",
                "SBANN_BATCHSCAN": "0",
            },
            "points": [
                point(
                    "r89",
                    8,
                    800,
                    hops=2,
                    beam=48,
                    graph=str(DATA / "hyb_t2i10m_k32.u32"),
                    edges=32,
                ),
                point(
                    "r92",
                    16,
                    1000,
                    hops=2,
                    beam=48,
                    graph=str(DATA / "hyb_t2i10m_k32.u32"),
                    edges=32,
                ),
                point(
                    "r94",
                    40,
                    1000,
                    hops=2,
                    beam=48,
                    graph=str(DATA / "hyb_t2i10m_k32.u32"),
                    edges=32,
                ),
                point(
                    "r98",
                    192,
                    3000,
                    hops=3,
                    beam=48,
                    graph=str(DATA / "hyb_t2i10m_k32.u32"),
                    edges=32,
                ),
                point(
                    "r985",
                    384,
                    6000,
                    hops=3,
                    beam=48,
                    graph=str(DATA / "hyb_t2i10m_k32.u32"),
                    edges=32,
                ),
            ],
        },
        "cohere10m": {
            "base": cohere / "base.i8bin",
            "query": cohere / "query.i8bin",
            "gt": cohere / "cohere_gt.ibin",
            "router": "hierkn",
            "compressor": "apq4",
            "a0": 3,
            "kf": 65536,
            "tmul": 8,
            "env": {
                "SBANN_INDEX_LOAD": str(
                    cohere / "eng_cohere10m_l2_65536_dpb4_em3_a3.idx"
                ),
                "SBANN_FLOAT_RERANK": "1",
                "SBANN_FBASE": str(cohere / "base.fbin"),
                "SBANN_FQUERY": str(cohere / "query.fbin"),
                "SBANN_IP": "1",
                "SBANN_NOMU": "1",
            },
            "points": [
                point(
                    "r95",
                    16,
                    500,
                    hops=1,
                    beam=24,
                    graph=str(cohere / "nd10m_r24_seed.u32"),
                    edges=16,
                    widths=(14, 16, 20, 32, 64),
                ),
                point(
                    "r97",
                    32,
                    1000,
                    hops=2,
                    beam=24,
                    graph=str(cohere / "nd10m_r24_seed.u32"),
                    edges=16,
                    widths=(14, 16, 20, 32, 64),
                ),
                point(
                    "r98",
                    64,
                    2000,
                    hops=3,
                    beam=24,
                    graph=str(cohere / "nd10m_r24_seed.u32"),
                    edges=16,
                    widths=(14, 16, 20, 32, 64),
                ),
                point(
                    "r99",
                    128,
                    4000,
                    hops=3,
                    beam=24,
                    graph=str(cohere / "nd10m_r24_seed.u32"),
                    edges=16,
                    widths=(14, 16, 20, 32, 64),
                ),
            ],
        },
        "mst30m": {
            "base": streaming / "msturing30M.i8bin",
            "query": streaming / "query10K.i8bin",
            "gt": streaming / "msturing30M_gt_f32.ibin",
            "router": "hierkn",
            "compressor": "apq4",
            "a0": 2,
            "kf": 131072,
            "tmul": 8,
            "env": {
                "SBANN_INDEX_LOAD": str(
                    streaming / "eng_msturing30M_kf131072.idx"
                ),
                "SBANN_FLOAT_RERANK": "1",
                "SBANN_FBASE": str(streaming / "30M-clustered64.fbin"),
                "SBANN_FQUERY": str(streaming / "testQuery10K.fbin"),
                "SBANN_GRAPH_BESTFIRST": "1",
            },
            "points": [
                point(
                    "r73",
                    1,
                    150,
                    hops=3,
                    beam=24,
                    graph=str(streaming / "vamana_mst_R32.u32"),
                    edges=32,
                    widths=(14, 16, 20, 32),
                ),
                point(
                    "r79",
                    2,
                    250,
                    hops=3,
                    beam=24,
                    graph=str(streaming / "vamana_mst_R32.u32"),
                    edges=32,
                    widths=(14, 16, 20, 32),
                ),
                point(
                    "r85",
                    4,
                    400,
                    hops=3,
                    beam=24,
                    graph=str(streaming / "vamana_mst_R32.u32"),
                    edges=32,
                    widths=(14, 16, 20, 32),
                ),
                point(
                    "r91",
                    16,
                    700,
                    hops=3,
                    beam=24,
                    graph=str(streaming / "vamana_mst_R32.u32"),
                    edges=32,
                    widths=(14, 16, 20, 32, 64),
                ),
                point(
                    "r95",
                    32,
                    1000,
                    hops=4,
                    beam=32,
                    graph=str(streaming / "vamana_mst_R32.u32"),
                    edges=32,
                    widths=(14, 16, 20, 32),
                ),
                point(
                    "r97",
                    64,
                    1800,
                    hops=5,
                    beam=48,
                    graph=str(streaming / "vamana_mst_R32.u32"),
                    edges=32,
                    widths=(14, 16, 20, 32, 64),
                ),
                point(
                    "r982",
                    160,
                    3500,
                    hops=6,
                    beam=64,
                    graph=str(streaming / "vamana_mst_R32.u32"),
                    edges=32,
                    widths=(14, 16, 20, 32, 64, 128),
                ),
                point(
                    "r988",
                    384,
                    7000,
                    hops=8,
                    beam=64,
                    graph=str(streaming / "vamana_mst_R32.u32"),
                    edges=32,
                    widths=(14, 16, 20, 32, 64, 128, 256),
                ),
            ],
        },
        "webvid": {
            "base": webvid / "base.2.5M.i8bin",
            "query": webvid / "query.2k.i8bin",
            "gt": webvid / "webvid_gt2k.ibin",
            "router": "hierkn",
            "compressor": "apq4",
            "a0": 3,
            "kf": 4096,
            "tmul": 8,
            "env": {
                "SBANN_INDEX_LOAD": str(webvid / "eng_webvid_kf4096.idx"),
                "SBANN_FLOAT_RERANK": "1",
                "SBANN_FBASE": str(webvid / "base.2.5M.fbin"),
                "SBANN_FQUERY": str(webvid / "query.2k.fbin"),
                "SBANN_IP": "1",
                "SBANN_ROUTE_GAMMA": "0.2",
                "SBANN_GRAPH_BESTFIRST": "1",
                "SBANN_RESIDENT_I8": "1",
                "SBANN_SQ4_NAV": "1",
                "SBANN_SQ4_INT8K": "192",
            },
            "points": [
                point(
                    "r81",
                    2,
                    150,
                    hops=2,
                    beam=24,
                    graph=str(webvid / "hyb_webvid_k64.u32"),
                    edges=48,
                    widths=(14, 16, 20, 32),
                    extra={
                        "SBANN_SEED_IDS_FILE": str(webvid / "qseed_T16_S64.u32"),
                    },
                ),
                point(
                    "r90",
                    2,
                    250,
                    hops=3,
                    beam=48,
                    graph=str(webvid / "hyb_webvid_k64.u32"),
                    edges=64,
                    widths=(14, 16, 20, 32, 48, 64),
                    extra={
                        "SBANN_SEED_IDS_FILE": str(webvid / "qseed_T16_S128.u32"),
                    },
                ),
                point(
                    "r81_k20_t200",
                    2,
                    200,
                    hops=2,
                    beam=24,
                    graph=str(webvid / "hyb_webvid_k64.u32"),
                    edges=48,
                    widths=(20,),
                    extra={
                        "SBANN_SEED_IDS_FILE": str(webvid / "qseed_T16_S64.u32"),
                    },
                ),
                point(
                    "r81_k20_p3",
                    3,
                    150,
                    hops=2,
                    beam=24,
                    graph=str(webvid / "hyb_webvid_k64.u32"),
                    edges=48,
                    widths=(20,),
                    extra={
                        "SBANN_SEED_IDS_FILE": str(webvid / "qseed_T16_S64.u32"),
                    },
                ),
                point(
                    "r81_k20_p3t200",
                    3,
                    200,
                    hops=2,
                    beam=24,
                    graph=str(webvid / "hyb_webvid_k64.u32"),
                    edges=48,
                    widths=(20,),
                    extra={
                        "SBANN_SEED_IDS_FILE": str(webvid / "qseed_T16_S64.u32"),
                    },
                ),
            ],
        },
        "t2i100m": {
            "base": DATA / "base100M.i8bin",
            "query": DATA / "query100K_s100m.i8bin",
            "gt": DATA / "t2i100m_gt.ibin",
            "router": "hierk3",
            "compressor": "apq4",
            "a0": 3,
            "kf": 524288,
            "tmul": 8,
            "env": {
                "SBANN_INDEX_LOAD": str(DATA / "eng_t2i100m_kf524288_em3.idx"),
                "SBANN_FLOAT_RERANK": "1",
                "SBANN_FBASE": str(DATA / "base100M.fbin"),
                "SBANN_FQUERY": str(DATA / "query.public.100K.fbin"),
                "SBANN_IP": "1",
                "SBANN_ROUTE_GAMMA": "0.5",
            },
            "points": [
                point(
                    "r88",
                    40,
                    8000,
                    hops=1,
                    beam=64,
                    graph=str(DATA / "t2i100m_graph_k16.u32"),
                    edges=16,
                ),
                point(
                    "r90",
                    64,
                    8000,
                    hops=1,
                    beam=64,
                    graph=str(DATA / "t2i100m_graph_k16.u32"),
                    edges=16,
                ),
                point(
                    "r92",
                    104,
                    8000,
                    hops=1,
                    beam=64,
                    graph=str(DATA / "t2i100m_graph_k16.u32"),
                    edges=16,
                ),
                point(
                    "frontier",
                    64,
                    8000,
                    hops=1,
                    beam=64,
                    graph=str(DATA / "t2i100m_graph_k16.u32"),
                    edges=16,
                    widths=(16, 20),
                    probes=(40, 52, 64, 80, 104),
                ),
            ],
        },
        "wiki35m": {
            "base": DATA / "wiki35m/base.i8bin",
            "query": DATA / "wiki35m/query.i8bin",
            "gt": DATA / "wiki35m/wiki35m_gt_clean.ibin",
            "router": "hierkn",
            "compressor": "apq4",
            "a0": 2,
            "kf": 65536,
            "tmul": 8,
            "env": {
                "SBANN_INDEX_LOAD": str(
                    DATA / "wiki35m/eng_wiki35m_fp16_65536_dpb4_a2.idx"
                ),
                "SBANN_FLOAT_RERANK": "1",
                "SBANN_FBASE": str(DATA / "wiki35m/base.fbin"),
                "SBANN_FQUERY": str(DATA / "wiki35m/query.fbin"),
                "SBANN_IP": "1",
                "SBANN_ROUTE_FP16": "1",
                "SBANN_RESIDENT_I8": "1",
                "SBANN_RERANK_F16": "1",
                "SBANN_SQ4_FILE": str(DATA / "wiki35m/sq4.side"),
                "SBANN_SQ4_NAV": "1",
                "SBANN_SQ4_INT8K": "256",
                "SBANN_GRAPH_BESTFIRST": "1",
            },
            "points": [
                point(
                    "r966_anchor",
                    24,
                    1200,
                    hops=1,
                    beam=48,
                    graph=str(DATA / "wiki35m/wiki35m_nd_k64s16.u32"),
                    edges=64,
                    widths=(24, 32),
                ),
                point(
                    "r968_candidate",
                    22,
                    1150,
                    hops=1,
                    beam=64,
                    graph=str(DATA / "wiki35m/wiki35m_nd_k64s16.u32"),
                    edges=64,
                    widths=(24, 32),
                ),
            ],
        },
    }


def csv_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in csv_names(value)]


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


def validate(name: str, spec: dict[str, Any], points: list[dict[str, Any]]) -> None:
    paths = [spec["base"], spec["query"], spec["gt"], Path(spec["env"]["SBANN_INDEX_LOAD"])]
    for key in ("SBANN_FBASE", "SBANN_FQUERY"):
        if key in spec["env"]:
            paths.append(Path(spec["env"][key]))
    for item in points:
        if item["graph"]:
            paths.append(Path(item["graph"]))
        for key, value in item["extra"].items():
            if key.endswith("FILE"):
                paths.append(Path(value))
    missing = sorted({str(path) for path in paths if not Path(path).is_file()})
    if missing:
        raise FileNotFoundError(f"{name}: missing inputs:\n" + "\n".join(missing))


def main() -> None:
    registry = dataset_registry()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="t2i1m,cohere1m",
        help=f"comma-separated subset of: {','.join(registry)}",
    )
    parser.add_argument(
        "--points",
        help="optional comma-separated point labels (applied to every dataset)",
    )
    parser.add_argument("--nq", type=int, default=500)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument(
        "--widths",
        help="override every point's KLIST; duplicates enable ABBA orders",
    )
    parser.add_argument("--core", type=int)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--max-load",
        type=float,
        help="wait until one-minute load is at or below this value before each run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "sbann-rs/experiments/law_transfer_screen.json",
    )
    args = parser.parse_args()

    names = csv_names(args.datasets)
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise ValueError(f"unknown datasets: {','.join(unknown)}")
    requested_points = set(csv_names(args.points)) if args.points else None
    width_override = csv_ints(args.widths) if args.widths else None
    if args.core is None:
        core, siblings, busy = choose_core()
    else:
        core, siblings, busy = args.core, [args.core], -1
    print(f"core={core} siblings={siblings} sampled_busy_ticks={busy}", flush=True)

    if args.out.exists():
        report: dict[str, Any] = json.loads(args.out.read_text())
    else:
        report = {"law": "W in {14,20}, incumbent brackets; B ~= 1.5 E", "runs": [], "samples": []}
    completed = {
        (
            run["dataset"],
            run["label"],
            tuple(run.get("probes", [run["p"]])),
            run["floor"],
            run["hops"],
            run["beam"],
            run["edges"],
            run["graph"],
            tuple(run["widths"]),
            run["nq"],
            run["reps"],
        )
        for run in report.get("runs", [])
        if run.get("returncode") == 0 and run.get("matches", 0) > 0
    }

    jobs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for name in names:
        spec = registry[name]
        points = [
            {**item, "widths": width_override or item["widths"]}
            for item in spec["points"]
            if requested_points is None or item["label"] in requested_points
        ]
        if not points:
            print(f"{name}: no selected points; skipping", flush=True)
            continue
        validate(name, spec, points)
        jobs.extend((name, spec, item) for item in points)

    started = time.time()
    for job_index, (name, spec, item) in enumerate(jobs, 1):
        key = (
            name,
            item["label"],
            tuple(item["probes"]),
            item["floor"],
            item["hops"],
            item["beam"],
            item["edges"],
            item["graph"],
            tuple(item["widths"]),
            args.nq,
            args.reps,
        )
        if key in completed:
            print(f"[{job_index}/{len(jobs)}] skip {name}/{item['label']}", flush=True)
            continue

        if args.max_load is not None:
            announced = False
            while os.getloadavg()[0] > args.max_load:
                if not announced:
                    print(
                        f"[{job_index}/{len(jobs)}] waiting: "
                        f"load1={os.getloadavg()[0]:.1f} > {args.max_load:.1f}",
                        flush=True,
                    )
                    announced = True
                time.sleep(30)
        env = os.environ.copy()
        env.update(spec["env"])
        env.update(item["extra"])
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "RAYON_NUM_THREADS": "1",
                "SBANN_KLIST": ",".join(map(str, item["widths"])),
                "SBANN_NQ": str(args.nq),
                "SBANN_PLIST": ",".join(map(str, item["probes"])),
                "SBANN_REPS": str(args.reps),
                "SBANN_TFLOOR": str(item["floor"]),
            }
        )
        if item["graph"]:
            env.update(
                {
                    "SBANN_GRAPH_FILE": item["graph"],
                    "SBANN_GRAPH_HOPS": str(item["hops"]),
                    "SBANN_GRAPH_KEDGE": str(item["edges"]),
                    "SBANN_GRAPH_M": str(item["beam"]),
                }
            )
        argv = [
            "nice",
            "-n",
            "5",
            "taskset",
            "-c",
            str(core),
            str(BIN),
            "run",
            str(spec["base"]),
            str(spec["query"]),
            str(spec["gt"]),
            spec["router"],
            spec["compressor"],
            str(spec["a0"]),
            str(spec["kf"]),
            str(spec["tmul"]),
        ]
        load = os.getloadavg()
        print(
            f"[{job_index}/{len(jobs)}] {name}/{item['label']} "
            f"p{item['probes']} f{item['floor']} widths={item['widths']} "
            f"load1={load[0]:.1f}",
            flush=True,
        )
        t0 = time.monotonic()
        try:
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
            returncode = proc.returncode
            output = proc.stdout
        except subprocess.TimeoutExpired as error:
            returncode = 124
            output = (error.stdout or "") + "\nTIMEOUT"
        wall = time.monotonic() - t0
        matches = list(RESULT_RE.finditer(output))
        run = {
            "dataset": name,
            "label": item["label"],
            "p": item["p"],
            "probes": item["probes"],
            "floor": item["floor"],
            "hops": item["hops"],
            "beam": item["beam"],
            "edges": item["edges"],
            "graph": item["graph"],
            "widths": item["widths"],
            "nq": args.nq,
            "reps": args.reps,
            "core": core,
            "load1": load[0],
            "load5": load[1],
            "load15": load[2],
            "wall_s": wall,
            "returncode": returncode,
            "matches": len(matches),
            "output_tail": output[-5000:],
        }
        report.setdefault("runs", []).append(run)
        for match in matches:
            report.setdefault("samples", []).append(
                {
                    "dataset": name,
                    "label": item["label"],
                    "p": int(match.group("p")),
                    "floor": item["floor"],
                    "hops": item["hops"],
                    "beam": item["beam"],
                    "edges": item["edges"],
                    "cascade_k": int(match.group("k")),
                    "recall": float(match.group("recall")),
                    "qps": float(match.group("qps")),
                    "core": core,
                    "load1": load[0],
                    "run_index": len(report["runs"]) - 1,
                }
            )
        report["frontiers"] = {
            dataset: pareto(
                [row for row in report["samples"] if row["dataset"] == dataset]
            )
            for dataset in sorted({row["dataset"] for row in report["samples"]})
        }
        report["elapsed_s"] = time.time() - started
        write_report(args.out, report)
        if returncode != 0 or not matches:
            raise RuntimeError(
                f"{name}/{item['label']} failed rc={returncode}, matches={len(matches)}\n"
                f"{output[-5000:]}"
            )
        for match in matches:
            print(
                f"  K={int(match.group('k')):3d} "
                f"r={float(match.group('recall')):.4f} "
                f"qps={float(match.group('qps')):.0f}",
                flush=True,
            )

    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
