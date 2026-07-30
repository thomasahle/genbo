#!/usr/bin/env python3
"""Rebuild every plotted genbo frontier under one uniform measurement protocol.

Each compatible (dataset, index, graph, entry-sidecar) family is loaded once.
Its exact operating points are run forward and then in reverse order; each arm
uses the same full query set and best-of-repetitions convention.  Results are
checkpointed after every family so the large 35M/100M runs are resumable.

This is a measurement harness, not a tuner.  Points whose historical knobs were
lost are replaced by small explicitly-labelled reconstruction brackets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abba_bench import choose_core
from sweep_transfer_laws import dataset_registry


ROOT = Path("/home/thomas-ahle/genbo")
DATA = Path("/home/thomas-ahle/big-ann-data")
DEEP = DATA / "deep10m"
WEBVID = Path("/home/thomas-ahle/RoarGraph/data/clip-webvid-2.5M")
BIN = ROOT / "sbann-rs/target/release/sbann"
DEFAULT_OUT = ROOT / "sbann-rs/experiments/uniform_frontier_results.json"
DEFAULT_LOG = ROOT / "sbann-rs/experiments/uniform_frontier.log"

SEARCH_RE = re.compile(
    r"p=\s*(?P<p>\d+).*?K=(?P<k>\d+):\s+"
    r"recall@10=(?P<recall>[0-9.]+)\s+QPS=(?P<qps>[0-9.]+).*?"
    r"\[plan=(?P<label>\S+)\s+M=(?P<beam>\d+)\s+ke=(?P<edges>\d+)\s+"
    r"h=(?P<hops>\d+)\s+bf=(?P<bestfirst>\d+)\s+tf=(?P<floor>\d+).*?"
    r"sq4k=(?P<sq4k>\d+)\]"
)
WALK_RE = re.compile(
    r"RW L=\s*(?P<walk_l>\d+).*?recall@10=(?P<recall>[0-9.]+)\s+"
    r"QPS=(?P<qps>[0-9.]+).*?"
    r"\[plan=(?P<label>\S+)\s+R=(?P<rounds>\d+)\s+B=(?P<frontier>\d+)\]"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Point:
    label: str
    p: int
    floor: int
    width: int
    beam: int = 0
    edges: int = 0
    hops: int = 0
    bestfirst: bool = False
    tmul: int = 8
    portal_keep: int = 1
    sq4k: int = 256

    def encoded(self, ordered_label: str) -> str:
        return ":".join(
            map(
                str,
                (
                    ordered_label,
                    self.p,
                    self.tmul,
                    self.floor,
                    self.width,
                    self.beam,
                    self.edges,
                    self.hops,
                    int(self.bestfirst),
                    self.portal_keep,
                    self.sq4k,
                ),
            )
        )


@dataclass(frozen=True)
class WalkPoint:
    label: str
    walk_l: int
    rounds: int
    frontier: int

    def encoded(self, ordered_label: str) -> str:
        return f"{ordered_label}:{self.walk_l}:{self.rounds}:{self.frontier}"


@dataclass
class Group:
    dataset: str
    family: str
    base: Path
    query: Path
    gt: Path
    router: str
    compressor: str
    a0: int
    kf: int
    tmul: int
    env: dict[str, str]
    points: list[Point] = field(default_factory=list)
    walk_points: list[WalkPoint] = field(default_factory=list)
    nq: int = 2000
    min_available_gib: float = 40.0

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.family}"

    def argv(self, core: int) -> list[str]:
        return [
            "nice",
            "-n",
            "5",
            "taskset",
            "-c",
            str(core),
            str(BIN),
            "run",
            str(self.base),
            str(self.query),
            str(self.gt),
            self.router,
            self.compressor,
            str(self.a0),
            str(self.kf),
            str(self.tmul),
        ]


def point(
    label: str,
    p: int,
    floor: int,
    width: int,
    beam: int = 0,
    edges: int = 0,
    hops: int = 0,
    bestfirst: bool = False,
    sq4k: int = 256,
) -> Point:
    return Point(label, p, floor, width, beam, edges, hops, bestfirst, 8, 1, sq4k)


def from_spec(
    dataset: str,
    family: str,
    spec: dict[str, Any],
    points: list[Point],
    *,
    env: dict[str, str] | None = None,
    nq: int = 2000,
    min_available_gib: float = 40.0,
) -> Group:
    merged = {str(key): str(value) for key, value in spec["env"].items()}
    if env:
        merged.update(env)
    return Group(
        dataset,
        family,
        Path(spec["base"]),
        Path(spec["query"]),
        Path(spec["gt"]),
        spec["router"],
        spec["compressor"],
        spec["a0"],
        spec["kf"],
        spec["tmul"],
        merged,
        points=points,
        nq=nq,
        min_available_gib=min_available_gib,
    )


def manual_group(
    dataset: str,
    family: str,
    *,
    base: Path,
    query: Path,
    gt: Path,
    index: Path,
    fbase: Path,
    fquery: Path,
    graph: Path | None,
    router: str,
    a0: int,
    kf: int,
    points: list[Point],
    ip: bool = False,
    gamma: str | None = None,
    extra: dict[str, str] | None = None,
    nq: int = 2000,
    min_available_gib: float = 40.0,
) -> Group:
    env = {
        "SBANN_INDEX_LOAD": str(index),
        "SBANN_FLOAT_RERANK": "1",
        "SBANN_FBASE": str(fbase),
        "SBANN_FQUERY": str(fquery),
        "SBANN_BATCHSCAN": "0",
    }
    if graph is not None:
        env["SBANN_GRAPH_FILE"] = str(graph)
    if ip:
        env["SBANN_IP"] = "1"
    if gamma is not None:
        env["SBANN_ROUTE_GAMMA"] = gamma
    if extra:
        env.update(extra)
    return Group(
        dataset,
        family,
        base,
        query,
        gt,
        router,
        "apq4",
        a0,
        kf,
        8,
        env,
        points=points,
        nq=nq,
        min_available_gib=min_available_gib,
    )


def groups() -> list[Group]:
    registry = dataset_registry()
    result: list[Group] = []

    result.append(
        from_spec(
            "cohere1m",
            "cascade",
            registry["cohere1m"],
            [
                point("r9732", 96, 3000, 20),
                point("r9814", 128, 4000, 16),
                point("r9900", 192, 6000, 20),
                point("r9925", 256, 8000, 16),
                point("r9967", 384, 12000, 20),
                point("r9971", 512, 16000, 16),
            ],
        )
    )
    result.append(
        from_spec(
            "cohere10m",
            "hybrid",
            registry["cohere10m"],
            [
                point("r9461", 16, 500, 16, 24, 16, 1),
                point("r9707", 32, 1000, 16, 24, 16, 2),
                point("r9832", 64, 2000, 16, 24, 16, 3),
                point("r9871", 96, 3000, 16, 24, 16, 3),
                point("r9904", 128, 4000, 16, 24, 16, 3),
                point("r9912", 128, 4000, 20, 24, 16, 3),
            ],
            env={"SBANN_GRAPH_FILE": str(DATA / "cohere/nd10m_r24_seed.u32")},
        )
    )

    result.append(
        from_spec(
            "mst30m",
            "cascade",
            registry["mst30m"],
            [
                point("r7268", 1, 150, 32, 24, 32, 3, True),
                point("r7900", 2, 250, 32, 24, 32, 3, True),
                point("r8471", 4, 400, 32, 24, 32, 3, True),
                point("r9109", 16, 700, 32, 24, 32, 3, True),
                point("r9483", 32, 1000, 32, 32, 32, 4, True),
                point("r9686", 64, 1800, 16, 48, 32, 5, True),
                point("r9819", 160, 3500, 16, 64, 32, 6, True),
                point("r9883", 384, 7000, 16, 64, 32, 8, True),
            ],
            env={
                "SBANN_GRAPH_FILE": str(
                    DATA / "streaming/vamana_mst_R32.u32"
                )
            },
            min_available_gib=90.0,
        )
    )

    result.append(
        from_spec(
            "wiki35m",
            "cascade",
            registry["wiki35m"],
            [
                point("r9328", 8, 250, 12, 48, 64, 1, True, 256),
                point("r9354", 10, 250, 16, 32, 64, 1, True, 256),
                point("r9421", 12, 250, 12, 32, 64, 1, True, 256),
                point("r9476", 14, 250, 16, 32, 64, 1, True, 256),
                point("r9505", 12, 250, 12, 64, 64, 1, True, 256),
                point("r9511", 16, 250, 12, 32, 64, 1, True, 256),
                point("r9531", 14, 250, 16, 48, 64, 1, True, 256),
                point("r9566", 16, 250, 12, 48, 64, 1, True, 256),
                point("r9590", 16, 250, 12, 64, 64, 1, True, 256),
                point("r9609", 16, 500, 16, 64, 64, 1, True, 256),
                point("r9677", 22, 1150, 32, 64, 64, 1, True, 256),
                point("r9723", 32, 1500, 32, 48, 64, 1, True, 512),
                point("r9860", 48, 2500, 64, 48, 64, 2, True, 512),
                point("r9925", 96, 4000, 64, 48, 64, 2, True, 512),
            ],
            env={
                "SBANN_GRAPH_FILE": str(DATA / "wiki35m/wiki35m_nd_k64s16.u32"),
                "SBANN_RERANK_F16_FILE": str(DATA / "wiki35m/base_f16.side"),
            },
            nq=1000,
            min_available_gib=150.0,
        )
    )

    deep_common = {
        "SBANN_GRAPH_BASE": str(DEEP / "graph_layout_gate.cellpair.aligned64.i8bin"),
        "SBANN_GRAPH_BASE_OFFSET": "64",
        "SBANN_GRAPH_RANK": str(DEEP / "graph_layout_gate.cellpair.u32"),
        "SBANN_RESIDENT_I8": "1",
    }
    result.append(
        manual_group(
            "deep10m",
            "cascade",
            base=DEEP / "base.10M.i8bin",
            query=DEEP / "query2k.i8bin",
            gt=DEEP / "deep10m_gt.ibin",
            index=DEEP / "eng_deep10m_kf65536.idx",
            fbase=DEEP / "base.10M.fbin",
            fquery=DEEP / "query2k.fbin",
            graph=DEEP / "graph_layout_gate.cellpair.graph.u32",
            router="hierkn",
            a0=2,
            kf=65536,
            points=[
                point("r9265", 20, 192, 16, 16, 24, 2, True),
                point("r9426", 24, 320, 32, 12, 32, 2, True),
                point("r9603", 28, 256, 48, 24, 32, 2, True),
                point("r9720", 32, 450, 32, 24, 32, 3, True),
                point("r9814", 64, 450, 32, 24, 32, 3, True),
                point("r9907", 128, 450, 32, 24, 32, 3, True),
                point("r9940", 192, 450, 64, 24, 32, 3, True),
                point("r9948", 224, 450, 64, 24, 32, 3, True),
                point("r9951", 256, 450, 64, 24, 32, 3, True),
                point("r9959", 320, 450, 64, 24, 32, 3, True),
                point("r9967", 384, 450, 64, 24, 32, 3, True),
                point("r9976", 512, 450, 64, 24, 32, 3, True),
                point("r9978", 608, 450, 64, 24, 32, 3, True),
                point("tail704", 704, 450, 64, 24, 32, 3, True),
                point("tail768", 768, 450, 64, 24, 32, 3, True),
                point("tail896", 896, 450, 64, 24, 32, 3, True),
                point("tail1024", 1024, 450, 64, 24, 32, 3, True),
            ],
            extra={
                **deep_common,
                "SBANN_RERANK_F16": "1",
                "SBANN_RERANK_F16_REFINE": "12",
            },
        )
    )
    deep_walk = manual_group(
        "deep10m",
        "walk",
        base=DEEP / "base.10M.i8bin",
        query=DEEP / "query2k.i8bin",
        gt=DEEP / "deep10m_gt.ibin",
        index=DEEP / "eng_deep10m_kf65536.idx",
        fbase=DEEP / "base.10M.fbin",
        fquery=DEEP / "query2k.fbin",
        graph=DEEP / "graph_layout_gate.cellpair.graph.u32",
        router="hierkn",
        a0=2,
        kf=65536,
        points=[],
        extra={
            **deep_common,
            "SBANN_CENTROID_GRAPH": str(DEEP / "deep10m_centroid_vamana16.u32"),
            "SBANN_CENTROID_LANDMARKS": str(DEEP / "deep10m_centroid_landmarks64.u32"),
            "SBANN_CENTROID_GRAPH_K": "16",
            "SBANN_CENTROID_GRAPH_EF": "8",
            "SBANN_PORTAL_FILE": str(DEEP / "deep10m_portals16.side"),
            "SBANN_PORTAL_SQ4_FILE": str(DEEP / "deep10m_portals16.sq4p64"),
            "SBANN_ROAR_PORTAL_CELLS": "8",
            "SBANN_ROAR_PORTAL_BUCKETS": "1",
            "SBANN_ROAR_PORTAL_ROWS": "1",
            "SBANN_GRAPH_M": "32",
            "SBANN_GRAPH_KEDGE": "32",
            "SBANN_IP": "1",
        },
    )
    deep_walk.walk_points = [
        WalkPoint("r8041", 25, 0, 16),
        WalkPoint("r8334", 14, 4, 11),
        WalkPoint("r8538", 16, 4, 15),
        WalkPoint("r8682", 16, 4, 19),
        WalkPoint("r8842", 14, 8, 10),
        WalkPoint("r8941", 20, 9, 10),
        WalkPoint("r9032", 14, 8, 14),
    ]
    result.append(deep_walk)

    result.append(
        from_spec(
            "t2i100m",
            "nd16",
            registry["t2i100m"],
            [
                point("r8796", 40, 8000, 20, 64, 16, 1),
                point("r8951", 52, 8000, 20, 64, 16, 1),
                point("r9049", 64, 8000, 20, 64, 16, 1),
                point("r9152", 80, 8000, 20, 64, 16, 1),
                point("r9233", 104, 8000, 20, 64, 16, 1),
            ],
            env={"SBANN_GRAPH_FILE": str(DATA / "t2i100m_graph_k16.u32")},
            min_available_gib=110.0,
        )
    )

    t2i1_common = dict(
        dataset="t2i1m",
        base=DATA / "base1M.i8bin",
        query=DATA / "query100K_s1m.i8bin",
        gt=DATA / "t2i1m_gt.ibin",
        fbase=DATA / "base1M.fbin",
        fquery=DATA / "query.public.100K.fbin",
        router="hierk",
        a0=3,
        ip=True,
        gamma="0.5",
    )
    result.extend(
        [
            manual_group(
                "t2i1m",
                "kf2048-hybrid",
                **{key: value for key, value in t2i1_common.items() if key != "dataset"},
                index=DATA / "eng_t2i1m_kf2048_c128_b32_em2.idx",
                graph=DATA / "hyb_t2i1m_k32.u32",
                kf=2048,
                points=[
                    # Historical loose rows predate exact-plan logging.  This
                    # reconstruction varies only the knobs that the first
                    # bracket showed actually change recall: floor and graph
                    # fanout (scan width did not).
                    point("p1h1m24f250", 1, 250, 16, 24, 32, 1),
                    point("p1h1m32f250", 1, 250, 16, 32, 32, 1),
                    point("p1h1m40f250", 1, 250, 16, 40, 32, 1),
                    point("p1h1m48f250", 1, 250, 16, 48, 32, 1),
                    point("p1h1m32f400", 1, 400, 16, 32, 32, 1),
                    point("p2m24f500", 2, 500, 16, 24, 32, 2),
                    point("p2m24f800", 2, 800, 16, 24, 32, 2),
                    point("p2m24f1200", 2, 1200, 16, 24, 32, 2),
                    point("p2m32f500", 2, 500, 16, 32, 32, 2),
                    point("p3m24f900", 3, 900, 16, 24, 32, 2),
                    point("p3m32f900", 3, 900, 16, 32, 32, 2),
                    point("p4m24f700", 4, 700, 16, 24, 32, 2),
                    point("p4m24f1100", 4, 1100, 16, 24, 32, 2),
                    point("p4m24f1600", 4, 1600, 16, 24, 32, 2),
                    point("p4m32f700", 4, 700, 16, 32, 32, 2),
                    point("p4m32f1200", 4, 1200, 16, 32, 32, 2),
                ],
            ),
            manual_group(
                "t2i1m",
                "kf2048-co16",
                **{key: value for key, value in t2i1_common.items() if key != "dataset"},
                index=DATA / "eng_t2i1m_kf2048_c128_b32_em2.idx",
                graph=DATA / "co_t2i1m_k16.u32",
                kf=2048,
                points=[
                    point("p1", 1, 400, 16, 24, 16, 2),
                    point("p2", 2, 500, 16, 24, 16, 2),
                    point("p3", 3, 600, 16, 24, 16, 2),
                    point("p4", 4, 700, 16, 24, 16, 2),
                    point("p1m16f250w16", 1, 250, 16, 16, 16, 2),
                    point("p1m16f250w32", 1, 250, 32, 16, 16, 2),
                    point("p2m16f400w16", 2, 400, 16, 16, 16, 2),
                    point("p2m16f400w32", 2, 400, 32, 16, 16, 2),
                    point("p3m16f500w16", 3, 500, 16, 16, 16, 2),
                    point("p3m16f500w32", 3, 500, 32, 16, 16, 2),
                    point("p4m16f600w16", 4, 600, 16, 16, 16, 2),
                    point("p4m16f600w32", 4, 600, 32, 16, 16, 2),
                ],
            ),
            manual_group(
                "t2i1m",
                "kf4096-co16",
                **{key: value for key, value in t2i1_common.items() if key != "dataset"},
                index=DATA / "eng_t2i1m_kf4096_c256_b48_em2.idx",
                graph=DATA / "co_t2i1m_k16.u32",
                kf=4096,
                points=[
                    point("p1", 1, 400, 16, 24, 16, 2),
                    point("p2", 2, 500, 16, 24, 16, 2),
                    point("p3", 3, 600, 16, 24, 16, 2),
                    point("p4", 4, 700, 16, 24, 16, 2),
                    point("p2m12f400", 2, 400, 16, 12, 16, 2),
                    point("p2m14f400", 2, 400, 16, 14, 16, 2),
                    point("p2m16f400", 2, 400, 16, 16, 16, 2),
                    point("p3m16f500", 3, 500, 16, 16, 16, 2),
                    point("p4m16f600", 4, 600, 16, 16, 16, 2),
                    point("p4m18f600", 4, 600, 16, 18, 16, 2),
                    point("p4m20f600", 4, 600, 16, 20, 16, 2),
                ],
            ),
            manual_group(
                "t2i1m",
                "kf4096-hybrid",
                **{key: value for key, value in t2i1_common.items() if key != "dataset"},
                index=DATA / "eng_t2i1m_kf4096_c256_b48_em2.idx",
                graph=DATA / "hyb_t2i1m_k32.u32",
                kf=4096,
                points=[
                    point("p2m24f500", 2, 500, 16, 24, 32, 2),
                    point("p2m24f800", 2, 800, 16, 24, 32, 2),
                    point("p2m24f1200", 2, 1200, 16, 24, 32, 2),
                    point("p2m32f500", 2, 500, 16, 32, 32, 2),
                    point("p2m32f900", 2, 900, 16, 32, 32, 2),
                    point("p4m24f700", 4, 700, 16, 24, 32, 2),
                    point("p4m24f1100", 4, 1100, 16, 24, 32, 2),
                    point("p4m24f1600", 4, 1600, 16, 24, 32, 2),
                    point("p4m32f700", 4, 700, 16, 32, 32, 2),
                    point("p4m32f1100", 4, 1100, 16, 32, 32, 2),
                    point("p4m32f1600", 4, 1600, 16, 32, 32, 2),
                    point("p8m32f800", 8, 800, 16, 32, 32, 2),
                    point("p8m32f1500", 8, 1500, 16, 32, 32, 2),
                    point("p8m32f2500", 8, 2500, 16, 32, 32, 2),
                ],
            ),
            manual_group(
                "t2i1m",
                "kf16384-nd16",
                **{key: value for key, value in t2i1_common.items() if key != "dataset"},
                index=DATA / "eng_scale_1m_kf16384.idx",
                graph=DATA / "t2i1m_graph_k16.u32",
                kf=16384,
                points=[
                    point("p40", 40, 1000, 16, 32, 16, 2),
                    point("p56", 56, 1500, 16, 32, 16, 2),
                    point("p80", 80, 2000, 16, 32, 16, 2),
                    point("p128", 128, 3000, 16, 32, 16, 3),
                ],
            ),
            manual_group(
                "t2i1m",
                "kf16384-hybrid",
                **{key: value for key, value in t2i1_common.items() if key != "dataset"},
                index=DATA / "eng_scale_1m_kf16384.idx",
                graph=DATA / "hyb_t2i1m_k32.u32",
                kf=16384,
                points=[
                    point("p40", 40, 1500, 20, 48, 32, 3),
                    point("p128", 128, 3000, 20, 48, 32, 3),
                ],
            ),
        ]
    )

    t2i10_common = dict(
        dataset="t2i10m",
        base=DATA / "base10M.i8bin",
        query=DATA / "query100K.i8bin",
        gt=DATA / "t2i10m_gt.ibin",
        index=DATA / "eng_t2i10m_kf65536_c4096_b128_a3_em2.idx",
        fbase=DATA / "base10M.fbin",
        fquery=DATA / "query.public.100K.fbin",
        router="hierk",
        a0=3,
        kf=65536,
        ip=True,
        gamma="0.5",
    )
    result.extend(
        [
            manual_group(
                "t2i10m",
                "co16",
                **{key: value for key, value in t2i10_common.items() if key != "dataset"},
                graph=DATA / "co_t2i10m_k16.u32",
                points=[
                    point("p8", 8, 800, 16, 32, 16, 2),
                    point("p16", 16, 1000, 16, 32, 16, 2),
                    point("p40", 40, 1000, 16, 32, 16, 2),
                ],
            ),
            manual_group(
                "t2i10m",
                "nd16",
                **{key: value for key, value in t2i10_common.items() if key != "dataset"},
                graph=DATA / "t2i10m_graph_k16.u32",
                points=[
                    point("p16", 16, 1000, 16, 32, 16, 2),
                    point("p24", 24, 1000, 16, 32, 16, 2),
                    point("p32", 32, 1000, 16, 32, 16, 2),
                    point("p56", 56, 1000, 16, 32, 16, 2),
                    point("p128", 128, 2000, 16, 32, 16, 3),
                    point("p192", 192, 3000, 16, 32, 16, 3),
                ],
            ),
            manual_group(
                "t2i10m",
                "nd32",
                **{key: value for key, value in t2i10_common.items() if key != "dataset"},
                graph=DATA / "nd_t2i10m_k32.u32",
                points=[
                    point("p192", 192, 3000, 16, 48, 32, 3),
                    point("p256", 256, 4000, 16, 48, 32, 4),
                ],
            ),
            manual_group(
                "t2i10m",
                "hybrid",
                **{key: value for key, value in t2i10_common.items() if key != "dataset"},
                graph=DATA / "hyb_t2i10m_k32.u32",
                points=[
                    point("p40", 40, 1000, 20, 48, 32, 2),
                    point("p192", 192, 3000, 20, 48, 32, 3),
                    point("p384", 384, 6000, 16, 48, 32, 3),
                    point("bf256m96", 256, 8000, 64, 96, 32, 6, True),
                    point("bf384m96", 384, 8000, 64, 96, 32, 4, True),
                    point("bf384m128", 384, 10000, 64, 128, 32, 5, True),
                    point("bf512m128", 512, 12000, 64, 128, 32, 5, True),
                ],
            ),
        ]
    )

    # P374: wide-graph (nd k64s16, KEDGE=64) family for t2i-1M — paired probe
    # (t2i1m_wide_probe.log) found one dominating point (0.9876@1879 vs plotted
    # 0.9859@1815 at p80/h2/M32); brackets around it for the envelope.
    t2i1m_wide_points = [
        point("wk64p64", 64, 1600, 32, 32, 64, 2, True, 256),
        point("wk64p80", 80, 2000, 32, 32, 64, 2, True, 256),
        point("wk64p96", 96, 2400, 32, 32, 64, 2, True, 256),
    ]
    result.append(
        from_spec(
            "t2i1m",
            "ndk64",
            registry["t2i1m"],
            t2i1m_wide_points,
            env={
                "SBANN_GRAPH_BESTFIRST": "1",
                "SBANN_GRAPH_FILE": "/home/thomas-ahle/big-ann-data/nd_t2i1m_k64s16.u32",
            },
        )
    )

    webvid_base_env = {
        "SBANN_GRAPH_BESTFIRST": "1",
        "SBANN_RESIDENT_I8": "1",
        "SBANN_SQ4_NAV": "1",
    }
    webvid_specs = [
        (
            "seed64-k64",
            WEBVID / "hyb_webvid_k64.u32",
            WEBVID / "qseed_T16_S64.u32",
            [point("r8062", 2, 150, 32, 24, 48, 2, True, 192)],
        ),
        (
            "seed128-k64",
            WEBVID / "hyb_webvid_k64.u32",
            WEBVID / "qseed_T16_S128.u32",
            [
                point("r9019", 2, 250, 32, 48, 64, 3, True, 192),
                point("r9024", 2, 250, 48, 48, 64, 3, True, 192),
            ],
        ),
        (
            "seed256-k64",
            WEBVID / "hyb_webvid_k64.u32",
            WEBVID / "qseed_dense_T32_S256.u32",
            [
                point("r9121", 2, 250, 48, 48, 64, 3, True, 192),
                point("r9322", 2, 300, 64, 64, 64, 4, True, 192),
            ],
        ),
        (
            "seed512-k32",
            WEBVID / "hyb_webvid_k32.u32",
            WEBVID / "qseed_dense_T48_S512.u32",
            [
                point("r9188", 2, 300, 64, 64, 32, 4, True, 384),
                point("r9383", 4, 400, 128, 80, 32, 6, True, 384),
                point("r9415", 8, 700, 128, 80, 32, 6, True, 384),
                point("r9523", 16, 1000, 128, 96, 32, 6, True, 384),
                point("r9626", 32, 1500, 128, 96, 32, 8, True, 384),
                point("r9677", 48, 2000, 192, 112, 32, 8, True, 384),
            ],
        ),
        (
            "seed768-k32",
            WEBVID / "hyb_webvid_k32.u32",
            WEBVID / "qseed_dense_T64_S768.u32",
            [
                point("p64f2000", 64, 2000, 192, 112, 32, 8, True, 384),
                point("p64f2500", 64, 2500, 192, 112, 32, 8, True, 384),
                point("p64f3000", 64, 3000, 192, 112, 32, 8, True, 384),
                point("p96f2500", 96, 2500, 256, 128, 32, 8, True, 384),
                point("p96f3000", 96, 3000, 256, 128, 32, 8, True, 384),
                point("p96f4000", 96, 4000, 256, 128, 32, 8, True, 384),
                point("p128f3000", 128, 3000, 256, 128, 32, 10, True, 384),
                point("p128f4000", 128, 4000, 256, 128, 32, 10, True, 384),
                point("p128f5000", 128, 5000, 256, 128, 32, 10, True, 384),
            ],
        ),
        # P373: wide-graph (k64, KEDGE=64) arms for the mid ladder — paired quiet ABBA
        # (webvid_wide_mid.log) showed they dominate the k32 rows 1.18-1.30x at fixed
        # recall (0.9409@1645 / 0.9556@1037 / 0.9657@777 vs the r9383..r9626 rungs).
        (
            "seed512-k64",
            WEBVID / "hyb_webvid_k64.u32",
            WEBVID / "qseed_dense_T48_S512.u32",
            [
                point("w9409", 2, 300, 64, 64, 64, 4, True, 384),
                point("w9556", 4, 400, 128, 80, 64, 6, True, 384),
                point("w9657", 16, 1000, 128, 96, 64, 6, True, 384),
                point("w48f2000", 48, 2000, 192, 112, 64, 8, True, 384),
            ],
        ),
        (
            "seed768-k64",
            WEBVID / "hyb_webvid_k64.u32",
            WEBVID / "qseed_dense_T64_S768.u32",
            [
                point("w64f2500", 64, 2500, 192, 112, 64, 8, True, 384),
                point("w96f3000", 96, 3000, 256, 128, 64, 8, True, 384),
                point("w128f4000", 128, 4000, 256, 128, 64, 10, True, 384),
            ],
        ),
    ]
    for family, graph, seed, plans in webvid_specs:
        result.append(
            from_spec(
                "webvid",
                family,
                registry["webvid"],
                plans,
                env={
                    **webvid_base_env,
                    "SBANN_GRAPH_FILE": str(graph),
                    "SBANN_SEED_IDS_FILE": str(seed),
                },
            )
        )
    return result


def available_gib() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0])
    return values["MemAvailable"] / 1024 / 1024


def wait_for_resources(max_load: float, min_available_gib: float, log) -> None:
    stable = 0
    while stable < 3:
        load1 = os.getloadavg()[0]
        available = available_gib()
        good = load1 <= max_load and available >= min_available_gib
        stable = stable + 1 if good else 0
        message = (
            f"[{now()}] gate load={load1:.2f}/{max_load:.2f} "
            f"available={available:.1f}/{min_available_gib:.1f}GiB stable={stable}/3"
        )
        print(message, flush=True)
        log.write(message + "\n")
        log.flush()
        if stable < 3:
            time.sleep(30)


def required_paths(group: Group) -> list[Path]:
    result = [group.base, group.query, group.gt, BIN]
    for key, value in group.env.items():
        if key.endswith("FILE") or key in {
            "SBANN_INDEX_LOAD",
            "SBANN_FBASE",
            "SBANN_FQUERY",
            "SBANN_GRAPH_BASE",
            "SBANN_GRAPH_RANK",
            "SBANN_CENTROID_GRAPH",
            "SBANN_CENTROID_LANDMARKS",
        }:
            result.append(Path(value))
    return result


def write_checkpoint(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def pareto(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = sorted(samples, key=lambda row: (-row["recall"], -row["qps"]))
    result: list[dict[str, Any]] = []
    best_qps = -1.0
    for row in candidates:
        if row["qps"] > best_qps:
            result.append(row)
            best_qps = row["qps"]
    return sorted(result, key=lambda row: row["recall"])


def latest_accepted_samples(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Use one coherent accepted run per family; never stitch family rows across windows."""
    latest_run: dict[str, int] = {}
    for run_index, run in enumerate(report.get("runs", [])):
        if run.get("accepted", True):
            latest_run[run["group"]] = run_index
    return [
        sample
        for sample in report.get("samples", [])
        if sample.get("accepted", True)
        and sample["run_index"] == latest_run.get(sample["group"])
    ]


def mirror_diagnostics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the forward/reverse occurrence of every semantic plan label."""
    by_label: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_label.setdefault(sample["label"], []).append(sample)
    rows = []
    for label, pair in sorted(by_label.items()):
        if len(pair) != 2:
            rows.append({"label": label, "count": len(pair), "spread": None})
            continue
        low, high = sorted((pair[0]["qps"], pair[1]["qps"]))
        rows.append(
            {
                "label": label,
                "count": 2,
                "recall_equal": pair[0]["recall"] == pair[1]["recall"],
                "qps_low": low,
                "qps_high": high,
                "spread": (high - low) / high if high else 0.0,
            }
        )
    valid_spreads = [row["spread"] for row in rows if row["spread"] is not None]
    return {
        "pairs": rows,
        "complete": all(row["count"] == 2 for row in rows),
        "recall_equal": all(row.get("recall_equal", False) for row in rows),
        "max_qps_spread": max(valid_spreads, default=1.0),
    }


def parse_samples(group: Group, output: str, run_index: int, load1: float) -> list[dict[str, Any]]:
    samples = []
    regex = WALK_RE if group.walk_points else SEARCH_RE
    for match in regex.finditer(output):
        values = match.groupdict()
        ordered_label = values["label"]
        direction, _, label = ordered_label.partition("_")
        if direction.startswith("W"):
            continue
        sample: dict[str, Any] = {
            "dataset": group.dataset,
            "family": group.family,
            "group": group.key,
            "label": label,
            "ordered_label": ordered_label,
            "direction": direction,
            "recall": float(values["recall"]),
            "qps": float(values["qps"]),
            "load1": load1,
            "run_index": run_index,
        }
        for key, value in values.items():
            if key not in {"label", "recall", "qps"}:
                sample[key] = int(value)
        samples.append(sample)
    return samples


def main() -> int:
    registry = groups()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        help="comma-separated dataset names; default is every dataset",
    )
    parser.add_argument(
        "--groups",
        help="comma-separated exact dataset/family keys",
    )
    parser.add_argument("--nq", type=int, help="override full-query count for smoke tests")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--core", type=int)
    parser.add_argument("--max-load", type=float, default=18.0)
    parser.add_argument(
        "--mirror-tolerance",
        type=float,
        default=0.05,
        help="maximum accepted forward/reverse QPS spread per semantic point",
    )
    parser.add_argument("--timeout", type=float, default=14400.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun selected groups even when an accepted checkpoint exists",
    )
    args = parser.parse_args()

    selected_datasets = (
        {value for value in args.datasets.split(",") if value}
        if args.datasets
        else None
    )
    selected_groups = (
        {value for value in args.groups.split(",") if value}
        if args.groups
        else None
    )
    jobs = [
        group
        for group in registry
        if (selected_datasets is None or group.dataset in selected_datasets)
        and (selected_groups is None or group.key in selected_groups)
    ]
    if not jobs:
        raise SystemExit("no groups selected")

    missing = {
        group.key: [str(path) for path in required_paths(group) if not path.is_file()]
        for group in jobs
    }
    missing = {key: value for key, value in missing.items() if value}
    if missing:
        raise FileNotFoundError(json.dumps(missing, indent=2))

    if args.core is None:
        core, siblings, busy = choose_core()
    else:
        core, siblings, busy = args.core, [args.core], -1
    print(f"core={core} siblings={siblings} sampled_busy_ticks={busy}", flush=True)

    if args.output.exists():
        report = json.loads(args.output.read_text())
    else:
        report = {
            "protocol": {
                "queries": "full official query set",
                "threads": 1,
                "repetitions": args.reps,
                "order": "forward then reverse within one loaded process",
                "timing": "best of repetitions; best of mirrored occurrences selected",
            },
            "started": now(),
            "runs": [],
            "samples": [],
        }
    completed = (
        set()
        if args.force
        else {
            run["group"]
            for run in report.get("runs", [])
            if run.get("returncode") == 0
            and run.get("matches", 0) > 0
            and run.get("accepted", True)
        }
    )

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", buffering=1) as log:
        for index, group in enumerate(jobs, 1):
            if group.key in completed:
                print(f"[{index}/{len(jobs)}] skip completed {group.key}", flush=True)
                continue
            if not args.skip_gate:
                wait_for_resources(args.max_load, group.min_available_gib, log)

            nq = args.nq or group.nq
            forward: list[str]
            reverse: list[str]
            if group.walk_points:
                warmup = group.walk_points[-1].encoded("W00_warmup")
                forward = [
                    item.encoded(f"F{i:02d}_{item.label}")
                    for i, item in enumerate(group.walk_points)
                ]
                reverse = [
                    item.encoded(f"R{i:02d}_{item.label}")
                    for i, item in enumerate(reversed(group.walk_points))
                ]
                plan_env = {
                    "SBANN_ROAR_PLAN": ";".join([warmup] + forward + reverse)
                }
            else:
                warmup = group.points[-1].encoded("W00_warmup")
                forward = [
                    item.encoded(f"F{i:02d}_{item.label}")
                    for i, item in enumerate(group.points)
                ]
                reverse = [
                    item.encoded(f"R{i:02d}_{item.label}")
                    for i, item in enumerate(reversed(group.points))
                ]
                plan_env = {
                    "SBANN_SEARCH_PLAN": ";".join([warmup] + forward + reverse)
                }

            env = os.environ.copy()
            env.update(group.env)
            env.update(plan_env)
            env.update(
                {
                    "OMP_NUM_THREADS": "1",
                    "RAYON_NUM_THREADS": "1",
                    "SBANN_NQ": str(nq),
                    "SBANN_REPS": str(args.reps),
                }
            )
            load = os.getloadavg()
            header = (
                f"=== [{index}/{len(jobs)}] {group.key} start {now()} "
                f"nq={nq} reps={args.reps} points={len(forward)} "
                f"load={load[0]:.2f} available={available_gib():.1f}GiB ==="
            )
            print(header, flush=True)
            log.write(header + "\n")
            started = time.monotonic()
            process = subprocess.Popen(
                group.argv(core),
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            output_lines = []
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                output_lines.append(line)
            returncode = process.wait()
            output = "".join(output_lines)
            samples = parse_samples(
                group, output, len(report.get("runs", [])), load[0]
            )
            mirror = mirror_diagnostics(samples)
            structurally_complete = (
                returncode == 0
                and len(samples) == 2 * len(forward)
                and mirror["complete"]
                and mirror["recall_equal"]
            )
            accepted = (
                structurally_complete
                and mirror["max_qps_spread"] <= args.mirror_tolerance
            )
            for sample in samples:
                sample["accepted"] = accepted
            run = {
                "dataset": group.dataset,
                "family": group.family,
                "group": group.key,
                "nq": nq,
                "reps": args.reps,
                "core": core,
                "load1": load[0],
                "load5": load[1],
                "load15": load[2],
                "available_gib": available_gib(),
                "wall_s": time.monotonic() - started,
                "returncode": returncode,
                "matches": len(samples),
                "expected_matches": 2 * len(forward),
                "mirror": mirror,
                "mirror_tolerance": args.mirror_tolerance,
                "accepted": accepted,
                "output_tail": output[-8000:],
                "finished": now(),
            }
            report.setdefault("runs", []).append(run)
            report.setdefault("samples", []).extend(samples)
            coherent_samples = latest_accepted_samples(report)
            report["frontiers"] = {
                dataset: pareto(
                    [
                        row
                        for row in coherent_samples
                        if row["dataset"] == dataset
                    ]
                )
                for dataset in sorted(
                    {row["dataset"] for row in coherent_samples}
                )
            }
            report["updated"] = now()
            write_checkpoint(args.output, report)
            footer = (
                f"=== {group.key} done rc={returncode} "
                f"matches={len(samples)}/{2 * len(forward)} "
                f"mirror_max={mirror['max_qps_spread']:.1%} "
                f"accepted={accepted} "
                f"wall={run['wall_s']:.1f}s ==="
            )
            print(footer, flush=True)
            log.write(footer + "\n")
            if not structurally_complete:
                raise RuntimeError(footer + "\n" + output[-8000:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
