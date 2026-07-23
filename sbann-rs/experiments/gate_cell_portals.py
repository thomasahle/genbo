#!/usr/bin/env python3
"""Cell-local portal oracle using the real DEEP-10M router output.

For each routed IVF cell we choose base-only farthest-first representatives,
assign cell members to their closest representative, and retain only buckets
whose portals are closest to the query.  We report true-GT coverage, rows
touched, and an exact-int8 three-hop graph continuation from the selected rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DATA = Path("/home/thomas-ahle/big-ann-data/deep10m")


def read_headered(path: Path, dtype: np.dtype, shape: tuple[int, int]) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=shape)


def farthest_first(rows: np.ndarray, count: int) -> np.ndarray:
    n = len(rows)
    count = min(count, n)
    if count == 0:
        return np.empty(0, dtype=np.int32)
    x = rows.astype(np.float32)
    chosen = [0]
    best = np.einsum("ij,ij->i", x - x[0], x - x[0])
    for _ in range(1, count):
        nxt = int(np.argmax(best))
        chosen.append(nxt)
        dist = np.einsum("ij,ij->i", x - x[nxt], x - x[nxt])
        best = np.minimum(best, dist)
    return np.asarray(chosen, dtype=np.int32)


def spherical_buckets(
    rows8: np.ndarray, initial: np.ndarray, iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    rows = rows8.astype(np.float32)
    cent = np.asarray(rows[initial], dtype=np.float32).copy()
    for _ in range(iterations):
        cn = np.linalg.norm(cent, axis=1, keepdims=True)
        cent /= np.maximum(cn, 1e-8)
        assign = np.argmax(rows @ cent.T, axis=1)
        for j in range(len(cent)):
            members = rows[assign == j]
            if len(members):
                cent[j] = members.mean(axis=0)
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-8)
    cent8 = np.clip(np.rint(127.0 * cent), -127, 127).astype(np.int8)
    assign = np.argmax(rows.astype(np.int32) @ cent8.astype(np.int32).T, axis=1)
    return cent8, assign


def graph_continue(
    seed: np.ndarray,
    query: np.ndarray,
    base8: np.ndarray,
    graph: np.ndarray,
    hops: int,
    beam: int,
    kedge: int,
) -> np.ndarray:
    seen = set(int(x) for x in seed)
    expanded: set[int] = set()
    for _ in range(hops):
        ids = np.fromiter(seen, dtype=np.int64)
        score = base8[ids].astype(np.int32) @ query.astype(np.int32)
        order = ids[np.argsort(-score)]
        frontier = [int(x) for x in order if int(x) not in expanded][:beam]
        if not frontier:
            break
        for orig in frontier:
            expanded.add(orig)
            for nb in graph[orig, :kedge]:
                if int(nb) < len(base8):
                    seen.add(int(nb))
    return np.fromiter(seen, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=500)
    ap.add_argument("--portals", default="4,8,16")
    ap.add_argument("--keeps", default="1,2,4")
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--beam", type=int, default=24)
    ap.add_argument("--kedge", type=int, default=32)
    ap.add_argument("--lloyd", type=int, default=3)
    ap.add_argument("--out", type=Path, default=DATA / "cell_portal_gate.json")
    args = ap.parse_args()

    portals_grid = [int(x) for x in args.portals.split(",")]
    keeps_grid = [int(x) for x in args.keeps.split(",")]
    d, n = 96, 10_000_000
    base8 = read_headered(DATA / "base.10M.i8bin", np.int8, (n, d))
    query8 = read_headered(DATA / "query2k.i8bin", np.int8, (2000, d))[: args.nq]
    gt = read_headered(DATA / "deep10m_gt.ibin", np.uint32, (2000, 100))[: args.nq, :10]
    graph = np.memmap(
        DATA / "vamana_deep10m_R32_a1.2.u32",
        dtype=np.uint32,
        mode="r",
        shape=(n, 32),
    )
    route_raw = np.fromfile(DATA / f"routes_p8_nq{args.nq}.u32", dtype=np.uint32)
    rnq, p = int(route_raw[0]), int(route_raw[1])
    assert rnq == args.nq
    routes = route_raw[2:].reshape(rnq, p)

    pairs = np.memmap(DATA / "deep10m_assign.u32", dtype=np.uint32, mode="r").reshape(-1, 2)
    orig = pairs[:, 0]
    cells = pairs[:, 1]
    nc = int(cells[-1]) + 1
    starts = np.searchsorted(cells, np.arange(nc + 1, dtype=np.uint32))

    unique_cells = np.unique(routes)
    max_portals = max(portals_grid)
    # cell -> (member ids, max-P representative positions)
    models: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ci, cell in enumerate(unique_cells):
        ids = np.asarray(orig[starts[cell] : starts[cell + 1]], dtype=np.int64)
        reps = farthest_first(base8[ids], max_portals)
        models[int(cell)] = (ids, reps)
        if (ci + 1) % 500 == 0:
            print(f"built {ci+1}/{len(unique_cells)} routed-cell models", flush=True)

    report: dict[str, dict[str, float]] = {}
    for nportal in portals_grid:
        bucket_models: dict[int, tuple[np.ndarray, list[np.ndarray], int]] = {}
        for cell, (ids, rep_pos_all) in models.items():
            rep_pos = rep_pos_all[: min(nportal, len(rep_pos_all))]
            reps, assign = spherical_buckets(base8[ids], rep_pos, args.lloyd)
            buckets = [ids[assign == j] for j in range(len(reps))]
            bucket_models[cell] = (reps, buckets, len(ids))
        for nkeep in keeps_grid:
            if nkeep > nportal:
                continue
            coverage = 0
            graph_coverage = 0
            selected_rows = []
            full_rows = []
            graph_rows = []
            for qi in range(args.nq):
                q = query8[qi].astype(np.float32)
                selected: list[np.ndarray] = []
                full = 0
                for cell in routes[qi]:
                    reps, buckets, nmember = bucket_models[int(cell)]
                    portal_order = np.argsort(-(reps @ q))
                    keep = portal_order[: min(nkeep, len(portal_order))]
                    selected.extend(buckets[int(j)] for j in keep)
                    full += nmember
                cand = np.unique(np.concatenate(selected)) if selected else np.empty(0, dtype=np.int64)
                truth = set(int(x) for x in gt[qi])
                coverage += len(truth.intersection(cand.tolist()))
                grown = graph_continue(cand, query8[qi], base8, graph, args.hops, args.beam, args.kedge)
                graph_coverage += len(truth.intersection(grown.tolist()))
                selected_rows.append(len(cand) + p * min(nportal, max_portals))
                full_rows.append(full)
                graph_rows.append(len(grown) + p * min(nportal, max_portals))
            key = f"p{p}_portals{nportal}_keep{nkeep}"
            report[key] = {
                "coverage": coverage / (args.nq * 10.0),
                "coverage_graph": graph_coverage / (args.nq * 10.0),
                "rows_mean": float(np.mean(selected_rows)),
                "rows_graph_mean": float(np.mean(graph_rows)),
                "full_cell_rows_mean": float(np.mean(full_rows)),
                "row_fraction": float(np.mean(np.asarray(selected_rows) / np.asarray(full_rows))),
            }
            print(key, report[key], flush=True)

    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
