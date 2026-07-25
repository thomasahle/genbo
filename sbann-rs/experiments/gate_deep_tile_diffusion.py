#!/usr/bin/env python3
"""Leak-free co-retrieval diffusion over cells, followed by portal streaming."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gate_deep_cell_rerank import load_routes as load_feature_routes
from gate_deep_graphfree_coverage import (
    Aggregate,
    ensure_row_buckets,
    hit_fraction,
    parse_ints,
)
from gate_deep_portal_representatives import (
    DATA,
    load_i8bin,
    load_portals,
    load_routes,
)


ROAR = Path("/home/thomas-ahle/RoarGraph/data/deep10m")
UINT32_MAX = np.uint32(2**32 - 1)


def build_transitions(
    path: Path,
    train_routes_path: Path,
    train_gt_path: Path,
    row_buckets: np.ndarray,
    portals: int,
    train_queries: int,
    source_cells: int,
    max_edges: int,
    nc: int,
) -> np.ndarray:
    if path.exists():
        saved = np.load(path)
        neighbors = saved["neighbors"]
        if neighbors.shape == (nc, max_edges):
            return neighbors

    train = load_feature_routes(train_routes_path)
    with train_gt_path.open("rb") as source:
        n_gt, width = np.frombuffer(source.read(8), dtype="<u4")
    train_gt = np.memmap(
        train_gt_path,
        dtype="<u4",
        mode="r",
        offset=8,
        shape=(int(n_gt), int(width)),
    )
    nq = min(train_queries, train.nq, len(train_gt))
    sources = np.asarray(
        train.records["cell"][:nq, :source_cells], dtype=np.uint32
    )
    targets = (
        np.asarray(row_buckets[np.asarray(train_gt[:nq, :10])])
        // portals
    ).reshape(nq, -1)
    source_flat = np.repeat(sources[:, :, None], targets.shape[1], axis=2)
    target_flat = np.broadcast_to(
        targets[:, None, :], source_flat.shape
    )
    source_flat = source_flat.reshape(-1)
    target_flat = target_flat.reshape(-1)
    keep = source_flat != target_flat
    keys = (
        source_flat[keep].astype(np.uint64) << 32
    ) | target_flat[keep].astype(np.uint64)
    unique, counts = np.unique(keys, return_counts=True)
    source = (unique >> 32).astype(np.uint32)
    target = (unique & np.uint64(2**32 - 1)).astype(np.uint32)

    neighbors = np.full((nc, max_edges), UINT32_MAX, dtype=np.uint32)
    starts = np.flatnonzero(
        np.r_[True, source[1:] != source[:-1], True]
    )
    for begin, end in zip(starts[:-1], starts[1:]):
        cell = int(source[begin])
        take = min(max_edges, end - begin)
        local = np.argpartition(counts[begin:end], -take)[-take:]
        local = local[np.argsort(counts[begin:end][local])[::-1]]
        neighbors[cell, :take] = target[begin:end][local]
    np.savez_compressed(
        path,
        neighbors=neighbors,
        train_queries=nq,
        source_cells=source_cells,
    )
    return neighbors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=Path, default=DATA / "query2k.i8bin")
    parser.add_argument("--gt", type=Path, default=DATA / "deep10m_gt.ibin")
    parser.add_argument(
        "--routes",
        type=Path,
        default=DATA / "deep_hier_routes_p128_public.u32",
    )
    parser.add_argument(
        "--portals", type=Path, default=DATA / "deep10m_portals16.side"
    )
    parser.add_argument(
        "--row-buckets",
        type=Path,
        default=DATA / "deep10m_portals16.rowbuckets.u32",
    )
    parser.add_argument(
        "--train-routes",
        type=Path,
        default=DATA / "deep10m_train60k_k128.rrf",
    )
    parser.add_argument(
        "--train-gt", type=Path, default=ROAR / "train.gt.bin"
    )
    parser.add_argument(
        "--transitions",
        type=Path,
        default=DATA / "deep10m_cell_coretrieval_edges32.npz",
    )
    parser.add_argument("--train-queries", type=int, default=60_000)
    parser.add_argument("--train-source-cells", type=int, default=4)
    parser.add_argument("--max-edges", type=int, default=32)
    parser.add_argument("--nq", type=int, default=500)
    parser.add_argument("--raw-cells", default="4,8,16")
    parser.add_argument("--source-cells", default="1,2,4,8")
    parser.add_argument("--edges", default="2,4,8,16,32")
    parser.add_argument("--buckets", default="16,32,64,96,128")
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA / "deep_tile_diffusion_gate.json",
    )
    args = parser.parse_args()

    started = time.monotonic()
    query = load_i8bin(args.query)
    gt = np.memmap(
        args.gt,
        dtype="<u4",
        mode="r",
        offset=8,
        shape=(2000, 100),
    )
    routes = load_routes(args.routes, 128)
    cent, offsets, ids, n, d, portals = load_portals(args.portals)
    row_buckets = ensure_row_buckets(args.row_buckets, offsets, ids, n)
    nc = cent.shape[0]
    neighbors = build_transitions(
        args.transitions,
        args.train_routes,
        args.train_gt,
        row_buckets,
        portals,
        args.train_queries,
        args.train_source_cells,
        args.max_edges,
        nc,
    )
    nq = min(args.nq, len(query), len(gt), len(routes))
    raw_grid = parse_ints(args.raw_cells)
    source_grid = parse_ints(args.source_cells)
    edge_grid = parse_ints(args.edges)
    bucket_grid = parse_ints(args.buckets)
    bucket_sizes = np.diff(np.asarray(offsets, dtype=np.int64))
    configs = [
        (raw, source, edge, bucket)
        for raw in raw_grid
        for source in source_grid
        if source <= raw
        for edge in edge_grid
        for bucket in bucket_grid
        if bucket <= (raw + source * edge) * portals
    ]
    aggregates = {config: Aggregate() for config in configs}
    cell_coverage: dict[tuple[int, int, int], list[float]] = {
        (raw, source, edge): []
        for raw, source, edge, _bucket in configs
    }

    for qi in range(nq):
        q = np.asarray(query[qi], dtype=np.int32)
        route = np.asarray(routes[qi], dtype=np.uint32)
        gt_buckets = np.asarray(row_buckets[np.asarray(gt[qi, :10])])
        gt_cells = gt_buckets // portals
        for raw in raw_grid:
            base_cells = route[:raw]
            for source in source_grid:
                if source > raw:
                    continue
                for edge in edge_grid:
                    expanded = np.concatenate(
                        [
                            base_cells,
                            neighbors[base_cells[:source], :edge].reshape(-1),
                        ]
                    )
                    expanded = np.unique(expanded[expanded != UINT32_MAX])
                    cell_coverage[raw, source, edge].append(
                        float(
                            np.isin(gt_cells, expanded)
                            .any(axis=1)
                            .mean()
                        )
                    )
                    buckets = (
                        expanded[:, None].astype(np.uint64) * portals
                        + np.arange(portals, dtype=np.uint64)[None, :]
                    ).reshape(-1)
                    score = (
                        np.asarray(cent[expanded], dtype=np.int32) @ q
                    ).reshape(-1)
                    order = np.argsort(score)[::-1]
                    for bucket in bucket_grid:
                        config = (raw, source, edge, bucket)
                        if config not in aggregates:
                            continue
                        chosen = buckets[order[:bucket]]
                        aggregates[config].add(
                            hit_fraction(gt_buckets, chosen),
                            int(bucket_sizes[chosen].sum()),
                            bucket,
                        )
        if (qi + 1) % 50 == 0:
            print(f"screened {qi + 1}/{nq} queries", flush=True)

    rows = []
    for config, aggregate in aggregates.items():
        raw, source, edge, bucket = config
        row: dict[str, int | float] = {
            "raw_cells": raw,
            "source_cells": source,
            "edges_per_source": edge,
            "unique_cells_mean": float(
                raw
                + source * edge
            ),
            "buckets": bucket,
            "cell_containment_recall_at_10": float(
                np.mean(cell_coverage[raw, source, edge])
            ),
        }
        row.update(aggregate.result())
        rows.append(row)
    output = {
        "date_utc": time.strftime("%Y-%m-%d", time.gmtime()),
        "dataset": "DEEP-10M",
        "protocol": {
            "training_queries": args.train_queries,
            "public_queries": nq,
            "edges": "top cell co-retrieval counts from disjoint training GT",
            "selection": "global portal-centroid ranking after one synchronous expansion",
        },
        "results": rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(
        f"wrote {args.out}; elapsed={output['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
