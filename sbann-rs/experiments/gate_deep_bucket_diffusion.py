#!/usr/bin/env python3
"""Leak-free co-retrieval diffusion directly between portal buckets."""

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


def selected_source_buckets(
    query: np.ndarray,
    cells: np.ndarray,
    cent: np.ndarray,
    portals: int,
) -> np.ndarray:
    q = np.asarray(query, dtype=np.int32)
    score = np.asarray(cent[cells], dtype=np.int32) @ q
    portal = score.argmax(axis=1).astype(np.uint32)
    return cells.astype(np.uint32) * portals + portal


def build_edges(
    path: Path,
    train_query: np.ndarray,
    train_cells: np.ndarray,
    train_gt: np.ndarray,
    row_buckets: np.ndarray,
    cent: np.ndarray,
    portals: int,
    train_queries: int,
    source_count: int,
    max_edges: int,
) -> tuple[np.ndarray, np.ndarray]:
    if path.exists():
        saved = np.load(path)
        return saved["sources"], saved["neighbors"]
    nq = min(train_queries, len(train_query), len(train_cells), len(train_gt))
    sources = np.empty((nq, source_count), dtype=np.uint32)
    for qi in range(nq):
        sources[qi] = selected_source_buckets(
            train_query[qi], train_cells[qi, :source_count], cent, portals
        )
    targets = np.asarray(
        row_buckets[np.asarray(train_gt[:nq, :10])], dtype=np.uint32
    ).reshape(nq, -1)
    source_flat = np.repeat(sources[:, :, None], targets.shape[1], axis=2)
    target_flat = np.broadcast_to(targets[:, None, :], source_flat.shape)
    source_flat = source_flat.reshape(-1)
    target_flat = target_flat.reshape(-1)
    keep = source_flat != target_flat
    keys = (
        source_flat[keep].astype(np.uint64) << 32
    ) | target_flat[keep].astype(np.uint64)
    unique, counts = np.unique(keys, return_counts=True)
    edge_source = (unique >> 32).astype(np.uint32)
    edge_target = (unique & np.uint64(2**32 - 1)).astype(np.uint32)
    starts = np.flatnonzero(
        np.r_[True, edge_source[1:] != edge_source[:-1], True]
    )
    compact_source = np.empty(len(starts) - 1, dtype=np.uint32)
    neighbors = np.full(
        (len(starts) - 1, max_edges), UINT32_MAX, dtype=np.uint32
    )
    for row, (begin, end) in enumerate(zip(starts[:-1], starts[1:])):
        compact_source[row] = edge_source[begin]
        take = min(max_edges, end - begin)
        local = np.argpartition(counts[begin:end], -take)[-take:]
        local = local[np.argsort(counts[begin:end][local])[::-1]]
        neighbors[row, :take] = edge_target[begin:end][local]
    np.savez_compressed(
        path,
        sources=compact_source,
        neighbors=neighbors,
        train_queries=nq,
        source_count=source_count,
    )
    return compact_source, neighbors


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
        "--train-query", type=Path, default=DATA / "query.train60k.i8bin"
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
        "--portals", type=Path, default=DATA / "deep10m_portals16.side"
    )
    parser.add_argument(
        "--row-buckets",
        type=Path,
        default=DATA / "deep10m_portals16.rowbuckets.u32",
    )
    parser.add_argument(
        "--edges-file",
        type=Path,
        default=DATA / "deep10m_bucket_coretrieval_edges32.npz",
    )
    parser.add_argument("--train-queries", type=int, default=60_000)
    parser.add_argument("--train-source-count", type=int, default=4)
    parser.add_argument("--max-edges", type=int, default=32)
    parser.add_argument("--nq", type=int, default=500)
    parser.add_argument("--source-count", default="1,2,4,8,16")
    parser.add_argument("--edges", default="2,4,8,16,32")
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA / "deep_bucket_diffusion_gate.json",
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
    routes = load_routes(args.routes, max(parse_ints(args.source_count)))
    train_query = load_i8bin(args.train_query)
    train_dump = load_feature_routes(args.train_routes)
    train_cells = np.asarray(train_dump.records["cell"], dtype=np.uint32)
    train_gt = np.memmap(
        args.train_gt,
        dtype="<u4",
        mode="r",
        offset=8,
        shape=(500_000, 100),
    )
    cent, offsets, ids, n, _d, portals = load_portals(args.portals)
    row_buckets = ensure_row_buckets(args.row_buckets, offsets, ids, n)
    edge_sources, edge_neighbors = build_edges(
        args.edges_file,
        train_query,
        train_cells,
        train_gt,
        row_buckets,
        cent,
        portals,
        args.train_queries,
        args.train_source_count,
        args.max_edges,
    )
    bucket_sizes = np.diff(np.asarray(offsets, dtype=np.int64))
    nq = min(args.nq, len(query), len(gt), len(routes))
    source_grid = parse_ints(args.source_count)
    edge_grid = parse_ints(args.edges)
    aggregates = {
        (source, edge): Aggregate()
        for source in source_grid
        for edge in edge_grid
    }
    source_seen = {
        (source, edge): [] for source in source_grid for edge in edge_grid
    }

    for qi in range(nq):
        all_sources = selected_source_buckets(
            query[qi], np.asarray(routes[qi]), cent, portals
        )
        targets = np.asarray(row_buckets[np.asarray(gt[qi, :10])])
        for source_count in source_grid:
            source = all_sources[:source_count]
            positions = np.searchsorted(edge_sources, source)
            found = (positions < len(edge_sources)) & (
                edge_sources[np.minimum(positions, len(edge_sources) - 1)]
                == source
            )
            for edge in edge_grid:
                expanded = [source]
                if np.any(found):
                    expanded.append(
                        edge_neighbors[positions[found], :edge].reshape(-1)
                    )
                chosen = np.unique(np.concatenate(expanded))
                chosen = chosen[chosen != UINT32_MAX]
                aggregates[source_count, edge].add(
                    hit_fraction(targets, chosen),
                    int(bucket_sizes[chosen].sum()),
                    len(chosen),
                )
                source_seen[source_count, edge].append(float(found.mean()))
        if (qi + 1) % 50 == 0:
            print(f"screened {qi + 1}/{nq} queries", flush=True)

    rows = []
    for (source, edge), aggregate in aggregates.items():
        row: dict[str, int | float] = {
            "source_buckets": source,
            "edges_per_source": edge,
            "source_seen_fraction": float(np.mean(source_seen[source, edge])),
        }
        row.update(aggregate.result())
        rows.append(row)
    output = {
        "date_utc": time.strftime("%Y-%m-%d", time.gmtime()),
        "dataset": "DEEP-10M",
        "protocol": {
            "training_queries": args.train_queries,
            "public_queries": nq,
            "edges": "portal-bucket co-retrieval from disjoint training GT",
            "query": "one synchronous expansion, then direct SQ4-stream budget",
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
