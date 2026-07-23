#!/usr/bin/env python3
"""Gate graph-local physical row layouts on exact engine union traces.

The experiment is deliberately query-independent: layouts are derived only from
the built index assignments and the base k-NN graph.  The trace is used solely to
measure the physical locality those layouts would give the real graph-rescore rows.

GUN1 format:
  magic[4], nq:u32, then nq * (pool_distinct:u32, union_len:u32, ids:u32[union_len]).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


DATA = Path("/home/thomas-ahle/big-ann-data/deep10m")


def read_trace(path: Path) -> list[tuple[int, np.ndarray]]:
    raw = path.read_bytes()
    if raw[:4] != b"GUN1":
        raise ValueError(f"{path}: not a GUN1 trace")
    nq = int.from_bytes(raw[4:8], "little")
    words = np.frombuffer(raw, dtype="<u4", offset=8)
    rows: list[tuple[int, np.ndarray]] = []
    off = 0
    for _ in range(nq):
        pool = int(words[off])
        size = int(words[off + 1])
        off += 2
        ids = words[off : off + size].copy()
        off += size
        rows.append((pool, ids))
    if off != len(words):
        raise ValueError(f"{path}: {len(words)-off} trailing u32 words")
    return rows


def assignment_keys(path: Path, n: int, chunk_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.memmap(path, dtype="<u4", mode="r").reshape(-1, 2)
    lo = np.full(n, np.iinfo(np.uint32).max, dtype=np.uint32)
    hi = np.zeros(n, dtype=np.uint32)
    for start in range(0, len(pairs), chunk_pairs):
        block = pairs[start : start + chunk_pairs]
        ids = block[:, 0]
        cells = block[:, 1]
        np.minimum.at(lo, ids, cells)
        np.maximum.at(hi, ids, cells)
    missing = int(np.count_nonzero(lo == np.iinfo(np.uint32).max))
    if missing:
        raise ValueError(f"{path}: {missing} base ids have no assignment")
    return lo, hi


def graph_cell_signature(
    graph_path: Path,
    primary_cell: np.ndarray,
    n: int,
    chunk_rows: int,
) -> np.ndarray:
    graph = np.memmap(graph_path, dtype="<u4", mode="r")
    if graph.size % n:
        raise ValueError(f"{graph_path}: size is not n*k")
    graph = graph.reshape(n, graph.size // n)
    signature = np.empty(n, dtype=np.uint32)
    mid = graph.shape[1] // 2
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        neighbor_cells = primary_cell[graph[start:stop]]
        signature[start:stop] = np.partition(neighbor_cells, mid, axis=1)[:, mid]
    return signature


def make_rank(*keys: np.ndarray) -> np.ndarray:
    # np.lexsort uses the final key as primary. Callers pass least -> most significant.
    order = np.lexsort(keys)
    rank = np.empty(len(order), dtype=np.uint32)
    rank[order] = np.arange(len(order), dtype=np.uint32)
    return rank


def materialize_base(
    source: Path,
    destination: Path,
    rank: np.ndarray,
    n: int,
    d: int,
    chunk_rows: int,
    data_offset: int,
) -> None:
    header = np.fromfile(source, dtype="<u4", count=2)
    if header.tolist() != [n, d]:
        raise ValueError(f"{source}: header {header.tolist()} != [{n}, {d}]")
    base = np.memmap(source, dtype=np.int8, mode="r", offset=8, shape=(n, d))
    order = np.empty(n, dtype=np.uint32)
    order[rank] = np.arange(n, dtype=np.uint32)
    with destination.open("wb") as out:
        header.tofile(out)
        out.write(bytes(data_offset - 8))
        for start in range(0, n, chunk_rows):
            stop = min(start + chunk_rows, n)
            np.ascontiguousarray(base[order[start:stop]]).tofile(out)


def materialize_graph(
    source: Path,
    destination: Path,
    rank: np.ndarray,
    n: int,
    chunk_rows: int,
    sort_neighbors: bool,
) -> None:
    flat = np.memmap(source, dtype="<u4", mode="r")
    if flat.size % n:
        raise ValueError(f"{source}: size is not n*k")
    graph = flat.reshape(n, flat.size // n)
    order = np.empty(n, dtype=np.uint32)
    order[rank] = np.arange(n, dtype=np.uint32)
    with destination.open("wb") as out:
        for start in range(0, n, chunk_rows):
            stop = min(start + chunk_rows, n)
            block = np.asarray(graph[order[start:stop]])
            if np.any(block >= n):
                mapped = np.full(block.shape, np.iinfo(np.uint32).max, dtype=np.uint32)
                valid = block < n
                mapped[valid] = rank[block[valid]]
            else:
                mapped = rank[block]
            if sort_neighbors:
                mapped.sort(axis=1)
            np.ascontiguousarray(mapped, dtype="<u4").tofile(out)


def trace_metrics(
    rows: list[tuple[int, np.ndarray]],
    rank: np.ndarray | None,
    row_bytes: int,
    page_bytes: int,
    data_offset: int,
) -> dict[str, float]:
    graph_rows = 0
    cache_lines = 0
    pages = 0
    near_4k = 0
    adjacent_8 = 0
    transitions = 0
    query_line_ratios: list[float] = []
    query_page_ratios: list[float] = []
    for pool, ids in rows:
        ids = ids[pool:]
        if rank is not None:
            ids = rank[ids]
        ids64 = ids.astype(np.uint64, copy=False)
        if len(ids64) == 0:
            continue
        starts = data_offset + ids64 * row_bytes
        line0 = starts // 64
        line1 = (starts + row_bytes - 1) // 64
        qlines = len(np.unique(np.concatenate((line0, line1))))
        page0 = starts // page_bytes
        page1 = (starts + row_bytes - 1) // page_bytes
        qpages = len(np.unique(np.concatenate((page0, page1))))
        graph_rows += len(ids64)
        cache_lines += qlines
        pages += qpages
        query_line_ratios.append(qlines / (2.0 * len(ids64)))
        query_page_ratios.append(qpages / len(ids64))
        if len(ids64) > 1:
            delta = np.abs(np.diff(ids64.astype(np.int64)))
            transitions += len(delta)
            near_4k += int(np.count_nonzero(delta * row_bytes < 4096))
            adjacent_8 += int(np.count_nonzero(delta <= 8))
    return {
        "queries": len(rows),
        "data_offset": data_offset,
        "graph_rows": graph_rows,
        "graph_rows_per_query": graph_rows / max(1, len(rows)),
        "cache_lines_per_graph_row": cache_lines / max(1, graph_rows),
        "pages_per_graph_row": pages / max(1, graph_rows),
        "query_cacheline_ratio_mean": float(np.mean(query_line_ratios)),
        "query_cacheline_ratio_p90": float(np.quantile(query_line_ratios, 0.9)),
        "query_page_ratio_mean": float(np.mean(query_page_ratios)),
        "near_4k_transition_fraction": near_4k / max(1, transitions),
        "within_8_rows_transition_fraction": adjacent_8 / max(1, transitions),
    }


def sampled_edge_metrics(
    graph_path: Path,
    rank: np.ndarray | None,
    n: int,
    sample_rows: int,
) -> dict[str, float]:
    flat = np.memmap(graph_path, dtype="<u4", mode="r")
    graph = flat.reshape(n, flat.size // n)
    step = max(1, n // sample_rows)
    src = np.arange(0, n, step, dtype=np.uint32)[:sample_rows]
    dst = np.asarray(graph[src]).reshape(-1)
    src = np.repeat(src, graph.shape[1])
    if rank is not None:
        src = rank[src]
        dst = rank[dst]
    delta = np.abs(src.astype(np.int64) - dst.astype(np.int64))
    return {
        "sampled_edges": len(delta),
        "within_8_rows": float(np.mean(delta <= 8)),
        "within_4k_rows": float(np.mean(delta * 96 < 4096)),
        "median_row_delta": float(np.median(delta)),
        "p90_row_delta": float(np.quantile(delta, 0.9)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, default=DATA / "deep10m_p2_h3m24_nq2000.gun")
    ap.add_argument("--assign", type=Path, default=DATA / "deep10m_assign.u32")
    ap.add_argument("--graph", type=Path, default=DATA / "vamana_deep10m_R32_a1.2.u32")
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--row-bytes", type=int, default=96)
    ap.add_argument("--chunk-pairs", type=int, default=2_000_000)
    ap.add_argument("--chunk-rows", type=int, default=250_000)
    ap.add_argument("--sample-rows", type=int, default=250_000)
    ap.add_argument("--out-prefix", type=Path, default=DATA / "graph_layout_gate")
    ap.add_argument("--base", type=Path, default=DATA / "base.10M.i8bin")
    ap.add_argument("--materialize-layout", choices=("none", "cellpair", "graphcell"), default="none")
    ap.add_argument("--materialize-chunk", type=int, default=250_000)
    ap.add_argument("--materialize-graph", action="store_true")
    ap.add_argument("--sort-graph-neighbors", action="store_true")
    ap.add_argument("--base-offset", type=int, choices=(8, 64), default=8)
    args = ap.parse_args()

    started = time.time()
    rows = read_trace(args.trace)
    print(f"trace: {len(rows)} queries in {time.time()-started:.1f}s", flush=True)

    t0 = time.time()
    primary, secondary = assignment_keys(args.assign, args.n, args.chunk_pairs)
    print(f"assignment keys: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    cell_rank = make_rank(secondary, primary)
    cell_path = args.out_prefix.with_suffix(".cellpair.u32")
    cell_rank.tofile(cell_path)
    print(f"cell-pair rank: {time.time()-t0:.1f}s -> {cell_path}", flush=True)

    t0 = time.time()
    graph_cell = graph_cell_signature(args.graph, primary, args.n, args.chunk_rows)
    print(f"graph-cell signature: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    graph_rank = make_rank(secondary, primary, graph_cell)
    graph_path = args.out_prefix.with_suffix(".graphcell.u32")
    graph_rank.tofile(graph_path)
    print(f"graph-cell rank: {time.time()-t0:.1f}s -> {graph_path}", flush=True)

    report: dict[str, object] = {
        "settings": {
            **vars(args),
            "trace": str(args.trace),
            "assign": str(args.assign),
            "graph": str(args.graph),
            "out_prefix": str(args.out_prefix),
        },
        "layouts": {},
    }
    for name, rank in (
        ("original", None),
        ("cellpair", cell_rank),
        ("graphcell", graph_rank),
    ):
        t0 = time.time()
        data_offset = 8 if name == "original" else 64
        report["layouts"][name] = {
            "trace": trace_metrics(
                rows, rank, args.row_bytes, 4096, data_offset
            ),
            "edges": sampled_edge_metrics(args.graph, rank, args.n, args.sample_rows),
        }
        print(f"metrics {name}: {time.time()-t0:.1f}s", flush=True)

    report["elapsed_seconds"] = time.time() - started
    out = args.out_prefix.with_suffix(".json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.materialize_layout != "none":
        rank = cell_rank if args.materialize_layout == "cellpair" else graph_rank
        alignment_tag = ".aligned64" if args.base_offset == 64 else ""
        base_out = args.out_prefix.with_suffix(
            f".{args.materialize_layout}{alignment_tag}.i8bin"
        )
        t0 = time.time()
        materialize_base(
            args.base,
            base_out,
            rank,
            args.n,
            args.row_bytes,
            args.materialize_chunk,
            args.base_offset,
        )
        print(f"materialized {args.materialize_layout}: {time.time()-t0:.1f}s -> {base_out}", flush=True)
        if args.materialize_graph:
            sort_tag = "sorted" if args.sort_graph_neighbors else ""
            graph_out = args.out_prefix.with_suffix(
                f".{args.materialize_layout}.graph{sort_tag}.u32"
            )
            t0 = time.time()
            materialize_graph(
                args.graph,
                graph_out,
                rank,
                args.n,
                args.materialize_chunk,
                args.sort_graph_neighbors,
            )
            print(
                f"materialized {args.materialize_layout} graph: "
                f"{time.time()-t0:.1f}s -> {graph_out}",
                flush=True,
            )
    print(f"wrote {out} in {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
