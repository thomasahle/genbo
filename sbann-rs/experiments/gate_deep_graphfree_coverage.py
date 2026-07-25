#!/usr/bin/env python3
"""Coverage/cost gates for graph-free DEEP-10M loose-recall search.

This intentionally stops before candidate scoring.  It measures whether direct
portal fan-out, product-code enumeration, or sparse component voting can expose
enough official top-10 neighbors at a plausible number of streamed assignments.
Only survivors should receive an SQ4 engine implementation.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from gate_deep_portal_representatives import (
    DATA,
    load_i8bin,
    load_portals,
    load_routes,
)


ROAR = Path("/home/thomas-ahle/RoarGraph/data/deep10m")
UINT32_MAX = np.uint32(2**32 - 1)


def load_bin(path: Path, dtype: str) -> np.memmap:
    with path.open("rb") as source:
        n, d = struct.unpack("<II", source.read(8))
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, d))


def parse_ints(spec: str) -> list[int]:
    values = sorted({int(value) for value in spec.split(",")})
    if not values or values[0] < 1:
        raise ValueError("grid values must be positive")
    return values


def ensure_row_buckets(
    path: Path, offsets: np.ndarray, ids: np.ndarray, n: int
) -> np.memmap:
    """Invert the two portal assignments per base row into a reusable raw file."""
    expected = n * 2 * 4
    if path.exists() and path.stat().st_size == expected:
        return np.memmap(path, dtype="<u4", mode="r", shape=(n, 2))

    print(f"building {path} ({expected / 1e6:.1f} MB)", flush=True)
    low = np.full(n, UINT32_MAX, dtype=np.uint32)
    high = np.zeros(n, dtype=np.uint32)
    nbuckets = len(offsets) - 1
    for b0 in range(0, nbuckets, 4096):
        b1 = min(b0 + 4096, nbuckets)
        counts = np.diff(np.asarray(offsets[b0 : b1 + 1], dtype=np.int64))
        buckets = np.repeat(
            np.arange(b0, b1, dtype=np.uint32), counts
        )
        lo, hi = int(offsets[b0]), int(offsets[b1])
        members = np.asarray(ids[lo:hi], dtype=np.uint32)
        np.minimum.at(low, members, buckets)
        np.maximum.at(high, members, buckets)
        if b1 % 131072 == 0:
            print(f"  inverted {b1}/{nbuckets} buckets", flush=True)
    if np.any(low == UINT32_MAX):
        raise ValueError("portal sidecar does not cover every base row")
    table = np.memmap(path, dtype="<u4", mode="w+", shape=(n, 2))
    table[:, 0] = low
    table[:, 1] = high
    table.flush()
    return np.memmap(path, dtype="<u4", mode="r", shape=(n, 2))


class Aggregate:
    def __init__(self) -> None:
        self.recall: list[float] = []
        self.rows: list[int] = []
        self.buckets: list[int] = []

    def add(self, recall: float, rows: int, buckets: int) -> None:
        self.recall.append(recall)
        self.rows.append(rows)
        self.buckets.append(buckets)

    def result(self) -> dict[str, float]:
        rows = np.asarray(self.rows)
        buckets = np.asarray(self.buckets)
        return {
            "containment_recall_at_10": float(np.mean(self.recall)),
            "mean_assignment_rows": float(rows.mean()),
            "p50_assignment_rows": float(np.median(rows)),
            "p95_assignment_rows": float(np.percentile(rows, 95)),
            "mean_buckets": float(buckets.mean()),
            "logical_sq4_kib": float(rows.mean() * 48 / 1024),
            "padded_sq4_kib": float(rows.mean() * 64 / 1024),
        }


def hit_fraction(gt_buckets: np.ndarray, selected: np.ndarray) -> float:
    return float(np.isin(gt_buckets, selected, assume_unique=False).any(axis=1).mean())


def route_member_ids(
    cells: np.ndarray, offsets: np.ndarray, ids: np.ndarray, portals: int
) -> np.ndarray:
    chunks = []
    for cell in cells:
        lo = int(offsets[int(cell) * portals])
        hi = int(offsets[(int(cell) + 1) * portals])
        chunks.append(np.asarray(ids[lo:hi], dtype=np.uint32))
    if not chunks:
        return np.empty(0, dtype=np.uint32)
    return np.concatenate(chunks)


def top_product_codes(
    query: np.ndarray, c0: np.ndarray, c1: np.ndarray, keep: int
) -> np.ndarray:
    d0 = ((c0 - query[: c0.shape[1]]) ** 2).sum(axis=1)
    d1 = ((c1 - query[c0.shape[1] :]) ** 2).sum(axis=1)
    score = (d0[:, None] + d1[None, :]).reshape(-1)
    keep = min(keep, len(score))
    top = np.argpartition(score, keep - 1)[:keep]
    return top.astype(np.uint32)


def top_component_codes(
    query: np.ndarray, centroids: np.ndarray, keep: int
) -> np.ndarray:
    score = ((centroids - query) ** 2).sum(axis=1)
    keep = min(keep, len(score))
    return np.argpartition(score, keep - 1)[:keep].astype(np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-i8", type=Path, default=DATA / "query2k.i8bin")
    parser.add_argument("--query-f32", type=Path, default=DATA / "query2k.fbin")
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
        "--imi-centroids",
        type=Path,
        default=DATA / "deep10m_imi256_centroids.npz",
    )
    parser.add_argument(
        "--imi-assign",
        type=Path,
        default=DATA / "deep10m_imi256_assign.u32",
    )
    parser.add_argument("--nq", type=int, default=500)
    parser.add_argument("--cells", default="8,16,32,64,128")
    parser.add_argument("--portals-per-cell", default="1,2,4,8,16")
    parser.add_argument("--global-buckets", default="8,16,32,64,128,256")
    parser.add_argument("--product-codes", default="8,32,128,512,2048")
    parser.add_argument("--component-codes", default="1,2,4,8,16")
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA / "deep_graphfree_coverage_gate.json",
    )
    args = parser.parse_args()

    started = time.monotonic()
    query_i8 = load_i8bin(args.query_i8)
    query_f32 = load_bin(args.query_f32, "<f4")
    gt = load_bin(args.gt, "<u4")
    routes = load_routes(args.routes, max(parse_ints(args.cells)))
    cent, offsets, ids, n, d, portals = load_portals(args.portals)
    if query_i8.shape[1] != d or query_f32.shape[1] != d:
        raise ValueError("query/portal dimension mismatch")
    nq = min(args.nq, len(query_i8), len(query_f32), len(gt), len(routes))
    row_buckets = ensure_row_buckets(args.row_buckets, offsets, ids, n)
    bucket_sizes = np.diff(np.asarray(offsets, dtype=np.int64))

    cell_grid = parse_ints(args.cells)
    portal_grid = parse_ints(args.portals_per_cell)
    global_grid = parse_ints(args.global_buckets)
    product_grid = parse_ints(args.product_codes)
    component_grid = parse_ints(args.component_codes)

    per_cell = {
        (cells, keep): Aggregate()
        for cells in cell_grid
        for keep in portal_grid
    }
    global_bucket = {
        (cells, keep): Aggregate()
        for cells in cell_grid
        for keep in global_grid
        if keep <= cells * portals
    }
    product = {
        (cells, keep): Aggregate()
        for cells in cell_grid
        for keep in product_grid
    }
    component = {
        (cells, keep, mode): Aggregate()
        for cells in cell_grid
        for keep in component_grid
        for mode in ("both", "either")
    }

    imi_centroids = np.load(args.imi_centroids)
    c0 = np.asarray(imi_centroids["c0"], dtype=np.float32)
    c1 = np.asarray(imi_centroids["c1"], dtype=np.float32)
    imi = np.memmap(args.imi_assign, dtype="<u4", mode="r", shape=(n,))

    for qi in range(nq):
        q32 = np.asarray(query_i8[qi], dtype=np.int32)
        qf = np.asarray(query_f32[qi], dtype=np.float32)
        route = np.asarray(routes[qi], dtype=np.uint32)
        gt_ids = np.asarray(gt[qi, :10], dtype=np.uint32)
        gt_buckets = np.asarray(row_buckets[gt_ids])
        gt_codes = np.asarray(imi[gt_ids], dtype=np.uint32)

        # The IMI was trained in the engine's int8 coordinate system.
        qi8f = q32.astype(np.float32)
        max_product = top_product_codes(qi8f, c0, c1, product_grid[-1])
        product_prefix = {
            keep: max_product[:keep] for keep in product_grid
        }
        comp0 = {
            keep: top_component_codes(qi8f[:48], c0, keep)
            for keep in component_grid
        }
        comp1 = {
            keep: top_component_codes(qi8f[48:], c1, keep)
            for keep in component_grid
        }

        for cells in cell_grid:
            selected_cells = route[:cells]
            portal_scores = (
                np.asarray(cent[selected_cells], dtype=np.int32) @ q32
            )
            order = np.argsort(portal_scores, axis=1)[:, ::-1]
            base_buckets = (
                selected_cells[:, None].astype(np.uint64) * portals
            )

            for keep in portal_grid:
                chosen = (
                    base_buckets + order[:, :keep].astype(np.uint64)
                ).reshape(-1)
                per_cell[cells, keep].add(
                    hit_fraction(gt_buckets, chosen),
                    int(bucket_sizes[chosen].sum()),
                    len(chosen),
                )

            flat_scores = portal_scores.reshape(-1)
            flat_buckets = (
                base_buckets
                + np.arange(portals, dtype=np.uint64)[None, :]
            ).reshape(-1)
            global_order = np.argsort(flat_scores)[::-1]
            for keep in global_grid:
                if (cells, keep) not in global_bucket:
                    continue
                chosen = flat_buckets[global_order[:keep]]
                global_bucket[cells, keep].add(
                    hit_fraction(gt_buckets, chosen),
                    int(bucket_sizes[chosen].sum()),
                    len(chosen),
                )

            members = route_member_ids(selected_cells, offsets, ids, portals)
            member_codes = np.asarray(imi[members], dtype=np.uint32)
            route_hit = np.isin(
                gt_buckets // portals, selected_cells, assume_unique=False
            ).any(axis=1)
            for keep in product_grid:
                chosen_codes = product_prefix[keep]
                code_hit = np.isin(gt_codes, chosen_codes)
                rows = int(np.isin(member_codes, chosen_codes).sum())
                product[cells, keep].add(
                    float((route_hit & code_hit).mean()), rows, keep
                )

            member0 = (member_codes >> 8).astype(np.uint16)
            member1 = (member_codes & 255).astype(np.uint16)
            gt0 = (gt_codes >> 8).astype(np.uint16)
            gt1 = (gt_codes & 255).astype(np.uint16)
            for keep in component_grid:
                hit0 = np.isin(gt0, comp0[keep])
                hit1 = np.isin(gt1, comp1[keep])
                rows0 = np.isin(member0, comp0[keep])
                rows1 = np.isin(member1, comp1[keep])
                for mode, hit, row_mask in (
                    ("both", hit0 & hit1, rows0 & rows1),
                    ("either", hit0 | hit1, rows0 | rows1),
                ):
                    component[cells, keep, mode].add(
                        float((route_hit & hit).mean()),
                        int(row_mask.sum()),
                        keep * (2 if mode == "either" else 1),
                    )

        if (qi + 1) % 50 == 0:
            print(f"screened {qi + 1}/{nq} queries", flush=True)

    def rows(
        values: dict[tuple, Aggregate], names: Iterable[str]
    ) -> list[dict[str, float | int | str]]:
        out = []
        for key, value in values.items():
            row = dict(zip(names, key))
            row.update(value.result())
            out.append(row)
        return out

    result = {
        "date_utc": time.strftime("%Y-%m-%d", time.gmtime()),
        "dataset": "DEEP-10M",
        "protocol": {
            "queries": nq,
            "metric": "official top-10 containment before approximate scoring",
            "rows": "materialized assignment rows; a0=2 duplicates are charged",
        },
        "portal_per_cell": rows(per_cell, ("cells", "portals_per_cell")),
        "portal_global": rows(global_bucket, ("cells", "buckets")),
        "product_enumeration": rows(product, ("cells", "product_codes")),
        "sparse_component_voting": rows(
            component, ("cells", "component_codes", "mode")
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"wrote {args.out}; elapsed={result['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
