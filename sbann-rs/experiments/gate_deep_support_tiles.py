#!/usr/bin/env python3
"""Gate support-witness and portal-local bound ranking of streamable tiles."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

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


def build_sphere_radii(
    path: Path,
    base: np.ndarray,
    cent: np.ndarray,
    offsets: np.ndarray,
    ids: np.ndarray,
    portals: int,
) -> np.memmap:
    nbuckets = len(offsets) - 1
    expected = nbuckets * 4
    if path.exists() and path.stat().st_size == expected:
        return np.memmap(path, dtype="<f4", mode="r", shape=(nbuckets,))
    radii2 = np.zeros(nbuckets, dtype=np.int32)
    for b0 in range(0, nbuckets, 4096):
        b1 = min(b0 + 4096, nbuckets)
        counts = np.diff(np.asarray(offsets[b0 : b1 + 1], dtype=np.int64))
        lo, hi = int(offsets[b0]), int(offsets[b1])
        buckets = np.repeat(
            np.arange(b0, b1, dtype=np.uint32), counts
        )
        members = np.asarray(ids[lo:hi], dtype=np.uint32)
        rows = np.asarray(base[members], dtype=np.int16)
        centers = np.asarray(
            cent[
                buckets // portals,
                buckets % portals,
            ],
            dtype=np.int16,
        )
        delta = rows - centers
        dist2 = (delta.astype(np.int32) ** 2).sum(axis=1)
        np.maximum.at(radii2, buckets, dist2)
        if b1 % 131072 == 0:
            print(f"  radii {b1}/{nbuckets} buckets", flush=True)
    radii = np.memmap(path, dtype="<f4", mode="w+", shape=(nbuckets,))
    radii[:] = np.sqrt(radii2.astype(np.float32))
    radii.flush()
    return np.memmap(path, dtype="<f4", mode="r", shape=(nbuckets,))


def normalize_rows(values: np.ndarray) -> np.ndarray:
    mean = values.mean()
    scale = values.std()
    return (values - mean) / max(float(scale), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DATA / "base.10M.i8bin")
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
        "--sphere-radii",
        type=Path,
        default=DATA / "deep10m_portals16.radii.f32",
    )
    parser.add_argument("--nq", type=int, default=500)
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--buckets", default="32,64,96,128,160,192,256")
    parser.add_argument("--witnesses", default="1,2,4,8")
    parser.add_argument("--blend", default="0,0.25,0.5,0.75,1")
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA / "deep_support_tiles_gate.json",
    )
    args = parser.parse_args()

    started = time.monotonic()
    base = load_i8bin(args.base)
    query = load_i8bin(args.query)
    gt = np.memmap(
        args.gt,
        dtype="<u4",
        mode="r",
        offset=8,
        shape=(2000, 100),
    )
    routes = load_routes(args.routes, args.cells)
    cent, offsets, ids, n, d, portals = load_portals(args.portals)
    row_buckets = ensure_row_buckets(args.row_buckets, offsets, ids, n)
    radii = build_sphere_radii(
        args.sphere_radii, base, cent, offsets, ids, portals
    )
    nq = min(args.nq, len(query), len(gt), len(routes))
    bucket_grid = parse_ints(args.buckets)
    witness_grid = parse_ints(args.witnesses)
    blend_grid = sorted(
        {float(value) for value in args.blend.split(",")}
    )
    max_witness = witness_grid[-1]
    methods = ["portal", "sphere"]
    methods += [
        f"support{witness}_blend{blend:g}"
        for witness in witness_grid
        for blend in blend_grid
    ]
    methods += [
        f"support{witness}_blend{blend:g}_pre{preselect}"
        for witness in witness_grid
        for blend in blend_grid
        for preselect in (128, 256, 512)
    ]
    aggregates = {
        (method, keep): Aggregate()
        for method in methods
        for keep in bucket_grid
    }
    bucket_sizes = np.diff(np.asarray(offsets, dtype=np.int64))

    for qi in range(nq):
        q = np.asarray(query[qi], dtype=np.int32)
        cells = np.asarray(routes[qi], dtype=np.uint32)
        buckets = (
            cells[:, None].astype(np.uint64) * portals
            + np.arange(portals, dtype=np.uint64)[None, :]
        ).reshape(-1)
        portal_score = (
            np.asarray(cent[cells], dtype=np.int32) @ q
        ).reshape(-1)
        size = bucket_sizes[buckets]
        lo = np.asarray(offsets[buckets], dtype=np.int64)
        valid = size > 0

        # Evenly spaced actual rows are a conservative fixed support sketch.
        fractions = (np.arange(max_witness) * 2 + 1) / (
            2 * max_witness
        )
        positions = lo[:, None] + np.minimum(
            (size[:, None] * fractions[None, :]).astype(np.int64),
            np.maximum(size[:, None] - 1, 0),
        )
        representative_ids = np.zeros(
            (len(buckets), max_witness), dtype=np.uint32
        )
        representative_ids[valid] = np.asarray(
            ids[positions[valid]], dtype=np.uint32
        )
        witness_score = np.full(
            (len(buckets), max_witness), np.iinfo(np.int32).min
        )
        witness_score[valid] = (
            np.asarray(base[representative_ids[valid]], dtype=np.int32) @ q
        )
        support_prefix = np.maximum.accumulate(witness_score, axis=1)

        qnorm = float(np.linalg.norm(q.astype(np.float32)))
        sphere_score = portal_score.astype(np.float64) + qnorm * np.asarray(
            radii[buckets], dtype=np.float64
        )
        score_by_method: dict[str, np.ndarray] = {
            "portal": portal_score,
            "sphere": sphere_score,
        }
        portal_z = normalize_rows(portal_score.astype(np.float64))
        for witness in witness_grid:
            support = support_prefix[:, witness - 1].astype(np.float64)
            support_z = normalize_rows(support)
            for blend in blend_grid:
                score_by_method[
                    f"support{witness}_blend{blend:g}"
                ] = blend * support_z + (1.0 - blend) * portal_z
                for preselect in (128, 256, 512):
                    if preselect > len(buckets):
                        continue
                    gated = np.full(len(buckets), -np.inf)
                    portal_top = np.argsort(portal_score)[::-1][:preselect]
                    gated[portal_top] = (
                        blend * support_z[portal_top]
                        + (1.0 - blend) * portal_z[portal_top]
                    )
                    score_by_method[
                        f"support{witness}_blend{blend:g}_pre{preselect}"
                    ] = gated

        gt_buckets = np.asarray(row_buckets[np.asarray(gt[qi, :10])])
        for method, score in score_by_method.items():
            order = np.argsort(score)[::-1]
            for keep in bucket_grid:
                chosen = buckets[order[:keep]]
                aggregates[method, keep].add(
                    hit_fraction(gt_buckets, chosen),
                    int(bucket_sizes[chosen].sum()),
                    keep,
                )
        if (qi + 1) % 50 == 0:
            print(f"screened {qi + 1}/{nq} queries", flush=True)

    rows = []
    for (method, keep), aggregate in aggregates.items():
        row: dict[str, str | int | float] = {
            "method": method,
            "cells": args.cells,
            "buckets": keep,
        }
        row.update(aggregate.result())
        rows.append(row)
    output = {
        "date_utc": time.strftime("%Y-%m-%d", time.gmtime()),
        "dataset": "DEEP-10M",
        "protocol": {
            "queries": nq,
            "support": "fixed evenly spaced actual rows per portal bucket",
            "sphere": "exact max radius around the stored portal centroid",
            "metric": "official top-10 containment before SQ4 selection",
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
