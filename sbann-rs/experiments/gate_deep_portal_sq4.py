#!/usr/bin/env python3
"""Gate portal-bucket entry selection with the resident SQ4 score.

Only query-selected buckets are encoded for this diagnostic.  A production
sidecar would store the two materialized assignments in portal order, making
the same nibble score a short sequential scan rather than scattered int8
gathers.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from gate_deep_portal_representatives import (
    DATA,
    load_i8bin,
    load_portals,
    load_routes,
    write_qseed,
)


def robust_ranges(base: np.ndarray, samples: int) -> tuple[np.ndarray, np.ndarray]:
    stride = max(1, len(base) // samples)
    sample = np.asarray(base[::stride][:samples], dtype=np.float32)
    lo = np.percentile(sample, 0.5, axis=0)
    hi = np.percentile(sample, 99.5, axis=0)
    step = np.maximum(hi - lo, 1.0) / 15.0
    return lo.astype(np.float32), step.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DATA / "base.10M.i8bin")
    parser.add_argument("--query", type=Path, default=DATA / "query2k.i8bin")
    parser.add_argument(
        "--routes",
        type=Path,
        default=DATA / "deep_centroid_graph_ef24_p15.dump",
    )
    parser.add_argument(
        "--portals",
        type=Path,
        default=DATA / "deep10m_portals16.side",
    )
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--range-samples", type=int, default=200_000)
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA / "deep_portalwalk_ef24_p8_sq4.u32",
    )
    args = parser.parse_args()

    started = time.monotonic()
    base = load_i8bin(args.base)
    query = load_i8bin(args.query)
    routes = load_routes(args.routes, args.cells)
    cent, offsets, ids, n, d, portals = load_portals(args.portals)
    if base.shape != (n, d) or query.shape[1] != d:
        raise ValueError("geometry mismatch")
    lo, step = robust_ranges(base, args.range_samples)
    nq = min(len(query), len(routes))
    table = np.full((nq, args.cells), np.uint32(0xFFFFFFFF), dtype=np.uint32)
    total_rows = 0
    for qi in range(nq):
        q = np.asarray(query[qi], dtype=np.int32)
        cells = routes[qi].astype(np.int64)
        portal_scores = cent[cells].astype(np.int32) @ q
        selected = portal_scores.argmax(axis=1)
        weighted = q.astype(np.float32) * step
        scale = 127.0 / max(float(np.abs(weighted).max()), 1e-9)
        q4 = np.rint(weighted * scale).clip(-127, 127).astype(np.int32)
        for ci, (cell, portal) in enumerate(zip(cells, selected)):
            bucket = int(cell) * portals + int(portal)
            loff, hoff = int(offsets[bucket]), int(offsets[bucket + 1])
            members = np.asarray(ids[loff:hoff], dtype=np.uint32)
            if not len(members):
                continue
            rows = np.asarray(base[members], dtype=np.float32)
            codes = np.rint((rows - lo) / step).clip(0, 15).astype(np.int32)
            scores = codes @ q4
            table[qi, ci] = members[int(scores.argmax())]
            total_rows += len(members)
    write_qseed(args.out, table)
    print(
        f"wrote {args.out}; nq={nq} cells={args.cells} "
        f"mean_rows/q={total_rows/nq:.1f} elapsed={time.monotonic()-started:.1f}s"
    )


if __name__ == "__main__":
    main()
