#!/usr/bin/env python3
"""Build the base-only spherical cell-portal sidecar consumed by SBANN_PORTAL_FILE."""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np

from gate_cell_portals import farthest_first, spherical_buckets


DATA = Path("/home/thomas-ahle/big-ann-data/deep10m")
MAGIC = b"SBPORT2\0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DATA / "base.10M.i8bin")
    ap.add_argument("--assign", type=Path, default=DATA / "deep10m_assign.u32")
    ap.add_argument("--out", type=Path, default=DATA / "deep10m_portals16.side")
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--cells", type=int, default=65_536)
    ap.add_argument("--portals", type=int, default=16)
    ap.add_argument("--lloyd", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    base = np.memmap(args.base, dtype=np.int8, mode="r", offset=8, shape=(args.n, args.d))
    pairs = np.memmap(args.assign, dtype=np.uint32, mode="r").reshape(-1, 2)
    orig, cell = pairs[:, 0], pairs[:, 1]
    starts = np.searchsorted(cell, np.arange(args.cells + 1, dtype=np.uint32))
    npairs = len(orig)

    cent = np.zeros((args.cells, args.portals, args.d), dtype=np.int8)
    offsets = np.empty(args.cells * args.portals + 1, dtype=np.uint32)
    ids_out = np.empty(npairs, dtype=np.uint32)
    cursor = 0
    for c in range(args.cells):
        ids = np.asarray(orig[starts[c] : starts[c + 1]], dtype=np.uint32)
        if len(ids):
            initial = farthest_first(base[ids], args.portals)
            c8, assign = spherical_buckets(base[ids], initial, args.lloyd)
            cent[c, : len(c8)] = c8
            # Cells smaller than P reuse their first centroid only for scoring;
            # their extra buckets stay empty.
            if len(c8) < args.portals:
                cent[c, len(c8) :] = c8[0]
        else:
            assign = np.empty(0, dtype=np.int64)
        for j in range(args.portals):
            offsets[c * args.portals + j] = cursor
            bucket = ids[assign == j] if len(ids) and j < len(c8) else np.empty(0, dtype=np.uint32)
            ids_out[cursor : cursor + len(bucket)] = bucket
            cursor += len(bucket)
        if (c + 1) % 4096 == 0:
            print(f"{c+1}/{args.cells} cells, {cursor}/{npairs} ids, {time.time()-t0:.1f}s", flush=True)
    offsets[-1] = cursor
    if cursor != npairs:
        raise AssertionError(f"lost ids: wrote {cursor}, expected {npairs}")

    header = struct.pack(
        "<8sQQIIII",
        MAGIC,
        args.n,
        npairs,
        args.d,
        args.cells,
        args.portals,
        args.lloyd,
    )
    with args.out.open("wb") as f:
        f.write(header)
        f.write(cent.tobytes(order="C"))
        f.write(offsets.tobytes(order="C"))
        f.write(ids_out.tobytes(order="C"))
    print(f"wrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
