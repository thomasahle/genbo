#!/usr/bin/env python3
"""Gate fixed portal representatives for DEEP scan-bypass graph entries.

The production portal entry path first chooses a spherical bucket in every
routed IVF cell, then scans every row in that bucket to find a query-specific
graph entry.  This diagnostic replaces the bucket scan with a small,
query-independent list of rows nearest the bucket centroid.  It writes QSEED
tables so entry quality can be measured independently from its eventual Rust
implementation.
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np


DATA = Path("/home/thomas-ahle/big-ann-data/deep10m")
PORTAL_MAGIC = b"SBPORT2\0"


def load_i8bin(path: Path) -> np.memmap:
    with path.open("rb") as source:
        n, d = struct.unpack("<II", source.read(8))
    return np.memmap(path, dtype=np.int8, mode="r", offset=8, shape=(n, d))


def load_routes(path: Path, keep: int) -> np.ndarray:
    with path.open("rb") as source:
        nq, width = struct.unpack("<II", source.read(8))
    if keep > width:
        raise ValueError(f"requested {keep} cells from a width-{width} route dump")
    routes = np.memmap(
        path, dtype=np.uint32, mode="r", offset=8, shape=(nq, width)
    )
    return np.asarray(routes[:, :keep])


def load_portals(
    path: Path,
) -> tuple[np.memmap, np.memmap, np.memmap, int, int, int]:
    with path.open("rb") as source:
        header = source.read(40)
    magic, n, npairs, d, nc, portals, _lloyd = struct.unpack(
        "<8sQQIIII", header
    )
    if magic != PORTAL_MAGIC:
        raise ValueError(f"{path}: bad portal magic {magic!r}")
    cent_offset = 40
    cent = np.memmap(
        path,
        dtype=np.int8,
        mode="r",
        offset=cent_offset,
        shape=(nc, portals, d),
    )
    offsets_offset = cent_offset + nc * portals * d
    offsets = np.memmap(
        path,
        dtype=np.uint32,
        mode="r",
        offset=offsets_offset,
        shape=(nc * portals + 1,),
    )
    ids_offset = offsets_offset + (nc * portals + 1) * 4
    ids = np.memmap(
        path,
        dtype=np.uint32,
        mode="r",
        offset=ids_offset,
        shape=(npairs,),
    )
    return cent, offsets, ids, n, d, portals


def write_qseed(path: Path, table: np.ndarray) -> None:
    table = np.asarray(table, dtype="<u4")
    with path.open("wb") as target:
        target.write(struct.pack("<II", *table.shape))
        target.write(table.tobytes(order="C"))


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
    parser.add_argument("--representatives", default="1,2,4,8")
    parser.add_argument(
        "--strategy",
        choices=("central", "diverse"),
        default="central",
        help="central rows or farthest-first rows within each portal bucket",
    )
    parser.add_argument(
        "--selection",
        choices=("chosen", "direct"),
        default="chosen",
        help="choose a portal centroid first, or score representatives from all portals",
    )
    parser.add_argument(
        "--emit",
        choices=("best", "all"),
        default="best",
        help="emit the query-best fixed row, or all fixed rows as walk entries",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=DATA / "deep_portalwalk_ef24_p8_fixedrep",
    )
    args = parser.parse_args()
    if args.selection == "direct" and args.emit == "all":
        parser.error("--emit all is only defined for --selection chosen")

    t0 = time.monotonic()
    base = load_i8bin(args.base)
    query = load_i8bin(args.query)
    routes = load_routes(args.routes, args.cells)
    cent, offsets, ids, n, d, portals = load_portals(args.portals)
    if base.shape != (n, d) or query.shape[1] != d:
        raise ValueError("base/query/portal geometry mismatch")
    nq = min(len(query), len(routes))
    requested = sorted({int(value) for value in args.representatives.split(",")})
    if not requested or requested[0] < 1:
        raise ValueError("representative counts must be positive")
    max_reps = requested[-1]

    # Identify the one query-selected portal bucket per routed cell.
    chosen = np.empty((nq, args.cells), dtype=np.uint32)
    for qi in range(nq):
        q32 = query[qi].astype(np.int32)
        c = routes[qi].astype(np.int64)
        scores = cent[c].astype(np.int32) @ q32
        chosen[qi] = c.astype(np.uint32) * portals + scores.argmax(axis=1)
    if args.selection == "chosen":
        unique_buckets = np.unique(chosen)
    else:
        unique_cells = np.unique(routes[:nq])
        unique_buckets = (
            unique_cells[:, None].astype(np.uint64) * portals
            + np.arange(portals, dtype=np.uint64)[None, :]
        ).reshape(-1)

    # Cache the rows nearest each selected bucket centroid.  Public evaluation
    # touches only ~nq*cells buckets, so the gate need not preprocess all 1M.
    fixed: dict[int, np.ndarray] = {}
    bucket_sizes: list[int] = []
    for index, bucket_value in enumerate(unique_buckets, 1):
        bucket = int(bucket_value)
        lo, hi = int(offsets[bucket]), int(offsets[bucket + 1])
        members = np.asarray(ids[lo:hi], dtype=np.uint32)
        bucket_sizes.append(len(members))
        if not len(members):
            fixed[bucket] = members
            continue
        portal = np.asarray(
            cent[bucket // portals, bucket % portals], dtype=np.int32
        )
        rows = base[members].astype(np.int32)
        take = min(max_reps, len(members))
        scores = rows @ portal
        first = int(scores.argmax())
        selected = [first]
        if args.strategy == "central":
            top = np.argpartition(scores, -take)[-take:]
            selected = list(top[np.argsort(scores[top])[::-1]])
        else:
            # Actual bucket rows are graph entries, so choose a diverse cover
            # directly.  DEEP vectors are normalized; minimum maximum dot is
            # the farthest point from the current representative set.
            nearest = rows @ rows[first]
            while len(selected) < take:
                nxt = int(nearest.argmin())
                selected.append(nxt)
                nearest = np.maximum(nearest, rows @ rows[nxt])
        fixed[bucket] = members[np.asarray(selected)]
        if index % 4096 == 0:
            print(
                f"cached {index}/{len(unique_buckets)} selected buckets",
                flush=True,
            )

    tables = {
        reps: np.full(
            (nq, args.cells * (reps if args.emit == "all" else 1)),
            np.uint32(0xFFFFFFFF),
            dtype=np.uint32,
        )
        for reps in requested
    }
    for qi in range(nq):
        q32 = query[qi].astype(np.int32)
        for ci, bucket_value in enumerate(chosen[qi]):
            for reps in requested:
                # The runtime design scores at most `reps` fixed rows and emits
                # one entry per cell.
                if args.selection == "chosen":
                    candidates = fixed[int(bucket_value)]
                    eligible = candidates[: min(reps, len(candidates))]
                else:
                    cell = int(routes[qi, ci])
                    eligible = np.concatenate(
                        [
                            fixed[cell * portals + portal][:reps]
                            for portal in range(portals)
                        ]
                    )
                if not len(eligible):
                    continue
                if args.emit == "all":
                    lo = ci * reps
                    tables[reps][qi, lo : lo + len(eligible)] = eligible
                else:
                    scores = base[eligible].astype(np.int32) @ q32
                    tables[reps][qi, ci] = eligible[int(scores.argmax())]

    for reps, table in tables.items():
        path = Path(f"{args.out_prefix}{reps}.u32")
        write_qseed(path, table)
        print(f"R={reps}: wrote {path} ({path.stat().st_size} bytes)")
    sizes = np.asarray(bucket_sizes)
    print(
        f"nq={nq} cells={args.cells} unique_buckets={len(unique_buckets)} "
        f"strategy={args.strategy} selection={args.selection} emit={args.emit} "
        f"bucket_size mean={sizes.mean():.1f} p50={np.median(sizes):.0f} "
        f"p95={np.percentile(sizes, 95):.0f} max={sizes.max()} "
        f"elapsed={time.monotonic()-t0:.1f}s"
    )


if __name__ == "__main__":
    main()
