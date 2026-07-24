#!/usr/bin/env python3
"""Build query-independent 4-bit displacement sketches for a fixed graph.

Each graph row stores k unsigned max-absolute displacement scales followed by
k nibble-packed signed displacement rows.  A nibble is q(delta)+8, where
q(delta) is the nearest integer in [-7, 7] after scaling the edge's maximum
absolute coordinate difference to 7.  At search time, edges from the same
source can be ranked without gathering their destination vectors:

    dot(query, destination - source)
      ~= max_abs / 7 * dot(query, q(delta)).

The source term and division by seven are common to all outgoing edges.
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np


MAGIC = b"EDS1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("graph", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-offset", type=int, default=8)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--chunk-nodes", type=int, default=8192)
    args = parser.parse_args()

    if args.d % 2:
        raise ValueError("edge sketches require an even dimension")
    expected_graph = args.n * args.k * 4
    if args.graph.stat().st_size != expected_graph:
        raise ValueError(
            f"{args.graph}: {args.graph.stat().st_size} bytes, "
            f"expected {expected_graph}"
        )
    expected_base = args.base_offset + args.n * args.d
    if args.base.stat().st_size < expected_base:
        raise ValueError(f"{args.base}: shorter than {expected_base} bytes")

    graph = np.memmap(
        args.graph, dtype="<u4", mode="r", shape=(args.n, args.k)
    )
    base = np.memmap(
        args.base,
        dtype=np.int8,
        mode="r",
        offset=args.base_offset,
        shape=(args.n, args.d),
    )
    half = args.d // 2
    stride = args.k + args.k * half
    with args.output.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<III", args.n, args.k, args.d))
        handle.truncate(16 + args.n * stride)
    output = np.memmap(
        args.output,
        dtype=np.uint8,
        mode="r+",
        offset=16,
        shape=(args.n, stride),
    )

    started = time.time()
    for begin in range(0, args.n, args.chunk_nodes):
        end = min(args.n, begin + args.chunk_nodes)
        neighbours = np.asarray(graph[begin:end], dtype=np.int64)
        if np.any(neighbours >= args.n):
            raise ValueError("edge-sketch graph contains padded/out-of-range ids")
        source = np.asarray(base[begin:end], dtype=np.int16)
        destination = np.asarray(base[neighbours], dtype=np.int16)
        delta = destination - source[:, None, :]
        magnitude = np.maximum(np.abs(delta).max(axis=2), 1).astype(np.uint16)
        absolute = np.abs(delta).astype(np.uint16)
        quantized = (
            (absolute * 7 + magnitude[:, :, None] // 2)
            // magnitude[:, :, None]
        ).astype(np.int16)
        quantized *= np.sign(delta).astype(np.int16)
        quantized = np.clip(quantized, -7, 7).astype(np.int8)
        codes = (
            (quantized[:, :, 0::2] + 8).astype(np.uint8)
            | ((quantized[:, :, 1::2] + 8).astype(np.uint8) << 4)
        )
        output[begin:end, : args.k] = magnitude.astype(np.uint8)
        output[begin:end, args.k :] = codes.reshape(end - begin, -1)
        if begin % (args.chunk_nodes * 64) == 0:
            print(
                f"{begin}/{args.n} nodes in {time.time() - started:.1f}s",
                flush=True,
            )
    output.flush()
    print(
        f"wrote {args.output} ({args.output.stat().st_size / 1e9:.2f} GB) "
        f"in {time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
