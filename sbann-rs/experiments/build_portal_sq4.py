#!/usr/bin/env python3
"""Build the portal-order, AVX-512-padded SQ4 entry sidecar."""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np

from gate_deep_portal_representatives import DATA, load_i8bin, load_portals
from gate_deep_portal_sq4 import robust_ranges


MAGIC = b"SBPSQ4\0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DATA / "base.10M.i8bin")
    parser.add_argument(
        "--portals",
        type=Path,
        default=DATA / "deep10m_portals16.side",
    )
    parser.add_argument(
        "--out", type=Path, default=DATA / "deep10m_portals16.sq4p64"
    )
    parser.add_argument("--range-samples", type=int, default=200_000)
    parser.add_argument("--chunk-assignments", type=int, default=250_000)
    args = parser.parse_args()

    started = time.monotonic()
    base = load_i8bin(args.base)
    _cent, _offsets, ids, n, d, _portals = load_portals(args.portals)
    if base.shape != (n, d) or d % 2:
        raise ValueError("base/portal geometry mismatch")
    lo, step = robust_ranges(base, args.range_samples)
    stride = 64  # d/2=48 padded to one full VNNI vector
    if d // 2 > stride:
        raise ValueError("code does not fit the selected padded stride")

    with args.out.open("wb") as target:
        target.write(struct.pack("<8sQII", MAGIC, len(ids), d, stride))
        target.write(step.astype("<f4").tobytes())
        for start in range(0, len(ids), args.chunk_assignments):
            stop = min(start + args.chunk_assignments, len(ids))
            members = np.asarray(ids[start:stop], dtype=np.uint32)
            rows = np.asarray(base[members], dtype=np.float32)
            quantized = np.rint((rows - lo) / step).clip(0, 15).astype(np.uint8)
            packed = np.zeros((len(members), stride), dtype=np.uint8)
            packed[:, : d // 2] = (
                quantized[:, 0::2] | (quantized[:, 1::2] << 4)
            )
            target.write(packed.tobytes(order="C"))
            if stop % 2_000_000 == 0:
                print(
                    f"{stop}/{len(ids)} assignments "
                    f"({time.monotonic()-started:.1f}s)",
                    flush=True,
                )
    print(
        f"wrote {args.out} ({args.out.stat().st_size/1e9:.2f} GB) "
        f"in {time.monotonic()-started:.1f}s"
    )


if __name__ == "__main__":
    main()
