#!/usr/bin/env python3
"""Extract a strided sample of an .i8bin into a smaller .i8bin (covers all clusters of a clustered
base, for cold-start cell/codebook training). Usage: strided_sample.py <in.i8bin> <out.i8bin> <n>"""
import sys, numpy as np
inp, outp, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(inp, "rb") as f:
    nb, d = np.fromfile(f, dtype=np.uint32, count=2)
nb, d = int(nb), int(d)
n = min(n, nb)
idx = np.linspace(0, nb - 1, n).astype(np.int64)
mm = np.memmap(inp, dtype=np.int8, mode="r", offset=8, shape=(nb, d))
with open(outp, "wb") as out:
    out.write(np.array([n, d], dtype=np.uint32).tobytes())
    CH = 200000
    for s in range(0, n, CH):
        out.write(np.ascontiguousarray(mm[idx[s:s+CH]]).tobytes())
print(f"[strided] {inp} nb={nb} -> {outp} n={n} d={d}")
