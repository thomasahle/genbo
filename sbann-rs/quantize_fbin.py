#!/usr/bin/env python3
"""Quantize a big-ann `.fbin` (float32) to `.i8bin` (int8) with a single global scale, chunked so the
30M base never materializes in RAM. The same scale must be used for base + queries so their int8 L2 is
consistent. Scale = 127 / (Q-th percentile abs over a sample) with clipping (robust to outliers); the
int8 index is only for CANDIDATE generation (exact float rerank fixes the ranking), so mild clipping is
fine. Prints the scale; pass --scale to reuse the base's scale for the queries.

Usage:
  quantize_fbin.py <in.fbin> <out.i8bin> [--scale S] [--sample N] [--pct P]
"""
import sys, numpy as np

inp, outp = sys.argv[1], sys.argv[2]
def arg(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
scale = arg("--scale", 0.0)
sample = arg("--sample", 2_000_000)
pct = arg("--pct", 99.99)

with open(inp, "rb") as f:
    nb, d = np.fromfile(f, dtype=np.uint32, count=2)
nb, d = int(nb), int(d)
print(f"[quantize] {inp}: nb={nb} d={d}")
mm = np.memmap(inp, dtype=np.float32, mode="r", offset=8, shape=(nb, d))

if scale <= 0.0:
    s = min(sample, nb)
    idx = np.linspace(0, nb - 1, s).astype(np.int64)
    a = np.abs(mm[idx])
    hi = np.percentile(a, pct)
    scale = 127.0 / float(hi)
    print(f"[quantize] sampled {s} rows, p{pct} abs={hi:.6g} -> scale={scale:.6g}")
else:
    print(f"[quantize] using provided scale={scale:.6g}")

with open(outp, "wb") as out:
    out.write(np.array([nb, d], dtype=np.uint32).tobytes())
    CH = 1_000_000
    clipped = 0
    for s0 in range(0, nb, CH):
        e0 = min(s0 + CH, nb)
        chunk = np.asarray(mm[s0:e0], dtype=np.float32) * scale
        clipped += int(np.sum(np.abs(chunk) > 127.0))
        q = np.clip(np.rint(chunk), -127, 127).astype(np.int8)
        out.write(q.tobytes())
        if s0 % (CH * 5) == 0:
            print(f"  {e0}/{nb}", flush=True)
    print(f"[quantize] done -> {outp}  (clipped {clipped} values, {100*clipped/(nb*d):.4f}%)")
