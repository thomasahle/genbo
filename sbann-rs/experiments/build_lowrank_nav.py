#!/usr/bin/env python3
"""Build a rank-R PCA int8 nav sidecar for LOW-RANK NAV (SBANN_LOWRANK_FILE).

Motivation (P365): on wiki-35M/d1024 every loose-recall attack fails on the same
Law-1 mechanism — scattered beam evals cost 512B/row even under SQ4.  A rank-256
int8 projection is 256B/row (4 cache lines); the recall law says mid-cascade
stages bind on CONTAINMENT, not precision, and every surfaced candidate still
passes the exact int8 rescore + float rerank downstream.

Train: PCA over an evenly-strided sample of the int8 base (covariance of the
centered sample, eigh, top-R eigenvectors by descending eigenvalue).
Encode: y = (x - mean) @ P, quantized per projected dim with a robust symmetric
scale (scale_t = 127 / p99.9 of |y_t| over the sample, clamped), streamed in
chunks so 35M x 1024 never materializes (~2-4 GB peak).

File format (little-endian), mirrored by vq::LowRankNav::load:
  magic   8 bytes  b"SBLRNAV\0"
  d       u32
  R       u32     (must be a multiple of 64 for the VNNI dot)
  n       u64
  mean    d   f32   (training-sample mean of the int8 rows)
  P       d*R f32   ROW-major (P[j, t]: eigenvectors are COLUMNS)
  scales  R   f32   (code_t = clamp(round(y_t * scale_t), -127, 127))
  codes   n*R i8    (row-major, orig-indexed)

Query-side convention (engine, vq::lowrank_fold_query): qy = (q_i8 - mean) @ P,
then qfold_t = qy_t / scale_t with ONE per-query global i8 requantization —
scaling both sides by scale_t would reweight projected dims by scale_t^2 (the
P346 rp8_engine_gate lesson).
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np


def read_i8bin_mmap(path: Path, n: int, d: int) -> np.ndarray:
    """Memory-map the first n rows of an .i8bin (8-byte header: u32 nb, u32 d)."""
    hdr = np.fromfile(path, dtype=np.uint32, count=2)
    nb, dd = int(hdr[0]), int(hdr[1])
    if dd != d:
        raise ValueError(f"{path}: header d={dd}, expected {d}")
    if nb < n:
        raise ValueError(f"{path}: header nb={nb} < requested n={n}")
    return np.memmap(path, dtype=np.int8, mode="r", offset=8, shape=(n, d))


def train_pca(base: np.ndarray, sample: int, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Evenly-strided sample -> (mean d f32, P d x R f32 row-major)."""
    step = max(1, len(base) // sample)
    smp = np.ascontiguousarray(base[::step][:sample]).astype(np.float32)
    mean = smp.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = (smp - mean).astype(np.float64)
    cov = centered.T @ centered / (len(centered) - 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1][:rank]
    p = np.ascontiguousarray(eigvec[:, order], dtype=np.float32)  # d x R
    explained = float(eigval[order].sum() / eigval.sum())
    print(f"  PCA: sample={len(smp)} rank={rank} explained_var={explained:.4f}", flush=True)
    return mean, p


def train_scales(base: np.ndarray, mean: np.ndarray, p: np.ndarray, sample: int) -> np.ndarray:
    """Robust symmetric per-projected-dim scale: 127 / p99.9(|y_t|) over the sample."""
    step = max(1, len(base) // sample)
    smp = np.ascontiguousarray(base[::step][:sample]).astype(np.float32)
    y = (smp - mean) @ p
    p999 = np.quantile(np.abs(y), 0.999, axis=0)
    return (127.0 / np.maximum(p999, 1e-8)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True, help="int8 .i8bin base")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=500_000, help="encode rows per chunk")
    args = ap.parse_args()
    if args.rank % 64 != 0:
        raise SystemExit(f"--rank {args.rank} must be a multiple of 64 (VNNI dot width)")

    t0 = time.time()
    base = read_i8bin_mmap(args.base, args.n, args.d)
    mean, p = train_pca(base, args.sample, args.rank)
    scales = train_scales(base, mean, p, args.sample)
    print(f"  train: {time.time()-t0:.1f}s", flush=True)

    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(b"SBLRNAV\0")
        f.write(struct.pack("<IIQ", args.d, args.rank, args.n))
        f.write(mean.astype("<f4").tobytes())
        f.write(p.astype("<f4").tobytes())  # row-major d x R
        f.write(scales.astype("<f4").tobytes())
        done = 0
        while done < args.n:
            hi = min(done + args.chunk, args.n)
            x = np.asarray(base[done:hi], dtype=np.float32)  # chunk x d, ~2GB at 500k/d1024
            y = (x - mean) @ p
            codes = np.clip(np.rint(y * scales), -127, 127).astype(np.int8)
            f.write(codes.tobytes())
            done = hi
            print(f"  encode: {done}/{args.n} rows  {time.time()-t0:.1f}s", flush=True)
    tmp.rename(args.out)
    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size/1e9:.2f} GB) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
