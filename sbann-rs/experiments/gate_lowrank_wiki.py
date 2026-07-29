#!/usr/bin/env python3
"""P341/P346-style containment gate for LOW-RANK NAV on a wiki-1M prefix.

Candidate set per query = exact float-IP top-U (U~2100) over the prefix — harder
to rank than the engine union because every row is a genuine near neighbour.
Metric = containment of the exact float top-10 inside the nav sidecar's top-K
ranking of that candidate set (K=128 is the gate anchor; the engine's beam then
int8-rescores and float-reranks whatever the nav score surfaces).

Compared arms, all query-side ENGINE-FAITHFUL (int8 query, per-dim fold, one
per-query global i8 requantization):
  - low-rank rank-R PCA int8 codes (R in --ranks; one rank-max PCA, prefixes
    sliced — a prefix of a rank-320 PCA IS the rank-R PCA);
  - the SQ4 nibble sidecar formula from main.rs (per-dim p0.5/p99.5 histogram
    range, step folded into the query) as the incumbent baseline.

PASS (spec): lowrank containment @R=256/K=128 >= SQ4 containment @K=128 - 0.01.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

WIKI = Path("/home/thomas-ahle/big-ann-data/wiki35m")


def read_fbin_mmap(path: Path, n: int, d: int) -> np.ndarray:
    hdr = np.fromfile(path, dtype=np.uint32, count=2)
    assert int(hdr[1]) == d, f"{path}: d={hdr[1]} != {d}"
    assert int(hdr[0]) >= n, f"{path}: nb={hdr[0]} < {n}"
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(n, d))


def read_i8bin_mmap(path: Path, n: int, d: int) -> np.ndarray:
    hdr = np.fromfile(path, dtype=np.uint32, count=2)
    assert int(hdr[1]) == d, f"{path}: d={hdr[1]} != {d}"
    assert int(hdr[0]) >= n, f"{path}: nb={hdr[0]} < {n}"
    return np.memmap(path, dtype=np.int8, mode="r", offset=8, shape=(n, d))


def exact_ip_topu(base_f: np.ndarray, query_f: np.ndarray, union: int, block: int) -> np.ndarray:
    """Exact inner-product top-U ids per query (SBANN_IP convention), blocked over the base."""
    nq = len(query_f)
    best_ids = np.zeros((nq, union), dtype=np.int64)
    best_scores = np.full((nq, union), -np.inf, dtype=np.float32)
    for lo in range(0, len(base_f), block):
        hi = min(lo + block, len(base_f))
        scores = np.asarray(base_f[lo:hi]) @ query_f.T  # (chunk, nq)
        for qi in range(nq):
            merged_scores = np.concatenate([best_scores[qi], scores[:, qi]])
            merged_ids = np.concatenate([best_ids[qi], np.arange(lo, hi)])
            keep = np.argpartition(-merged_scores, union - 1)[:union]
            best_scores[qi] = merged_scores[keep]
            best_ids[qi] = merged_ids[keep]
    order = np.argsort(-best_scores, axis=1)
    return np.take_along_axis(best_ids, order, axis=1)


def fold_query_global(qfold: np.ndarray) -> np.ndarray:
    """One per-query global i8 requantization of an already per-dim-folded float query."""
    g = 127.0 / np.maximum(np.max(np.abs(qfold), axis=1, keepdims=True), 1e-9)
    return np.clip(np.rint(qfold * g), -127, 127).astype(np.int8)


def containment(topu: np.ndarray, nav_scores: list[np.ndarray], keeps: list[int]) -> dict[str, float]:
    """Mean |float top-10 ∩ nav top-K(candidates)| / 10. nav_scores[qi] aligns with topu[qi]."""
    out = {}
    for keep in keeps:
        hit = 0
        for qi, cand in enumerate(topu):
            order = np.argsort(-nav_scores[qi])
            hit += len(set(cand[:10].tolist()) & set(cand[order[:keep]].tolist()))
        out[str(keep)] = hit / (len(topu) * 10.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbase", type=int, default=1_000_000)
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--union", type=int, default=2100)
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--ranks", type=int, nargs="+", default=[128, 192, 256, 320])
    ap.add_argument("--keeps", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--block", type=int, default=200_000)
    ap.add_argument("--out", type=Path, default=WIKI / "lowrank_gate_1m.json")
    args = ap.parse_args()
    d = args.d
    t0 = time.time()

    base_f = read_fbin_mmap(WIKI / "base.fbin", args.nbase, d)
    base_i8 = np.asarray(read_i8bin_mmap(WIKI / "base.i8bin", args.nbase, d))
    query_f = np.asarray(read_fbin_mmap(WIKI / "query.fbin", args.nq, d)).copy()
    query_i8 = np.asarray(read_i8bin_mmap(WIKI / "query.i8bin", args.nq, d)).copy()

    topu = exact_ip_topu(base_f, query_f, args.union, args.block)
    print(f"exact IP top-{args.union} for {args.nq} queries: {time.time()-t0:.1f}s", flush=True)

    report: dict = {
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "lowrank": {},
    }

    # ---- LOW-RANK arms: one PCA at max rank, prefixes sliced ----
    rmax = max(args.ranks)
    step = max(1, args.nbase // args.sample)
    smp = base_i8[::step][: args.sample].astype(np.float32)
    mean = smp.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = (smp - mean).astype(np.float64)
    cov = centered.T @ cov_rhs(centered)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1][:rmax]
    p = np.ascontiguousarray(eigvec[:, order], dtype=np.float32)
    print(f"PCA rank-{rmax} trained ({time.time()-t0:.1f}s), "
          f"explained={float(eigval[order].sum()/eigval.sum()):.4f}", flush=True)

    # project the union candidates only (approx 200*2100 rows) + the sample for scales
    y_sample = (smp - mean) @ p
    qy_full = (query_i8.astype(np.float32) - mean) @ p
    cand_rows = {}
    for qi, cand in enumerate(topu):
        cand_rows[qi] = (base_i8[cand].astype(np.float32) - mean) @ p

    for rank in args.ranks:
        scales = (127.0 / np.maximum(np.quantile(np.abs(y_sample[:, :rank]), 0.999, axis=0), 1e-8)).astype(np.float32)
        q8 = fold_query_global(qy_full[:, :rank] / scales)
        nav = []
        for qi in range(args.nq):
            c8 = np.clip(np.rint(cand_rows[qi][:, :rank] * scales), -127, 127).astype(np.int8)
            nav.append(c8.astype(np.int32) @ q8[qi].astype(np.int32))
        report["lowrank"][str(rank)] = containment(topu, nav, args.keeps)
        print(f"rank {rank}: {report['lowrank'][str(rank)]}  ({time.time()-t0:.1f}s)", flush=True)

    # ---- SQ4 baseline: engine formula (main.rs SQ4-RUNG block), engine mstep sampling ----
    mstep = max(1, args.nbase // 200_000)
    ssmp = base_i8[::mstep]
    lo_q = np.quantile(ssmp, 0.005, axis=0)
    hi_q = np.quantile(ssmp, 0.995, axis=0)
    hi_q = np.maximum(hi_q, lo_q + 1)
    sq4_step = ((hi_q - lo_q) / 15.0).astype(np.float32)
    sq4_lo = lo_q.astype(np.float32)
    q8 = fold_query_global(query_i8.astype(np.float32) * sq4_step)  # engine folds step only
    nav = []
    for qi, cand in enumerate(topu):
        nib = np.clip(np.rint((base_i8[cand].astype(np.float32) - sq4_lo) / sq4_step), 0, 15).astype(np.int32)
        nav.append(nib @ q8[qi].astype(np.int32))
    report["sq4"] = containment(topu, nav, args.keeps)
    print(f"sq4 baseline: {report['sq4']}  ({time.time()-t0:.1f}s)", flush=True)

    lr256 = report["lowrank"].get("256", {}).get("128")
    sq4_128 = report["sq4"]["128"]
    report["gate"] = {
        "lowrank_r256_k128": lr256,
        "sq4_k128": sq4_128,
        "pass": bool(lr256 is not None and lr256 >= sq4_128 - 0.01),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["gate"], indent=2))
    print(f"total {time.time()-t0:.1f}s; wrote {args.out}", flush=True)


def cov_rhs(centered: np.ndarray) -> np.ndarray:
    """Split out so the big f64 temporary is explicit: cov = X^T X / (m-1)."""
    return centered / (len(centered) - 1)


if __name__ == "__main__":
    main()
