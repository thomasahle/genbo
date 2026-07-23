#!/usr/bin/env python3
"""Cheap gates for the post-P345 ideas on DEEP-1M.

The candidate set is the exact float top-U.  That is deliberately harder to rank
than the engine union because all U rows are genuine near neighbours.

Gates:
  RP8: base-only random orthogonal projection, per-dimension symmetric int8,
       containment of the float top-10 after culling U to K.
  BOUND: oracle lower-bound selectivity of an exact striped dot-product cascade.
  COHORT: candidate-row reuse after clustering a waiting query batch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np


DATA = Path("/home/thomas-ahle/big-ann-data")


def read_fbin(path: Path, n: int, d: int) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float32, offset=8, count=n * d)
    if a.size != n * d:
        raise ValueError(f"{path}: got {a.size} floats, expected {n*d}")
    return a.reshape(n, d)


def read_i8bin(path: Path, n: int, d: int) -> np.ndarray:
    a = np.fromfile(path, dtype=np.int8, offset=8, count=n * d)
    if a.size != n * d:
        raise ValueError(f"{path}: got {a.size} bytes, expected {n*d}")
    return a.reshape(n, d)


def exact_union(base: np.ndarray, query: np.ndarray, union: int) -> np.ndarray:
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    _, ids = index.search(np.ascontiguousarray(query), union)
    return ids


def rp8_gate(
    base: np.ndarray,
    query: np.ndarray,
    topu: np.ndarray,
    dims: list[int],
    keeps: list[int],
    seed: int,
    basis: str,
) -> dict:
    dmax = max(dims)
    if basis == "random":
        rng = np.random.default_rng(seed)
        # Columns are orthonormal; every tested prefix is itself orthonormal.
        mat, _ = np.linalg.qr(rng.standard_normal((base.shape[1], dmax), dtype=np.float32))
    elif basis == "pca":
        sample = base[:: max(1, len(base) // 200_000)]
        cov = np.cov(sample, rowvar=False)
        _, vec = np.linalg.eigh(cov)
        mat = np.ascontiguousarray(vec[:, ::-1][:, :dmax], dtype=np.float32)
    else:
        raise ValueError(basis)
    bp = np.ascontiguousarray(base @ mat, dtype=np.float32)
    qp = np.ascontiguousarray(query @ mat, dtype=np.float32)
    out: dict[str, dict[str, float]] = {}
    for rd in dims:
        # A single symmetric scale per projected dimension makes the engine score
        # ||b-q||^2 as norm(b)-2 dot(b,q)+const with int32 VNNI products.
        sample = bp[:: max(1, len(bp) // 200_000), :rd]
        scale = 127.0 / np.maximum(np.quantile(np.abs(sample), 0.995, axis=0), 1e-8)
        b8 = np.clip(np.rint(bp[:, :rd] * scale), -127, 127).astype(np.int8)
        q8 = np.clip(np.rint(qp[:, :rd] * scale), -127, 127).astype(np.int8)
        hit = {k: 0 for k in keeps}
        for qi, cand in enumerate(topu):
            delta = b8[cand].astype(np.int16) - q8[qi].astype(np.int16)
            dist = np.einsum("ij,ij->i", delta, delta, dtype=np.int32)
            order = np.argsort(dist)
            truth = set(cand[:10].tolist())
            for keep in keeps:
                hit[keep] += len(truth.intersection(cand[order[:keep]].tolist()))
        out[str(rd)] = {str(k): hit[k] / (len(query) * 10.0) for k in keeps}
        del b8, q8
    return out


def bound_gate(
    base: np.ndarray,
    query: np.ndarray,
    topu: np.ndarray,
    stripes: list[int],
    ks: list[int],
) -> dict:
    """Best possible exact partial-dot rejection rate.

    For observed dimensions S, dot(q,x) <= dot_S + ||q_tail|| ||x_tail||.
    With unit-normalized DEEP, 2-2*upper_dot is a lower bound on L2.  We use
    the true kth full distance as the admission threshold, so this is an
    optimistic oracle: an implementation cannot reject more rows than reported.
    """

    d = base.shape[1]
    out: dict[str, dict[str, float]] = {}
    for width in stripes:
        if d % width != 0:
            continue
        nstripe = d // width
        fracs = {k: [] for k in ks}
        for qi, cand in enumerate(topu):
            q = query[qi]
            rows = base[cand]
            full = np.einsum("ij,ij->i", rows - q, rows - q)
            # Query-adaptive stripe: read the stripe with the largest query norm.
            qnorms = [
                float(np.dot(q[s * width : (s + 1) * width], q[s * width : (s + 1) * width]))
                for s in range(nstripe)
            ]
            s = int(np.argmax(qnorms))
            lo, hi = s * width, (s + 1) * width
            partial_dot = rows[:, lo:hi] @ q[lo:hi]
            qtail2 = max(0.0, float(np.dot(q, q)) - qnorms[s])
            rownorm2 = np.einsum("ij,ij->i", rows, rows)
            seen_norm2 = np.einsum("ij,ij->i", rows[:, lo:hi], rows[:, lo:hi])
            upper_dot = partial_dot + np.sqrt(np.maximum(0.0, qtail2 * (rownorm2 - seen_norm2)))
            lower_l2 = rownorm2 + float(np.dot(q, q)) - 2.0 * upper_dot
            for k in ks:
                kth = np.partition(full, k - 1)[k - 1]
                fracs[k].append(float(np.mean(lower_l2 <= kth)))
        out[str(width)] = {
            str(k): float(np.mean(fracs[k])) for k in ks
        } | {
            f"{k}_p90": float(np.quantile(fracs[k], 0.9)) for k in ks
        }
    return out


def rp8_engine_gate(
    base8: np.ndarray,
    query8: np.ndarray,
    topu: np.ndarray,
    dims: list[int],
    keeps: list[int],
    seed: int,
    reference_k: int = 32,
) -> dict:
    """Containment of the engine's full-int8 top-K by an RP8 preselection band."""

    dmax = max(dims)
    rng = np.random.default_rng(seed)
    mat, _ = np.linalg.qr(rng.standard_normal((base8.shape[1], dmax), dtype=np.float32))
    bp = np.ascontiguousarray(base8.astype(np.float32) @ mat, dtype=np.float32)
    qp = np.ascontiguousarray(query8.astype(np.float32) @ mat, dtype=np.float32)
    out: dict[str, dict[str, float]] = {}
    for rd in dims:
        sample = bp[:: max(1, len(bp) // 200_000), :rd]
        scale = 127.0 / np.maximum(np.quantile(np.abs(sample), 0.995, axis=0), 1e-8)
        b8 = np.clip(np.rint(bp[:, :rd] * scale), -127, 127).astype(np.int8)
        # Asymmetric fold: b8_j ~= b_j*scale_j, therefore quantize q_j/scale_j
        # with one per-query global multiplier.  Scaling both sides by scale_j
        # would silently reweight projected dimensions by scale_j^2.
        qfold = qp[:, :rd] / scale
        qglobal = 127.0 / np.maximum(np.max(np.abs(qfold), axis=1, keepdims=True), 1e-8)
        q8 = np.clip(np.rint(qfold * qglobal), -127, 127).astype(np.int8)
        hit = {k: 0 for k in keeps}
        for qi, cand in enumerate(topu):
            full_score = base8[cand].astype(np.int32) @ query8[qi].astype(np.int32)
            ref = set(cand[np.argsort(-full_score)[:reference_k]].tolist())
            rp_score = b8[cand].astype(np.int32) @ q8[qi].astype(np.int32)
            order = np.argsort(-rp_score)
            for keep in keeps:
                hit[keep] += len(ref.intersection(cand[order[:keep]].tolist()))
        out[str(rd)] = {
            str(k): hit[k] / (len(query8) * reference_k) for k in keeps
        }
        del b8, q8
    return out


def cohort_gate(query: np.ndarray, topu: np.ndarray, sizes: list[int], seed: int) -> dict:
    """Optimistic queue scheduler: cluster queries, then form nearby cohorts."""

    out: dict[str, dict[str, float]] = {}
    for size in sizes:
        ncohort = max(1, len(query) // size)
        km = faiss.Kmeans(query.shape[1], ncohort, niter=20, nredo=1, seed=seed, verbose=False)
        km.train(np.ascontiguousarray(query))
        _, assign = km.index.search(np.ascontiguousarray(query), 1)
        groups: list[list[int]] = [[] for _ in range(ncohort)]
        for i, c in enumerate(assign[:, 0]):
            groups[int(c)].append(i)
        ratios = []
        overlaps = []
        for group in groups:
            for start in range(0, len(group), size):
                g = group[start : start + size]
                if len(g) < 2:
                    continue
                rows = topu[g].reshape(-1)
                unique = np.unique(rows)
                ratios.append(len(unique) / len(rows))
                # Mean pairwise overlap fraction relative to one query's U.
                sets = [set(topu[i].tolist()) for i in g]
                for a in range(len(sets)):
                    for b in range(a + 1, len(sets)):
                        overlaps.append(len(sets[a] & sets[b]) / topu.shape[1])
        out[str(size)] = {
            "unique_ratio_mean": float(np.mean(ratios)) if ratios else 1.0,
            "unique_ratio_p90": float(np.quantile(ratios, 0.9)) if ratios else 1.0,
            "pair_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
            "cohorts": len(ratios),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbase", type=int, default=1_000_000)
    ap.add_argument("--nq", type=int, default=500)
    ap.add_argument("--union", type=int, default=2100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--out", type=Path, default=DATA / "deep10m" / "next_rungs_gate.json")
    args = ap.parse_args()

    faiss.omp_set_num_threads(args.threads)
    d = 96
    t0 = time.time()
    base = read_fbin(DATA / "deep1m" / "base.1M.fbin", args.nbase, d)
    query = read_fbin(DATA / "deep10m" / "query2k.fbin", 2000, d)[: args.nq].copy()
    base8 = read_i8bin(DATA / "deep1m" / "base.1M.i8bin", args.nbase, d)
    query8 = read_i8bin(DATA / "deep10m" / "query2k.i8bin", 2000, d)[: args.nq].copy()
    topu = exact_union(base, query, args.union)
    print(f"exact top-{args.union}: {time.time()-t0:.1f}s", flush=True)

    report = {
        "settings": vars(args) | {"out": str(args.out)},
        "rp8_containment": {
            basis: rp8_gate(
                base,
                query,
                topu,
                [24, 32, 48, 64, 80, 96],
                [32, 64, 128, 256],
                args.seed,
                basis,
            )
            for basis in ("random", "pca")
        },
        "bound_fullscore_fraction": bound_gate(base, query, topu, [16, 24, 32, 48], [10, 32, 64]),
        "cohort": cohort_gate(query, topu, [2, 4, 8, 16], args.seed),
        "rp8_engine_top32_containment": rp8_engine_gate(
            base8, query8, topu, [32, 48, 64, 80], [64, 128, 256], args.seed
        ),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"total: {time.time()-t0:.1f}s; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
