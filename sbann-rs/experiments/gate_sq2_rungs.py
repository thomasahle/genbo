#!/usr/bin/env python3
"""SQ2/SQ3 containment gate — the pre-build pilot DESIGN_LAWS Law 4 mandates
("no sub-4-bit tier without offline containment >= 0.995 on real unions").

Protocol mirrors the P341/P366 gates exactly:
  wiki arm: wiki-1M prefix, 200 queries, exact float-IP top-2100 candidate
    unions (gate_lowrank_wiki.py convention); float top-10 = union[:10].
  deep arm: DEEP-1M carve, 500 queries (deep10m/query2k), exact float-L2
    top-2100 unions (gate_next_rungs.py convention). DEEP rows/queries are
    unit-norm (verified min=max=1.0000), so the engine's dot kernel ranks
    identically to L2 up to quantization noise — which is what we measure.

Arms, all query-side ENGINE-FAITHFUL (per-dim step folded into the query,
one per-query global i8 requantization — the SQ4_STEP convention; the -lo
offset is a per-query rank constant):
  sqB      : B-bit per-dim asymmetric affine, robust p0.5..p99.5 range
             (main.rs SQ4-RUNG formula with (hi-lo)/(2^B-1) steps)
  sqB_sym  : same, range forced symmetric (m = max(|lo|,|hi|), [-m, +m])
  sq2/3_p98, sq2_p95: tighter-clip variants (p2..p98, p5..p95) — coarse
             rungs may prefer harder clipping.

Metric: containment of the float top-10 inside the quantized-dot top-K of
the union, K in {64,128,192,256,384,512}. Bar: >= 0.995 at K <= 256.
SQ4 is recomputed with the same harness so arms are comparable (engine gate
had it at 1.000@64 on the wiki set, P366).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DATA = Path("/home/thomas-ahle/big-ann-data")
KEEPS = [64, 128, 192, 256, 384, 512]
# name -> (bits, plo, phi, symmetric)
ARMS: dict[str, tuple[int, float, float, bool]] = {
    "sq4": (4, 0.005, 0.995, False),
    "sq3": (3, 0.005, 0.995, False),
    "sq2": (2, 0.005, 0.995, False),
    "sq4_sym": (4, 0.005, 0.995, True),
    "sq3_sym": (3, 0.005, 0.995, True),
    "sq2_sym": (2, 0.005, 0.995, True),
    "sq3_p98": (3, 0.02, 0.98, False),
    "sq2_p98": (2, 0.02, 0.98, False),
    "sq2_p95": (2, 0.05, 0.95, False),
}


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


def exact_topu(base_f: np.ndarray, query_f: np.ndarray, union: int, block: int,
               metric: str) -> np.ndarray:
    """Exact top-U ids per query, blocked over the base.

    metric='ip': larger dot better (wiki, SBANN_IP convention).
    metric='l2': smaller ||x-q||^2 better; scored as -(||x||^2 - 2 x.q).
    """
    nq = len(query_f)
    best_ids = np.zeros((nq, union), dtype=np.int64)
    best_scores = np.full((nq, union), -np.inf, dtype=np.float32)
    for lo in range(0, len(base_f), block):
        hi = min(lo + block, len(base_f))
        chunk = np.asarray(base_f[lo:hi])
        scores = chunk @ query_f.T  # (chunk, nq)
        if metric == "l2":
            norms = np.einsum("ij,ij->i", chunk, chunk, dtype=np.float32)
            scores = 2.0 * scores - norms[:, None]  # = -(||x||^2 - 2 x.q)
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
        out[str(keep)] = round(hit / (len(topu) * 10.0), 4)
    return out


def sq_range(ssmp: np.ndarray, bits: int, plo: float, phi: float, symmetric: bool
             ) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim (lo, step) for a B-bit affine rung, engine sampling convention."""
    levels = (1 << bits) - 1
    lo_q = np.quantile(ssmp, plo, axis=0)
    hi_q = np.quantile(ssmp, phi, axis=0)
    hi_q = np.maximum(hi_q, lo_q + 1)
    if symmetric:
        m = np.maximum(np.abs(lo_q), np.abs(hi_q))
        m = np.maximum(m, 0.5)
        lo_q, hi_q = -m, m
    step = ((hi_q - lo_q) / levels).astype(np.float32)
    return lo_q.astype(np.float32), step


def run_dataset(name: str, base_f: np.ndarray, base_i8: np.ndarray,
                query_f: np.ndarray, query_i8: np.ndarray, union: int,
                block: int, metric: str, cache: Path, t0: float) -> dict:
    if cache.exists():
        topu = np.load(cache)
        assert topu.shape == (len(query_f), union)
        print(f"[{name}] union cache hit {cache}", flush=True)
    else:
        topu = exact_topu(base_f, query_f, union, block, metric)
        np.save(cache, topu)
        print(f"[{name}] exact {metric} top-{union} for {len(query_f)} queries: "
              f"{time.time()-t0:.1f}s", flush=True)

    # engine sampling convention for the range histogram (main.rs mstep)
    mstep = max(1, len(base_i8) // 200_000)
    ssmp = np.asarray(base_i8[::mstep])
    qf32 = query_i8.astype(np.float32)

    # gather candidate rows once (nq x union x d int8)
    cand_rows = [np.asarray(base_i8[cand]).astype(np.float32) for cand in topu]

    out: dict[str, dict[str, float]] = {}
    for arm, (bits, plo, phi, sym) in ARMS.items():
        levels = (1 << bits) - 1
        lo, step = sq_range(ssmp, bits, plo, phi, sym)
        q8 = fold_query_global(qf32 * step)  # engine folds step only
        nav = []
        for qi in range(len(topu)):
            codes = np.clip(np.rint((cand_rows[qi] - lo) / step), 0, levels).astype(np.int32)
            nav.append(codes @ q8[qi].astype(np.int32))
        out[arm] = containment(topu, nav, KEEPS)
        print(f"[{name}] {arm:8s}: {out[arm]}  ({time.time()-t0:.1f}s)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union", type=int, default=2100)
    ap.add_argument("--nq-wiki", type=int, default=200)
    ap.add_argument("--nq-deep", type=int, default=500)
    ap.add_argument("--block", type=int, default=200_000)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/tmp/claude-20035/-home-thomas-ahle-genbo/"
                                 "5c1e0b4b-c28c-4023-8c75-af9d17832b0d/scratchpad"))
    ap.add_argument("--out", type=Path, default=DATA / "wiki35m" / "sq2_gate.json")
    args = ap.parse_args()
    t0 = time.time()
    report: dict = {
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "arms": {k: {"bits": b, "plo": pl, "phi": ph, "symmetric": sy}
                 for k, (b, pl, ph, sy) in ARMS.items()},
        "keeps": KEEPS,
        "bar": "containment >= 0.995 at K <= 256 (DESIGN_LAWS Law 4)",
    }

    # ---- wiki-1M prefix, IP unions (gate_lowrank_wiki.py protocol) ----
    W = DATA / "wiki35m"
    d = 1024
    base_f = read_fbin_mmap(W / "base.fbin", 1_000_000, d)
    base_i8 = read_i8bin_mmap(W / "base.i8bin", 1_000_000, d)
    query_f = np.asarray(read_fbin_mmap(W / "query.fbin", args.nq_wiki, d)).copy()
    query_i8 = np.asarray(read_i8bin_mmap(W / "query.i8bin", args.nq_wiki, d)).copy()
    report["wiki1m"] = run_dataset(
        "wiki1m", base_f, base_i8, query_f, query_i8, args.union, args.block,
        "ip", args.cache_dir / f"topu_wiki1m_u{args.union}_q{args.nq_wiki}.npy", t0)
    del base_f, base_i8

    # ---- DEEP-1M, L2 unions (gate_next_rungs.py protocol; rows unit-norm) ----
    d = 96
    base_f = read_fbin_mmap(DATA / "deep1m" / "base.1M.fbin", 1_000_000, d)
    base_i8 = read_i8bin_mmap(DATA / "deep1m" / "base.1M.i8bin", 1_000_000, d)
    query_f = np.asarray(read_fbin_mmap(DATA / "deep10m" / "query2k.fbin", 2000, d))[: args.nq_deep].copy()
    query_i8 = np.asarray(read_i8bin_mmap(DATA / "deep10m" / "query2k.i8bin", 2000, d))[: args.nq_deep].copy()
    report["deep1m"] = run_dataset(
        "deep1m", base_f, base_i8, query_f, query_i8, args.union, args.block,
        "l2", args.cache_dir / f"topu_deep1m_u{args.union}_q{args.nq_deep}.npy", t0)

    # ---- verdicts vs the 0.995 bar ----
    verdict = {}
    for ds in ("wiki1m", "deep1m"):
        verdict[ds] = {}
        for arm in ARMS:
            c = report[ds][arm]
            k_pass = next((k for k in KEEPS if k <= 256 and c[str(k)] >= 0.995), None)
            verdict[ds][arm] = {
                "pass": k_pass is not None,
                "first_passing_K<=256": k_pass,
                "c@128": c["128"], "c@256": c["256"],
            }
    report["gate"] = verdict
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"total {time.time()-t0:.1f}s; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
