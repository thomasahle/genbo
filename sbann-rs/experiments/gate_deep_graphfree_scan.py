#!/usr/bin/env python3
"""Engine-faithful SQ4 selection gate for direct global portal fan-out."""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

from gate_deep_graphfree_coverage import load_bin, parse_ints
from gate_deep_portal_representatives import (
    DATA,
    load_i8bin,
    load_portals,
    load_routes,
)


SQ4_MAGIC = b"SBPSQ4\0\0"


def load_sq4(path: Path) -> tuple[np.ndarray, np.memmap, int]:
    with path.open("rb") as source:
        magic, nassign, d, stride = struct.unpack("<8sQII", source.read(24))
        step = np.frombuffer(source.read(d * 4), dtype="<f4").copy()
    if magic != SQ4_MAGIC:
        raise ValueError(f"{path}: bad SQ4 magic")
    codes = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=24 + d * 4,
        shape=(nassign, stride),
    )
    return step, codes, stride


def top_unique_ids(
    scores: np.ndarray, ids: np.ndarray, keep: int
) -> np.ndarray:
    if not len(ids):
        return np.empty(0, dtype=np.uint32)
    take = min(len(ids), keep * 3)
    selected = np.argpartition(scores, len(scores) - take)[-take:]
    selected = selected[np.argsort(scores[selected])[::-1]]
    out: list[int] = []
    seen: set[int] = set()
    for index in selected:
        value = int(ids[index])
        if value not in seen:
            seen.add(value)
            out.append(value)
            if len(out) == keep:
                break
    return np.asarray(out, dtype=np.uint32)


class Aggregate:
    def __init__(self) -> None:
        self.recall: list[float] = []
        self.rows: list[int] = []
        self.unique: list[int] = []

    def add(self, recall: float, rows: int, unique: int) -> None:
        self.recall.append(recall)
        self.rows.append(rows)
        self.unique.append(unique)

    def result(self) -> dict[str, float]:
        rows = np.asarray(self.rows)
        return {
            "recall_at_10": float(np.mean(self.recall)),
            "mean_scanned_assignments": float(rows.mean()),
            "p95_scanned_assignments": float(np.percentile(rows, 95)),
            "mean_unique_survivors": float(np.mean(self.unique)),
            "logical_sq4_kib": float(rows.mean() * 48 / 1024),
            "padded_sq4_kib": float(rows.mean() * 64 / 1024),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-i8", type=Path, default=DATA / "query2k.i8bin")
    parser.add_argument("--query-f32", type=Path, default=DATA / "query2k.fbin")
    parser.add_argument("--base-f32", type=Path, default=DATA / "base.10M.fbin")
    parser.add_argument("--gt", type=Path, default=DATA / "deep10m_gt.ibin")
    parser.add_argument(
        "--routes",
        type=Path,
        default=DATA / "deep_hier_routes_p128_public.u32",
    )
    parser.add_argument(
        "--portals", type=Path, default=DATA / "deep10m_portals16.side"
    )
    parser.add_argument(
        "--sq4", type=Path, default=DATA / "deep10m_portals16.sq4p64"
    )
    parser.add_argument("--nq", type=int, default=500)
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument(
        "--ranker", choices=("portal", "support"), default="portal"
    )
    parser.add_argument("--support-witnesses", type=int, default=8)
    parser.add_argument("--support-blend", type=float, default=0.25)
    parser.add_argument("--size-penalty", type=float, default=0.0)
    parser.add_argument("--buckets", default="64,96,128,160,192,256")
    parser.add_argument("--survivors", default="64,128,256,512")
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA / "deep_graphfree_scan_gate.json",
    )
    args = parser.parse_args()

    started = time.monotonic()
    query_i8 = load_i8bin(args.query_i8)
    query_f32 = load_bin(args.query_f32, "<f4")
    base_f32 = load_bin(args.base_f32, "<f4")
    gt = load_bin(args.gt, "<u4")
    routes = load_routes(args.routes, args.cells)
    cent, offsets, ids, n, d, portals = load_portals(args.portals)
    step, codes, stride = load_sq4(args.sq4)
    if base_f32.shape != (n, d) or codes.shape[0] != len(ids):
        raise ValueError("base/portal/SQ4 geometry mismatch")
    nq = min(args.nq, len(query_i8), len(query_f32), len(gt), len(routes))
    bucket_grid = parse_ints(args.buckets)
    survivor_grid = parse_ints(args.survivors)
    if bucket_grid[-1] > args.cells * portals:
        raise ValueError("bucket grid exceeds routed portal candidates")
    results = {
        (bucket, survivor): Aggregate()
        for bucket in bucket_grid
        for survivor in survivor_grid
    }

    for qi in range(nq):
        q32 = np.asarray(query_i8[qi], dtype=np.int32)
        qf = np.asarray(query_f32[qi], dtype=np.float32)
        route = np.asarray(routes[qi], dtype=np.uint32)
        portal_scores = (
            np.asarray(cent[route], dtype=np.int32) @ q32
        ).reshape(-1)
        bucket_ids = (
            route[:, None].astype(np.uint64) * portals
            + np.arange(portals, dtype=np.uint64)[None, :]
        ).reshape(-1)
        rank_score: np.ndarray = portal_scores
        size = np.asarray(
            offsets[bucket_ids + 1] - offsets[bucket_ids],
            dtype=np.int64,
        )
        if args.ranker == "support":
            lo = np.asarray(offsets[bucket_ids], dtype=np.int64)
            valid = size > 0
            fractions = (
                np.arange(args.support_witnesses) * 2 + 1
            ) / (2 * args.support_witnesses)
            positions = lo[:, None] + np.minimum(
                (size[:, None] * fractions[None, :]).astype(np.int64),
                np.maximum(size[:, None] - 1, 0),
            )
            support = np.full(
                (len(bucket_ids), args.support_witnesses),
                np.iinfo(np.int32).min,
            )
            representative_ids = np.asarray(
                ids[positions[valid]], dtype=np.uint32
            )
            support[valid] = (
                np.asarray(base_f32[representative_ids], dtype=np.float32)
                @ qf
                * 64_000
            ).astype(np.int32)
            support_score = support.max(axis=1).astype(np.float64)
            portal_z = (
                portal_scores - portal_scores.mean()
            ) / max(float(portal_scores.std()), 1.0)
            support_z = (
                support_score - support_score.mean()
            ) / max(float(support_score.std()), 1.0)
            rank_score = (
                args.support_blend * support_z
                + (1.0 - args.support_blend) * portal_z
            )
        if args.size_penalty:
            rank_score = (
                rank_score - np.mean(rank_score)
            ) / max(float(np.std(rank_score)), 1.0)
            log_size_z = (np.log1p(size) - 3.0) / 1.5
            rank_score = rank_score - args.size_penalty * log_size_z
        order = np.argsort(rank_score)[::-1][: bucket_grid[-1]]
        ranked_buckets = bucket_ids[order]

        weighted = q32.astype(np.float32) * step
        scale = 127.0 / max(float(np.abs(weighted).max()), 1e-9)
        q4 = np.rint(weighted * scale).clip(-127, 127).astype(np.int32)
        qe, qo = q4[0::2], q4[1::2]

        id_chunks: list[np.ndarray] = []
        score_chunks: list[np.ndarray] = []
        ends: list[int] = []
        total = 0
        for bucket in ranked_buckets:
            lo, hi = int(offsets[bucket]), int(offsets[bucket + 1])
            packed = np.asarray(codes[lo:hi, : d // 2], dtype=np.uint8)
            if len(packed):
                score = (
                    (packed & 15).astype(np.int32) @ qe
                    + (packed >> 4).astype(np.int32) @ qo
                )
                id_chunks.append(np.asarray(ids[lo:hi], dtype=np.uint32))
                score_chunks.append(score)
                total += hi - lo
            ends.append(total)
        all_ids = (
            np.concatenate(id_chunks)
            if id_chunks
            else np.empty(0, dtype=np.uint32)
        )
        all_scores = (
            np.concatenate(score_chunks)
            if score_chunks
            else np.empty(0, dtype=np.int32)
        )
        gt10 = np.asarray(gt[qi, :10], dtype=np.uint32)

        for bucket in bucket_grid:
            end = ends[bucket - 1]
            prefix_ids = all_ids[:end]
            prefix_scores = all_scores[:end]
            for survivor in survivor_grid:
                candidates = top_unique_ids(
                    prefix_scores, prefix_ids, survivor
                )
                if len(candidates):
                    exact = np.asarray(base_f32[candidates]) @ qf
                    take = min(10, len(exact))
                    top = np.argpartition(exact, len(exact) - take)[-take:]
                    answer = candidates[top]
                    recall = float(np.isin(gt10, answer).sum() / 10)
                else:
                    recall = 0.0
                results[bucket, survivor].add(
                    recall, len(prefix_ids), len(candidates)
                )

        if (qi + 1) % 50 == 0:
            print(f"scored {qi + 1}/{nq} queries", flush=True)

    rows = []
    for (bucket, survivor), aggregate in results.items():
        row: dict[str, float | int] = {
            "cells": args.cells,
            "buckets": bucket,
            "sq4_survivors": survivor,
        }
        row.update(aggregate.result())
        rows.append(row)
    output = {
        "date_utc": time.strftime("%Y-%m-%d", time.gmtime()),
        "dataset": "DEEP-10M",
        "protocol": {
            "queries": nq,
            "bucket_ranker": (
                "global int8 portal-centroid dot"
                if args.ranker == "portal"
                else (
                    f"portal/support blend={args.support_blend:g}, "
                    f"witnesses={args.support_witnesses}"
                )
            ),
            "selector": "portal-order robust SQ4, then exact f32 top-10",
            "size_penalty": args.size_penalty,
        },
        "results": rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(
        f"wrote {args.out}; elapsed={output['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
