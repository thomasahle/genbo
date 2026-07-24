#!/usr/bin/env python3
"""Leak-free supervised-cell-reranking gate for DEEP-10M.

The model sees only the router's raw top-K fine cells. Training uses RoarGraph's
disjoint DEEP training queries/ground truth; the public 2,000 queries are held
out from model/rank/policy selection.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


DATA = Path("/home/thomas-ahle/big-ann-data/deep10m")
ROAR = Path("/home/thomas-ahle/RoarGraph/data/deep10m")
RECORD_DTYPE = np.dtype(
    [
        ("cell", "<u4"),
        ("parent", "<u4"),
        ("fine_score", "<i4"),
        ("parent_score", "<i4"),
        ("fine_norm", "<i4"),
        ("parent_norm", "<i4"),
        ("occupancy", "<u4"),
    ]
)


def prepare_inputs(args: argparse.Namespace) -> None:
    query = np.memmap(
        args.train_fbin,
        dtype="<f4",
        mode="r",
        offset=8,
        shape=(500_000, 96),
    )
    quantized = np.clip(
        np.rint(np.asarray(query[: args.dump_train_nq]) * args.query_scale),
        -127,
        127,
    ).astype(np.int8)
    with args.train_i8.open("wb") as handle:
        handle.write(struct.pack("<II", args.dump_train_nq, 96))
        handle.write(quantized.tobytes())
    environment = os.environ.copy()
    environment["SBANN_INDEX_LOAD"] = str(args.index)
    subprocess.run(
        [str(args.binary), "dumproutermeta", str(args.router_meta)],
        env=environment,
        check=True,
    )
    subprocess.run(
        [
            str(args.binary),
            "dumproutefeat",
            str(args.train_i8),
            str(args.train_routes),
            "128",
            str(args.dump_train_nq),
        ],
        env=environment,
        check=True,
    )
    subprocess.run(
        [
            str(args.binary),
            "dumproutefeat",
            str(args.eval_i8),
            str(args.eval_routes),
            "128",
            "2000",
        ],
        env=environment,
        check=True,
    )


@dataclass
class RouteDump:
    backing: np.memmap
    query: np.ndarray
    records: np.ndarray
    nq: int
    keep: int
    d: int


def load_routes(path: Path) -> RouteDump:
    with path.open("rb") as handle:
        header = handle.read(20)
    if header[:4] != b"RRF1":
        raise ValueError(f"{path}: not an RRF1 route-feature dump")
    nq, keep, d, words = struct.unpack("<4I", header[4:])
    if words * 4 != RECORD_DTYPE.itemsize:
        raise ValueError(f"{path}: record width {words}, expected 7")
    stride = d + keep * RECORD_DTYPE.itemsize
    expected = 20 + nq * stride
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: size {path.stat().st_size}, expected {expected}")
    backing = np.memmap(path, dtype=np.uint8, mode="r")
    query = np.ndarray(
        (nq, d),
        dtype=np.int8,
        buffer=backing,
        offset=20,
        strides=(stride, 1),
    )
    records = np.ndarray(
        (nq, keep),
        dtype=RECORD_DTYPE,
        buffer=backing,
        offset=20 + d,
        strides=(stride, RECORD_DTYPE.itemsize),
    )
    return RouteDump(backing, query, records, nq, keep, d)


def load_gt(path: Path, nq: int) -> np.ndarray:
    header = np.fromfile(path, dtype="<u4", count=2)
    if len(header) != 2 or int(header[0]) < nq:
        raise ValueError(f"{path}: insufficient ground truth")
    return np.memmap(
        path,
        dtype="<u4",
        mode="r",
        offset=8,
        shape=(int(header[0]), int(header[1])),
    )[:nq, :10]


def load_router_centroids(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.read(12)
    if header[:4] != b"RCM1":
        raise ValueError(f"{path}: not an RCM1 router-metadata dump")
    n_cells, d = struct.unpack("<2I", header[4:])
    stride = 8 + d
    if path.stat().st_size != 12 + n_cells * stride:
        raise ValueError(f"{path}: malformed router metadata")
    backing = np.memmap(path, dtype=np.uint8, mode="r")
    return np.ndarray(
        (n_cells, d),
        dtype=np.int8,
        buffer=backing,
        offset=20,
        strides=(stride, 1),
    ).copy()


def build_assignments(path: Path, nb: int) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.memmap(path, dtype="<u4", mode="r").reshape(-1, 2)
    low = np.full(nb, np.uint32(2**32 - 1), dtype=np.uint32)
    high = np.zeros(nb, dtype=np.uint32)
    for start in range(0, len(pairs), 2_000_000):
        chunk = pairs[start : start + 2_000_000]
        np.minimum.at(low, chunk[:, 0], chunk[:, 1])
        np.maximum.at(high, chunk[:, 0], chunk[:, 1])
    if np.any(low == np.uint32(2**32 - 1)):
        raise ValueError("assignment dump does not cover every base row")
    return low, high


def hit_mask(
    records: np.ndarray,
    gt: np.ndarray,
    assign_low: np.ndarray,
    assign_high: np.ndarray,
) -> np.ndarray:
    cells = records["cell"]
    low = assign_low[gt]
    high = assign_high[gt]
    return (cells[:, :, None] == low[:, None, :]) | (
        cells[:, :, None] == high[:, None, :]
    )


def greedy_targets(hit: np.ndarray, budget: int = 10) -> np.ndarray:
    """Marginal-gain labels for a nonredundant oracle cell cover."""
    nq, keep, ngt = hit.shape
    remaining = np.ones((nq, ngt), dtype=bool)
    target = np.zeros((nq, keep), dtype=np.uint8)
    rows = np.arange(nq)
    for _ in range(budget):
        gain = (hit & remaining[:, None, :]).sum(axis=2)
        chosen = gain.argmax(axis=1)
        best = gain[rows, chosen]
        active = best > 0
        if not np.any(active):
            break
        active_rows = rows[active]
        active_chosen = chosen[active]
        target[active_rows, active_chosen] = best[active].astype(np.uint8)
        remaining[active_rows] &= ~hit[active_rows, active_chosen]
    return target


def scalar_features(records: np.ndarray) -> np.ndarray:
    """Cheap features available after the existing router has scored its shortlist."""

    def zscore(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32)
        mean = values.mean(axis=1, keepdims=True)
        scale = values.std(axis=1, keepdims=True)
        return (values - mean) / np.maximum(scale, 1.0)

    fine = records["fine_score"].astype(np.float32)
    parent = records["parent_score"].astype(np.float32)
    gap = np.empty_like(fine)
    gap[:, 0] = 0
    gap[:, 1:] = fine[:, 1:] - fine[:, :-1]
    rank = np.linspace(0.0, 1.0, records.shape[1], dtype=np.float32)
    rank = np.broadcast_to(rank, fine.shape)
    log_occupancy = np.log1p(records["occupancy"].astype(np.float32))
    return np.stack(
        [
            rank,
            zscore(fine),
            zscore(np.log1p(np.maximum(fine - fine[:, :1], 0.0))),
            zscore(gap),
            zscore(parent),
            zscore(fine - parent),
            zscore(records["fine_norm"]),
            zscore(records["parent_norm"]),
            zscore(log_occupancy),
        ],
        axis=2,
    )


class CellRanker(nn.Module):
    """Tiny deployable low-rank query/cell interaction plus scalar correction."""

    def __init__(
        self,
        d: int,
        n_cells: int,
        n_parents: int,
        n_scalar: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scalar = nn.Linear(n_scalar, 1)
        with torch.no_grad():
            self.scalar.weight.zero_()
            self.scalar.weight[0, 1] = -1.0
            self.scalar.bias.zero_()
        if rank:
            self.query_projection = nn.Linear(d, rank, bias=False)
            self.cell_embedding = nn.Embedding(n_cells, rank)
            self.parent_embedding = nn.Embedding(n_parents, rank)
            self.cell_bias = nn.Embedding(n_cells, 1)
            nn.init.normal_(self.query_projection.weight, std=0.01)
            nn.init.normal_(self.cell_embedding.weight, std=0.01)
            nn.init.normal_(self.parent_embedding.weight, std=0.01)
            nn.init.zeros_(self.cell_bias.weight)

    def forward(
        self,
        query: torch.Tensor,
        cell: torch.Tensor,
        parent: torch.Tensor,
        scalar: torch.Tensor,
    ) -> torch.Tensor:
        score = self.scalar(scalar).squeeze(-1)
        if self.rank:
            qproj = self.query_projection(query)
            candidate = self.cell_embedding(cell) + self.parent_embedding(parent)
            score = score + (candidate * qproj[:, None, :]).sum(dim=2)
            score = score + self.cell_bias(cell).squeeze(-1)
        return score


class GeometryRanker(nn.Module):
    """Shared geometric correction: diagonal or full bilinear q×centroid map."""

    def __init__(
        self,
        centroids: np.ndarray,
        n_scalar: int,
        kind: str,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.scalar = nn.Linear(n_scalar, 1)
        with torch.no_grad():
            self.scalar.weight.zero_()
            self.scalar.weight[0, 1] = -1.0
            self.scalar.bias.zero_()
        self.register_buffer(
            "centroids",
            torch.from_numpy(centroids.astype(np.float32) / 127.0),
        )
        d = centroids.shape[1]
        if kind == "diag":
            self.dimension_weight = nn.Parameter(torch.zeros(d))
        elif kind == "full":
            self.query_projection = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.query_projection.weight)
        else:
            raise ValueError(kind)

    def forward(
        self,
        query: torch.Tensor,
        cell: torch.Tensor,
        _parent: torch.Tensor,
        scalar: torch.Tensor,
    ) -> torch.Tensor:
        score = self.scalar(scalar).squeeze(-1)
        centroid = self.centroids[cell]
        if self.kind == "diag":
            correction = (
                centroid * query[:, None, :] * self.dimension_weight
            ).sum(dim=2)
        else:
            transformed = self.query_projection(query)
            correction = (centroid * transformed[:, None, :]).sum(dim=2)
        return score + correction / np.sqrt(query.shape[1])


def coverage_at(
    scores: np.ndarray,
    hit: np.ndarray,
    budget: int,
) -> tuple[float, np.ndarray]:
    chosen = np.argpartition(scores, -budget, axis=1)[:, -budget:]
    selected = np.take_along_axis(hit, chosen[:, :, None], axis=1)
    per_query = selected.any(axis=1).mean(axis=1)
    return float(per_query.mean()), per_query


def raw_coverage(hit: np.ndarray, budget: int) -> tuple[float, np.ndarray]:
    per_query = hit[:, :budget].any(axis=1).mean(axis=1)
    return float(per_query.mean()), per_query


def apply_selection_policy(
    model_scores: np.ndarray,
    records: np.ndarray,
    cap: int,
    raw_weight: float,
) -> np.ndarray:
    raw_score = -scalar_features(records)[:, :, 1]
    scores = model_scores + raw_weight * raw_score
    if cap < scores.shape[1]:
        scores[:, cap:] = -np.inf
    return scores


def tune_selection_policy(
    model_scores: np.ndarray,
    records: np.ndarray,
    hit: np.ndarray,
) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    raw_score = -scalar_features(records)[:, :, 1]
    for cap in (10, 12, 15, 20, 24, 32, 48, 64, records.shape[1]):
        for raw_weight in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            scores = model_scores + raw_weight * raw_score
            if cap < scores.shape[1]:
                scores = scores.copy()
                scores[:, cap:] = -np.inf
            coverage, _ = coverage_at(scores, hit, 10)
            candidate = {
                "cap": cap,
                "raw_weight": raw_weight,
                "coverage_at_10": coverage,
            }
            if best is None or (
                coverage,
                -cap,
                -raw_weight,
            ) > (
                float(best["coverage_at_10"]),
                -int(best["cap"]),
                -float(best["raw_weight"]),
            ):
                best = candidate
    assert best is not None
    return best


def score_model(
    model: nn.Module,
    dump: RouteDump,
    start: int,
    end: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    result = np.empty((end - start, dump.keep), dtype=np.float32)
    with torch.no_grad():
        for left in range(start, end, batch_size):
            right = min(end, left + batch_size)
            scalar = scalar_features(dump.records[left:right])
            scores = model(
                torch.from_numpy(np.asarray(dump.query[left:right], dtype=np.float32) / 127.0).to(device),
                torch.from_numpy(np.asarray(dump.records["cell"][left:right], dtype=np.int64)).to(device),
                torch.from_numpy(np.asarray(dump.records["parent"][left:right], dtype=np.int64)).to(device),
                torch.from_numpy(scalar).to(device),
            )
            result[left - start : right - start] = scores.cpu().numpy()
    return result


def train_one(
    dump: RouteDump,
    target: np.ndarray,
    train_nq: int,
    val_hit: np.ndarray,
    rank: int | None,
    geometry: str | None,
    centroids: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_cells = int(dump.records["cell"].max()) + 1
    n_parents = int(dump.records["parent"].max()) + 1
    if geometry is not None:
        model: nn.Module = GeometryRanker(centroids, 9, geometry).to(device)
        model_name = geometry
    else:
        assert rank is not None
        model = CellRanker(dump.d, n_cells, n_parents, 9, rank).to(device)
        model_name = "scalar" if rank == 0 else f"cell{rank}"
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_coverage = -1.0
    best_policy: dict[str, float | int] | None = None

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_nq)
        losses: list[float] = []
        for offset in range(0, train_nq, batch_size):
            index = order[offset : offset + batch_size]
            scalar = scalar_features(dump.records[index])
            weights = np.asarray(target[index], dtype=np.float32)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
            scores = model(
                torch.from_numpy(np.asarray(dump.query[index], dtype=np.float32) / 127.0).to(device),
                torch.from_numpy(np.asarray(dump.records["cell"][index], dtype=np.int64)).to(device),
                torch.from_numpy(np.asarray(dump.records["parent"][index], dtype=np.int64)).to(device),
                torch.from_numpy(scalar).to(device),
            )
            truth = torch.from_numpy(weights).to(device)
            loss = -(truth * torch.log_softmax(scores, dim=1)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_scores = score_model(
            model, dump, train_nq, dump.nq, device, batch_size
        )
        val_coverage, _ = coverage_at(val_scores, val_hit, 10)
        policy = tune_selection_policy(
            val_scores, dump.records[train_nq:], val_hit
        )
        row = {
            "epoch": float(epoch + 1),
            "loss": float(np.mean(losses)),
            "val_coverage_at_10": val_coverage,
            "val_policy_coverage_at_10": float(policy["coverage_at_10"]),
        }
        history.append(row)
        print(
            f"model={model_name:6s} epoch={epoch + 1:2d} "
            f"loss={row['loss']:.4f} val_cov10={val_coverage:.4f} "
            f"policy={row['val_policy_coverage_at_10']:.4f}"
            f"/cap{policy['cap']}/w{policy['raw_weight']}",
            flush=True,
        )
        if float(policy["coverage_at_10"]) > best_coverage:
            best_coverage = float(policy["coverage_at_10"])
            best_policy = policy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    assert best_state is not None and best_policy is not None
    model.load_state_dict(best_state)
    return model, {
        "model": model_name,
        "best_val_coverage_at_10": best_coverage,
        "selection_policy": best_policy,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-routes", type=Path, default=DATA / "deep10m_train60k_k128.rrf"
    )
    parser.add_argument(
        "--eval-routes", type=Path, default=DATA / "deep10m_eval_k128.rrf"
    )
    parser.add_argument("--train-gt", type=Path, default=ROAR / "train.gt.bin")
    parser.add_argument("--eval-gt", type=Path, default=DATA / "deep10m_gt.ibin")
    parser.add_argument(
        "--assign", type=Path, default=DATA / "deep10m_assign.u32"
    )
    parser.add_argument(
        "--router-meta", type=Path, default=DATA / "deep10m_router.rcm"
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("/home/thomas-ahle/genbo/sbann-rs/target/release/sbann"),
    )
    parser.add_argument(
        "--index", type=Path, default=DATA / "eng_deep10m_kf65536.idx"
    )
    parser.add_argument(
        "--train-fbin", type=Path, default=ROAR / "query.train.fbin"
    )
    parser.add_argument(
        "--train-i8", type=Path, default=DATA / "query.train60k.i8bin"
    )
    parser.add_argument(
        "--eval-i8", type=Path, default=DATA / "query2k.i8bin"
    )
    parser.add_argument("--dump-train-nq", type=int, default=60_000)
    parser.add_argument("--query-scale", type=float, default=252.92)
    parser.add_argument("--nb", type=int, default=10_000_000)
    parser.add_argument("--train-nq", type=int, default=50_000)
    parser.add_argument("--ranks", default="0,4,8,16")
    parser.add_argument("--geometry-models", default="diag,full")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--out", type=Path, default=DATA / "deep_cell_rerank_gate.json"
    )
    parser.add_argument(
        "--model-out", type=Path, default=DATA / "deep_cell_rerank_best.pt"
    )
    args = parser.parse_args()

    if args.prepare:
        prepare_inputs(args)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    started = time.monotonic()
    train_dump = load_routes(args.train_routes)
    eval_dump = load_routes(args.eval_routes)
    if train_dump.keep != eval_dump.keep or train_dump.d != eval_dump.d:
        raise ValueError("train/eval route dump geometry mismatch")
    if not 0 < args.train_nq < train_dump.nq:
        raise ValueError("--train-nq must leave a nonempty validation split")

    assign_low, assign_high = build_assignments(args.assign, args.nb)
    centroids = load_router_centroids(args.router_meta)
    train_gt = load_gt(args.train_gt, train_dump.nq)
    eval_gt = load_gt(args.eval_gt, eval_dump.nq)
    train_target = np.zeros((train_dump.nq, train_dump.keep), dtype=np.uint8)
    for start in range(0, train_dump.nq, 2_000):
        end = min(train_dump.nq, start + 2_000)
        hit = hit_mask(
            train_dump.records[start:end],
            train_gt[start:end],
            assign_low,
            assign_high,
        )
        train_target[start:end] = greedy_targets(hit)
    val_hit = hit_mask(
        train_dump.records[args.train_nq :],
        train_gt[args.train_nq :],
        assign_low,
        assign_high,
    )
    eval_hit = hit_mask(
        eval_dump.records,
        eval_gt,
        assign_low,
        assign_high,
    )

    val_raw10, _ = raw_coverage(val_hit, 10)
    val_raw15, _ = raw_coverage(val_hit, 15)
    val_oracle10, _ = coverage_at(
        train_target[args.train_nq :].astype(np.float32), val_hit, 10
    )
    report: dict[str, Any] = {
        "protocol": {
            "train_queries": args.train_nq,
            "validation_queries": train_dump.nq - args.train_nq,
            "evaluation_queries": eval_dump.nq,
            "candidate_keep": train_dump.keep,
            "selection_budget": 10,
            "target": "match raw top-15 true-neighbor cell coverage",
        },
        "validation": {
            "raw_coverage_at_10": val_raw10,
            "raw_coverage_at_15": val_raw15,
            "oracle_coverage_at_10": val_oracle10,
        },
        "models": [],
    }
    print(
        f"validation raw10={val_raw10:.4f} raw15={val_raw15:.4f} "
        f"oracle10={val_oracle10:.4f}",
        flush=True,
    )

    best_model: nn.Module | None = None
    best_summary: dict[str, Any] | None = None
    for rank in [int(value) for value in args.ranks.split(",") if value]:
        model, summary = train_one(
            train_dump,
            train_target,
            args.train_nq,
            val_hit,
            rank,
            None,
            centroids,
            args.epochs,
            args.batch_size,
            device,
            args.seed,
        )
        report["models"].append(summary)
        if best_summary is None or summary["best_val_coverage_at_10"] > best_summary[
            "best_val_coverage_at_10"
        ]:
            best_model = model
            best_summary = summary
    for geometry in [
        value for value in args.geometry_models.split(",") if value
    ]:
        model, summary = train_one(
            train_dump,
            train_target,
            args.train_nq,
            val_hit,
            None,
            geometry,
            centroids,
            args.epochs,
            args.batch_size,
            device,
            args.seed,
        )
        report["models"].append(summary)
        if best_summary is None or summary["best_val_coverage_at_10"] > best_summary[
            "best_val_coverage_at_10"
        ]:
            best_model = model
            best_summary = summary

    assert best_model is not None and best_summary is not None
    # Public queries are evaluated only after selecting model capacity, epoch,
    # shortlist cap, and raw-score blend on the disjoint validation split.
    eval_scores = score_model(
        best_model, eval_dump, 0, eval_dump.nq, device, args.batch_size
    )
    policy = best_summary["selection_policy"]
    eval_policy_scores = apply_selection_policy(
        eval_scores,
        eval_dump.records,
        int(policy["cap"]),
        float(policy["raw_weight"]),
    )
    eval_learned, per_query_learned = coverage_at(
        eval_policy_scores, eval_hit, 10
    )
    eval_raw10, per_query_raw10 = raw_coverage(eval_hit, 10)
    eval_raw15, per_query_raw15 = raw_coverage(eval_hit, 15)
    eval_oracle, _ = coverage_at(
        greedy_targets(eval_hit).astype(np.float32), eval_hit, 10
    )
    report["selected"] = best_summary
    report["evaluation"] = {
        "raw_coverage_at_10": eval_raw10,
        "raw_coverage_at_15": eval_raw15,
        "learned_coverage_at_10": eval_learned,
        "oracle_coverage_at_10": eval_oracle,
        "learned_minus_raw15": eval_learned - eval_raw15,
        "queries_learned_ge_raw15": float(
            np.mean(per_query_learned >= per_query_raw15)
        ),
        "queries_learned_gt_raw10": float(
            np.mean(per_query_learned > per_query_raw10)
        ),
        "gate_pass": bool(eval_learned >= eval_raw15),
    }
    report["elapsed_s"] = time.monotonic() - started
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "model": best_summary["model"],
            "d": train_dump.d,
            "n_cells": int(train_dump.records["cell"].max()) + 1,
            "n_parents": int(train_dump.records["parent"].max()) + 1,
            "n_scalar": 9,
        },
        args.model_out,
    )
    print(json.dumps(report["evaluation"], indent=2, sort_keys=True))
    print(f"wrote {args.out} and {args.model_out}", flush=True)


if __name__ == "__main__":
    main()
