#!/usr/bin/env python3
"""Leak-free learned maximum-relevance router for portal tiles."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from gate_deep_cell_rerank import load_routes as load_feature_routes
from gate_deep_graphfree_coverage import (
    Aggregate,
    ensure_row_buckets,
    hit_fraction,
    parse_ints,
)
from gate_deep_portal_representatives import (
    DATA,
    load_i8bin,
    load_portals,
    load_routes,
)


ROAR = Path("/home/thomas-ahle/RoarGraph/data/deep10m")


def tile_features(
    query: np.ndarray,
    cells: np.ndarray,
    cent: np.ndarray,
    bucket_sizes: np.ndarray,
    radii: np.ndarray,
    portals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(query, dtype=np.int32)
    buckets = (
        cells[:, None].astype(np.uint64) * portals
        + np.arange(portals, dtype=np.uint64)[None, :]
    )
    score = np.asarray(cent[cells], dtype=np.int32) @ q
    flat_score = score.reshape(-1).astype(np.float32)
    score_z = (flat_score - flat_score.mean()) / max(
        float(flat_score.std()), 1.0
    )
    within_order = np.argsort(np.argsort(-score, axis=1), axis=1)
    cell_rank = np.repeat(
        np.arange(len(cells), dtype=np.float32) / max(len(cells) - 1, 1),
        portals,
    )
    portal_rank = (
        within_order.reshape(-1).astype(np.float32) / (portals - 1)
    )
    size = np.asarray(bucket_sizes[buckets.reshape(-1)], dtype=np.float32)
    log_size = np.log1p(size)
    log_size_z = (log_size - 3.0) / 1.5
    cell_size = np.asarray(
        [
            bucket_sizes[int(cell) * portals : (int(cell) + 1) * portals].sum()
            for cell in cells
        ],
        dtype=np.float32,
    )
    log_cell_size = np.repeat(np.log1p(cell_size), portals)
    log_radius = np.log1p(
        np.asarray(radii[buckets.reshape(-1)], dtype=np.float32)
    )
    features = np.column_stack(
        [
            score_z,
            cell_rank,
            portal_rank,
            log_size_z,
            log_cell_size,
            log_radius,
            score_z * log_size_z,
            score_z * (1.0 - cell_rank),
            (1.0 - cell_rank) * (1.0 - portal_rank),
        ]
    ).astype(np.float32)
    return features, buckets.reshape(-1), flat_score


def train_model(
    train_query: np.ndarray,
    train_cells: np.ndarray,
    train_gt: np.ndarray,
    row_buckets: np.ndarray,
    cent: np.ndarray,
    bucket_sizes: np.ndarray,
    radii: np.ndarray,
    portals: int,
    hard_negatives: int,
    random_negatives: int,
) -> HistGradientBoostingClassifier:
    rng = np.random.default_rng(0x47454E424F)
    feature_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for qi in range(len(train_query)):
        feature, buckets, portal_score = tile_features(
            train_query[qi],
            train_cells[qi],
            cent,
            bucket_sizes,
            radii,
            portals,
        )
        targets = np.asarray(row_buckets[np.asarray(train_gt[qi, :10])])
        positive = np.isin(buckets, targets)
        positive_index = np.flatnonzero(positive)
        negative_index = np.flatnonzero(~positive)
        hard_order = negative_index[
            np.argsort(portal_score[negative_index])[::-1][
                :hard_negatives
            ]
        ]
        random_take = min(random_negatives, len(negative_index))
        random_index = rng.choice(
            negative_index, size=random_take, replace=False
        )
        chosen = np.unique(
            np.concatenate([positive_index, hard_order, random_index])
        )
        feature_rows.append(feature[chosen])
        labels.append(positive[chosen].astype(np.uint8))
        if (qi + 1) % 500 == 0:
            print(f"prepared {qi + 1}/{len(train_query)} training queries", flush=True)
    x = np.concatenate(feature_rows)
    y = np.concatenate(labels)
    print(
        f"fitting {len(y)} tile examples; positives={int(y.sum())}",
        flush=True,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=100,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=0,
    )
    model.fit(x, y)
    return model


def evaluate(
    name: str,
    query: np.ndarray,
    cells: np.ndarray,
    gt: np.ndarray,
    model: HistGradientBoostingClassifier,
    row_buckets: np.ndarray,
    cent: np.ndarray,
    bucket_sizes: np.ndarray,
    radii: np.ndarray,
    portals: int,
    bucket_grid: list[int],
) -> dict[str, object]:
    blend_grid = (0.5, 0.75, 0.9, 1.0)
    penalty_grid = (0.0, 0.1, 0.25, 0.5, 1.0)
    methods = {
        (blend, penalty): {keep: Aggregate() for keep in bucket_grid}
        for blend in blend_grid
        for penalty in penalty_grid
    }
    for qi in range(len(query)):
        feature, buckets, portal_score = tile_features(
            query[qi],
            cells[qi],
            cent,
            bucket_sizes,
            radii,
            portals,
        )
        prediction = model.predict_proba(feature)[:, 1]
        prediction = np.log(
            np.maximum(prediction, 1e-6)
            / np.maximum(1.0 - prediction, 1e-6)
        )
        prediction = (prediction - prediction.mean()) / max(
            float(prediction.std()), 1e-6
        )
        portal_z = (portal_score - portal_score.mean()) / max(
            float(portal_score.std()), 1.0
        )
        orders = {
            (blend, penalty): np.argsort(
                blend * portal_z
                + (1.0 - blend) * prediction
                - penalty * feature[:, 3]
            )[::-1]
            for blend in blend_grid
            for penalty in penalty_grid
        }
        targets = np.asarray(row_buckets[np.asarray(gt[qi, :10])])
        for keep in bucket_grid:
            for method_key, order in orders.items():
                aggregate = methods[method_key][keep]
                chosen = buckets[order[:keep]]
                aggregate.add(
                    hit_fraction(targets, chosen),
                    int(bucket_sizes[chosen].sum()),
                    keep,
                )
        if (qi + 1) % 100 == 0:
            print(f"{name}: evaluated {qi + 1}/{len(query)}", flush=True)
    rows = []
    for (blend, penalty), values in methods.items():
        method = (
            "portal"
            if blend == 1.0 and penalty == 0.0
            else (
                f"portal_learned_blend{blend:g}"
                f"_cost{penalty:g}"
            )
        )
        for keep, aggregate in values.items():
            row: dict[str, str | int | float] = {
                "method": method,
                "buckets": keep,
            }
            row.update(aggregate.result())
            rows.append(row)
    return {"name": name, "queries": len(query), "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DATA / "base.10M.i8bin")
    parser.add_argument("--public-query", type=Path, default=DATA / "query2k.i8bin")
    parser.add_argument("--public-gt", type=Path, default=DATA / "deep10m_gt.ibin")
    parser.add_argument(
        "--public-routes",
        type=Path,
        default=DATA / "deep_hier_routes_p128_public.u32",
    )
    parser.add_argument(
        "--train-query", type=Path, default=DATA / "query.train60k.i8bin"
    )
    parser.add_argument(
        "--train-gt", type=Path, default=ROAR / "train.gt.bin"
    )
    parser.add_argument(
        "--train-routes",
        type=Path,
        default=DATA / "deep10m_train60k_k128.rrf",
    )
    parser.add_argument(
        "--portals", type=Path, default=DATA / "deep10m_portals16.side"
    )
    parser.add_argument(
        "--row-buckets",
        type=Path,
        default=DATA / "deep10m_portals16.rowbuckets.u32",
    )
    parser.add_argument(
        "--sphere-radii",
        type=Path,
        default=DATA / "deep10m_portals16.radii.f32",
    )
    parser.add_argument("--train-queries", type=int, default=5_000)
    parser.add_argument("--validation-start", type=int, default=5_000)
    parser.add_argument("--validation-queries", type=int, default=2_000)
    parser.add_argument("--public-queries", type=int, default=500)
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--hard-negatives", type=int, default=64)
    parser.add_argument("--random-negatives", type=int, default=64)
    parser.add_argument("--buckets", default="32,64,96,128,160,192,256")
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA / "deep_learned_tiles_gate.json",
    )
    args = parser.parse_args()

    started = time.monotonic()
    base = load_i8bin(args.base)
    train_query = load_i8bin(args.train_query)
    public_query = load_i8bin(args.public_query)
    train_route_dump = load_feature_routes(args.train_routes)
    train_cells = np.asarray(
        train_route_dump.records["cell"][:, : args.cells],
        dtype=np.uint32,
    )
    public_cells = load_routes(args.public_routes, args.cells)
    train_gt = np.memmap(
        args.train_gt,
        dtype="<u4",
        mode="r",
        offset=8,
        shape=(500_000, 100),
    )
    public_gt = np.memmap(
        args.public_gt,
        dtype="<u4",
        mode="r",
        offset=8,
        shape=(2000, 100),
    )
    cent, offsets, ids, n, _d, portals = load_portals(args.portals)
    row_buckets = ensure_row_buckets(args.row_buckets, offsets, ids, n)
    bucket_sizes = np.diff(np.asarray(offsets, dtype=np.int64))
    radii = np.memmap(
        args.sphere_radii,
        dtype="<f4",
        mode="r",
        shape=(len(bucket_sizes),),
    )
    bucket_grid = parse_ints(args.buckets)

    model = train_model(
        train_query[: args.train_queries],
        train_cells[: args.train_queries],
        train_gt[: args.train_queries],
        row_buckets,
        cent,
        bucket_sizes,
        radii,
        portals,
        args.hard_negatives,
        args.random_negatives,
    )
    start = args.validation_start
    stop = start + args.validation_queries
    validation = evaluate(
        "disjoint-training-validation",
        train_query[start:stop],
        train_cells[start:stop],
        train_gt[start:stop],
        model,
        row_buckets,
        cent,
        bucket_sizes,
        radii,
        portals,
        bucket_grid,
    )
    public = evaluate(
        "official-public",
        public_query[: args.public_queries],
        public_cells[: args.public_queries],
        public_gt[: args.public_queries],
        model,
        row_buckets,
        cent,
        bucket_sizes,
        radii,
        portals,
        bucket_grid,
    )
    output = {
        "date_utc": time.strftime("%Y-%m-%d", time.gmtime()),
        "dataset": "DEEP-10M",
        "protocol": {
            "train_queries": args.train_queries,
            "validation_range": [start, stop],
            "public_queries": args.public_queries,
            "features": "portal score/ranks, occupancy, sphere radius and interactions",
            "labels": "bucket contains an official training top-10 neighbor",
        },
        "validation": validation,
        "public": public,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(
        f"wrote {args.out}; elapsed={output['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
