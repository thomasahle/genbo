#!/usr/bin/env python3
"""Held-out gate for confidence-dispatching DEEP search policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_results(path: Path, nq: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint32)
    if int(raw[0]) != nq:
        raise ValueError(f"{path}: nq={raw[0]}, expected {nq}")
    return raw[1:].reshape(nq, 10)


def hits(ids: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return (ids[:, :, None] == gt[:, None, :]).any(axis=2).sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("confidence", type=Path)
    parser.add_argument("low_results", type=Path)
    parser.add_argument("high_results", nargs="+", type=Path)
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path("/home/thomas-ahle/big-ann-data/deep10m/deep10m_gt.ibin"),
    )
    parser.add_argument("--max-promote", type=float, default=0.30)
    args = parser.parse_args()

    table = np.genfromtxt(args.confidence, delimiter=",", names=True)
    nq = len(table)
    features = np.column_stack(
        [table[name] for name in table.dtype.names if name != "qid"]
    ).astype(np.float64)
    # Log-compress non-negative heavy tails while preserving zero boundaries.
    features = np.log1p(features)

    gt_raw = np.fromfile(args.gt, dtype=np.int32)
    gt_nq, gt_k = map(int, gt_raw[:2])
    if gt_nq < nq:
        raise ValueError(f"ground truth has {gt_nq} queries, need {nq}")
    gt = gt_raw[2:].reshape(gt_nq, gt_k)[:nq, :10].astype(np.uint32)
    low = hits(load_results(args.low_results, nq), gt)

    models = {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "hist": lambda: HistGradientBoostingRegressor(
            max_iter=150, max_depth=3, learning_rate=0.05, l2_regularization=5.0
        ),
        "forest": lambda: RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=17,
            n_jobs=1,
        ),
    }
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    cap = int(np.floor(args.max_promote * nq))

    print(f"low recall={low.sum() / (nq * 10):.4f}; cap={cap}/{nq}")
    for high_path in args.high_results:
        high = hits(load_results(high_path, nq), gt)
        gain = high - low
        target_hits = int(high.sum())
        print(
            f"\ntarget={high_path.stem} recall={target_hits / (nq * 10):.4f} "
            f"oracle-positive={np.mean(gain > 0):.3f}"
        )
        oracle_order = np.argsort(-gain, kind="stable")
        oracle_curve = low.sum() + np.cumsum(gain[oracle_order])
        oracle_at = np.flatnonzero(oracle_curve >= target_hits)
        oracle_n = int(oracle_at[0] + 1) if len(oracle_at) else nq + 1
        print(f"  oracle promotes {oracle_n}/{nq} ({oracle_n / nq:.3f})")

        for name, factory in models.items():
            prediction = np.empty(nq, dtype=np.float64)
            for train, test in folds.split(features):
                model = factory()
                model.fit(features[train], gain[train])
                prediction[test] = model.predict(features[test])
            order = np.argsort(-prediction, kind="stable")
            curve = low.sum() + np.cumsum(gain[order])
            reached = np.flatnonzero(curve[:cap] >= target_hits)
            if len(reached):
                promoted = int(reached[0] + 1)
                verdict = f"PASS at {promoted}/{nq} ({promoted / nq:.3f})"
            else:
                promoted = cap
                verdict = (
                    f"FAIL; recall@cap={curve[cap - 1] / (nq * 10):.4f}"
                    if cap
                    else "FAIL; zero promotion cap"
                )
            corr = np.corrcoef(prediction, gain)[0, 1]
            print(f"  {name:6s} {verdict}; oof-corr={corr:.3f}")


if __name__ == "__main__":
    main()
