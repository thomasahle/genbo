#!/usr/bin/env python3
"""Build the opt-in finest-centroid graph-router sidecars.

The input is the `RCM1` file emitted by `sbann dumproutermeta`.  This script
extracts its centroid rows, invokes genbo's own NN-descent and Vamana pruning,
and writes deterministic farthest-point landmark ids.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
from pathlib import Path

import numpy as np


def load_centroids(path: Path) -> np.ndarray:
    header = path.read_bytes()[:12]
    if header[:4] != b"RCM1":
        raise ValueError(f"{path}: not an RCM1 router-metadata dump")
    n, d = struct.unpack("<II", header[4:])
    stride = 8 + d
    if path.stat().st_size != 12 + n * stride:
        raise ValueError(f"{path}: malformed RCM1 length")
    backing = np.memmap(path, dtype=np.uint8, mode="r")
    return np.ndarray(
        (n, d),
        dtype=np.int8,
        buffer=backing,
        offset=20,
        strides=(stride, 1),
    )


def write_i8bin(path: Path, centroids: np.ndarray) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<II", *centroids.shape))
        handle.write(np.asarray(centroids).tobytes())


def farthest_landmarks(centroids: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(centroids, dtype=np.int16)
    landmarks = np.empty(count, dtype=np.uint32)
    landmarks[0] = 0
    delta = points - points[0]
    nearest = (delta.astype(np.int32) ** 2).sum(axis=1)
    for index in range(1, count):
        chosen = int(nearest.argmax())
        landmarks[index] = chosen
        delta = points - points[chosen]
        distance = (delta.astype(np.int32) ** 2).sum(axis=1)
        np.minimum(nearest, distance, out=nearest)
    return landmarks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("router_meta", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("/home/thomas-ahle/genbo/sbann-rs/target/release/sbann"),
    )
    parser.add_argument("--pool-k", type=int, default=32)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--landmarks", type=int, default=64)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    centroids = load_centroids(args.router_meta)
    i8bin = args.output_prefix.with_suffix(".i8bin")
    pool = args.output_prefix.with_name(
        args.output_prefix.name + f"_knn{args.pool_k}.u32"
    )
    graph = args.output_prefix.with_name(
        args.output_prefix.name + f"_vamana{args.degree}.u32"
    )
    landmarks = args.output_prefix.with_name(
        args.output_prefix.name + f"_landmarks{args.landmarks}.u32"
    )
    write_i8bin(i8bin, centroids)
    environment = os.environ.copy()
    environment["SBANN_ND_L2"] = "1"
    environment["RAYON_NUM_THREADS"] = str(args.threads)
    subprocess.run(
        [
            str(args.binary),
            "nndescent",
            str(i8bin),
            str(pool),
            str(args.pool_k),
        ],
        env=environment,
        check=True,
    )
    environment["SBANN_PRUNE_REVERSE"] = "1"
    subprocess.run(
        [
            str(args.binary),
            "prune",
            str(i8bin),
            str(pool),
            str(graph),
            str(args.pool_k),
            str(args.degree),
            str(args.alpha),
        ],
        env=environment,
        check=True,
    )
    farthest_landmarks(centroids, args.landmarks).tofile(landmarks)
    print(f"centroids: {i8bin}")
    print(f"graph: {graph}")
    print(f"landmarks: {landmarks}")


if __name__ == "__main__":
    main()
