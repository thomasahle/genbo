"""Stress compact centroid decoding for the smoothed score-max IVF route.

The existing IVF diagnostic scores every facility centroid at query time.  This
script keeps that exact scoring path as the control and asks the sharper
question: can a compact routing index over the centroids recover the same top
few facility buckets while scoring far fewer than all centers?

This is still a diagnostic, not a new theorem form.  Failure means that this
particular centroid-decoder implementation does not discharge the open
``asm:smoothed-score-max`` primitive.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from smoothed_scoremax_ivf_stress import (
    CentroidRouter,
    _normalize_rows,
    parse_csv_list,
)
from smoothed_scoremax_stress import (
    _candidate_best_loss,
    _quantile_with_inf,
    promised_spherical_instance,
    rho_for_c,
    smoothed_scores,
)


FIELDNAMES = [
    "n",
    "d",
    "c",
    "rho",
    "near_corr",
    "far_corr_max",
    "t_scale",
    "router",
    "centers",
    "center_exponent",
    "center_mult",
    "lloyd_iters",
    "decoder",
    "center_trees",
    "center_fanout",
    "center_leaf_size",
    "center_beam",
    "center_hash_bits",
    "center_hash_probes",
    "center_graph_degree",
    "center_graph_pivots",
    "center_graph_entries",
    "center_graph_ef",
    "probes",
    "bucket_mean",
    "bucket_q90",
    "query_trials",
    "score_gap_fail_rate",
    "oracle_top1_valid_rate",
    "exact_p_bucket_rank",
    "exact_scoremax_bucket_rank",
    "exact_p_rate",
    "exact_scoremax_rate",
    "exact_additive_success_rate",
    "exact_candidate_mean",
    "exact_candidate_q90",
    "exact_candidate_over_nrho",
    "exact_best_loss_over_margin_q90",
    "center_top1_rate",
    "center_probe_recall_mean",
    "decoder_p_rate",
    "decoder_scoremax_rate",
    "decoder_additive_success_rate",
    "decoder_candidate_mean",
    "decoder_candidate_q90",
    "decoder_candidate_over_nrho",
    "decoder_best_loss_over_margin_q90",
    "center_work_mean",
    "center_work_q90",
    "center_work_over_centers",
    "center_leaf_candidate_mean",
    "center_leaf_candidate_q90",
    "seed",
]


@dataclass(frozen=True)
class DecodeStats:
    work: int
    leaf_candidates: int
    child_scores: int


class ExactCenterDecoder:
    """Control decoder that scores every centroid."""

    name = "exact"

    def __init__(self, centers: np.ndarray) -> None:
        self.centers = np.asarray(centers, dtype=float)

    def query_top(self, y: np.ndarray, top: int) -> tuple[np.ndarray, DecodeStats]:
        scores = self.centers @ y
        k = min(max(1, top), len(scores))
        ids = np.argpartition(-scores, k - 1)[:k]
        ids = ids[np.argsort(-scores[ids])]
        stats = DecodeStats(
            work=len(scores),
            leaf_candidates=len(scores),
            child_scores=0,
        )
        return ids.astype(int), stats


class _CenterNode:
    __slots__ = ("idx", "children", "centroids")

    def __init__(self, idx: np.ndarray) -> None:
        self.idx = idx
        self.children: list[_CenterNode] | None = None
        self.centroids: np.ndarray | None = None


class RPCenterDecoder:
    """Balanced random-projection beam tree over facility centroids."""

    name = "rptree"

    def __init__(
        self,
        centers: np.ndarray,
        *,
        trees: int,
        fanout: int,
        leaf_size: int,
        beam: int,
        seed: int,
    ) -> None:
        if trees < 1:
            raise ValueError("trees must be positive")
        if fanout < 2:
            raise ValueError("fanout must be at least two")
        if leaf_size < 1:
            raise ValueError("leaf_size must be positive")
        if beam < 1:
            raise ValueError("beam must be positive")
        self.centers = np.asarray(centers, dtype=float)
        self.trees = trees
        self.fanout = fanout
        self.leaf_size = leaf_size
        self.beam = beam
        rng = np.random.default_rng(seed)
        ids = np.arange(len(self.centers), dtype=int)
        self.roots = [self._build(ids, rng, depth=0) for _ in range(trees)]

    def _build(
        self,
        idx: np.ndarray,
        rng: np.random.Generator,
        depth: int,
    ) -> _CenterNode:
        node = _CenterNode(idx)
        if len(idx) <= self.leaf_size or depth >= 40:
            return node

        direction = rng.standard_normal(self.centers.shape[1])
        projection = self.centers[idx] @ direction
        order = idx[np.argsort(projection, kind="stable")]
        chunks = [chunk for chunk in np.array_split(order, min(self.fanout, len(order))) if len(chunk)]
        if len(chunks) <= 1:
            return node

        node.idx = np.empty(0, dtype=int)
        node.children = [self._build(chunk, rng, depth + 1) for chunk in chunks]
        node.centroids = _normalize_rows(
            np.asarray([self.centers[chunk].mean(axis=0) for chunk in chunks])
        )
        return node

    def query_top(self, y: np.ndarray, top: int) -> tuple[np.ndarray, DecodeStats]:
        if top < 1:
            raise ValueError("top must be positive")
        leaf_parts = []
        child_scores = 0

        for root in self.roots:
            frontier = [root]
            while frontier:
                scored: list[tuple[float, _CenterNode]] = []
                for node in frontier:
                    if node.children is None:
                        leaf_parts.append(node.idx)
                        continue
                    assert node.centroids is not None
                    scores = node.centroids @ y
                    child_scores += len(scores)
                    scored.extend(
                        (float(score), child)
                        for score, child in zip(scores, node.children)
                    )
                if not scored:
                    break
                scored.sort(key=lambda item: item[0], reverse=True)
                frontier = [child for _score, child in scored[: self.beam]]

        if not leaf_parts:
            return np.empty(0, dtype=int), DecodeStats(
                work=child_scores,
                leaf_candidates=0,
                child_scores=child_scores,
            )

        candidates = np.unique(np.concatenate(leaf_parts))
        scores = self.centers[candidates] @ y
        k = min(top, len(candidates))
        ids = candidates[np.argpartition(-scores, k - 1)[:k]]
        ids = ids[np.argsort(-(self.centers[ids] @ y))]
        stats = DecodeStats(
            work=child_scores + len(candidates),
            leaf_candidates=len(candidates),
            child_scores=child_scores,
        )
        return ids.astype(int), stats


def _pack_bits(bits: np.ndarray) -> np.ndarray:
    width = bits.shape[1]
    weights = np.uint64(1) << np.arange(width, dtype=np.uint64)
    return bits.astype(np.uint64) @ weights


class LSHCenterDecoder:
    """Flat random-hyperplane multiprobe index over facility centroids."""

    name = "lsh"

    def __init__(
        self,
        centers: np.ndarray,
        *,
        tables: int,
        bits: int,
        hash_probes: int,
        seed: int,
    ) -> None:
        if tables < 1:
            raise ValueError("tables must be positive")
        if bits < 1 or bits > 63:
            raise ValueError("bits must be in [1, 63]")
        if hash_probes < 1:
            raise ValueError("hash_probes must be positive")
        self.centers = np.asarray(centers, dtype=float)
        self.tables_count = tables
        self.bits = bits
        self.hash_probes = hash_probes
        rng = np.random.default_rng(seed)
        self.planes = [
            rng.standard_normal((bits, self.centers.shape[1]))
            for _ in range(tables)
        ]
        self.tables: list[dict[int, np.ndarray]] = []
        for planes in self.planes:
            signatures = _pack_bits(self.centers @ planes.T >= 0)
            order = np.argsort(signatures, kind="stable")
            cuts = np.flatnonzero(np.diff(signatures[order])) + 1
            groups = np.split(order, cuts)
            self.tables.append({int(signatures[group[0]]): group for group in groups})

    def query_top(self, y: np.ndarray, top: int) -> tuple[np.ndarray, DecodeStats]:
        if top < 1:
            raise ValueError("top must be positive")
        pool = []
        low_bits = min(self.bits, 8)
        projection_work = self.tables_count * self.bits
        for planes, table in zip(self.planes, self.tables):
            proj = planes @ y
            key = int(_pack_bits((proj >= 0)[None, :])[0])
            margins = np.abs(proj)
            order = np.argsort(margins)
            pool.append((0.0, table, key))
            for bit in order:
                bit = int(bit)
                pool.append((float(margins[bit]), table, key ^ (1 << bit)))
            low = order[:low_bits]
            for a in range(len(low)):
                for b in range(a + 1, len(low)):
                    i = int(low[a])
                    j = int(low[b])
                    pool.append(
                        (
                            float(margins[i] + margins[j]),
                            table,
                            key ^ (1 << i) ^ (1 << j),
                        )
                    )
        pool.sort(key=lambda item: item[0])

        parts = []
        for _cost, table, key in pool[: self.hash_probes]:
            group = table.get(key)
            if group is not None:
                parts.append(group)
        if not parts:
            return np.empty(0, dtype=int), DecodeStats(
                work=projection_work,
                leaf_candidates=0,
                child_scores=projection_work,
            )

        candidates = np.unique(np.concatenate(parts))
        scores = self.centers[candidates] @ y
        k = min(top, len(candidates))
        ids = candidates[np.argpartition(-scores, k - 1)[:k]]
        ids = ids[np.argsort(-(self.centers[ids] @ y))]
        stats = DecodeStats(
            work=projection_work + len(candidates),
            leaf_candidates=len(candidates),
            child_scores=projection_work,
        )
        return ids.astype(int), stats


class PivotGraphCenterDecoder:
    """Exact centroid-neighbor graph with query-scored pivot entries."""

    name = "pivotgraph"

    def __init__(
        self,
        centers: np.ndarray,
        *,
        degree: int,
        pivots: int,
        entries: int,
        ef: int,
        seed: int,
    ) -> None:
        if degree < 1:
            raise ValueError("degree must be positive")
        if pivots < 1:
            raise ValueError("pivots must be positive")
        if entries < 1:
            raise ValueError("entries must be positive")
        if ef < 1:
            raise ValueError("ef must be positive")
        self.centers = np.asarray(centers, dtype=float)
        n = len(self.centers)
        self.degree = min(degree, max(1, n - 1))
        self.pivot_count = min(pivots, n)
        self.entries = min(entries, self.pivot_count)
        self.ef = min(max(ef, self.entries), n)
        rng = np.random.default_rng(seed)
        self.pivots = rng.choice(n, size=self.pivot_count, replace=False)
        self.neighbors = self._build_exact_neighbor_graph()

    def _build_exact_neighbor_graph(self) -> list[np.ndarray]:
        n = len(self.centers)
        if n <= 1:
            return [np.empty(0, dtype=int)]
        scores = self.centers @ self.centers.T
        np.fill_diagonal(scores, -np.inf)
        k = min(self.degree, n - 1)
        raw = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        return [
            raw[i][np.argsort(-scores[i, raw[i]])].astype(int)
            for i in range(n)
        ]

    def query_top(self, y: np.ndarray, top: int) -> tuple[np.ndarray, DecodeStats]:
        if top < 1:
            raise ValueError("top must be positive")
        pivot_scores = self.centers[self.pivots] @ y
        entry_local = np.argpartition(
            -pivot_scores, self.entries - 1)[: self.entries]
        entries = self.pivots[entry_local]

        score_cache = {
            int(pivot): float(score)
            for pivot, score in zip(self.pivots, pivot_scores)
        }
        work = len(self.pivots)

        def score(idx: int) -> float:
            nonlocal work
            idx = int(idx)
            if idx not in score_cache:
                score_cache[idx] = float(self.centers[idx] @ y)
                work += 1
            return score_cache[idx]

        visited = set()
        frontier: list[tuple[float, int]] = []
        for idx in entries:
            idx = int(idx)
            visited.add(idx)
            heapq.heappush(frontier, (-score(idx), idx))

        while frontier and len(visited) < self.ef:
            _neg_score, idx = heapq.heappop(frontier)
            for nb in self.neighbors[idx]:
                nb = int(nb)
                if nb in visited:
                    continue
                visited.add(nb)
                heapq.heappush(frontier, (-score(nb), nb))
                if len(visited) >= self.ef:
                    break

        if not visited:
            return np.empty(0, dtype=int), DecodeStats(
                work=work,
                leaf_candidates=0,
                child_scores=len(self.pivots),
            )

        candidates = np.fromiter(visited, dtype=int)
        scores = np.asarray([score(int(i)) for i in candidates])
        k = min(top, len(candidates))
        ids = candidates[np.argpartition(-scores, k - 1)[:k]]
        ids = ids[np.argsort(-(self.centers[ids] @ y))]
        stats = DecodeStats(
            work=work,
            leaf_candidates=len(candidates),
            child_scores=len(self.pivots),
        )
        return ids.astype(int), stats


def _make_decoder(
    decoder: str,
    centers: np.ndarray,
    *,
    trees: int,
    fanout: int,
    leaf_size: int,
    beam: int,
    hash_bits: int,
    hash_probes: int,
    graph_degree: int,
    graph_pivots: int,
    graph_entries: int,
    graph_ef: int,
    seed: int,
) -> ExactCenterDecoder | RPCenterDecoder | LSHCenterDecoder | PivotGraphCenterDecoder:
    if decoder == "exact":
        return ExactCenterDecoder(centers)
    if decoder == "rptree":
        return RPCenterDecoder(
            centers,
            trees=trees,
            fanout=fanout,
            leaf_size=leaf_size,
            beam=beam,
            seed=seed,
        )
    if decoder == "lsh":
        return LSHCenterDecoder(
            centers,
            tables=trees,
            bits=hash_bits,
            hash_probes=hash_probes,
            seed=seed,
        )
    if decoder == "pivotgraph":
        pivots = graph_pivots or max(1, int(math.ceil(len(centers) ** 0.5)))
        return PivotGraphCenterDecoder(
            centers,
            degree=graph_degree,
            pivots=pivots,
            entries=graph_entries,
            ef=graph_ef,
            seed=seed,
        )
    raise ValueError("decoder must be 'exact', 'rptree', 'lsh', or 'pivotgraph'")


def _bucket_candidates(ivf: CentroidRouter, center_ids: np.ndarray) -> set[int]:
    parts = [ivf.buckets[int(j)] for j in center_ids if len(ivf.buckets[int(j)])]
    if not parts:
        return set()
    return set(int(i) for i in np.concatenate(parts))


def run_trial(
    *,
    n: int,
    d: int,
    c: float,
    far_corr_max: float,
    t_scale: float,
    router: str,
    center_exponent: float,
    center_mult: float,
    lloyd_iters: int,
    decoder: str,
    center_trees: int,
    center_fanout: int,
    center_leaf_size: int,
    center_beam: int,
    center_hash_bits: int = 10,
    center_hash_probes: int = 16,
    center_graph_degree: int = 16,
    center_graph_pivots: int = 0,
    center_graph_entries: int = 8,
    center_graph_ef: int = 128,
    probes: int,
    query_trials: int,
    seed: int,
) -> dict[str, float | int | str]:
    if n < 2:
        raise ValueError("n must be at least two")
    if d < 2:
        raise ValueError("d must be at least two")
    if center_exponent <= 0.0 or center_mult <= 0.0:
        raise ValueError("center parameters must be positive")
    if probes < 1:
        raise ValueError("probes must be positive")
    if query_trials < 1:
        raise ValueError("query_trials must be positive")

    rng = np.random.default_rng(seed)
    rho = rho_for_c(c)
    x, q, _p, p_index = promised_spherical_instance(
        n_far=n - 1,
        d=d,
        c=c,
        far_corr_max=far_corr_max,
        rng=rng,
    )
    center_count = max(1, min(n, int(math.ceil(center_mult * n ** center_exponent))))
    ivf = CentroidRouter(
        centers=center_count,
        mode=router,
        lloyd_iters=lloyd_iters,
        seed=seed + 31,
    ).build(x)
    center_decoder = _make_decoder(
        decoder,
        ivf.center_vecs,
        trees=center_trees,
        fanout=center_fanout,
        leaf_size=center_leaf_size,
        beam=center_beam,
        hash_bits=center_hash_bits,
        hash_probes=center_hash_probes,
        graph_degree=center_graph_degree,
        graph_pivots=center_graph_pivots,
        graph_entries=center_graph_entries,
        graph_ef=center_graph_ef,
        seed=seed + 79,
    )

    score_gap_failures = 0
    oracle_valid = 0
    exact_p = 0
    exact_scoremax = 0
    exact_additive = 0
    decoder_p = 0
    decoder_scoremax = 0
    decoder_additive = 0
    center_top1 = 0
    center_probe_recalls = []
    exact_counts = []
    decoder_counts = []
    exact_losses = []
    decoder_losses = []
    p_bucket_ranks = []
    scoremax_bucket_ranks = []
    center_work = []
    center_leaf_candidates = []

    for _ in range(query_trials):
        scores, y, _t, margin = smoothed_scores(
            x, q, c=c, t_scale=t_scale, rng=rng)
        p_score = float(scores[p_index])
        if np.max(scores[:p_index]) > p_score - margin:
            score_gap_failures += 1
        top_point = int(np.argmax(scores))
        oracle_valid += int(top_point == p_index)

        center_scores = ivf.center_vecs @ y
        exact_k = min(probes, len(center_scores))
        exact_center_ids = np.argpartition(-center_scores, exact_k - 1)[:exact_k]
        exact_center_ids = exact_center_ids[np.argsort(-center_scores[exact_center_ids])]
        exact_top_center = int(exact_center_ids[0])
        p_center = int(ivf.labels[p_index])
        top_center = int(ivf.labels[top_point])
        p_bucket_ranks.append(1 + int(np.sum(center_scores > center_scores[p_center])))
        scoremax_bucket_ranks.append(
            1 + int(np.sum(center_scores > center_scores[top_center]))
        )

        decoded_center_ids, stats = center_decoder.query_top(y, top=probes)
        center_work.append(stats.work)
        center_leaf_candidates.append(stats.leaf_candidates)
        decoded_set = set(int(j) for j in decoded_center_ids)
        exact_set = set(int(j) for j in exact_center_ids)
        center_top1 += int(exact_top_center in decoded_set)
        center_probe_recalls.append(len(decoded_set & exact_set) / max(1, len(exact_set)))

        exact_candidates = _bucket_candidates(ivf, exact_center_ids)
        decoder_candidates = _bucket_candidates(ivf, decoded_center_ids)
        exact_counts.append(len(exact_candidates))
        decoder_counts.append(len(decoder_candidates))
        exact_loss = _candidate_best_loss(scores, exact_candidates)
        decoder_loss = _candidate_best_loss(scores, decoder_candidates)
        exact_losses.append(exact_loss)
        decoder_losses.append(decoder_loss)
        exact_p += int(p_index in exact_candidates)
        exact_scoremax += int(top_point in exact_candidates)
        exact_additive += int(exact_loss <= margin)
        decoder_p += int(p_index in decoder_candidates)
        decoder_scoremax += int(top_point in decoder_candidates)
        decoder_additive += int(decoder_loss <= margin)

    exact_counts_arr = np.asarray(exact_counts, dtype=float)
    decoder_counts_arr = np.asarray(decoder_counts, dtype=float)
    exact_losses_arr = np.asarray(exact_losses, dtype=float)
    decoder_losses_arr = np.asarray(decoder_losses, dtype=float)
    center_work_arr = np.asarray(center_work, dtype=float)
    center_leaf_arr = np.asarray(center_leaf_candidates, dtype=float)
    n_rho = n ** rho
    near_corr = 1.0 - 1.0 / (c * c)
    return {
        "n": n,
        "d": d,
        "c": c,
        "rho": rho,
        "near_corr": near_corr,
        "far_corr_max": far_corr_max,
        "t_scale": t_scale,
        "router": router,
        "centers": center_count,
        "center_exponent": center_exponent,
        "center_mult": center_mult,
        "lloyd_iters": lloyd_iters,
        "decoder": decoder,
        "center_trees": center_trees,
        "center_fanout": center_fanout,
        "center_leaf_size": center_leaf_size,
        "center_beam": center_beam,
        "center_hash_bits": center_hash_bits,
        "center_hash_probes": center_hash_probes,
        "center_graph_degree": center_graph_degree,
        "center_graph_pivots": center_graph_pivots,
        "center_graph_entries": center_graph_entries,
        "center_graph_ef": center_graph_ef,
        "probes": probes,
        "bucket_mean": float(np.mean(ivf.bucket_sizes)),
        "bucket_q90": float(np.quantile(ivf.bucket_sizes, 0.9)),
        "query_trials": query_trials,
        "score_gap_fail_rate": score_gap_failures / query_trials,
        "oracle_top1_valid_rate": oracle_valid / query_trials,
        "exact_p_bucket_rank": float(np.median(p_bucket_ranks)),
        "exact_scoremax_bucket_rank": float(np.median(scoremax_bucket_ranks)),
        "exact_p_rate": exact_p / query_trials,
        "exact_scoremax_rate": exact_scoremax / query_trials,
        "exact_additive_success_rate": exact_additive / query_trials,
        "exact_candidate_mean": float(np.mean(exact_counts_arr)),
        "exact_candidate_q90": float(np.quantile(exact_counts_arr, 0.9)),
        "exact_candidate_over_nrho": float(np.mean(exact_counts_arr) / n_rho),
        "exact_best_loss_over_margin_q90": _quantile_with_inf(
            exact_losses_arr / margin, 0.9),
        "center_top1_rate": center_top1 / query_trials,
        "center_probe_recall_mean": float(np.mean(center_probe_recalls)),
        "decoder_p_rate": decoder_p / query_trials,
        "decoder_scoremax_rate": decoder_scoremax / query_trials,
        "decoder_additive_success_rate": decoder_additive / query_trials,
        "decoder_candidate_mean": float(np.mean(decoder_counts_arr)),
        "decoder_candidate_q90": float(np.quantile(decoder_counts_arr, 0.9)),
        "decoder_candidate_over_nrho": float(np.mean(decoder_counts_arr) / n_rho),
        "decoder_best_loss_over_margin_q90": _quantile_with_inf(
            decoder_losses_arr / margin, 0.9),
        "center_work_mean": float(np.mean(center_work_arr)),
        "center_work_q90": float(np.quantile(center_work_arr, 0.9)),
        "center_work_over_centers": float(np.mean(center_work_arr) / center_count),
        "center_leaf_candidate_mean": float(np.mean(center_leaf_arr)),
        "center_leaf_candidate_q90": float(np.quantile(center_leaf_arr, 0.9)),
        "seed": seed,
    }


def _rows(args: argparse.Namespace) -> Iterable[dict[str, float | int | str]]:
    rho = rho_for_c(args.c)
    default_exp = 1.0 - rho
    for n in parse_csv_list(args.n, int):
        for d in parse_csv_list(args.d, int):
            for router in parse_csv_list(args.routers, str):
                for decoder in parse_csv_list(args.decoders, str):
                    for center_mult in parse_csv_list(args.center_mults, float):
                        for probes in parse_csv_list(args.probes, int):
                            for seed in parse_csv_list(args.seeds, int):
                                yield run_trial(
                                    n=n,
                                    d=d,
                                    c=args.c,
                                    far_corr_max=args.far_corr_max,
                                    t_scale=args.t_scale,
                                    router=router,
                                    center_exponent=args.center_exponent or default_exp,
                                    center_mult=center_mult,
                                    lloyd_iters=args.lloyd_iters,
                                    decoder=decoder,
                                    center_trees=args.center_trees,
                                    center_fanout=args.center_fanout,
                                    center_leaf_size=args.center_leaf_size,
                                    center_beam=args.center_beam,
                                    center_hash_bits=args.center_hash_bits,
                                    center_hash_probes=args.center_hash_probes,
                                    center_graph_degree=args.center_graph_degree,
                                    center_graph_pivots=args.center_graph_pivots,
                                    center_graph_entries=args.center_graph_entries,
                                    center_graph_ef=args.center_graph_ef,
                                    probes=probes,
                                    query_trials=args.query_trials,
                                    seed=seed,
                                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default="1000,3000")
    parser.add_argument("--d", default="24")
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--far-corr-max", type=float, default=-0.05)
    parser.add_argument("--t-scale", type=float, default=64.0)
    parser.add_argument("--routers", default="kmeanspp")
    parser.add_argument("--center-exponent", type=float, default=0.0)
    parser.add_argument("--center-mults", default="1.0")
    parser.add_argument("--lloyd-iters", type=int, default=4)
    parser.add_argument("--decoders", default="exact,rptree")
    parser.add_argument("--center-trees", type=int, default=4)
    parser.add_argument("--center-fanout", type=int, default=4)
    parser.add_argument("--center-leaf-size", type=int, default=8)
    parser.add_argument("--center-beam", type=int, default=4)
    parser.add_argument("--center-hash-bits", type=int, default=10)
    parser.add_argument("--center-hash-probes", type=int, default=16)
    parser.add_argument("--center-graph-degree", type=int, default=16)
    parser.add_argument("--center-graph-pivots", type=int, default=0)
    parser.add_argument("--center-graph-entries", type=int, default=8)
    parser.add_argument("--center-graph-ef", type=int, default=128)
    parser.add_argument("--probes", default="1,2,4")
    parser.add_argument("--query-trials", type=int, default=60)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--csv")
    args = parser.parse_args()

    rows = list(_rows(args))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    else:
        print(",".join(FIELDNAMES))
        for row in rows:
            print(",".join(str(row[name]) for name in FIELDNAMES))


if __name__ == "__main__":
    main()
