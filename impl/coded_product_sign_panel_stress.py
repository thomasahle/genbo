"""Stress coded product-sign panels against the top-label entropy barrier.

The previous product-sign obstruction used the full sign cube.  This harness
keeps the same score normal form but restricts labels to a small binary linear
code, such as a Reed-Muller panel.  For small panels we enumerate the code
exactly and measure the concrete finite question:

* report the top L codewords under the query bit LLRs;
* store a point in a reported label when its product-sign score exceeds a far
  threshold calibrated to the target scan load;
* compare near hit rate, query/tilted top-L mass, storage, and far load.

This is deliberately a finite stress test, not a decoder benchmark.  If exact
enumeration fails at these sizes, an SCL/Fano implementation has no theorem to
rescue.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from product_sign_margin_stress import bit_score, log_cosh


FIELDNAMES = [
    "code",
    "family",
    "r_bits",
    "dimension",
    "labels",
    "local_m",
    "c",
    "rho",
    "target_label_exponent",
    "label_exponent",
    "top_l",
    "sigma",
    "corr",
    "alpha",
    "threshold_tail",
    "threshold",
    "empirical_tail",
    "threshold_samples",
    "query_trials",
    "far_load_budget",
    "far_load_over_budget",
    "storage_per_point",
    "storage_over_linear_budget",
    "query_top_l_mass_q05",
    "query_top_l_mass_median",
    "query_top1_mass_median",
    "query_top1_score_level_q05",
    "query_top1_score_level_median",
    "tilted_top_l_mass_q05",
    "tilted_top_l_mass_median",
    "near_hit_rate",
    "oracle_near_hit_rate",
    "top1_near_hit_rate",
    "reported_score_level_median",
    "oracle_score_level_median",
    "near_margin_q05",
    "near_margin_median",
    "oracle_margin_q05",
    "decoder_loss_q95",
]

SADDLEPOINT_FIELDNAMES = [
    "code",
    "family",
    "r_bits",
    "dimension",
    "labels",
    "local_m",
    "c",
    "rho",
    "target_label_exponent",
    "label_exponent",
    "top_l",
    "sigma",
    "corr",
    "threshold_tail",
    "tail_rate",
    "far_threshold_level",
    "query_level_median",
    "query_level_asymptotic",
    "median_shifted_score_mean",
    "median_shifted_score_variance",
    "median_mean_gap",
    "median_normal_hit",
    "median_decay_rate",
    "asymptotic_shifted_score_mean",
    "asymptotic_shifted_score_variance",
    "asymptotic_mean_gap",
    "asymptotic_normal_hit",
    "asymptotic_decay_rate",
]


@dataclass(frozen=True)
class CodePanel:
    name: str
    family: str
    generator: np.ndarray
    signs: np.ndarray


def parse_csv_list(text: str) -> list[str]:
    out = [part.strip() for part in text.split(",") if part.strip()]
    if not out:
        raise ValueError("list must contain at least one item")
    return out


def gf2_rank(matrix: np.ndarray) -> int:
    mat = np.array(matrix, dtype=np.uint8, copy=True) & 1
    if mat.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    rows, cols = mat.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if mat[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            mat[[rank, pivot]] = mat[[pivot, rank]]
        for row in range(rows):
            if row != rank and mat[row, col]:
                mat[row] ^= mat[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def reed_muller_generator(*, m: int, degree: int) -> np.ndarray:
    if m <= 0:
        raise ValueError("m must be positive")
    if degree < 0 or degree > m:
        raise ValueError("degree must lie between 0 and m")
    n = 1 << m
    points = ((np.arange(n, dtype=np.uint32)[:, None] >> np.arange(m)) & 1).astype(
        np.uint8
    )
    rows = []
    for deg in range(degree + 1):
        for coords in itertools.combinations(range(m), deg):
            if not coords:
                row = np.ones(n, dtype=np.uint8)
            else:
                row = np.prod(points[:, coords], axis=1).astype(np.uint8)
            rows.append(row)
    return np.vstack(rows).astype(np.uint8)


def polar_transform_generator(*, m: int) -> np.ndarray:
    if m <= 0:
        raise ValueError("m must be positive")
    gen = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    out = np.array([[1]], dtype=np.uint8)
    for _ in range(m):
        out = np.kron(out, gen).astype(np.uint8)
    return out


def polar_weight_generator(*, m: int, dimension: int) -> np.ndarray:
    base = polar_transform_generator(m=m)
    n = base.shape[0]
    if dimension <= 0 or dimension > n:
        raise ValueError("dimension must lie in [1, 2^m]")
    weights = np.sum(base, axis=1)
    order = np.lexsort((np.arange(n), -weights))
    return base[order[:dimension]].astype(np.uint8)


def random_full_rank_generator(
    *, r_bits: int, dimension: int, seed: int, max_tries: int = 10_000
) -> np.ndarray:
    if r_bits <= 0:
        raise ValueError("r_bits must be positive")
    if dimension <= 0 or dimension > r_bits:
        raise ValueError("dimension must lie in [1, r_bits]")
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        gen = rng.integers(0, 2, size=(dimension, r_bits), dtype=np.uint8)
        if gf2_rank(gen) == dimension:
            return gen
    raise RuntimeError("failed to sample a full-rank generator")


def enumerate_binary_linear_code(generator: np.ndarray) -> np.ndarray:
    gen = np.asarray(generator, dtype=np.uint8) & 1
    if gen.ndim != 2:
        raise ValueError("generator must be two-dimensional")
    dimension, r_bits = gen.shape
    if dimension > 22:
        raise ValueError("exact enumeration is capped at dimension 22")
    messages = (
        (np.arange(1 << dimension, dtype=np.uint32)[:, None] >> np.arange(dimension))
        & 1
    ).astype(np.int16)
    code_bits = (messages @ gen.astype(np.int16)) % 2
    return (1 - 2 * code_bits.astype(np.int8)).astype(np.int8)


def code_from_spec(spec: str, *, seed: int) -> CodePanel:
    parts = spec.split(":")
    if parts[0] == "rm" and len(parts) == 3:
        m = int(parts[1])
        degree = int(parts[2])
        gen = reed_muller_generator(m=m, degree=degree)
        return CodePanel(
            name=f"rm({degree},{m})",
            family="rm",
            generator=gen,
            signs=enumerate_binary_linear_code(gen),
        )
    if parts[0] == "full" and len(parts) == 2:
        r_bits = int(parts[1])
        gen = np.eye(r_bits, dtype=np.uint8)
        return CodePanel(
            name=f"full({r_bits})",
            family="full",
            generator=gen,
            signs=enumerate_binary_linear_code(gen),
        )
    if parts[0] == "random" and len(parts) in (3, 4):
        r_bits = int(parts[1])
        dimension = int(parts[2])
        local_seed = int(parts[3]) if len(parts) == 4 else seed
        gen = random_full_rank_generator(
            r_bits=r_bits, dimension=dimension, seed=local_seed
        )
        return CodePanel(
            name=f"random({dimension},{r_bits},{local_seed})",
            family="random",
            generator=gen,
            signs=enumerate_binary_linear_code(gen),
        )
    if parts[0] == "iid" and len(parts) in (3, 4):
        r_bits = int(parts[1])
        dimension = int(parts[2])
        local_seed = int(parts[3]) if len(parts) == 4 else seed
        if dimension <= 0 or dimension > 22:
            raise ValueError("iid dimension must lie in [1, 22]")
        rng = np.random.default_rng(local_seed)
        signs = rng.choice(
            np.array([-1, 1], dtype=np.int8),
            size=(1 << dimension, r_bits),
        )
        placeholder = np.zeros((dimension, r_bits), dtype=np.uint8)
        return CodePanel(
            name=f"iid({dimension},{r_bits},{local_seed})",
            family="iid",
            generator=placeholder,
            signs=signs,
        )
    if parts[0] == "polar-weight" and len(parts) == 3:
        m = int(parts[1])
        dimension = int(parts[2])
        gen = polar_weight_generator(m=m, dimension=dimension)
        return CodePanel(
            name=f"polar-weight({dimension},{m})",
            family="polar_weight",
            generator=gen,
            signs=enumerate_binary_linear_code(gen),
        )
    raise ValueError(
        "code spec must be rm:m:degree, full:r, random:r:dimension[:seed], "
        "iid:r:dimension[:seed], or polar-weight:m:dimension"
    )


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    max_value = float(np.max(values))
    return max_value + math.log(float(np.sum(np.exp(values - max_value))))


def _mass_of_indices(scores: np.ndarray, indices: np.ndarray) -> float:
    log_total = _logsumexp(scores)
    log_part = _logsumexp(scores[indices])
    return float(math.exp(log_part - log_total))


def _top_indices(scores: np.ndarray, top_l: int) -> np.ndarray:
    if top_l <= 0:
        raise ValueError("top_l must be positive")
    if top_l >= scores.shape[0]:
        return np.arange(scores.shape[0])
    top = np.argpartition(-scores, top_l - 1)[:top_l]
    return top[np.argsort(-scores[top])]


def _quantile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_upper_tail(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def normal_ppf(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie in (0, 1)")
    low = -12.0
    high = 12.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        if normal_cdf(mid) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def normal_upper_tail_ppf(tail: float) -> float:
    if not (0.0 < tail < 1.0):
        raise ValueError("tail must lie in (0, 1)")
    low = -12.0
    high = 12.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        if normal_upper_tail(mid) > tail:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _normal_nodes(mean: float, sigma: float, quadrature: int) -> tuple[np.ndarray, np.ndarray]:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if quadrature < 8:
        raise ValueError("quadrature must be at least 8")
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature)
    z = mean + math.sqrt(2.0) * sigma * nodes
    prob_weights = weights / math.sqrt(math.pi)
    return z, prob_weights


def shifted_score_moments(
    *, mean: float, sigma: float, quadrature: int = 96
) -> tuple[float, float]:
    z, weights = _normal_nodes(mean, sigma, quadrature)
    scores = bit_score(z)
    score_mean = float(np.dot(weights, scores))
    variance = float(np.dot(weights, (scores - score_mean) ** 2))
    return score_mean, max(0.0, variance)


def shifted_score_rate(
    level: float,
    *,
    mean: float,
    sigma: float,
    quadrature: int = 96,
    theta_max: float = 80.0,
    theta_grid: int = 2000,
) -> float:
    score_mean, _variance = shifted_score_moments(
        mean=mean, sigma=sigma, quadrature=quadrature
    )
    if level <= score_mean:
        return 0.0
    theta = np.linspace(0.0, theta_max, theta_grid)
    z, weights = _normal_nodes(mean, sigma, quadrature)
    scores = bit_score(z)
    vals = np.log(weights)[None, :] + theta[:, None] * scores[None, :]
    shifted = vals - np.max(vals, axis=1, keepdims=True)
    log_mgf = np.max(vals, axis=1) + np.log(np.sum(np.exp(shifted), axis=1))
    return max(0.0, float(np.max(theta * level - log_mgf)))


def score_level_for_rate(
    rate: float,
    *,
    mean: float,
    sigma: float,
    quadrature: int = 96,
    theta_max: float = 80.0,
    theta_grid: int = 2000,
) -> float:
    if rate < 0.0:
        raise ValueError("rate must be nonnegative")
    score_mean, _variance = shifted_score_moments(
        mean=mean, sigma=sigma, quadrature=quadrature
    )
    if rate == 0.0:
        return score_mean
    low = score_mean
    high = math.log(2.0) - 1e-10
    for _ in range(80):
        mid = 0.5 * (low + high)
        mid_rate = shifted_score_rate(
            mid,
            mean=mean,
            sigma=sigma,
            quadrature=quadrature,
            theta_max=theta_max,
            theta_grid=theta_grid,
        )
        if mid_rate < rate:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def random_code_query_levels(*, r_bits: int, dimension: int, sigma: float) -> tuple[float, float]:
    if r_bits <= 0:
        raise ValueError("r_bits must be positive")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    log_labels = dimension * math.log(2.0)
    asymptotic = sigma * math.sqrt(2.0 * log_labels / r_bits)
    labels = 1 << dimension
    median_tail = -math.expm1(math.log(0.5) / labels)
    max_median_z = normal_upper_tail_ppf(median_tail)
    median = sigma * max_median_z / math.sqrt(r_bits)
    return median, asymptotic


def _normal_hit_probability(
    *,
    r_bits: int,
    threshold_level: float,
    score_mean: float,
    score_variance: float,
) -> float:
    if score_variance <= 0.0:
        return 1.0 if score_mean >= threshold_level else 0.0
    z = math.sqrt(r_bits) * (threshold_level - score_mean) / math.sqrt(score_variance)
    return normal_upper_tail(z)


def random_code_saddlepoint_summary(
    code: CodePanel,
    *,
    local_m: int,
    c: float,
    sigma: float,
    corr: float,
    top_l: int,
    threshold_tail: float | None = None,
    quadrature: int = 96,
    theta_grid: int = 2000,
) -> dict:
    return random_code_saddlepoint_summary_for_shape(
        name=code.name,
        family=code.family,
        r_bits=int(code.signs.shape[1]),
        dimension=int(code.generator.shape[0]),
        labels=int(code.signs.shape[0]),
        local_m=local_m,
        c=c,
        sigma=sigma,
        corr=corr,
        top_l=top_l,
        threshold_tail=threshold_tail,
        quadrature=quadrature,
        theta_grid=theta_grid,
    )


def random_code_saddlepoint_summary_for_shape(
    *,
    name: str,
    family: str,
    r_bits: int,
    dimension: int,
    labels: int,
    local_m: int,
    c: float,
    sigma: float,
    corr: float,
    top_l: int,
    threshold_tail: float | None = None,
    quadrature: int = 96,
    theta_grid: int = 2000,
) -> dict:
    if not (0.0 < corr < 1.0):
        raise ValueError("corr must lie in (0, 1)")
    row = summarize_panel_shape(
        name=name,
        family=family,
        r_bits=r_bits,
        dimension=dimension,
        labels=labels,
        local_m=local_m,
        c=c,
        top_l=top_l,
    )
    tail_prob = (
        float(threshold_tail)
        if threshold_tail is not None
        else local_m ** (row["rho"] - 1.0) / top_l
    )
    tail_rate = -math.log(tail_prob) / r_bits
    threshold_level = score_level_for_rate(
        tail_rate,
        mean=0.0,
        sigma=sigma,
        quadrature=quadrature,
        theta_grid=theta_grid,
    )
    query_median, query_asymptotic = random_code_query_levels(
        r_bits=r_bits, dimension=row["dimension"], sigma=sigma
    )
    out = {
        **row,
        "sigma": float(sigma),
        "corr": float(corr),
        "threshold_tail": float(tail_prob),
        "tail_rate": float(tail_rate),
        "far_threshold_level": float(threshold_level),
        "query_level_median": float(query_median),
        "query_level_asymptotic": float(query_asymptotic),
    }
    for prefix, query_level in (
        ("median", query_median),
        ("asymptotic", query_asymptotic),
    ):
        shifted_mean, shifted_variance = shifted_score_moments(
            mean=corr * query_level,
            sigma=sigma,
            quadrature=quadrature,
        )
        hit = _normal_hit_probability(
            r_bits=r_bits,
            threshold_level=threshold_level,
            score_mean=shifted_mean,
            score_variance=shifted_variance,
        )
        decay_rate = shifted_score_rate(
            threshold_level,
            mean=corr * query_level,
            sigma=sigma,
            quadrature=quadrature,
            theta_grid=theta_grid,
        )
        out.update(
            {
                f"{prefix}_shifted_score_mean": float(shifted_mean),
                f"{prefix}_shifted_score_variance": float(shifted_variance),
                f"{prefix}_mean_gap": float(shifted_mean - threshold_level),
                f"{prefix}_normal_hit": float(hit),
                f"{prefix}_decay_rate": float(decay_rate),
            }
        )
    return out


def empirical_far_threshold(
    *,
    r_bits: int,
    sigma: float,
    tail_prob: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if r_bits <= 0:
        raise ValueError("r_bits must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not (0.0 < tail_prob < 1.0):
        raise ValueError("tail_prob must lie in (0, 1)")
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    logits = sigma * rng.standard_normal((samples, r_bits))
    scores = np.sum(bit_score(logits), axis=1)
    threshold = float(np.quantile(scores, 1.0 - tail_prob, method="higher"))
    empirical_tail = float(np.mean(scores >= threshold))
    return threshold, empirical_tail


def summarize_code(code: CodePanel, *, local_m: int, c: float, top_l: int) -> dict:
    return summarize_panel_shape(
        name=code.name,
        family=code.family,
        r_bits=int(code.signs.shape[1]),
        dimension=int(code.generator.shape[0]),
        labels=int(code.signs.shape[0]),
        local_m=local_m,
        c=c,
        top_l=top_l,
    )


def summarize_panel_shape(
    *,
    name: str,
    family: str,
    r_bits: int,
    dimension: int,
    labels: int,
    local_m: int,
    c: float,
    top_l: int,
) -> dict:
    if local_m <= 1:
        raise ValueError("local_m must exceed one")
    if c <= 1:
        raise ValueError("c must exceed one")
    if top_l <= 0:
        raise ValueError("top_l must be positive")
    if r_bits <= 0:
        raise ValueError("r_bits must be positive")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if labels <= 0:
        raise ValueError("labels must be positive")
    rho = 1.0 / (2.0 * c * c - 1.0)
    return {
        "code": name,
        "family": family,
        "r_bits": int(r_bits),
        "dimension": int(dimension),
        "labels": int(labels),
        "local_m": int(local_m),
        "c": float(c),
        "rho": rho,
        "target_label_exponent": 1.0 - rho,
        "label_exponent": math.log(labels) / math.log(local_m),
        "top_l": int(top_l),
    }


def run_code_trial(
    code: CodePanel,
    *,
    local_m: int,
    c: float,
    sigma: float,
    corr: float,
    top_l: int,
    threshold_tail: float | None,
    threshold_samples: int,
    query_trials: int,
    seed: int,
    alpha: float = 0.5,
) -> dict:
    if not (0.0 < corr < 1.0):
        raise ValueError("corr must lie in (0, 1)")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie in [0, 1]")
    if query_trials <= 0:
        raise ValueError("query_trials must be positive")

    row = summarize_code(code, local_m=local_m, c=c, top_l=top_l)
    rho = row["rho"]
    labels, r_bits = code.signs.shape
    tail_prob = (
        float(threshold_tail)
        if threshold_tail is not None
        else local_m ** (rho - 1.0) / top_l
    )
    threshold, empirical_tail = empirical_far_threshold(
        r_bits=r_bits,
        sigma=sigma,
        tail_prob=tail_prob,
        samples=threshold_samples,
        seed=seed + 1_000_003,
    )

    rng = np.random.default_rng(seed)
    signs_float = code.signs.astype(float, copy=False)
    query_masses = []
    query_top1_masses = []
    query_top1_score_levels = []
    tilted_masses = []
    hits = []
    oracle_hits = []
    top1_hits = []
    reported_score_levels = []
    oracle_score_levels = []
    margins = []
    oracle_margins = []
    decoder_losses = []

    for _ in range(query_trials):
        uq = sigma * rng.standard_normal(r_bits)
        up = corr * uq + sigma * math.sqrt(1.0 - corr * corr) * rng.standard_normal(
            r_bits
        )
        q_scores = signs_float @ uq
        top = _top_indices(q_scores, top_l)
        top1 = top[:1]
        query_masses.append(_mass_of_indices(q_scores, top))
        query_top1_masses.append(_mass_of_indices(q_scores, top1))
        query_top1_score_levels.append(float(q_scores[top[0]]) / r_bits)

        tilt_logits = alpha * up + (1.0 - alpha) * uq
        tilt_scores = signs_float @ tilt_logits
        tilted_masses.append(_mass_of_indices(tilt_scores, top))

        point_offset = float(np.sum(log_cosh(up)))
        near_scores = signs_float @ up - point_offset
        reported_best = float(np.max(near_scores[top]))
        oracle_best = float(np.max(near_scores))
        reported_score_levels.append(reported_best / r_bits)
        oracle_score_levels.append(oracle_best / r_bits)
        margins.append(reported_best - threshold)
        oracle_margins.append(oracle_best - threshold)
        hits.append(reported_best >= threshold)
        oracle_hits.append(oracle_best >= threshold)
        top1_hits.append(float(near_scores[top[0]]) >= threshold)
        decoder_losses.append(oracle_best - reported_best)

    far_load_budget = local_m * top_l * empirical_tail
    row.update(
        {
            "sigma": float(sigma),
            "corr": float(corr),
            "alpha": float(alpha),
            "threshold_tail": float(tail_prob),
            "threshold": float(threshold),
            "empirical_tail": float(empirical_tail),
            "threshold_samples": int(threshold_samples),
            "query_trials": int(query_trials),
            "far_load_budget": float(far_load_budget),
            "far_load_over_budget": float(far_load_budget / (local_m**rho)),
            "storage_per_point": float(labels * empirical_tail),
            "storage_over_linear_budget": float(labels * empirical_tail),
            "query_top_l_mass_q05": _quantile(query_masses, 0.05),
            "query_top_l_mass_median": _quantile(query_masses, 0.5),
            "query_top1_mass_median": _quantile(query_top1_masses, 0.5),
            "query_top1_score_level_q05": _quantile(query_top1_score_levels, 0.05),
            "query_top1_score_level_median": _quantile(query_top1_score_levels, 0.5),
            "tilted_top_l_mass_q05": _quantile(tilted_masses, 0.05),
            "tilted_top_l_mass_median": _quantile(tilted_masses, 0.5),
            "near_hit_rate": float(np.mean(hits)),
            "oracle_near_hit_rate": float(np.mean(oracle_hits)),
            "top1_near_hit_rate": float(np.mean(top1_hits)),
            "reported_score_level_median": _quantile(reported_score_levels, 0.5),
            "oracle_score_level_median": _quantile(oracle_score_levels, 0.5),
            "near_margin_q05": _quantile(margins, 0.05),
            "near_margin_median": _quantile(margins, 0.5),
            "oracle_margin_q05": _quantile(oracle_margins, 0.05),
            "decoder_loss_q95": _quantile(decoder_losses, 0.95),
        }
    )
    return row


def run_sweep(args: argparse.Namespace) -> list[dict]:
    rows = []
    for spec in parse_csv_list(args.codes):
        for seed in args.seeds:
            code = code_from_spec(spec, seed=seed)
            row = run_code_trial(
                code,
                local_m=args.local_m,
                c=args.c,
                sigma=args.sigma,
                corr=args.corr,
                top_l=args.top_l,
                threshold_tail=args.threshold_tail,
                threshold_samples=args.threshold_samples,
                query_trials=args.query_trials,
                seed=seed,
                alpha=args.alpha,
            )
            rows.append(row)
            print(
                " ".join(
                    [
                        f"code={row['code']}",
                        f"seed={seed}",
                        f"label_exp={row['label_exponent']:.3f}",
                        f"hit={row['near_hit_rate']:.3f}",
                        f"oracle={row['oracle_near_hit_rate']:.3f}",
                        f"qmass50={row['query_top_l_mass_median']:.3g}",
                        f"tmass05={row['tilted_top_l_mass_q05']:.3g}",
                        f"margin05={row['near_margin_q05']:.3f}",
                        f"storage={row['storage_per_point']:.3g}",
                        f"load/budget={row['far_load_over_budget']:.3f}",
                    ]
                ),
                file=sys.stderr,
            )
    return rows


def run_saddlepoint_sweep(args: argparse.Namespace) -> list[dict]:
    rows = []
    if args.saddlepoint_shapes:
        for spec in parse_csv_list(args.saddlepoint_shapes):
            parts = spec.split(":")
            if len(parts) != 2:
                raise ValueError("saddlepoint shape specs must be r_bits:dimension")
            r_bits = int(parts[0])
            dimension = int(parts[1])
            row = random_code_saddlepoint_summary_for_shape(
                name=f"shape({dimension},{r_bits})",
                family="shape",
                r_bits=r_bits,
                dimension=dimension,
                labels=1 << dimension,
                local_m=args.local_m,
                c=args.c,
                sigma=args.sigma,
                corr=args.corr,
                top_l=args.top_l,
                threshold_tail=args.threshold_tail,
                quadrature=args.rate_quadrature,
                theta_grid=args.rate_theta_grid,
            )
            rows.append(row)
            print(
                " ".join(
                    [
                        f"code={row['code']}",
                        f"label_exp={row['label_exponent']:.3f}",
                        f"tail_rate={row['tail_rate']:.3f}",
                        f"threshold={row['far_threshold_level']:.3f}",
                        f"qmed={row['query_level_median']:.3f}",
                        f"gap_med={row['median_mean_gap']:.3f}",
                        f"hit_med={row['median_normal_hit']:.3f}",
                        f"gap_asym={row['asymptotic_mean_gap']:.3f}",
                        f"hit_asym={row['asymptotic_normal_hit']:.3f}",
                    ]
                ),
                file=sys.stderr,
            )
        return rows
    for spec in parse_csv_list(args.codes):
        for seed in args.seeds:
            code = code_from_spec(spec, seed=seed)
            row = random_code_saddlepoint_summary(
                code,
                local_m=args.local_m,
                c=args.c,
                sigma=args.sigma,
                corr=args.corr,
                top_l=args.top_l,
                threshold_tail=args.threshold_tail,
                quadrature=args.rate_quadrature,
                theta_grid=args.rate_theta_grid,
            )
            rows.append(row)
            print(
                " ".join(
                    [
                        f"code={row['code']}",
                        f"seed={seed}",
                        f"label_exp={row['label_exponent']:.3f}",
                        f"tail_rate={row['tail_rate']:.3f}",
                        f"threshold={row['far_threshold_level']:.3f}",
                        f"qmed={row['query_level_median']:.3f}",
                        f"gap_med={row['median_mean_gap']:.3f}",
                        f"hit_med={row['median_normal_hit']:.3f}",
                        f"gap_asym={row['asymptotic_mean_gap']:.3f}",
                        f"hit_asym={row['asymptotic_normal_hit']:.3f}",
                    ]
                ),
                file=sys.stderr,
            )
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in FIELDNAMES})


def write_saddlepoint_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SADDLEPOINT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in SADDLEPOINT_FIELDNAMES})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codes",
        default="rm:4:1,rm:4:2,random:16:11:0,full:16",
        help=(
            "Comma-separated specs: rm:m:degree, full:r, "
            "random:r:dimension[:seed], iid:r:dimension[:seed], "
            "polar-weight:m:dimension"
        ),
    )
    parser.add_argument("--local-m", type=int, default=8192)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=3.5)
    parser.add_argument("--corr", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top-l", type=int, default=8)
    parser.add_argument(
        "--threshold-tail",
        type=float,
        default=None,
        help="Override per-reported-label far tail; default is m^(rho-1)/L.",
    )
    parser.add_argument("--threshold-samples", type=int, default=300_000)
    parser.add_argument("--query-trials", type=int, default=300)
    parser.add_argument("--rate-quadrature", type=int, default=96)
    parser.add_argument("--rate-theta-grid", type=int, default=2000)
    parser.add_argument(
        "--seeds",
        type=lambda text: [int(part) for part in parse_csv_list(text)],
        default=[0, 1, 2],
    )
    parser.add_argument("--csv", default="")
    parser.add_argument("--saddlepoint-csv", default="")
    parser.add_argument(
        "--saddlepoint-shapes",
        default="",
        help="Optional comma-separated r_bits:dimension specs for saddlepoint-only sweeps.",
    )
    parser.add_argument(
        "--saddlepoint-only",
        action="store_true",
        help="Only run the iid random-code saddlepoint approximation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.saddlepoint_only:
        rows = []
    else:
        rows = run_sweep(args)
    saddlepoint_rows = (
        run_saddlepoint_sweep(args)
        if args.saddlepoint_csv or args.saddlepoint_only
        else []
    )
    if args.csv and rows:
        write_csv(args.csv, rows)
    if args.saddlepoint_csv and saddlepoint_rows:
        write_saddlepoint_csv(args.saddlepoint_csv, saddlepoint_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
