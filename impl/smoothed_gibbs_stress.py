"""Stress the frozen smoothed-Gibbs mass-capture target.

The smoothed-Gibbs endpoint asks a query-time ``m^rho polylog(m)`` reporter to
return a list carrying inverse-polylogarithmic Gibbs mass.  This script checks
the most basic necessary condition: even the best list of that size, namely
the top scores under the Gibbs weights, must carry that much mass.

The medium-valid-plateau construction is a direct obstruction to the target as
stated.  Put ``m^a`` already-valid points at the query, with
``rho < a < 1-rho``.  The guard density is still below ``m^-rho``, but a charged
list of size ``m^rho polylog(m)`` captures only
``m^(rho-a) polylog(m)`` Gibbs mass from the plateau.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Callable, Iterable

import numpy as np


FIELDNAMES = [
    "m",
    "c",
    "rho",
    "valid_exponent",
    "list_log_power",
    "capture_log_power",
    "valid_count",
    "list_size",
    "guard_density",
    "guard_threshold",
    "guard_power_gap",
    "topk_mass_upper",
    "capture_threshold",
    "capture_fails_finite",
    "mass_power_gap",
]

SAMPLE_FIELDNAMES = FIELDNAMES + [
    "seed",
    "t_constant",
    "invalid_radius",
    "invalid_mass",
    "invalid_max_score",
    "sampled_topk_mass",
]


def parse_csv_list(text: str, cast: Callable[[str], object] = str) -> list[object]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    if not out:
        raise ValueError("list must contain at least one item")
    return out


def rho_for_c(c: float) -> float:
    if c <= 1.0:
        raise ValueError("c must be greater than one")
    return 1.0 / (2.0 * c * c - 1.0)


def _safe_exp(log_value: float) -> float:
    if log_value <= -745.0:
        return 0.0
    if log_value >= 709.0:
        return float("inf")
    return math.exp(log_value)


def topk_gibbs_mass_from_scores(scores: np.ndarray, k: int) -> float:
    """Return the Gibbs mass of the k largest scores."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if len(scores) == 0:
        raise ValueError("scores must be nonempty")
    if k <= 0:
        return 0.0
    if k >= len(scores):
        return 1.0

    split = len(scores) - k
    top = np.partition(scores, split)[split:]
    shift = float(np.max(scores))
    numerator = float(np.sum(np.exp(top - shift)))
    denominator = float(np.sum(np.exp(scores - shift)))
    return numerator / denominator


def plateau_bound(
    *,
    m: int,
    c: float,
    valid_exponent: float,
    list_log_power: float,
    capture_log_power: float,
) -> dict[str, float | int | bool]:
    """Return the asymptotic and finite top-K bounds for a valid plateau."""
    if m < 2:
        raise ValueError("m must be at least two")
    rho = rho_for_c(c)
    if not (rho < valid_exponent < 1.0 - rho):
        raise ValueError("valid_exponent must lie in (rho, 1-rho)")
    if list_log_power < 0.0 or capture_log_power < 0.0:
        raise ValueError("log powers must be nonnegative")

    log_m = math.log(m)
    log_log_m = math.log(log_m)
    valid_log = valid_exponent * log_m
    list_log = rho * log_m + list_log_power * log_log_m
    valid_count = max(1, min(m, int(math.floor(_safe_exp(valid_log)))))
    list_size = max(1, min(m, int(math.ceil(_safe_exp(list_log)))))

    guard_density = valid_count / m
    guard_threshold = _safe_exp(-rho * log_m)
    topk_mass_upper = min(1.0, list_size / valid_count)
    capture_threshold = _safe_exp(-capture_log_power * log_log_m)

    return {
        "m": m,
        "c": c,
        "rho": rho,
        "valid_exponent": valid_exponent,
        "list_log_power": list_log_power,
        "capture_log_power": capture_log_power,
        "valid_count": valid_count,
        "list_size": list_size,
        "guard_density": guard_density,
        "guard_threshold": guard_threshold,
        "guard_power_gap": valid_exponent + rho - 1.0,
        "topk_mass_upper": topk_mass_upper,
        "capture_threshold": capture_threshold,
        "capture_fails_finite": topk_mass_upper < capture_threshold,
        "mass_power_gap": rho - valid_exponent,
    }


def sample_plateau_trial(
    *,
    m: int,
    c: float,
    valid_exponent: float,
    list_log_power: float,
    capture_log_power: float,
    seed: int,
    t_constant: float = 32.0,
    invalid_radius: float | None = None,
) -> dict[str, float | int | bool]:
    """Sample smoothed scores for the plateau plus far invalid points.

    The query and the plateau are at the origin.  Invalid points are placed on
    a shell of radius ``invalid_radius * r`` with ``r=1``.  Their score
    distribution is

        N(-t R^2/2, t R^2),    t = t_constant log(m).

    The plateau scores are exactly zero.
    """
    base = plateau_bound(
        m=m,
        c=c,
        valid_exponent=valid_exponent,
        list_log_power=list_log_power,
        capture_log_power=capture_log_power,
    )
    if t_constant <= 0.0:
        raise ValueError("t_constant must be positive")
    if invalid_radius is None:
        invalid_radius = c + 1.0
    if invalid_radius <= c:
        raise ValueError("invalid_radius must be greater than c")

    valid_count = int(base["valid_count"])
    invalid_count = m - valid_count
    rng = np.random.default_rng(seed)
    t = t_constant * math.log(m)
    radius = float(invalid_radius)
    invalid_scores = rng.normal(
        loc=-0.5 * t * radius * radius,
        scale=math.sqrt(t) * radius,
        size=invalid_count,
    )
    scores = np.concatenate([np.zeros(valid_count), invalid_scores])
    sampled_topk_mass = topk_gibbs_mass_from_scores(scores, int(base["list_size"]))
    shift = float(np.max(scores))
    invalid_mass = float(
        np.sum(np.exp(invalid_scores - shift)) / np.sum(np.exp(scores - shift))
    )

    row = dict(base)
    row.update(
        {
            "seed": seed,
            "t_constant": t_constant,
            "invalid_radius": radius,
            "invalid_mass": invalid_mass,
            "invalid_max_score": float(np.max(invalid_scores)) if invalid_count else -math.inf,
            "sampled_topk_mass": sampled_topk_mass,
        }
    )
    return row


def _rows(args: argparse.Namespace) -> Iterable[dict[str, float | int | bool]]:
    for m in parse_csv_list(args.m, int):
        for valid_exponent in parse_csv_list(args.valid_exponents, float):
            if args.sample:
                for seed in parse_csv_list(args.seeds, int):
                    yield sample_plateau_trial(
                        m=m,
                        c=args.c,
                        valid_exponent=valid_exponent,
                        list_log_power=args.list_log_power,
                        capture_log_power=args.capture_log_power,
                        seed=seed,
                        t_constant=args.t_constant,
                        invalid_radius=args.invalid_radius,
                    )
            else:
                yield plateau_bound(
                    m=m,
                    c=args.c,
                    valid_exponent=valid_exponent,
                    list_log_power=args.list_log_power,
                    capture_log_power=args.capture_log_power,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", default="1000000,1000000000000,1000000000000000000")
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--valid-exponents", default="0.4,0.5,0.7")
    parser.add_argument("--list-log-power", type=float, default=2.0)
    parser.add_argument("--capture-log-power", type=float, default=2.0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--t-constant", type=float, default=32.0)
    parser.add_argument("--invalid-radius", type=float)
    parser.add_argument("--csv")
    args = parser.parse_args()

    fieldnames = SAMPLE_FIELDNAMES if args.sample else FIELDNAMES
    rows = list(_rows(args))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        print(",".join(fieldnames))
        for row in rows:
            print(",".join(str(row[name]) for name in fieldnames))


if __name__ == "__main__":
    main()
