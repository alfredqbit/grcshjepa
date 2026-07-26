from __future__ import annotations

import numpy as np


def bootstrap_ci(values, alpha: float = 0.05, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return (float("nan"), float("nan"))
    boots = [np.mean(rng.choice(values, size=values.size, replace=True)) for _ in range(n_boot)]
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def paired_noninferiority(candidate, reference, margin: float, alpha: float = 0.05, seed: int = 0) -> dict:
    diff = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    lo, hi = bootstrap_ci(diff, alpha=2 * alpha, seed=seed)  # one-sided alpha as two-sided 2 alpha
    return {
        "mean_diff": float(np.mean(diff)),
        "lower_bound": lo,
        "upper_bound": hi,
        "margin": float(margin),
        "noninferior": bool(lo > -margin),
    }


def equivalence_interval(candidate, reference, margin: float, alpha: float = 0.05, seed: int = 0) -> dict:
    diff = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    lo, hi = bootstrap_ci(diff, alpha=2 * alpha, seed=seed)
    return {
        "mean_diff": float(np.mean(diff)),
        "lower_bound": lo,
        "upper_bound": hi,
        "margin": float(margin),
        "equivalent": bool(lo > -margin and hi < margin),
    }


def classify_interval(lo: float, hi: float, support_region: tuple[float, float], null_region: tuple[float, float]) -> str:
    s_lo, s_hi = support_region
    n_lo, n_hi = null_region
    if lo >= s_lo and hi <= s_hi:
        return "supported"
    if lo >= n_lo and hi <= n_hi:
        return "equivalent_to_null"
    if hi < s_lo or lo > s_hi:
        return "contradicted_or_too_small"
    return "inconclusive"


def damage_degradation(baseline, damaged) -> np.ndarray:
    return np.asarray(baseline, dtype=float) - np.asarray(damaged, dtype=float)
