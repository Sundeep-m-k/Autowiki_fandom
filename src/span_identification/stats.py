"""Statistical utilities: mean±std, paired bootstrap, significance testing."""
from __future__ import annotations

import random
from typing import Callable


def bootstrap_significance(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> float:
    """
    Paired permutation test for H0: mean(scores_b) = mean(scores_a).
    Under H0, each pair (a_i, b_i) can be swapped. Returns two-sided p-value.
    Low p-value => significant difference.
    """
    if len(scores_a) != len(scores_b) or len(scores_a) == 0:
        return 1.0
    rng = random.Random(seed)
    n = len(scores_a)
    observed_diff = sum(scores_b) / n - sum(scores_a) / n
    count_extreme = 0
    for _ in range(n_bootstrap):
        perm_diff = 0
        for i in range(n):
            if rng.random() < 0.5:
                perm_diff += (scores_a[i] - scores_b[i]) / n
            else:
                perm_diff += (scores_b[i] - scores_a[i]) / n
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    return (count_extreme + 1) / (n_bootstrap + 1)


def mean_std(values: list[float]) -> tuple[float, float]:
    """Compute mean and standard deviation."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    std = var ** 0.5
    return mean, std


def paired_bootstrap(
    gold_per_sample: list,
    pred_a_per_sample: list,
    pred_b_per_sample: list,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> float:
    """Paired bootstrap test. Returns p-value."""
    rng = random.Random(seed)
    n = len(gold_per_sample)
    count = 0
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        ga = sum(metric_fn(gold_per_sample[i], pred_a_per_sample[i]) for i in indices) / n
        gb = sum(metric_fn(gold_per_sample[i], pred_b_per_sample[i]) for i in indices) / n
        if (gb - ga) >= 0:
            count += 1
    return (count + 1) / (n_bootstrap + 1)


def aggregate_seed_metrics(seed_metrics: list[dict]) -> dict:
    """Aggregate metrics across seeds: mean and std for each key."""
    if not seed_metrics:
        return {}
    keys = seed_metrics[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in seed_metrics if k in m]
        mean, std = mean_std(vals)
        out[k] = mean
        out[f"{k}_std"] = std
        if vals:
            out[f"{k}_min"] = min(vals)
            out[f"{k}_max"] = max(vals)
    return out
