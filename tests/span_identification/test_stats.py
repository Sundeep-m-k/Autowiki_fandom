"""Tests for stats utilities."""
from src.span_identification.stats import mean_std, aggregate_seed_metrics, bootstrap_significance


def test_mean_std():
    vals = [1.0, 2.0, 3.0]
    m, s = mean_std(vals)
    assert m == 2.0
    assert s == (2/3) ** 0.5


def test_mean_std_empty():
    m, s = mean_std([])
    assert m == 0.0
    assert s == 0.0


def test_aggregate_seed_metrics():
    seed_metrics = [
        {"span_f1": 0.8, "span_precision": 0.9},
        {"span_f1": 1.0, "span_precision": 0.95},
    ]
    agg = aggregate_seed_metrics(seed_metrics)
    assert "span_f1" in agg
    assert "span_f1_std" in agg
    assert agg["span_f1_min"] == 0.8
    assert agg["span_f1_max"] == 1.0


def test_bootstrap_significance():
    a = [0.5] * 5
    b = [0.9] * 5
    p = bootstrap_significance(a, b, n_bootstrap=100)
    assert p < 0.1
