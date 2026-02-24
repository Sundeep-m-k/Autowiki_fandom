"""Tests for evaluator metrics."""
import pytest

from src.span_identification.evaluator import (
    span_f1,
    token_f1,
    exact_match_pct,
    overlap_f1,
    evaluate_example,
    aggregate_metrics,
)


def test_span_f1_exact_perfect():
    gold = [(0, 10), (20, 30)]
    pred = [(0, 10), (20, 30)]
    p, r, f = span_f1(gold, pred)
    assert p == 1.0
    assert r == 1.0
    assert f == 1.0


def test_span_f1_exact_partial():
    gold = [(0, 10), (20, 30)]
    pred = [(0, 10)]
    p, r, f = span_f1(gold, pred)
    assert p == 1.0
    assert r == 0.5
    assert f == 2 * 0.5 / 1.5


def test_span_f1_empty_gold():
    gold = []
    pred = [(0, 10)]
    p, r, f = span_f1(gold, pred)
    assert r == 0.0
    assert p == 0.0


def test_token_f1():
    gold = [(0, 5)]
    pred = [(0, 3)]
    p, r, f = token_f1(gold, pred, text_length=50)
    assert p > 0
    assert r > 0


def test_exact_match_pct():
    gold = [(0, 10), (20, 30)]
    pred = [(0, 10), (20, 30)]
    assert exact_match_pct(gold, pred) == 1.0
    pred = [(0, 10)]
    assert exact_match_pct(gold, pred) == 0.5


def test_aggregate_metrics():
    m1 = {"span_f1": 0.8, "span_precision": 0.9}
    m2 = {"span_f1": 1.0, "span_precision": 1.0}
    agg = aggregate_metrics([m1, m2])
    assert agg["span_f1"] == 0.9
    assert agg["span_precision"] == 0.95
