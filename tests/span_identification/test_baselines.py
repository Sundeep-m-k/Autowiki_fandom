"""Tests for baseline span predictors."""
import pytest

from src.span_identification.baselines import (
    baseline_rule_capitalized,
    baseline_heuristic_anchor,
    baseline_random,
    run_baseline,
)


def test_rule_capitalized():
    text = "Axel Foley goes to Beverly Hills."
    spans = baseline_rule_capitalized(text, [])
    assert len(spans) >= 2  # "Axel Foley", "Beverly Hills"
    assert any(s[1] - s[0] == 10 for s in spans)  # "Axel Foley"


def test_heuristic_anchor():
    text = "Eddie Murphy stars in the film."
    spans = baseline_heuristic_anchor(text, [])
    assert len(spans) >= 1


def test_random_baseline():
    text = "Some text here."
    gold = [(0, 4), (10, 14)]
    spans = baseline_random(text, gold, seed=42)
    assert len(spans) <= len(gold) + 2
    for s, e in spans:
        assert 0 <= s < e <= len(text)


def test_run_baseline():
    examples = [{"text": "Axel Foley.", "gold_spans": [(0, 10)]}]
    out = run_baseline("rule_capitalized", examples)
    assert len(out) == 1
    assert "pred_spans" in out[0]
    assert isinstance(out[0]["pred_spans"], list)
