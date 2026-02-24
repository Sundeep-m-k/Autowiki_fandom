"""Tests for error analysis."""
from src.span_identification.error_analysis import categorize_errors, sample_errors


def test_categorize_errors():
    gold = [(0, 10), (20, 30)]
    pred = [(0, 10), (15, 25)]
    cat = categorize_errors(gold, pred)
    assert "tp_count" in cat
    assert "fp_count" in cat
    assert "fn_count" in cat


def test_sample_errors():
    examples = [
        {"text": "Axel Foley.", "gold_spans": [(0, 10)], "pred_spans": [(0, 10)]},
        {"text": "No links.", "gold_spans": [], "pred_spans": [(0, 3)]},
    ]
    fp, fn = sample_errors(examples, max_fp=5, max_fn=5)
    assert isinstance(fp, list)
    assert isinstance(fn, list)
