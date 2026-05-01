"""Tests for error_categorization (Task 1 v2)."""

from src.span_identification.error_categorization import (
    categorize_example,
    classify_matched_pair,
    match_greedy_iou,
    span_iou,
)


def test_span_iou_identical():
    assert span_iou((0, 5), (0, 5)) == 1.0


def test_span_iou_disjoint():
    assert span_iou((0, 2), (5, 7)) == 0.0


def test_match_greedy_exact():
    gold = [(0, 10), (20, 30)]
    pred = [(0, 10), (20, 30)]
    pairs = match_greedy_iou(gold, pred, iou_threshold=0.5)
    assert set(pairs) == {(0, 0), (1, 1)}


def test_categorize_exact_and_spurious():
    text = "Hello world here"
    gold = [(0, 5)]
    pred = [(0, 5), (6, 11)]
    out = categorize_example(text, gold, pred, iou_threshold=0.5)
    assert out["counts"]["exact"] == 1
    assert out["counts"]["spurious"] == 1
    assert out["counts"]["missed"] == 0


def test_classify_matched_pair():
    text = "The quick brown fox"
    assert classify_matched_pair((4, 9), (4, 9), text) == "exact"
    # same surface string "quick" but different bounds — simulate with duplicate
    # substring at same position only
    assert classify_matched_pair((4, 9), (4, 9), text) == "exact"
    # boundary shift: same text slice for gold and pred
    g, p = (4, 9), (4, 9)
    assert classify_matched_pair(g, p, text) == "exact"


def test_aggregate_missed():
    text = "ab"
    gold = [(0, 1)]
    pred = []
    out = categorize_example(text, gold, pred, iou_threshold=0.5)
    assert out["counts"]["missed"] == 1
    assert out["counts"]["exact"] == 0
