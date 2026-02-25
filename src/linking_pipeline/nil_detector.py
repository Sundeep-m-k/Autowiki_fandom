"""NIL detection for the linking pipeline.

A span is assigned NIL (no link produced) when its top-1 retrieval/reranking
score is below the configured threshold. This prevents low-confidence links
from appearing in the rendered HTML.

Threshold=0.0 (default) means link everything — no NIL filtering.
Ablation over thresholds [0.0, 0.1, 0.2, 0.5, 1.0] lets you study the
precision/recall trade-off of the linking system.
"""
from __future__ import annotations


def should_link(score: float, threshold: float) -> bool:
    """
    Return True if the span should be linked (score >= threshold).
    Return False to assign NIL (no link).
    """
    return score >= threshold


def apply_nil_filter(
    predicted_links: list[dict],
    threshold: float,
) -> list[dict]:
    """
    Apply NIL threshold to a list of predicted link dicts.
    Each dict must have a "retrieval_score" key.
    Sets "linked": True/False on each record in-place and returns the list.
    """
    for link in predicted_links:
        score = link.get("retrieval_score", 0.0)
        link["linked"] = should_link(score, threshold)
    return predicted_links
