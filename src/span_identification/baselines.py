"""Rule-based and heuristic baselines for span identification."""
from __future__ import annotations

import re
import random
from typing import Callable

from src.span_identification.dataset import extract_spans_from_links


def baseline_rule_capitalized(text: str, _gold_spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Link spans that look like capitalized phrases (Title Case)."""
    spans = []
    for m in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
        spans.append((m.start(), m.end()))
    return spans


def baseline_heuristic_anchor(text: str, _gold_spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Use a simple heuristic: phrases that could be anchor text.
    Link words that appear to be nouns (capitalized or common wiki-like patterns).
    """
    spans = []
    for m in re.finditer(r"\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Za-z][a-zA-Z0-9_]*)*\b", text):
        if len(m.group()) >= 2:
            spans.append((m.start(), m.end()))
    return spans


def baseline_random(
    text: str,
    gold_spans: list[tuple[int, int]],
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Random baseline: random spans with similar count and length distribution."""
    rng = random.Random(seed)
    n = len(gold_spans)
    if n == 0:
        return []
    lengths = [e - s for s, e in gold_spans]
    mean_len = sum(lengths) / len(lengths) if lengths else 10
    preds = []
    max_tries = n * 3
    tried = 0
    while len(preds) < n and tried < max_tries:
        try:
            start = rng.randint(0, max(0, len(text) - int(mean_len)))
            length = max(2, int(mean_len) + rng.randint(-2, 2))
            end = min(start + length, len(text))
            if end > start and (start, end) not in preds:
                preds.append((start, end))
        except Exception:
            pass
        tried += 1
    return sorted(preds)


def run_baseline(
    name: str,
    examples: list[dict],
) -> list[dict]:
    """
    Run a baseline on examples.
    examples: list of {"text", "gold_spans", ...}
    Returns list of {"text", "gold_spans", "pred_spans", ...}
    """
    if name == "rule_capitalized":
        fn: Callable = baseline_rule_capitalized
    elif name == "heuristic_anchor":
        fn = baseline_heuristic_anchor
    elif name == "random":
        fn = lambda t, g: baseline_random(t, g, seed=42)
    else:
        raise ValueError(f"Unknown baseline: {name}")

    out = []
    for ex in examples:
        pred = fn(ex["text"], ex["gold_spans"])
        out.append({**ex, "pred_spans": pred})
    return out
