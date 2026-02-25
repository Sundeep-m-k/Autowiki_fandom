"""Evaluation metrics for span identification: span F1, token F1, exact match, overlap."""
from __future__ import annotations


def span_f1(
    gold_spans: list[tuple[int, int]],
    pred_spans: list[tuple[int, int]],
    match: str = "exact",
) -> tuple[float, float, float]:
    """
    Compute span-level precision, recall, F1.
    match: "exact" = span must match exactly, "overlap" = any overlap counts.
    """
    gold_set = {tuple(s) for s in gold_spans}
    pred_set = {tuple(s) for s in pred_spans}

    def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    if match == "exact":
        tp = len(gold_set & pred_set)
    else:
        tp_p = sum(1 for p in pred_set if any(overlaps(p, g) for g in gold_set))
        tp_g = sum(1 for g in gold_set if any(overlaps(p, g) for p in pred_set))
        precision = tp_p / len(pred_set) if pred_set else 0.0
        recall = tp_g / len(gold_set) if gold_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def char_f1(
    gold_spans: list[tuple[int, int]],
    pred_spans: list[tuple[int, int]],
    text_length: int,
) -> tuple[float, float, float]:
    """Character-level F1: treat each character position as a unit."""
    gold_chars = set()
    for s, e in gold_spans:
        for i in range(s, min(e, text_length)):
            gold_chars.add(i)
    pred_chars = set()
    for s, e in pred_spans:
        for i in range(s, min(e, text_length)):
            pred_chars.add(i)

    tp = len(gold_chars & pred_chars)
    precision = tp / len(pred_chars) if pred_chars else 0.0
    recall = tp / len(gold_chars) if gold_chars else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def exact_match_pct(
    gold_spans: list[tuple[int, int]],
    pred_spans: list[tuple[int, int]],
) -> float | None:
    """
    Percentage of gold spans exactly matched by a predicted span.
    Returns None when there are no gold spans so the caller can exclude
    such examples from the mean rather than inflating it.
    """
    if not gold_spans:
        return None
    gold_set = {tuple(s) for s in gold_spans}
    pred_set = {tuple(s) for s in pred_spans}
    matched = len(gold_set & pred_set)
    return matched / len(gold_set)


def overlap_f1(
    gold_spans: list[tuple[int, int]],
    pred_spans: list[tuple[int, int]],
) -> tuple[float, float, float]:
    """Span F1 with overlap-based matching."""
    return span_f1(gold_spans, pred_spans, match="overlap")


def evaluate_example(
    gold_spans: list[tuple[int, int]],
    pred_spans: list[tuple[int, int]],
    text_length: int,
) -> dict[str, float | None]:
    """Compute all metrics for one example."""
    p, r, f = span_f1(gold_spans, pred_spans)
    cp, cr, cf = char_f1(gold_spans, pred_spans, text_length)
    em = exact_match_pct(gold_spans, pred_spans)
    op, oa, of = overlap_f1(gold_spans, pred_spans)
    return {
        "span_precision": p,
        "span_recall": r,
        "span_f1": f,
        "char_precision": cp,
        "char_recall": cr,
        "char_f1": cf,
        "exact_match_pct": em,
        "overlap_precision": op,
        "overlap_recall": oa,
        "overlap_f1": of,
    }


def aggregate_metrics(example_metrics: list[dict[str, float | None]]) -> dict[str, float]:
    """Average metrics across examples, skipping None values (e.g. exact_match_pct on empty gold)."""
    if not example_metrics:
        return {}
    keys = example_metrics[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in example_metrics if k in m and m[k] is not None]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out
