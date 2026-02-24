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


def token_f1(
    gold_spans: list[tuple[int, int]],
    pred_spans: list[tuple[int, int]],
    text_length: int,
) -> tuple[float, float, float]:
    """Token-level F1: treat each char position as a token."""
    gold_tokens = set()
    for s, e in gold_spans:
        for i in range(s, min(e, text_length)):
            gold_tokens.add(i)
    pred_tokens = set()
    for s, e in pred_spans:
        for i in range(s, min(e, text_length)):
            pred_tokens.add(i)

    tp = len(gold_tokens & pred_tokens)
    precision = tp / len(pred_tokens) if pred_tokens else 0.0
    recall = tp / len(gold_tokens) if gold_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def exact_match_pct(
    gold_spans: list[tuple[int, int]],
    pred_spans: list[tuple[int, int]],
) -> float:
    """Percentage of gold spans exactly matched by a predicted span."""
    if not gold_spans:
        return 1.0
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
) -> dict[str, float]:
    """Compute all metrics for one example."""
    p, r, f = span_f1(gold_spans, pred_spans)
    tp, tr, tf = token_f1(gold_spans, pred_spans, text_length)
    em = exact_match_pct(gold_spans, pred_spans)
    op, oa, of = overlap_f1(gold_spans, pred_spans)
    return {
        "span_precision": p,
        "span_recall": r,
        "span_f1": f,
        "token_precision": tp,
        "token_recall": tr,
        "token_f1": tf,
        "exact_match_pct": em,
        "overlap_precision": op,
        "overlap_recall": oa,
        "overlap_f1": of,
    }


def aggregate_metrics(example_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average metrics across examples (micro averaging)."""
    if not example_metrics:
        return {}
    keys = example_metrics[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in example_metrics if k in m]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out
