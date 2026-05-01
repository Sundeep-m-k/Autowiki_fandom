"""Structured error categorization for span identification (Task 1).

Uses one-to-one greedy matching by descending IoU (common in detection-style eval).
Pairs with IoU < ``iou_threshold`` are left unmatched (counted as missed gold /
spurious pred).

Categories for matched pairs (IoU >= threshold):
  * exact            — identical (start, end)
  * boundary_shift   — non-exact but same surface string as gold span
  * overlap_confusion — any other overlap (wrong extent / wrong mention)

Unmatched predictions: spurious. Unmatched gold: missed.
"""
from __future__ import annotations

import random
from typing import Any, Literal

Span = tuple[int, int]
MatchCategory = Literal["exact", "boundary_shift", "overlap_confusion"]


def span_iou(a: Span, b: Span) -> float:
    """Intersection-over-union on character intervals [start, end)."""
    a0, a1 = a[0], a[1]
    b0, b1 = b[0], b[1]
    if a1 <= a0 or b1 <= b0:
        return 0.0
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def match_greedy_iou(
    gold_spans: list[Span],
    pred_spans: list[Span],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    """
    One-to-one matching: sort all (i, j) with IoU >= threshold by IoU descending,
    then greedily take pairs that do not reuse a gold or pred index.
    Deterministic tie-break: then by i, then by j.
    """
    n, m = len(gold_spans), len(pred_spans)
    if n == 0 or m == 0:
        return []
    candidates: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(m):
            iou = span_iou(gold_spans[i], pred_spans[j])
            if iou >= iou_threshold:
                candidates.append((iou, i, j))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_g: set[int] = set()
    used_p: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _iou, i, j in candidates:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        pairs.append((i, j))
    return pairs


def classify_matched_pair(
    gold: Span,
    pred: Span,
    text: str,
) -> MatchCategory:
    """Classify a matched pair (already IoU-qualified)."""
    if gold == pred:
        return "exact"
    g_txt = text[gold[0] : gold[1]] if 0 <= gold[0] < gold[1] <= len(text) else ""
    p_txt = text[pred[0] : pred[1]] if 0 <= pred[0] < pred[1] <= len(text) else ""
    if g_txt == p_txt:
        return "boundary_shift"
    return "overlap_confusion"


def categorize_example(
    text: str,
    gold_spans: list[Span],
    pred_spans: list[Span],
    iou_threshold: float = 0.5,
    unit_id: str | int | None = None,
) -> dict[str, Any]:
    """
    Per-example categorization + flat records for JSONL export.

    Returns:
      counts: exact, boundary_shift, overlap_confusion, spurious, missed
      records: list of dicts (one per matched pair, spurious pred, or missed gold)
    """
    gold_spans = [tuple(s) for s in gold_spans]
    pred_spans = [tuple(s) for s in pred_spans]

    pairs = match_greedy_iou(gold_spans, pred_spans, iou_threshold)
    matched_g = {i for i, _ in pairs}
    matched_p = {j for _, j in pairs}

    counts = {
        "exact": 0,
        "boundary_shift": 0,
        "overlap_confusion": 0,
        "spurious": 0,
        "missed": 0,
    }
    records: list[dict[str, Any]] = []

    for i, j in pairs:
        g, p = gold_spans[i], pred_spans[j]
        cat = classify_matched_pair(g, p, text)
        counts[cat] += 1
        iou = span_iou(g, p)
        records.append({
            "record_type": "matched",
            "category": cat,
            "iou": round(iou, 6),
            "gold_span": list(g),
            "pred_span": list(p),
            "gold_text": text[g[0] : g[1]] if g[1] <= len(text) else "",
            "pred_text": text[p[0] : p[1]] if p[1] <= len(text) else "",
            "unit_id": unit_id,
        })

    for j, p in enumerate(pred_spans):
        if j not in matched_p:
            counts["spurious"] += 1
            records.append({
                "record_type": "spurious",
                "category": "spurious",
                "pred_span": list(p),
                "pred_text": text[p[0] : p[1]] if p[1] <= len(text) else "",
                "unit_id": unit_id,
            })

    for i, g in enumerate(gold_spans):
        if i not in matched_g:
            counts["missed"] += 1
            records.append({
                "record_type": "missed",
                "category": "missed",
                "gold_span": list(g),
                "gold_text": text[g[0] : g[1]] if g[1] <= len(text) else "",
                "unit_id": unit_id,
            })

    return {"counts": counts, "records": records}


def aggregate_categorization(
    per_example: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sum counts across examples; collect flat list of all records."""
    total = {
        "exact": 0,
        "boundary_shift": 0,
        "overlap_confusion": 0,
        "spurious": 0,
        "missed": 0,
    }
    all_records: list[dict[str, Any]] = []
    for ex in per_example:
        c = ex["counts"]
        for k in total:
            total[k] += c[k]
        for r in ex["records"]:
            all_records.append(r)
    return {"counts": total, "records": all_records, "num_examples": len(per_example)}


def sample_stratified(
    records: list[dict[str, Any]],
    max_per_category: int = 15,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample up to ``max_per_category`` records per (record_type, category)."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        key = f"{r.get('record_type', '')}:{r.get('category', '')}"
        buckets.setdefault(key, []).append(r)
    out: list[dict[str, Any]] = []
    for _key, items in sorted(buckets.items()):
        k = min(len(items), max_per_category)
        out.extend(rng.sample(items, k) if k < len(items) else list(items))
    return out
