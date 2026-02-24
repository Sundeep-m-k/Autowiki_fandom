"""Span and seqeval metrics for HF Trainer compute_metrics."""
from __future__ import annotations

from typing import Any

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

from src.span_identification.preprocess import ID2LABEL, LABEL2ID


def _mask_and_convert_to_tags(
    pred_ids: np.ndarray,
    label_ids: np.ndarray,
    id2label: dict[int, str],
    ignore_index: int = -100,
) -> tuple[list[list[str]], list[list[str]]]:
    """Filter padding, convert ids to tag sequences for seqeval."""
    pred_tags = []
    true_tags = []
    for pred_seq, label_seq in zip(pred_ids.tolist(), label_ids.tolist()):
        p_tags = []
        t_tags = []
        for pid, lid in zip(pred_seq, label_seq):
            if lid == ignore_index:
                continue
            p_tags.append(id2label.get(int(pid), "O"))
            t_tags.append(id2label.get(int(lid), "O"))
        pred_tags.append(p_tags)
        true_tags.append(t_tags)
    return pred_tags, true_tags


def _spans_from_bilou_labels(labels: list[str]) -> list[tuple[int, int]]:
    """Extract token spans from BILOU/B-SPAN labels (start, end) exclusive end."""
    spans = []
    i = 0
    n = len(labels)
    while i < n:
        if labels[i] == "B-SPAN":
            start = i
            i += 1
            while i < n and labels[i] in ("I-SPAN", "L-SPAN"):
                i += 1
            spans.append((start, i))
            continue
        if labels[i] == "U-SPAN":
            spans.append((i, i + 1))
        i += 1
    return spans


def _span_level_metrics(
    pred_tags: list[list[str]],
    true_tags: list[list[str]],
    match: str = "exact",
) -> dict[str, float]:
    """Compute span-level precision, recall, F1 (exact or overlap)."""
    def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    pred_count = 0
    gold_count = 0
    tp_p = 0  # preds that overlap some gold
    tp_g = 0  # golds that overlap some pred

    for p_seq, t_seq in zip(pred_tags, true_tags):
        pred_spans = list(_spans_from_bilou_labels(p_seq))
        gold_spans = list(_spans_from_bilou_labels(t_seq))
        pred_count += len(pred_spans)
        gold_count += len(gold_spans)

        if match == "exact":
            ps = set(pred_spans)
            gs = set(gold_spans)
            hits = len(ps & gs)
            tp_p += hits
            tp_g += hits
        else:
            for p in pred_spans:
                if any(overlaps(p, g) for g in gold_spans):
                    tp_p += 1
            for g in gold_spans:
                if any(overlaps(p, g) for p in pred_spans):
                    tp_g += 1

    precision = tp_p / pred_count if pred_count else 0.0
    recall = tp_g / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    prefix = "exact_span" if match == "exact" else "relaxed_span"
    return {
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
    }


def compute_span_metrics_for_trainer(
    eval_pred: tuple[np.ndarray, np.ndarray],
    id2label: dict[int, str] | None = None,
    ignore_index: int = -100,
) -> dict[str, float]:
    """
    Compute seqeval + span metrics for HF Trainer.
    eval_pred: (predictions, label_ids) where predictions are logits.
    """
    id2label = id2label or ID2LABEL
    logits, labels = eval_pred
    pred_ids = np.argmax(logits, axis=-1)
    labels = np.array(labels)

    pred_tags, true_tags = _mask_and_convert_to_tags(
        pred_ids, labels, id2label, ignore_index
    )

    metrics = {
        "eval_f1_seqeval": f1_score(true_tags, pred_tags),
        "eval_precision_seqeval": precision_score(true_tags, pred_tags),
        "eval_recall_seqeval": recall_score(true_tags, pred_tags),
    }

    exact = _span_level_metrics(pred_tags, true_tags, match="exact")
    relaxed = _span_level_metrics(pred_tags, true_tags, match="overlap")

    for k, v in exact.items():
        metrics[f"eval_{k}"] = v
    for k, v in relaxed.items():
        metrics[f"eval_{k}"] = v

    return metrics
