"""Span and seqeval metrics for HF Trainer compute_metrics.

The key entry point is ``compute_span_metrics_for_trainer``, which now accepts
a ``label_scheme`` argument so it works correctly for both BIO and BILOU.
"""
from __future__ import annotations

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

from src.span_identification.tokenization import LabelScheme


def _mask_and_convert_to_tags(
    pred_ids: np.ndarray,
    label_ids: np.ndarray,
    id2label: dict[int, str],
    ignore_index: int = -100,
) -> tuple[list[list[str]], list[list[str]]]:
    """Filter padding tokens and convert ids to tag sequences for seqeval."""
    pred_tags = []
    true_tags = []
    for pred_seq, label_seq in zip(pred_ids.tolist(), label_ids.tolist()):
        p_tags, t_tags = [], []
        for pid, lid in zip(pred_seq, label_seq):
            if lid == ignore_index:
                continue
            p_tags.append(id2label.get(int(pid), "O"))
            t_tags.append(id2label.get(int(lid), "O"))
        pred_tags.append(p_tags)
        true_tags.append(t_tags)
    return pred_tags, true_tags


def _spans_from_labels(labels: list[str], label_scheme: LabelScheme) -> list[tuple[int, int]]:
    """
    Extract token-level spans from a label sequence.

    Works for both BIO (B-SPAN / I-SPAN) and BILOU (B-SPAN / I-SPAN / L-SPAN / U-SPAN).
    Returns a list of (start, end) with exclusive end index.
    """
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(labels)

    if label_scheme == "BILOU":
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

    elif label_scheme == "BIO":
        while i < n:
            if labels[i] == "B-SPAN":
                start = i
                i += 1
                while i < n and labels[i] == "I-SPAN":
                    i += 1
                spans.append((start, i))
                continue
            i += 1

    else:
        raise ValueError(f"_spans_from_labels: unsupported scheme {label_scheme!r}. "
                         "Add a case here to support it.")
    return spans


def _span_level_metrics(
    pred_tags: list[list[str]],
    true_tags: list[list[str]],
    label_scheme: LabelScheme,
    match: str = "exact",
) -> dict[str, float]:
    """Compute span-level precision, recall, F1 (exact or overlap matching)."""
    def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    pred_count = gold_count = tp_p = tp_g = 0

    for p_seq, t_seq in zip(pred_tags, true_tags):
        pred_spans = _spans_from_labels(p_seq, label_scheme)
        gold_spans = _spans_from_labels(t_seq, label_scheme)
        pred_count += len(pred_spans)
        gold_count += len(gold_spans)

        if match == "exact":
            hits = len(set(pred_spans) & set(gold_spans))
            tp_p += hits
            tp_g += hits
        else:
            tp_p += sum(1 for p in pred_spans if any(overlaps(p, g) for g in gold_spans))
            tp_g += sum(1 for g in gold_spans if any(overlaps(p, g) for p in pred_spans))

    precision = tp_p / pred_count if pred_count else 0.0
    recall    = tp_g / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    prefix = "exact_span" if match == "exact" else "relaxed_span"
    return {f"{prefix}_precision": precision, f"{prefix}_recall": recall, f"{prefix}_f1": f1}


def compute_span_metrics_for_trainer(
    eval_pred: tuple[np.ndarray, np.ndarray],
    id2label: dict[int, str],
    label_scheme: LabelScheme = "BILOU",
    ignore_index: int = -100,
) -> dict[str, float]:
    """
    Compute seqeval + span metrics for HF Trainer's ``compute_metrics`` callback.

    Args:
        eval_pred:    (logits, label_ids) arrays from the Trainer.
        id2label:     Mapping from label id to tag string (e.g. ``{0: "O", 1: "B-SPAN", ...}``).
        label_scheme: ``"BIO"`` or ``"BILOU"`` — controls how spans are decoded from the tags.
        ignore_index: Label id used for padding/special tokens (default -100).
    """
    logits, labels = eval_pred
    pred_ids = np.argmax(np.array(logits), axis=-1)
    labels   = np.array(labels)

    pred_tags, true_tags = _mask_and_convert_to_tags(pred_ids, labels, id2label, ignore_index)

    metrics = {
        "eval_f1_seqeval":        f1_score(true_tags, pred_tags),
        "eval_precision_seqeval": precision_score(true_tags, pred_tags),
        "eval_recall_seqeval":    recall_score(true_tags, pred_tags),
    }

    for k, v in _span_level_metrics(pred_tags, true_tags, label_scheme, match="exact").items():
        metrics[f"eval_{k}"] = v
    for k, v in _span_level_metrics(pred_tags, true_tags, label_scheme, match="overlap").items():
        metrics[f"eval_{k}"] = v

    return metrics
