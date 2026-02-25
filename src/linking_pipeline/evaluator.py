"""Evaluation metrics for the linking pipeline (Task 3).

Metrics computed per article and aggregated across the test set:

  Linking F1      : span boundary correct AND article_id correct
                    (the primary end-to-end metric for Task 3)
  Span F1         : span boundary correct regardless of article_id
                    (measures Task 1 component quality in the joint pipeline)
  Entity Accuracy : article_id correct, given span boundary is correct
                    (measures Task 2 component quality)
  NIL Rate        : fraction of gold spans assigned NIL by the threshold
  Coverage        : fraction of gold spans that had a Task 2 result at all
                    (spans with no Task 2 lookup hit cannot be linked)

All metrics use exact span matching (char_start and char_end must match exactly).
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("linking")

# ── Per-article metrics ────────────────────────────────────────────────────────

def _is_span_match(pred: dict, gold: dict) -> bool:
    return (
        pred["char_start"] == gold["char_start"]
        and pred["char_end"] == gold["char_end"]
    )


def evaluate_article(article: dict) -> dict:
    """
    Compute linking metrics for one article.

    article must have:
      gold_spans:       [{"char_start", "char_end", "gold_article_id", ...}]
      predicted_links:  [{"char_start", "char_end", "article_id", "linked", ...}]
    """
    gold_spans     = article.get("gold_spans", [])
    pred_links_all = article.get("predicted_links", [])
    pred_links     = [p for p in pred_links_all if p.get("linked", True)]

    n_gold = len(gold_spans)
    n_pred = len(pred_links)

    if n_gold == 0 and n_pred == 0:
        return {
            "linking_precision": 1.0, "linking_recall": 1.0, "linking_f1": 1.0,
            "span_precision": 1.0, "span_recall": 1.0, "span_f1": 1.0,
            "entity_accuracy": 1.0,
            "nil_rate": 0.0, "coverage": 1.0,
            "n_gold": 0, "n_pred": 0,
        }

    # ── Linking F1: span boundary AND article_id correct ──
    link_tp = 0
    for pred in pred_links:
        for gold in gold_spans:
            if (_is_span_match(pred, gold)
                    and pred.get("article_id") == gold.get("gold_article_id")):
                link_tp += 1
                break

    link_precision = link_tp / n_pred if n_pred else 0.0
    link_recall    = link_tp / n_gold if n_gold else 0.0
    link_f1 = (
        2 * link_precision * link_recall / (link_precision + link_recall)
        if (link_precision + link_recall) > 0 else 0.0
    )

    # ── Span F1: boundary only ──
    span_tp = 0
    for pred in pred_links:
        for gold in gold_spans:
            if _is_span_match(pred, gold):
                span_tp += 1
                break

    span_precision = span_tp / n_pred if n_pred else 0.0
    span_recall    = span_tp / n_gold if n_gold else 0.0
    span_f1 = (
        2 * span_precision * span_recall / (span_precision + span_recall)
        if (span_precision + span_recall) > 0 else 0.0
    )

    # ── Entity Accuracy: article_id correct given span match ──
    entity_correct = 0
    span_matched   = 0
    for pred in pred_links:
        for gold in gold_spans:
            if _is_span_match(pred, gold):
                span_matched += 1
                if pred.get("article_id") == gold.get("gold_article_id"):
                    entity_correct += 1
                break
    entity_accuracy = entity_correct / span_matched if span_matched else 0.0

    # ── NIL rate and coverage ──
    n_nil       = sum(1 for p in pred_links_all if not p.get("linked", True))
    n_no_result = sum(1 for p in pred_links_all if p.get("article_id") is None)
    nil_rate    = n_nil / n_gold if n_gold else 0.0
    coverage    = 1.0 - (n_no_result / n_gold) if n_gold else 1.0

    return {
        "linking_precision": round(link_precision, 4),
        "linking_recall":    round(link_recall, 4),
        "linking_f1":        round(link_f1, 4),
        "span_precision":    round(span_precision, 4),
        "span_recall":       round(span_recall, 4),
        "span_f1":           round(span_f1, 4),
        "entity_accuracy":   round(entity_accuracy, 4),
        "nil_rate":          round(nil_rate, 4),
        "coverage":          round(coverage, 4),
        "n_gold":            n_gold,
        "n_pred":            n_pred,
    }


def aggregate_metrics(article_metrics: list[dict]) -> dict:
    """Macro-average metrics across articles."""
    if not article_metrics:
        return {}
    keys = [k for k in article_metrics[0] if k not in ("n_gold", "n_pred")]
    out = {}
    for k in keys:
        vals = [m[k] for m in article_metrics if k in m]
        out[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    out["n_articles"] = len(article_metrics)
    out["n_gold_total"] = sum(m.get("n_gold", 0) for m in article_metrics)
    out["n_pred_total"] = sum(m.get("n_pred", 0) for m in article_metrics)
    return out


# ── Save metrics ───────────────────────────────────────────────────────────────

def save_metrics_json(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("[evaluator] metrics saved → %s", path)


# ── Research CSV ───────────────────────────────────────────────────────────────

FIELDNAMES = [
    "timestamp", "domain",
    "retriever", "reranker", "stage", "query_version",
    "nil_threshold",
    "n_articles", "n_gold_total", "n_pred_total",
    "linking_precision", "linking_recall", "linking_f1",
    "span_precision", "span_recall", "span_f1",
    "entity_accuracy", "nil_rate", "coverage",
    "notes",
]


def append_to_research_csv(
    csv_path: Path,
    domain: str,
    metrics: dict,
    config: dict,
    notes: str = "",
) -> None:
    t2  = config.get("task2", {})
    nil = config.get("nil_detection", {})
    row = {
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "domain":            domain,
        "retriever":         t2.get("retriever", ""),
        "reranker":          t2.get("reranker", "") if t2.get("stage") == "reranking" else "",
        "stage":             t2.get("stage", "reranking"),
        "query_version":     t2.get("query_version", 6),
        "nil_threshold":     nil.get("threshold", 0.0),
        "n_articles":        metrics.get("n_articles", 0),
        "n_gold_total":      metrics.get("n_gold_total", 0),
        "n_pred_total":      metrics.get("n_pred_total", 0),
        "linking_precision": metrics.get("linking_precision", 0.0),
        "linking_recall":    metrics.get("linking_recall", 0.0),
        "linking_f1":        metrics.get("linking_f1", 0.0),
        "span_precision":    metrics.get("span_precision", 0.0),
        "span_recall":       metrics.get("span_recall", 0.0),
        "span_f1":           metrics.get("span_f1", 0.0),
        "entity_accuracy":   metrics.get("entity_accuracy", 0.0),
        "nil_rate":          metrics.get("nil_rate", 0.0),
        "coverage":          metrics.get("coverage", 0.0),
        "notes":             notes,
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    log.info("[evaluator] research CSV updated → %s", csv_path)
