"""Evaluation metrics for the article retrieval pipeline.

Implements:
  - Recall@K  : Is the gold article in the top-K retrieved results?
  - MRR       : Mean Reciprocal Rank — 1/rank_of_first_correct_hit

These metrics apply identically to retrieval and reranking results
because both share the same output JSONL format.

Corpus-granularity note:
  For paragraph/sentence granularity (Phase 2), multiple retrieved
  entries may belong to the same article. The is_hit() function
  considers a hit if any retrieved entry matches the gold article_id,
  which is correct for both article-level and sub-article granularity.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("article_retrieval")


# ── Per-query metrics ─────────────────────────────────────────────────────────

def _filter_source(result: dict) -> list[dict]:
    """Return the retrieved list with the source article removed and ranks re-assigned.

    Queries are derived from anchor text inside the source article, so that
    article always scores highest in retrieval — excluding it ensures metrics
    reflect genuine link-target discovery ability.
    """
    source_id = result.get("source_article_id")
    items = [r for r in result.get("retrieved", []) if r.get("article_id") != source_id]
    return [{"article_id": r["article_id"], "score": r.get("score", 0.0), "rank": i + 1}
            for i, r in enumerate(items)]


def reciprocal_rank(result: dict) -> float:
    """
    Compute 1/rank for the first retrieved item matching gold_article_id.
    Returns 0.0 if the gold article is not in the retrieved list.

    Uses the source-filtered retrieved list so the source article cannot
    contribute to the MRR calculation.
    """
    gold = result.get("gold_article_id")
    if gold is None:
        return 0.0
    for item in _filter_source(result):
        if item.get("article_id") == gold:
            rank = item.get("rank", 0)
            return 1.0 / rank if rank > 0 else 0.0
    return 0.0


def is_hit_at_k(result: dict, k: int) -> bool:
    """True if gold_article_id appears in the top-k retrieved articles.

    Uses the source-filtered retrieved list so the source article cannot
    artificially inflate recall.
    """
    gold = result.get("gold_article_id")
    if gold is None:
        return False
    for item in _filter_source(result)[:k]:
        if item.get("article_id") == gold:
            return True
    return False


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def compute_metrics(
    results: list[dict],
    recall_at_k: list[int] = None,
) -> dict[str, float]:
    """
    Compute aggregate Recall@K and MRR over a list of retrieval/reranking results.

    K values are automatically capped at the number of results to avoid
    misleading Recall@100 on a 50-article corpus.
    """
    if recall_at_k is None:
        recall_at_k = [1, 3, 5, 10, 20, 50, 100]

    if not results:
        return {f"recall_at_{k}": 0.0 for k in recall_at_k} | {"mrr": 0.0, "n_queries": 0}

    n = len(results)

    metrics: dict[str, float] = {}
    for k in recall_at_k:
        hits = sum(1 for r in results if is_hit_at_k(r, k))
        metrics[f"recall_at_{k}"] = hits / n

    rrs = [reciprocal_rank(r) for r in results]
    metrics["mrr"] = sum(rrs) / n
    metrics["n_queries"] = n

    return metrics


# ── Save metrics ──────────────────────────────────────────────────────────────

def save_metrics_json(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("[evaluator] metrics saved → %s", path)


# ── Research CSV ──────────────────────────────────────────────────────────────

FIELDNAMES = [
    "timestamp",
    "domain",
    "retriever",
    "reranker",
    "stage",
    "version",
    "corpus_representation",
    "corpus_granularity",
    "query_context_mode",
    "anchor_preprocessing",
    "n_queries",
    "n_articles",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "recall_at_50",
    "recall_at_100",
    "mrr",
    "notes",
]


def append_to_research_csv(
    csv_path: Path,
    domain: str,
    retriever: str,
    metrics: dict,
    config: dict,
    stage: str = "retrieval",
    reranker: str = "",
    version: int = 0,
    n_articles: int = 0,
    notes: str = "",
) -> None:
    """
    Append one experiment row to the research CSV.
    Writes a header row only if the file is empty or new.
    """
    ai_cfg   = config.get("article_index", {})
    q_cfg    = config.get("queries", {})

    row = {
        "timestamp":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "domain":               domain,
        "retriever":            retriever,
        "reranker":             reranker,
        "stage":                stage,
        "version":              version,
        "corpus_representation": ai_cfg.get("corpus_representation", "title_full"),
        "corpus_granularity":   ai_cfg.get("corpus_granularity", "article"),
        "query_context_mode":   q_cfg.get("query_context_mode", "anchor_sentence"),
        "anchor_preprocessing": q_cfg.get("anchor_preprocessing", "raw"),
        "n_queries":            int(metrics.get("n_queries", 0)),
        "n_articles":           n_articles,
        "recall_at_1":          round(metrics.get("recall_at_1", 0.0), 4),
        "recall_at_3":          round(metrics.get("recall_at_3", 0.0), 4),
        "recall_at_5":          round(metrics.get("recall_at_5", 0.0), 4),
        "recall_at_10":         round(metrics.get("recall_at_10", 0.0), 4),
        "recall_at_20":         round(metrics.get("recall_at_20", 0.0), 4),
        "recall_at_50":         round(metrics.get("recall_at_50", 0.0), 4),
        "recall_at_100":        round(metrics.get("recall_at_100", 0.0), 4),
        "mrr":                  round(metrics.get("mrr", 0.0), 4),
        "notes":                notes,
    }

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    log.info(
        "[evaluator] research CSV updated → %s (stage=%s, retriever=%s, version=v%d)",
        csv_path, stage, retriever, version,
    )


# ── Summary CSV ───────────────────────────────────────────────────────────────

def save_summary_csv(all_metrics: list[dict], path: Path) -> None:
    """Save a flat summary CSV aggregating all (retriever, version, stage) runs."""
    if not all_metrics:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(all_metrics[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_metrics)
    log.info("[evaluator] summary saved → %s (%d rows)", path, len(all_metrics))
