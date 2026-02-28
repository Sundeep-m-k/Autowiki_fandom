"""Error analysis for the article retrieval and reranking pipeline.

Provides pure functions (no I/O dependencies) for:
  - Categorizing each query as not_retrieved / low_rank / top10_hit
  - Aggregating error counts and per-category MRR
  - Sampling miss records for human review
  - Comparing retrieval vs reranking rank changes per query
  - Saving outputs to disk (errors_summary.json, miss_samples.jsonl,
    rank_change_summary.json)
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

log = logging.getLogger("article_retrieval")

# Default K threshold that separates "low_rank" from "top10_hit"
_DEFAULT_HIT_K = 10


# ── Source filtering (mirrors evaluator._filter_source) ───────────────────────

def _filter_source(result: dict) -> list[dict]:
    """Return the retrieved list with the source article removed and ranks re-assigned."""
    source_id = result.get("source_article_id")
    items = [r for r in result.get("retrieved", []) if r.get("article_id") != source_id]
    return [
        {"article_id": r["article_id"], "score": r.get("score", 0.0), "rank": i + 1}
        for i, r in enumerate(items)
    ]


# ── Per-query categorization ───────────────────────────────────────────────────

def categorize_query(
    result: dict,
    hit_k: int = _DEFAULT_HIT_K,
) -> dict[str, Any]:
    """
    Classify one retrieval/reranking result record.

    Returns a dict with:
      category       : "top10_hit" | "low_rank" | "not_retrieved"
      gold_rank      : int rank of gold article (None if not retrieved)
      gold_score     : float score of gold article (None if not retrieved)
      top1_article_id: article_id of the top-ranked candidate
      top1_score     : score of the top-ranked candidate
      score_gap      : top1_score - gold_score (None if gold not retrieved)
      reciprocal_rank: 1/gold_rank (0.0 if not retrieved)
    """
    gold_id = result.get("gold_article_id")
    filtered = _filter_source(result)

    gold_rank: int | None = None
    gold_score: float | None = None
    for item in filtered:
        if item["article_id"] == gold_id:
            gold_rank = item["rank"]
            gold_score = item["score"]
            break

    top1 = filtered[0] if filtered else None
    top1_article_id = top1["article_id"] if top1 else None
    top1_score = top1["score"] if top1 else None

    if gold_rank is None:
        category = "not_retrieved"
        rr = 0.0
        score_gap = None
    elif gold_rank <= hit_k:
        category = "top10_hit"
        rr = 1.0 / gold_rank
        score_gap = (top1_score - gold_score) if top1_score is not None else None
    else:
        category = "low_rank"
        rr = 1.0 / gold_rank
        score_gap = (top1_score - gold_score) if top1_score is not None else None

    return {
        "query_id": result.get("query_id"),
        "version": result.get("version"),
        "gold_article_id": gold_id,
        "gold_rank": gold_rank,
        "gold_score": gold_score,
        "top1_article_id": top1_article_id,
        "top1_score": top1_score,
        "score_gap": score_gap,
        "category": category,
        "reciprocal_rank": rr,
    }


# ── Aggregate statistics ───────────────────────────────────────────────────────

def aggregate_errors(
    results: list[dict],
    hit_k: int = _DEFAULT_HIT_K,
) -> dict[str, Any]:
    """
    Compute aggregate error counts and per-category MRR over all results.

    Returns a summary dict suitable for JSON serialisation.
    """
    cats = [categorize_query(r, hit_k=hit_k) for r in results]

    n = len(cats)
    counts: dict[str, int] = {"not_retrieved": 0, "low_rank": 0, "top10_hit": 0}
    rrs: dict[str, list[float]] = {"not_retrieved": [], "low_rank": [], "top10_hit": []}
    score_gaps: list[float] = []

    for c in cats:
        cat = c["category"]
        counts[cat] += 1
        rrs[cat].append(c["reciprocal_rank"])
        if c["score_gap"] is not None:
            score_gaps.append(c["score_gap"])

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    all_rrs = [c["reciprocal_rank"] for c in cats]

    return {
        "n_queries": n,
        "hit_k": hit_k,
        "not_retrieved": counts["not_retrieved"],
        "low_rank": counts["low_rank"],
        "top10_hit": counts["top10_hit"],
        "mrr": round(mean(all_rrs), 4),
        "mrr_not_retrieved": round(mean(rrs["not_retrieved"]), 4),
        "mrr_low_rank": round(mean(rrs["low_rank"]), 4),
        "mrr_top10_hit": round(mean(rrs["top10_hit"]), 4),
        "score_gap_mean": round(mean(score_gaps), 4) if score_gaps else None,
    }


# ── Miss sampling ─────────────────────────────────────────────────────────────

def sample_misses(
    results: list[dict],
    max_samples: int = 50,
    seed: int = 42,
    hit_k: int = _DEFAULT_HIT_K,
    anchor_lookup: dict[str, str] | None = None,
) -> list[dict]:
    """
    Randomly sample miss records (not_retrieved + low_rank) for human review.

    Each returned record contains the query metadata, gold rank, top-10
    candidates, and optionally the anchor_text from `anchor_lookup`
    (a dict mapping query_id → anchor_text).
    """
    rng = random.Random(seed)
    misses = []
    for result in results:
        cat_info = categorize_query(result, hit_k=hit_k)
        if cat_info["category"] == "top10_hit":
            continue
        filtered = _filter_source(result)
        record: dict[str, Any] = {
            "query_id": result.get("query_id"),
            "version": result.get("version"),
            "gold_article_id": result.get("gold_article_id"),
            "gold_rank": cat_info["gold_rank"],
            "gold_score": cat_info["gold_score"],
            "top1_article_id": cat_info["top1_article_id"],
            "top1_score": cat_info["top1_score"],
            "score_gap": cat_info["score_gap"],
            "category": cat_info["category"],
            "top10": filtered[:10],
        }
        if anchor_lookup is not None:
            record["anchor_text"] = anchor_lookup.get(result.get("query_id", ""), "")
        misses.append(record)

    return rng.sample(misses, min(len(misses), max_samples))


# ── Retrieval vs reranking comparison ─────────────────────────────────────────

def compare_retrieval_reranking(
    ret_results: list[dict],
    rer_results: list[dict],
) -> dict[str, Any]:
    """
    Pair retrieval and reranking results by query_id and compute rank-change stats.

    Returns a summary with counts for:
      rank_improved  — gold rank went up (lower number) after reranking
      rank_degraded  — gold rank went down after reranking
      rank_unchanged — gold rank stayed the same
      rank_dropped   — gold was in retrieval top-K but absent from reranking output
      not_in_retrieval — gold was absent from retrieval entirely
    """
    ret_by_id = {r["query_id"]: r for r in ret_results if "query_id" in r}
    rer_by_id = {r["query_id"]: r for r in rer_results if "query_id" in r}

    counts = {
        "rank_improved": 0,
        "rank_degraded": 0,
        "rank_unchanged": 0,
        "rank_dropped": 0,
        "not_in_retrieval": 0,
    }
    gain_values: list[int] = []  # positive = improved, negative = degraded

    for qid, rer in rer_by_id.items():
        ret = ret_by_id.get(qid)
        if ret is None:
            continue

        ret_cat = categorize_query(ret)
        rer_cat = categorize_query(rer)

        ret_rank = ret_cat["gold_rank"]
        rer_rank = rer_cat["gold_rank"]

        if ret_rank is None:
            counts["not_in_retrieval"] += 1
            continue

        if rer_rank is None:
            counts["rank_dropped"] += 1
            continue

        gain = ret_rank - rer_rank  # positive means rank improved (moved up)
        gain_values.append(gain)

        if gain > 0:
            counts["rank_improved"] += 1
        elif gain < 0:
            counts["rank_degraded"] += 1
        else:
            counts["rank_unchanged"] += 1

    n = len(rer_by_id)
    mean_gain = sum(gain_values) / len(gain_values) if gain_values else 0.0

    return {
        "n_queries": n,
        **counts,
        "mean_rank_gain": round(mean_gain, 3),
    }


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_error_analysis(
    output_dir: Path,
    summary: dict,
    miss_samples: list[dict],
    rank_change_summary: dict | None = None,
) -> None:
    """
    Save error analysis outputs to disk.

    Always writes:
      errors_summary.json
      miss_samples.jsonl

    Writes only when provided:
      rank_change_summary.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "errors_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "miss_samples.jsonl", "w", encoding="utf-8") as f:
        for rec in miss_samples:
            f.write(json.dumps(rec) + "\n")

    if rank_change_summary is not None:
        with open(output_dir / "rank_change_summary.json", "w", encoding="utf-8") as f:
            json.dump(rank_change_summary, f, indent=2)

    log.info("[error_analysis] saved → %s", output_dir)
