"""Bridge: Task 1 span → Task 2 pre-computed retrieval result.

Rather than re-running retrieval at linking time, Task 3 looks up the
pre-computed result from Task 2 using the key:
    (source_article_id, anchor_text)  →  query_id  →  top-1 retrieved article

This module:
  1. Loads the Task 2 query dataset to build the lookup key mapping.
  2. Loads the Task 2 retrieval or reranking JSONL to build
     query_id → top-1 (article_id, score) mapping.
  3. Exposes a single `build_lookup` function that returns a dict:
       (source_article_id, anchor_text_lower) → {"article_id": int, "score": float}

The anchor_text key is lowercased for case-insensitive matching.
If a span's anchor text appears multiple times in the same source article
(same anchor linking different targets), the one with the highest retrieval
score is kept — this is the correct behaviour for a linking system.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("linking")


def build_lookup(
    query_dataset_path: Path,
    results_path: Path,
) -> dict[tuple[int, str], dict]:
    """
    Build a (source_article_id, anchor_text_lower) → {article_id, score} lookup.

    Args:
        query_dataset_path: Task 2 query_dataset_*.jsonl
        results_path:       Task 2 retrieval_*.jsonl or reranking_*.jsonl

    Returns:
        Dict mapping (source_article_id, anchor_text_lower) to the top-1
        retrieved/reranked article_id and score for that query.
    """
    if not query_dataset_path.exists():
        log.error("[span_to_query] query dataset not found: %s", query_dataset_path)
        return {}
    if not results_path.exists():
        log.error("[span_to_query] Task 2 results not found: %s", results_path)
        return {}

    # Step 1: query_id → (source_article_id, anchor_text_lower)
    qid_to_key: dict[str, tuple[int, str]] = {}
    with open(query_dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("query_id")
            if qid is None:
                continue
            src_id  = int(rec.get("source_article_id", -1))
            anchor  = rec.get("anchor_text", "").lower()
            qid_to_key[qid] = (src_id, anchor)

    log.info("[span_to_query] loaded %d query records", len(qid_to_key))

    # Step 2: query_id → top-1 result (highest score from retrieved list)
    lookup: dict[tuple[int, str], dict] = {}
    n_loaded = 0

    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("query_id")
            if qid not in qid_to_key:
                continue

            retrieved = rec.get("retrieved", [])
            if not retrieved:
                continue

            # Top-1 is rank=1 (already sorted by retriever/reranker)
            top1 = retrieved[0]
            key  = qid_to_key[qid]

            # If same (source_article_id, anchor) appears multiple times,
            # keep the entry with the highest top-1 score.
            if key not in lookup or top1["score"] > lookup[key]["score"]:
                lookup[key] = {
                    "article_id": int(top1["article_id"]),
                    "score":      float(top1["score"]),
                }
            n_loaded += 1

    log.info(
        "[span_to_query] built lookup for %d unique (source_article_id, anchor) keys "
        "from %d result records",
        len(lookup), n_loaded,
    )
    return lookup


def lookup_span(
    lookup: dict[tuple[int, str], dict],
    source_article_id: int,
    anchor_text: str,
) -> dict | None:
    """
    Look up the top-1 retrieved article for a span.
    Returns {"article_id": int, "score": float} or None if not found.
    """
    key = (source_article_id, anchor_text.lower())
    return lookup.get(key)
