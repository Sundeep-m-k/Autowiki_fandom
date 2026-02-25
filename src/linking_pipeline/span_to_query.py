"""Bridge: Task 1 span → Task 2 pre-computed retrieval result.

Rather than re-running retrieval at linking time, Task 3 looks up the
pre-computed result from Task 2 using the key:
    (source_article_id, char_start, char_end)  →  query_id  →  top-1 retrieved article

Using char offsets as the key (instead of anchor text) eliminates the span
recall gap that occurred when the same anchor text appeared multiple times in
one article linking to different targets.

Falls back to (source_article_id, anchor_text_lower) for query records that
pre-date char-offset storage (legacy query datasets built before this fix).

This module:
  1. Loads the Task 2 query dataset to build the lookup key mapping.
  2. Loads the Task 2 retrieval or reranking JSONL to build
     query_id → top-1 (article_id, score) mapping.
  3. Exposes a single `build_lookup` function that returns a dict:
       (source_article_id, char_start, char_end) → {"article_id": int, "score": float}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("linking")


def build_lookup(
    query_dataset_path: Path,
    results_path: Path,
) -> dict[tuple, dict]:
    """
    Build a span-key → {article_id, score} lookup.

    Primary key:  (source_article_id, char_start, char_end)
    Fallback key: (source_article_id, anchor_text_lower)  — for legacy datasets
                  that do not store char offsets.

    Args:
        query_dataset_path: Task 2 query_dataset_*.jsonl
        results_path:       Task 2 retrieval_*.jsonl or reranking_*.jsonl

    Returns:
        Dict mapping span-key to the top-1 retrieved/reranked article_id and score.
    """
    if not query_dataset_path.exists():
        log.error("[span_to_query] query dataset not found: %s", query_dataset_path)
        return {}
    if not results_path.exists():
        log.error("[span_to_query] Task 2 results not found: %s", results_path)
        return {}

    # Step 1: query_id → span key
    qid_to_key: dict[str, tuple] = {}
    n_with_offsets = 0
    with open(query_dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("query_id")
            if qid is None:
                continue
            src_id     = int(rec.get("source_article_id", -1))
            char_start = rec.get("char_start")
            char_end   = rec.get("char_end")
            if char_start is not None and char_end is not None:
                # New-format key: offset-based, unambiguous
                qid_to_key[qid] = (src_id, int(char_start), int(char_end))
                n_with_offsets += 1
            else:
                # Legacy key: anchor-text-based
                anchor = rec.get("anchor_text", "").lower()
                qid_to_key[qid] = (src_id, anchor)

    log.info(
        "[span_to_query] loaded %d query records (%d with char offsets, %d legacy)",
        len(qid_to_key), n_with_offsets, len(qid_to_key) - n_with_offsets,
    )

    # Step 2: query_id → top-1 result
    lookup: dict[tuple, dict] = {}
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

            # For legacy anchor-text keys, keep entry with highest top-1 score
            # (multiple anchors with same text in one article). For offset keys
            # each key is unique, so this branch is never taken.
            if key not in lookup or top1["score"] > lookup[key]["score"]:
                lookup[key] = {
                    "article_id": int(top1["article_id"]),
                    "score":      float(top1["score"]),
                }
            n_loaded += 1

    log.info(
        "[span_to_query] built lookup for %d unique span keys from %d result records",
        len(lookup), n_loaded,
    )
    return lookup


def lookup_span(
    lookup: dict[tuple, dict],
    source_article_id: int,
    anchor_text: str,
    char_start: int | None = None,
    char_end: int | None = None,
) -> dict | None:
    """
    Look up the top-1 retrieved article for a span.

    Tries the char-offset key first (unambiguous); falls back to the legacy
    anchor-text key so that lookups still work against older query datasets.

    Returns {"article_id": int, "score": float} or None if not found.
    """
    if char_start is not None and char_end is not None:
        result = lookup.get((source_article_id, int(char_start), int(char_end)))
        if result is not None:
            return result
    # Legacy fallback
    return lookup.get((source_article_id, anchor_text.lower()))
