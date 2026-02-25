"""Zero-shot cross-encoder re-ranker for the article retrieval pipeline.

Responsibilities (I/O only — no metric computation):
  - Load top-K retrieval results (from retriever.py output)
  - Load article texts from persisted article JSONL
  - Score (query, article) pairs using a cross-encoder
  - Save re-ranked results to JSONL

Output JSONL format is identical to retrieval results for a clean evaluator interface:
  {
    "query_id": "...",
    "gold_article_id": 42,
    "source_article_id": 17,
    "version": 3,
    "retriever": "bm25",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "retrieved": [
      {"article_id": 42, "score": 3.21, "rank": 1},
      ...
    ]
  }

Exp 6  (re-ranker model):      model_name selects which cross-encoder to use
Exp 7  (re-ranker input size): top_k_input controls how many candidates are re-ranked

Stage 3 — Fine-tuned re-ranker is deferred. The NegativeMiningAndDatasetGenerator
module and ReRankerTrainer module are planned but not yet implemented.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("article_retrieval")


def _load_cross_encoder(model_name: str):
    """Load a CrossEncoder model (lazy import)."""
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for re-ranking. "
            "Install with: pip install sentence-transformers"
        ) from e
    log.info("[reranker] loading cross-encoder: %s", model_name)
    return CrossEncoder(model_name)


def rerank(
    retrieval_results: list[dict],
    article_lookup: dict[int, str],
    model_name: str,
    top_k_input: int,
    version: int,
    query_records_by_id: dict[str, dict],
    version_key_prefix: str = "v",
) -> list[dict]:
    """
    Re-rank the top-K retrieved articles for each query using a cross-encoder.

    Args:
        retrieval_results:    List of retrieval result records (from retriever.py).
        article_lookup:       Mapping from article_id → article text (for scoring).
        model_name:           Cross-encoder model name (Exp 6).
        top_k_input:          Number of candidates to re-rank (Exp 7).
        version:              Query version integer.
        query_records_by_id:  Mapping from query_id → query record
                              (used to get the query text for version).
        version_key_prefix:   Prefix for the query version key, default "v".

    Returns:
        List of re-ranked result records.
    """
    cross_encoder = _load_cross_encoder(model_name)
    version_key   = f"{version_key_prefix}{version}"
    reranked_results = []

    for rec in retrieval_results:
        query_id = rec["query_id"]
        qrec     = query_records_by_id.get(query_id)
        if qrec is None:
            log.warning("[reranker] query record not found for query_id=%s; skipping", query_id)
            continue

        query_text = qrec.get("queries", {}).get(version_key, "")
        if not query_text:
            continue

        # Take top-K candidates from retrieval (Exp 7)
        candidates = rec.get("retrieved", [])[:top_k_input]
        if not candidates:
            reranked_results.append({**rec, "reranker": model_name, "retrieved": []})
            continue

        # Build (query, article_text) pairs for scoring
        pairs = []
        for cand in candidates:
            article_text = article_lookup.get(cand["article_id"], "")
            pairs.append([query_text, article_text])

        scores = cross_encoder.predict(pairs, convert_to_numpy=True)

        # Sort by descending score
        scored = sorted(
            zip(candidates, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        reranked_retrieved = [
            {
                "article_id": cand["article_id"],
                "score": score,
                "rank": rank,
            }
            for rank, (cand, score) in enumerate(scored, start=1)
        ]

        reranked_results.append({
            "query_id":         rec["query_id"],
            "gold_article_id":  rec["gold_article_id"],
            "source_article_id": rec["source_article_id"],
            "version":          version,
            "retriever":        rec["retriever"],
            "reranker":         model_name,
            "retrieved":        reranked_retrieved,
        })

    log.info(
        "[reranker] re-ranked %d queries (model=%s, top_k_input=%d)",
        len(reranked_results), model_name, top_k_input,
    )
    return reranked_results


# ── Save / load ───────────────────────────────────────────────────────────────

def save_reranking_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("[reranker] saved %d reranking results → %s", len(results), path)


def load_reranking_results(path: Path) -> list[dict]:
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


# ── Article text lookup helper ─────────────────────────────────────────────────

def build_article_lookup(articles_jsonl_path: Path) -> dict[int, str]:
    """Build article_id → text mapping for use in scoring pairs."""
    lookup: dict[int, str] = {}
    with open(articles_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            aid = rec.get("article_id")
            if aid is not None:
                lookup[int(aid)] = rec.get("text", "")
    return lookup
