"""Article retrieval for the article retrieval pipeline.

Responsibilities (I/O only — no metric computation):
  - BM25 retrieval for one query version
  - TF-IDF retrieval for one query version
  - Dense FAISS retrieval for one query version
  - Save retrieval results to JSONL

Output JSONL format per line (one record per query):
  {
    "query_id": "...",
    "gold_article_id": 42,
    "source_article_id": 17,
    "version": 3,
    "retriever": "bm25",
    "retrieved": [
      {"article_id": 42, "score": 12.3, "rank": 1},
      {"article_id": 99, "score": 11.1, "rank": 2},
      ...
    ]
  }

This clean interface allows the reranker and evaluator to read any retrieval
result without knowing which retriever produced it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("article_retrieval")


# ── BM25 ─────────────────────────────────────────────────────────────────────

def _rerank_after_filter(retrieved: list[dict]) -> list[dict]:
    """Re-assign contiguous ranks after any items have been removed."""
    return [
        {"article_id": r["article_id"], "score": r["score"], "rank": i + 1}
        for i, r in enumerate(retrieved)
    ]


def retrieve_bm25(
    bm25_index,
    article_ids: list[int],
    query_records: list[dict],
    version: int,
    top_k: int,
    preprocessing: str = "raw",
) -> list[dict]:
    """
    Run BM25 retrieval for all queries for a given query version.
    Returns list of result records (not yet saved — caller decides).

    The source article is excluded from results: because queries are derived
    from anchor text inside the source article, that article always scores
    highest and would trivially occupy rank 1 for most queries.
    """
    from article_retrieval.article_index import _tokenise

    results = []
    version_key = f"v{version}"

    for rec in query_records:
        query_text = rec.get("queries", {}).get(version_key, "")
        if not query_text:
            continue
        tokens = _tokenise(query_text, preprocessing)
        scores = bm25_index.get_scores(tokens)
        # Retrieve extra candidates to absorb the source article exclusion
        top_indices = np.argsort(scores)[::-1][:top_k + 1]

        source_id = rec.get("source_article_id")
        retrieved = []
        for idx in top_indices:
            if article_ids[idx] == source_id:
                continue
            retrieved.append({
                "article_id": article_ids[idx],
                "score": float(scores[idx]),
                "rank": len(retrieved) + 1,
            })
            if len(retrieved) == top_k:
                break

        results.append({
            "query_id": rec["query_id"],
            "gold_article_id": rec["gold_article_id"],
            "source_article_id": rec["source_article_id"],
            "version": version,
            "retriever": "bm25",
            "retrieved": retrieved,
        })

    return results


# ── TF-IDF ────────────────────────────────────────────────────────────────────

def retrieve_tfidf(
    vectorizer,
    matrix,
    article_ids: list[int],
    query_records: list[dict],
    version: int,
    top_k: int,
    preprocessing: str = "raw",
) -> list[dict]:
    """Run TF-IDF retrieval for all queries for a given query version.

    The source article is excluded from results for the same reason as BM25.
    """
    from article_retrieval.article_index import _preprocess_text

    version_key = f"v{version}"
    query_texts = []
    filtered_records = []

    for rec in query_records:
        qt = rec.get("queries", {}).get(version_key, "")
        if qt:
            query_texts.append(_preprocess_text(qt, preprocessing))
            filtered_records.append(rec)

    if not query_texts:
        return []

    query_matrix = vectorizer.transform(query_texts)
    # Cosine similarity via dot product (TF-IDF vectors are not normalised here)
    scores_matrix = (query_matrix @ matrix.T).toarray()

    results = []
    for rec, scores in zip(filtered_records, scores_matrix):
        source_id = rec.get("source_article_id")
        top_indices = np.argsort(scores)[::-1][:top_k + 1]
        retrieved = []
        for idx in top_indices:
            if article_ids[idx] == source_id:
                continue
            retrieved.append({
                "article_id": article_ids[idx],
                "score": float(scores[idx]),
                "rank": len(retrieved) + 1,
            })
            if len(retrieved) == top_k:
                break
        results.append({
            "query_id": rec["query_id"],
            "gold_article_id": rec["gold_article_id"],
            "source_article_id": rec["source_article_id"],
            "version": version,
            "retriever": "tfidf",
            "retrieved": retrieved,
        })

    return results


# ── Dense FAISS ───────────────────────────────────────────────────────────────

def retrieve_dense(
    faiss_index,
    article_ids: list[int],
    query_embeddings: np.ndarray,
    query_records: list[dict],
    version: int,
    top_k: int,
    model_name: str,
) -> list[dict]:
    """
    Run dense retrieval for all queries for a given (model, query version).
    query_embeddings must match query_records order.

    The source article is excluded from results for the same reason as BM25.
    """
    # L2-normalise query vectors to match how FAISS index was built
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = (query_embeddings / norms).astype(np.float32)

    # Retrieve one extra to absorb the source article exclusion
    scores_batch, indices_batch = faiss_index.search(normed, top_k + 1)

    results = []
    for rec, scores, indices in zip(query_records, scores_batch, indices_batch):
        source_id = rec.get("source_article_id")
        retrieved = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(article_ids):
                continue
            if article_ids[idx] == source_id:
                continue
            retrieved.append({
                "article_id": article_ids[idx],
                "score": float(score),
                "rank": len(retrieved) + 1,
            })
            if len(retrieved) == top_k:
                break
        results.append({
            "query_id": rec["query_id"],
            "gold_article_id": rec["gold_article_id"],
            "source_article_id": rec["source_article_id"],
            "version": version,
            "retriever": model_name,
            "retrieved": retrieved,
        })

    return results


# ── Save / load ───────────────────────────────────────────────────────────────

def save_retrieval_results(results: list[dict], path: Path) -> None:
    """Save retrieval results to JSONL (one record per query)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("[retriever] saved %d retrieval results → %s", len(results), path)


def load_retrieval_results(path: Path) -> list[dict]:
    """Load retrieval results from JSONL."""
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results
