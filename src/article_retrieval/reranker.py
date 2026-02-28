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

Performance note
----------------
Use rerank_all_versions() instead of calling rerank() in a loop.
The all-versions function loads the cross-encoder once, then processes every
(retriever, version) pair with a single large batched predict() call.
This is 10–15× faster because model loading and CUDA warm-up only happen once.

Stage 3 — Fine-tuned re-ranker is deferred. The NegativeMiningAndDatasetGenerator
module and ReRankerTrainer module are planned but not yet implemented.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from article_retrieval.retriever import load_retrieval_results

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
    cross_encoder=None,
) -> list[dict]:
    """
    Re-rank the top-K retrieved articles for each query using a cross-encoder.

    Args:
        retrieval_results:    List of retrieval result records (from retriever.py).
        article_lookup:       Mapping from article_id → article text (for scoring).
        model_name:           Cross-encoder model name (Exp 6).
        top_k_input:          Number of candidates to re-rank (Exp 7).
        version:              Query version integer.
        query_records_by_id:  Mapping from query_id → query record.
        version_key_prefix:   Prefix for the query version key, default "v".
        cross_encoder:        Pre-loaded CrossEncoder instance. If None, loads from
                              model_name (slow — prefer passing a loaded instance).

    Returns:
        List of re-ranked result records.
    """
    if cross_encoder is None:
        cross_encoder = _load_cross_encoder(model_name)

    version_key   = f"{version_key_prefix}{version}"

    # ── Build all pairs for this version in one batch ────────────────────────
    # Collect (query_text, candidate_list) per record so we can do one big
    # cross_encoder.predict() call instead of one per query.
    records_with_pairs: list[tuple[dict, list[dict], int, int]] = []
    all_pairs: list[list[str]] = []

    for rec in retrieval_results:
        query_id = rec["query_id"]
        qrec     = query_records_by_id.get(query_id)
        if qrec is None:
            log.warning("[reranker] query record not found for query_id=%s; skipping", query_id)
            continue

        query_text = qrec.get("queries", {}).get(version_key, "")
        if not query_text:
            continue

        source_id = rec.get("source_article_id")
        candidates = [
            c for c in rec.get("retrieved", [])[:top_k_input]
            if c.get("article_id") != source_id
        ]
        if not candidates:
            continue

        pair_start = len(all_pairs)
        for cand in candidates:
            article_text = article_lookup.get(cand["article_id"], "")
            all_pairs.append([query_text, article_text])
        pair_end = len(all_pairs)

        records_with_pairs.append((rec, candidates, pair_start, pair_end))

    if not all_pairs:
        return []

    # One batched predict call for ALL queries in this version
    all_scores = cross_encoder.predict(all_pairs, convert_to_numpy=True, show_progress_bar=False)

    reranked_results = []
    for rec, candidates, pair_start, pair_end in records_with_pairs:
        scores = all_scores[pair_start:pair_end]
        scored = sorted(
            zip(candidates, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        reranked_retrieved = [
            {"article_id": cand["article_id"], "score": score, "rank": rank}
            for rank, (cand, score) in enumerate(scored, start=1)
        ]
        reranked_results.append({
            "query_id":          rec["query_id"],
            "gold_article_id":   rec["gold_article_id"],
            "source_article_id": rec["source_article_id"],
            "version":           version,
            "retriever":         rec["retriever"],
            "reranker":          model_name,
            "retrieved":         reranked_retrieved,
        })

    log.info(
        "[reranker] re-ranked %d queries (model=%s, top_k_input=%d, n_pairs=%d)",
        len(reranked_results), model_name, top_k_input, len(all_pairs),
    )
    return reranked_results


def rerank_all_versions(
    retrievers: list[str],
    versions: list[int],
    article_lookup: dict[int, str],
    model_name: str,
    top_k_input: int,
    query_records_by_id: dict[str, dict],
    get_ret_path_fn,
    get_out_path_fn,
    force: bool = False,
) -> None:
    """
    Re-rank all (retriever, version) combinations using ONE loaded cross-encoder.

    This is the fast path: the cross-encoder is loaded once, then all
    (retriever × version) pairs are processed with batched predict() calls.
    On Kudremukh (4× RTX 6000 Ada) this is ~12× faster than loading the model
    per version.

    Args:
        retrievers:           List of retriever names to process.
        versions:             List of version integers to process.
        article_lookup:       article_id → article text mapping.
        model_name:           Cross-encoder model name.
        top_k_input:          Number of top candidates to re-rank per query.
        query_records_by_id:  query_id → query record mapping.
        get_ret_path_fn:      Callable(retriever, version) → Path of retrieval JSONL.
        get_out_path_fn:      Callable(retriever, version) → Path to write reranking JSONL.
        force:                Recompute even if output already exists.
    """
    # Identify which (retriever, version) pairs actually need computing
    jobs: list[tuple[str, int]] = []
    for retriever in retrievers:
        for version in versions:
            out_path = get_out_path_fn(retriever, version)
            if out_path.exists() and not force:
                log.info("[reranker] skip (cached): %s", out_path)
                continue
            ret_path = get_ret_path_fn(retriever, version)
            if not ret_path.exists():
                log.warning("[reranker] retrieval results not found: %s — skip", ret_path)
                continue
            jobs.append((retriever, version))

    if not jobs:
        log.info("[reranker] all (retriever, version) pairs already cached — skip")
        return

    log.info(
        "[reranker] loading cross-encoder once for %d jobs: model=%s",
        len(jobs), model_name,
    )
    cross_encoder = _load_cross_encoder(model_name)

    for retriever, version in jobs:
        ret_path = get_ret_path_fn(retriever, version)
        out_path = get_out_path_fn(retriever, version)

        retrieval_results = load_retrieval_results(ret_path)
        reranked = rerank(
            retrieval_results=retrieval_results,
            article_lookup=article_lookup,
            model_name=model_name,
            top_k_input=top_k_input,
            version=version,
            query_records_by_id=query_records_by_id,
            cross_encoder=cross_encoder,
        )
        save_reranking_results(reranked, out_path)
        log.info("[reranker] done — retriever=%s v%d → %s", retriever, version, out_path)


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
