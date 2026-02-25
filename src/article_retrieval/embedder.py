"""Text embedder using sentence-transformers for the article retrieval pipeline.

Responsibilities (I/O only — no retrieval logic):
  - Encode article texts to dense vectors, with caching
  - Encode query texts for a given (model, query version) pair, with caching
  - Support batched encoding for GPU efficiency

Exp 2 (retriever model) is expressed here — the model_name passed in selects
the encoder used. All dense retrieval models go through this module.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("article_retrieval")


def _load_st_model(model_name: str):
    """Load a sentence-transformers model (lazy import to avoid hard dependency)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for dense retrieval. "
            "Install with: pip install sentence-transformers"
        ) from e
    log.info("[embedder] loading model: %s", model_name)
    return SentenceTransformer(model_name)


def embed_articles(
    texts: list[str],
    article_ids: list[int],
    model_name: str,
    emb_path: Path,
    ids_path: Path,
    batch_size: int = 64,
    force: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """
    Encode article texts and save embeddings to disk.

    If emb_path already exists and force=False, loads from disk (cache hit).
    Returns (embeddings [N × D], article_ids [N]).
    """
    if emb_path.exists() and ids_path.exists() and not force:
        log.info("[embedder] cache hit — loading article embeddings from %s", emb_path)
        embeddings = np.load(str(emb_path))
        with open(ids_path, encoding="utf-8") as f:
            cached_ids = json.load(f)
        return embeddings, cached_ids

    model = _load_st_model(model_name)
    log.info("[embedder] encoding %d articles (batch_size=%d)", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # FAISS normalises before indexing
    )
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(emb_path), embeddings)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(article_ids, f)
    log.info("[embedder] article embeddings saved → %s (%d vectors)", emb_path, len(article_ids))
    return embeddings, article_ids


def embed_queries(
    query_texts: list[str],
    query_ids: list[str],
    model_name: str,
    emb_path: Path,
    ids_path: Path,
    batch_size: int = 64,
    force: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Encode query texts for one (model, query version) pair.

    If emb_path already exists and force=False, loads from disk (cache hit).
    Returns (embeddings [N × D], query_ids [N]).
    """
    if emb_path.exists() and ids_path.exists() and not force:
        log.info("[embedder] cache hit — loading query embeddings from %s", emb_path)
        embeddings = np.load(str(emb_path))
        with open(ids_path, encoding="utf-8") as f:
            cached_ids = json.load(f)
        return embeddings, cached_ids

    model = _load_st_model(model_name)
    log.info("[embedder] encoding %d queries (batch_size=%d)", len(query_texts), batch_size)
    embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(emb_path), embeddings)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(query_ids, f)
    log.info("[embedder] query embeddings saved → %s (%d vectors)", emb_path, len(query_ids))
    return embeddings, query_ids
