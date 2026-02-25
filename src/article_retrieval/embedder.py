"""Text embedder using sentence-transformers for the article retrieval pipeline.

Responsibilities (I/O only — no retrieval logic):
  - Encode article texts to dense vectors, with caching
  - Encode query texts for a given (model, query version) pair, with caching
  - Encode ALL query versions in one model-load pass (fast path)
  - Support batched encoding for GPU efficiency, multi-GPU via device_map

Exp 2 (retriever model) is expressed here — the model_name passed in selects
the encoder used. All dense retrieval models go through this module.

Performance note
----------------
Use embed_queries_all_versions() instead of calling embed_queries() in a loop.
The all-versions function loads the model once and encodes every version's
texts in a single large batched call, which is 10–20× faster on GPU because
model loading dominates runtime when there are 24 versions × N models.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("article_retrieval")


def _load_st_model(model_name: str, device: str | None = None):
    """Load a sentence-transformers model (lazy import to avoid hard dependency)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for dense retrieval. "
            "Install with: pip install sentence-transformers"
        ) from e
    log.info("[embedder] loading model: %s  device=%s", model_name, device or "auto")
    kwargs = {}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


def _best_device() -> str:
    """Return 'cuda' if any GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def embed_articles(
    texts: list[str],
    article_ids: list[int],
    model_name: str,
    emb_path: Path,
    ids_path: Path,
    batch_size: int = 64,
    force: bool = False,
    device: str | None = None,
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

    model = _load_st_model(model_name, device or _best_device())
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
    device: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Encode query texts for one (model, query version) pair.

    If emb_path already exists and force=False, loads from disk (cache hit).
    Returns (embeddings [N × D], query_ids [N]).

    Prefer embed_queries_all_versions() when encoding multiple versions for the
    same model — it loads the model only once.
    """
    if emb_path.exists() and ids_path.exists() and not force:
        log.info("[embedder] cache hit — loading query embeddings from %s", emb_path)
        embeddings = np.load(str(emb_path))
        with open(ids_path, encoding="utf-8") as f:
            cached_ids = json.load(f)
        return embeddings, cached_ids

    model = _load_st_model(model_name, device or _best_device())
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


def embed_queries_all_versions(
    query_records: list[dict],
    versions: list[int],
    model_name: str,
    get_emb_path_fn,
    get_ids_path_fn,
    batch_size: int = 256,
    force: bool = False,
    device: str | None = None,
) -> dict[int, tuple[np.ndarray, list[str]]]:
    """
    Encode query texts for ALL versions of a single model in one model-load pass.

    This is the fast path: rather than loading the model 24 times (once per
    version), we load it once, encode every version's texts in a single
    batched call, then split the resulting embeddings back per version and
    cache each version to disk.

    Args:
        query_records:    Full list of query records (each has a "queries" dict).
        versions:         List of version integers to encode (e.g. list(range(1,25))).
        model_name:       Sentence-transformers model name.
        get_emb_path_fn:  Callable(version) → Path for embeddings .npy file.
        get_ids_path_fn:  Callable(version) → Path for ids .json file.
        batch_size:       Encoding batch size (use 256–512 on A100/RTX 6000).
        force:            Recompute even if cached files exist.
        device:           'cuda', 'cpu', or None (auto-detect).

    Returns:
        Dict mapping version → (embeddings [N×D], query_ids [N]).
        Only versions with at least one valid query text are included.
    """
    # ── 1. Determine which versions need computing ────────────────────────────
    versions_to_compute: list[int] = []
    cached: dict[int, tuple[np.ndarray, list[str]]] = {}

    for v in versions:
        ep = get_emb_path_fn(v)
        ip = get_ids_path_fn(v)
        if ep.exists() and ip.exists() and not force:
            log.info("[embedder] cache hit v%d — %s", v, ep)
            embs = np.load(str(ep))
            with open(ip, encoding="utf-8") as f:
                ids = json.load(f)
            cached[v] = (embs, ids)
        else:
            versions_to_compute.append(v)

    if not versions_to_compute:
        log.info("[embedder] all %d versions cached — skip encoding for %s", len(versions), model_name)
        return cached

    # ── 2. Build one big flat list of (version, query_id, text) ──────────────
    # We encode them all at once, then split by version using slice boundaries.
    flat_texts:  list[str] = []
    flat_ids:    list[str] = []
    flat_vers:   list[int] = []   # which version each row belongs to
    version_slices: dict[int, tuple[int, int]] = {}  # version → (start, end)

    for v in versions_to_compute:
        vkey = f"v{v}"
        start = len(flat_texts)
        for rec in query_records:
            text = rec.get("queries", {}).get(vkey, "")
            if text:
                flat_texts.append(text)
                flat_ids.append(rec["query_id"])
                flat_vers.append(v)
        end = len(flat_texts)
        if end > start:
            version_slices[v] = (start, end)
        else:
            log.warning("[embedder] no query texts for v%d — skipping", v)

    if not flat_texts:
        log.warning("[embedder] no query texts to encode for model=%s", model_name)
        return cached

    # ── 3. Encode everything in one pass ─────────────────────────────────────
    model = _load_st_model(model_name, device or _best_device())
    log.info(
        "[embedder] encoding %d query texts across %d versions "
        "(model=%s, batch_size=%d)",
        len(flat_texts), len(version_slices), model_name, batch_size,
    )
    all_embeddings = model.encode(
        flat_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    log.info("[embedder] encoding complete — splitting and caching per version")

    # ── 4. Split and cache per version ───────────────────────────────────────
    results: dict[int, tuple[np.ndarray, list[str]]] = dict(cached)
    for v, (start, end) in version_slices.items():
        embs = all_embeddings[start:end]
        ids  = flat_ids[start:end]

        ep = get_emb_path_fn(v)
        ip = get_ids_path_fn(v)
        ep.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(ep), embs)
        with open(ip, "w", encoding="utf-8") as f:
            json.dump(ids, f)
        log.info("[embedder] v%d saved → %s (%d vectors)", v, ep, len(ids))
        results[v] = (embs, ids)

    return results
