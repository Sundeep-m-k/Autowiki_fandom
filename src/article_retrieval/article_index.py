"""Article index builder for the article retrieval pipeline.

Responsibilities (I/O only — no metric computation):
  - Load and clean articles from processed JSONL
  - Build and persist BM25, TF-IDF, and FAISS indexes
  - Load indexes from disk for retrieval

Each index artifact encodes the corpus_representation and corpus_granularity
in its filename (via config_utils) so that multiple configurations coexist.

Exp 3 (corpus representation) and Exp 5 (corpus granularity) are implemented here.
Exp 10 (FAISS index type: flat/ivf/hnsw) is prepared but only flat is active;
ivf and hnsw are commented out pending corpora of 100k+ articles.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("article_retrieval")


# ── Article record ─────────────────────────────────────────────────────────────

@dataclass
class ArticleRecord:
    """One entry in the article index — one retrieval 'document'."""
    article_id: int
    title: str
    page_name: str
    text: str          # ready-to-encode text (not truncated — embedder handles that)


# ── Loading articles from processed JSONL ─────────────────────────────────────

def _build_article_text(
    record: dict,
    corpus_representation: str,
    max_chars: int,
) -> str:
    """
    Build the indexable text for one article according to corpus_representation.

    Exp 3 — corpus_representation options:
      title_only  : title string only
      title_lead  : title + first paragraph text
      title_full  : title + all paragraph texts concatenated (default)
    """
    title = record.get("title", record.get("page_name", "")).strip()

    if corpus_representation == "title_only":
        return title

    # Collect paragraph texts from links array structure
    links = record.get("links", [])
    paragraphs: list[str] = []

    # The article JSONL stores full plain text in article_plain_text
    full_text = record.get("article_plain_text", "").strip()

    if corpus_representation == "title_lead":
        # Take the first 500 characters of the article text as the lead
        lead = full_text[:500].strip()
        return f"{title}. {lead}"[:max_chars] if lead else title

    # title_full: title + full article plain text
    body = full_text[:max_chars].strip()
    return f"{title}. {body}" if body else title


def load_articles(
    jsonl_path: Path,
    corpus_representation: str = "title_full",
    corpus_granularity: str = "article",
    max_chars: int = 2000,
) -> list[ArticleRecord]:
    """
    Load and clean articles from articles_page_granularity_<domain>.jsonl.

    Exp 5 — corpus_granularity:
      article   : one ArticleRecord per article (default)
      paragraph : one ArticleRecord per paragraph (multiple per article)
      sentence  : one ArticleRecord per sentence  (multiple per article)

    For paragraph/sentence granularity, the article_id field still refers to the
    source article — deduplication at eval time is handled in evaluator.py.
    """
    records: list[ArticleRecord] = []
    seen_ids: set[int] = set()

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            article_id = rec.get("article_id")
            if article_id is None:
                continue
            article_id = int(article_id)
            title     = rec.get("title", rec.get("page_name", "")).strip()
            page_name = rec.get("page_name", "").strip()

            if corpus_granularity == "article":
                if article_id in seen_ids:
                    continue
                seen_ids.add(article_id)
                text = _build_article_text(rec, corpus_representation, max_chars)
                records.append(ArticleRecord(
                    article_id=article_id,
                    title=title,
                    page_name=page_name,
                    text=text,
                ))

            elif corpus_granularity == "paragraph":
                # Phase 2: one record per paragraph
                # Each paragraph is stored as a link context in the article JSONL.
                # For now we fall back to article-level and log a warning.
                # TODO (Phase 2): read paragraphs_<domain>.jsonl instead.
                if article_id not in seen_ids:
                    seen_ids.add(article_id)
                    text = _build_article_text(rec, corpus_representation, max_chars)
                    records.append(ArticleRecord(
                        article_id=article_id,
                        title=title,
                        page_name=page_name,
                        text=text,
                    ))
                    log.debug(
                        "corpus_granularity=paragraph not yet fully implemented; "
                        "using article-level text. Use paragraphs_<domain>.jsonl for full support."
                    )

            elif corpus_granularity == "sentence":
                # Phase 2: one record per sentence — requires sentences_<domain>.jsonl.
                # TODO (Phase 2): read sentences_<domain>.jsonl instead.
                if article_id not in seen_ids:
                    seen_ids.add(article_id)
                    text = _build_article_text(rec, corpus_representation, max_chars)
                    records.append(ArticleRecord(
                        article_id=article_id,
                        title=title,
                        page_name=page_name,
                        text=text,
                    ))
                    log.debug(
                        "corpus_granularity=sentence not yet fully implemented; "
                        "using article-level text. Use sentences_<domain>.jsonl for full support."
                    )

    log.info(
        "[article_index] loaded %d article records (repr=%s, gran=%s)",
        len(records), corpus_representation, corpus_granularity,
    )
    return records


def save_articles_jsonl(records: list[ArticleRecord], path: Path) -> None:
    """Persist clean article records to JSONL for later lookup during reranking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "article_id": r.article_id,
                "title": r.title,
                "page_name": r.page_name,
                "text": r.text,
            }, ensure_ascii=False) + "\n")
    log.info("[article_index] saved %d article records → %s", len(records), path)


def load_articles_jsonl(path: Path) -> list[ArticleRecord]:
    """Load persisted article records from JSONL."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(ArticleRecord(
                article_id=int(d["article_id"]),
                title=d["title"],
                page_name=d["page_name"],
                text=d["text"],
            ))
    return records


# ── BM25 index ────────────────────────────────────────────────────────────────

def build_bm25_index(records: list[ArticleRecord], preprocessing: str = "raw"):
    """
    Build a BM25Okapi index from article texts.

    Exp 11 — preprocessing applied to article text before tokenising:
      raw              : split on whitespace (default)
      lowercase        : lowercase then split
      stopword_removed : lowercase + remove English stopwords then split
    """
    from rank_bm25 import BM25Okapi
    tokenised = [_tokenise(r.text, preprocessing) for r in records]
    index = BM25Okapi(tokenised)
    log.info("[article_index] BM25 index built (%d documents)", len(records))
    return index


def save_bm25_index(index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)
    log.info("[article_index] BM25 index saved → %s", path)


def load_bm25_index(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── TF-IDF index ──────────────────────────────────────────────────────────────

def build_tfidf_index(records: list[ArticleRecord], preprocessing: str = "raw"):
    """
    Build a TF-IDF vectorizer + document matrix from article texts.
    Returns (vectorizer, matrix) tuple.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [_preprocess_text(r.text, preprocessing) for r in records]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    log.info(
        "[article_index] TF-IDF index built (%d docs, %d features)",
        matrix.shape[0], matrix.shape[1],
    )
    return vectorizer, matrix


def save_tfidf_index(vectorizer, matrix, vec_path: Path, mat_path: Path) -> None:
    import scipy.sparse
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    scipy.sparse.save_npz(str(mat_path), matrix)
    log.info("[article_index] TF-IDF index saved → %s + %s", vec_path, mat_path)


def load_tfidf_index(vec_path: Path, mat_path: Path):
    import scipy.sparse
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    matrix = scipy.sparse.load_npz(str(mat_path))
    return vectorizer, matrix


# ── FAISS index ───────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray, index_type: str = "flat"):
    """
    Build a FAISS index from article embeddings.

    Exp 10 — index_type:
      flat : exact L2 search — correct for Fandom wiki corpus sizes (default)
      ivf  : approximate, faster on large corpora (Future — 100k+ articles)
      hnsw : graph-based approximate search (Future — 100k+ articles)
    """
    import faiss
    dim = embeddings.shape[1]

    # L2-normalise for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = (embeddings / norms).astype(np.float32)

    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
        index.add(normed)

    elif index_type == "ivf":
        # Future (Exp 10): only worthwhile for corpora of 100k+ articles.
        # nlist = max(4, int(np.sqrt(len(embeddings))))
        # quantizer = faiss.IndexFlatIP(dim)
        # index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        # index.train(normed)
        # index.add(normed)
        raise NotImplementedError(
            "IVF index not yet active. Set faiss_index_type: flat in config. "
            "IVF is only beneficial for corpora of 100k+ articles."
        )

    elif index_type == "hnsw":
        # Future (Exp 10): graph-based approximate search.
        # index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        # index.add(normed)
        raise NotImplementedError(
            "HNSW index not yet active. Set faiss_index_type: flat in config."
        )

    else:
        raise ValueError(f"Unknown faiss_index_type: {index_type!r}")

    log.info(
        "[article_index] FAISS index built (type=%s, dim=%d, n=%d)",
        index_type, dim, index.ntotal,
    )
    return index, normed


def save_faiss_index(index, path: Path, meta: dict, meta_path: Path) -> None:
    import faiss
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("[article_index] FAISS index saved → %s", path)


def load_faiss_index(path: Path):
    import faiss
    return faiss.read_index(str(path))


def save_embeddings(
    embeddings: np.ndarray,
    article_ids: list[int],
    emb_path: Path,
    ids_path: Path,
) -> None:
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(emb_path), embeddings)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(article_ids, f)
    log.info("[article_index] embeddings saved → %s (%d vectors)", emb_path, len(article_ids))


def load_embeddings(emb_path: Path, ids_path: Path) -> tuple[np.ndarray, list[int]]:
    embeddings = np.load(str(emb_path))
    with open(ids_path, encoding="utf-8") as f:
        article_ids = json.load(f)
    return embeddings, article_ids


# ── Index metadata ─────────────────────────────────────────────────────────────

def save_index_meta(path: Path, domain: str, n_articles: int, config: dict) -> None:
    ai = config.get("article_index", {})
    meta = {
        "domain": domain,
        "n_articles": n_articles,
        "corpus_representation": ai.get("corpus_representation", "title_full"),
        "corpus_granularity": ai.get("corpus_granularity", "article"),
        "faiss_index_type": config.get("faiss_index_type", "flat"),
        "created_at": datetime.now().isoformat(),
        "dense_models": config.get("retrievers", {}).get("dense", []),
        "sparse_models": config.get("retrievers", {}).get("sparse", []),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ── Text preprocessing helpers ────────────────────────────────────────────────

_STOPWORDS: set[str] | None = None


def _get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            _STOPWORDS = set(ENGLISH_STOP_WORDS)
        except ImportError:
            _STOPWORDS = set()
    return _STOPWORDS


def _preprocess_text(text: str, preprocessing: str) -> str:
    """
    Apply text preprocessing for sparse retrievers.
    Exp 11 — preprocessing:
      raw              : no change (default; dense models normalise internally)
      lowercase        : lowercase only
      stopword_removed : lowercase + remove common English stopwords
    """
    if preprocessing == "raw":
        return text
    text = text.lower()
    if preprocessing == "stopword_removed":
        stopwords = _get_stopwords()
        tokens = [w for w in text.split() if w not in stopwords]
        return " ".join(tokens)
    return text


def _tokenise(text: str, preprocessing: str) -> list[str]:
    """Tokenise preprocessed text for BM25."""
    return _preprocess_text(text, preprocessing).split()
