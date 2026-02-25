"""Step 0: Build article index.

This script:
  1. Loads articles from articles_page_granularity_<domain>.jsonl
  2. Builds and saves BM25 and TF-IDF indexes (sparse baselines)
  3. Encodes articles with each dense model and builds FAISS indexes
  4. Saves persisted article records (for reranker text lookup)

All artifacts are named to encode the active experiment dimensions via config_utils.

Run:
  python scripts/03_Article_retrieval/00_build_article_index.py
  python scripts/03_Article_retrieval/00_build_article_index.py --config configs/article_retrieval/kudremukh.yaml
  python scripts/03_Article_retrieval/00_build_article_index.py --domain money-heist --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import article_retrieval.config_utils as cu
import article_retrieval.article_index as ai
from article_retrieval.logging_utils import setup_logger


def build_for_domain(config: dict, domain: str, force: bool) -> None:
    log_dir = cu.get_log_dir(config, domain)
    log     = setup_logger(log_dir, script_name="00_build_article_index")

    # ── Load raw articles ──────────────────────────────────────────────────────
    raw_path = cu.get_articles_page_granularity_path(config, domain)
    if not raw_path.exists():
        log.error("Articles JSONL not found: %s — run data scraping first.", raw_path)
        return

    ai_cfg = config.get("article_index", {})
    corpus_repr = ai_cfg.get("corpus_representation", "title_full")
    corpus_gran = ai_cfg.get("corpus_granularity", "article")
    max_chars   = ai_cfg.get("max_chars", 2000)

    records = ai.load_articles(raw_path, corpus_repr, corpus_gran, max_chars)
    if not records:
        log.error("No article records loaded from %s", raw_path)
        return

    article_ids = [r.article_id for r in records]
    texts       = [r.text for r in records]

    # ── Save persisted article JSONL (for reranker lookup) ────────────────────
    articles_out = cu.get_articles_jsonl_path(config, domain)
    if not articles_out.exists() or force:
        ai.save_articles_jsonl(records, articles_out)

    # ── BM25 ──────────────────────────────────────────────────────────────────
    q_cfg   = config.get("queries", {})
    preproc = q_cfg.get("anchor_preprocessing", "raw")

    bm25_path = cu.get_bm25_index_path(config, domain)
    if not bm25_path.exists() or force:
        bm25_index = ai.build_bm25_index(records, preprocessing=preproc)
        ai.save_bm25_index(bm25_index, bm25_path)
    else:
        log.info("[00] BM25 index exists — skip (use --force to rebuild): %s", bm25_path)

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    tfidf_path = cu.get_tfidf_index_path(config, domain)
    tfidf_mat  = cu.get_tfidf_matrix_path(config, domain)
    if not tfidf_path.exists() or force:
        vectorizer, matrix = ai.build_tfidf_index(records, preprocessing=preproc)
        ai.save_tfidf_index(vectorizer, matrix, tfidf_path, tfidf_mat)
    else:
        log.info("[00] TF-IDF index exists — skip (use --force to rebuild): %s", tfidf_path)

    # ── Dense embedding + FAISS (one per model) ───────────────────────────────
    retrievers    = config.get("retrievers", {})
    dense_models  = retrievers.get("dense", [])
    batch_size    = config.get("parallel", {}).get("embedding_batch_size", 64)
    index_type    = config.get("faiss_index_type", "flat")

    for model_name in dense_models:
        emb_path  = cu.get_embeddings_path(config, domain, model_name)
        ids_path  = cu.get_embeddings_ids_path(config, domain, model_name)
        fais_path = cu.get_faiss_index_path(config, domain, model_name)
        fais_meta = cu.get_faiss_meta_path(config, domain, model_name)

        if not emb_path.exists() or force:
            from article_retrieval.embedder import embed_articles
            embeddings, _ = embed_articles(
                texts, article_ids, model_name, emb_path, ids_path,
                batch_size=batch_size, force=force,
            )
        else:
            log.info("[00] embeddings exist — skip: %s", emb_path)
            embeddings, _ = ai.load_embeddings(emb_path, ids_path)

        if not fais_path.exists() or force:
            import numpy as np
            index, _ = ai.build_faiss_index(embeddings, index_type=index_type)
            meta = {
                "model": model_name,
                "n_articles": len(article_ids),
                "index_type": index_type,
            }
            ai.save_faiss_index(index, fais_path, meta, fais_meta)
        else:
            log.info("[00] FAISS index exists — skip: %s", fais_path)

    # ── Index metadata ─────────────────────────────────────────────────────────
    meta_path = cu.get_index_meta_path(config, domain)
    ai.save_index_meta(meta_path, domain, len(records), config)
    log.info("[00] Article index build complete for domain: %s", domain)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build article indexes for retrieval.")
    parser.add_argument("--config", default="configs/article_retrieval/article_retrieval.yaml")
    parser.add_argument("--domain", help="Override domain list (single domain).")
    parser.add_argument("--force", action="store_true", help="Rebuild all indexes even if cached.")
    args = parser.parse_args()

    config  = cu.load_config(ROOT / args.config)
    domains = [args.domain] if args.domain else config.get("domains", [])

    for domain in domains:
        build_for_domain(config, domain, force=args.force)


if __name__ == "__main__":
    main()
