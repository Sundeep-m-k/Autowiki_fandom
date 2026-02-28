"""Step 2: Run retrieval.

For each (domain, retriever, query version):
  1. Load the appropriate index (BM25 / TF-IDF / FAISS)
  2. Retrieve top-K articles for every query in the query dataset
  3. Save retrieval results to JSONL

Supports parallel execution across query versions via a multiprocessing pool.
Skip logic: if output JSONL already exists and --force not given, the run is skipped.

Run:
  python scripts/03_Article_retrieval/02_run_retrieval.py
  python scripts/03_Article_retrieval/02_run_retrieval.py --domain money-heist --retriever bm25
  python scripts/03_Article_retrieval/02_run_retrieval.py --versions 1,2,3 --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import article_retrieval.config_utils as cu
import article_retrieval.article_index as ai_mod
import article_retrieval.retriever as ret
from article_retrieval.query_builder import load_query_dataset
from article_retrieval.logging_utils import setup_logger

log = logging.getLogger("article_retrieval")


def run_sparse_retriever(
    name: str,
    config: dict,
    domain: str,
    query_records: list[dict],
    article_ids: list[int],
    versions: list[int],
    top_k: int,
    force: bool,
) -> None:
    """Run BM25 or TF-IDF for all requested query versions."""
    q_cfg   = config.get("queries", {})
    preproc = q_cfg.get("anchor_preprocessing", "raw")

    if name == "bm25":
        bm25_path = cu.get_bm25_index_path(config, domain)
        if not bm25_path.exists():
            log.error("BM25 index not found: %s — run step 00 first.", bm25_path)
            return
        index = ai_mod.load_bm25_index(bm25_path)
    else:
        tfidf_path = cu.get_tfidf_index_path(config, domain)
        tfidf_mat  = cu.get_tfidf_matrix_path(config, domain)
        if not tfidf_path.exists():
            log.error("TF-IDF index not found: %s — run step 00 first.", tfidf_path)
            return
        vectorizer, matrix = ai_mod.load_tfidf_index(tfidf_path, tfidf_mat)

    for version in versions:
        out_path = cu.get_retrieval_path(config, domain, name, version, top_k)
        if out_path.exists() and not force:
            log.info("[02] skip (cached): %s", out_path)
            continue

        if name == "bm25":
            results = ret.retrieve_bm25(
                index, article_ids, query_records, version, top_k, preprocessing=preproc,
            )
        else:
            results = ret.retrieve_tfidf(
                vectorizer, matrix, article_ids, query_records, version, top_k,
                preprocessing=preproc,
            )

        ret.save_retrieval_results(results, out_path)


def run_dense_retriever(
    model_name: str,
    config: dict,
    domain: str,
    query_records: list[dict],
    article_ids: list[int],
    versions: list[int],
    top_k: int,
    force: bool,
) -> None:
    """Run dense FAISS retrieval for all requested query versions.

    Uses embed_queries_all_versions() to load the model once and encode
    every version's queries in a single batched call — much faster than
    loading the model once per version.
    """
    from article_retrieval.embedder import embed_queries_all_versions

    faiss_path = cu.get_faiss_index_path(config, domain, model_name)
    if not faiss_path.exists():
        log.error("FAISS index not found: %s — run step 00 first.", faiss_path)
        return
    faiss_index = ai_mod.load_faiss_index(faiss_path)

    batch_size = config.get("parallel", {}).get("embedding_batch_size", 256)

    # Encode all versions in one model-load pass
    version_embeddings = embed_queries_all_versions(
        query_records=query_records,
        versions=versions,
        model_name=model_name,
        get_emb_path_fn=lambda v: cu.get_query_embeddings_path(config, domain, model_name, v),
        get_ids_path_fn=lambda v: cu.get_query_embeddings_ids_path(config, domain, model_name, v),
        batch_size=batch_size,
        force=force,
    )

    # Retrieve for each version using the cached embeddings
    for version in versions:
        out_path = cu.get_retrieval_path(config, domain, model_name, version, top_k)
        if out_path.exists() and not force:
            log.info("[02] skip (cached): %s", out_path)
            continue

        if version not in version_embeddings:
            log.warning("[02] no embeddings for v%d (no query texts?) — skip", version)
            continue

        query_embeddings, cached_qids = version_embeddings[version]
        version_key = f"v{version}"
        filtered = [r for r in query_records if r.get("queries", {}).get(version_key)]

        if not filtered:
            log.warning("[02] no query records for v%d — skip", version)
            continue

        results = ret.retrieve_dense(
            faiss_index, article_ids, query_embeddings, filtered, version, top_k, model_name,
        )
        ret.save_retrieval_results(results, out_path)


def run_for_domain(config: dict, domain: str, retrievers: list[str], versions: list[int], force: bool) -> None:
    log_dir = cu.get_log_dir(config, domain)
    setup_logger(log_dir, script_name="02_run_retrieval")

    # Load query dataset
    query_path = cu.get_query_dataset_path(config, domain)
    if not query_path.exists():
        log.error("Query dataset not found: %s — run step 01 first.", query_path)
        return
    query_records = load_query_dataset(query_path)
    log.info("[02] loaded %d query records", len(query_records))

    # Load article IDs from persisted article JSONL
    articles_path = cu.get_articles_jsonl_path(config, domain)
    if not articles_path.exists():
        log.error("Article JSONL not found: %s — run step 00 first.", articles_path)
        return
    from article_retrieval.article_index import load_articles_jsonl
    article_records = load_articles_jsonl(articles_path)
    article_ids = [r.article_id for r in article_records]
    log.info("[02] %d articles in corpus", len(article_ids))

    top_k = config.get("retrieval", {}).get("top_k", 100)
    sparse_models = config.get("retrievers", {}).get("sparse", [])
    dense_models  = config.get("retrievers", {}).get("dense", [])

    for name in retrievers:
        if name in sparse_models:
            run_sparse_retriever(name, config, domain, query_records, article_ids, versions, top_k, force)
        elif name in dense_models:
            run_dense_retriever(name, config, domain, query_records, article_ids, versions, top_k, force)
        else:
            log.warning("[02] retriever '%s' not in config — skipping", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run article retrieval.")
    parser.add_argument("--config", default="configs/article_retrieval/article_retrieval.yaml")
    parser.add_argument("--domain", help="Single domain override.")
    parser.add_argument(
        "--retriever",
        help="Comma-separated retriever names. Default: all from config.",
    )
    parser.add_argument(
        "--versions",
        help="Comma-separated version numbers. Default: all from config.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config  = cu.resolve_config(cu.load_config(ROOT / args.config))
    domains = [args.domain] if args.domain else config.get("domains", [])

    cfg_sparse = config.get("retrievers", {}).get("sparse", [])
    cfg_dense  = config.get("retrievers", {}).get("dense", [])
    all_retrievers = cfg_sparse + cfg_dense
    retrievers = (
        [r.strip() for r in args.retriever.split(",")] if args.retriever else all_retrievers
    )

    cfg_versions = config.get("queries", {}).get("versions", list(range(1, 25)))
    versions = (
        [int(v.strip()) for v in args.versions.split(",")] if args.versions else cfg_versions
    )

    for domain in domains:
        run_for_domain(config, domain, retrievers, versions, force=args.force)


if __name__ == "__main__":
    main()
