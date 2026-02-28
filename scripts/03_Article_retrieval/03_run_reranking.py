"""Step 3: Run zero-shot cross-encoder re-ranking.

For each (domain, retriever, query version):
  1. Load retrieval results (from step 02)
  2. Load article text lookup (from persisted article JSONL)
  3. Re-rank the top-K_input candidates using the cross-encoder
  4. Save re-ranked results to JSONL

Skip logic: if reranking JSONL already exists and --force not given, the run is skipped.

Run:
  python scripts/03_Article_retrieval/03_run_reranking.py
  python scripts/03_Article_retrieval/03_run_reranking.py --retriever bm25 --versions 1,2,3
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import article_retrieval.config_utils as cu
from article_retrieval.retriever import load_retrieval_results
from article_retrieval.reranker import rerank_all_versions, save_reranking_results, build_article_lookup
from article_retrieval.query_builder import load_query_dataset
from article_retrieval.logging_utils import setup_logger

log = logging.getLogger("article_retrieval")


def run_for_domain(
    config: dict,
    domain: str,
    retrievers: list[str],
    versions: list[int],
    force: bool,
) -> None:
    log_dir = cu.get_log_dir(config, domain)
    setup_logger(log_dir, script_name="03_run_reranking")

    reranking_cfg = config.get("reranking", {})
    if not reranking_cfg.get("enabled", True):
        log.info("[03] reranking disabled in config — skipping domain: %s", domain)
        return

    reranker_name = reranking_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_k_input   = reranking_cfg.get("top_k_input", 20)
    top_k         = config.get("retrieval", {}).get("top_k", 100)

    # Load article text lookup for cross-encoder scoring
    articles_path = cu.get_articles_jsonl_path(config, domain)
    if not articles_path.exists():
        log.error("Article JSONL not found: %s — run step 00 first.", articles_path)
        return
    article_lookup = build_article_lookup(articles_path)
    log.info("[03] loaded %d article texts for reranking", len(article_lookup))

    # Load query dataset for query text lookup
    query_path = cu.get_query_dataset_path(config, domain)
    if not query_path.exists():
        log.error("Query dataset not found: %s — run step 01 first.", query_path)
        return
    query_records = load_query_dataset(query_path)
    query_by_id   = {r["query_id"]: r for r in query_records}

    log.info(
        "[03] reranking %d retrievers × %d versions with model=%s "
        "(cross-encoder loaded ONCE for all jobs)",
        len(retrievers), len(versions), reranker_name,
    )

    # Load the cross-encoder once and process all (retriever × version) jobs
    rerank_all_versions(
        retrievers=retrievers,
        versions=versions,
        article_lookup=article_lookup,
        model_name=reranker_name,
        top_k_input=top_k_input,
        query_records_by_id=query_by_id,
        get_ret_path_fn=lambda r, v: cu.get_retrieval_path(config, domain, r, v, top_k),
        get_out_path_fn=lambda r, v: cu.get_reranking_path(config, domain, r, reranker_name, v),
        force=force,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-encoder re-ranking.")
    parser.add_argument("--config", default="configs/article_retrieval/article_retrieval.yaml")
    parser.add_argument("--domain", help="Single domain override.")
    parser.add_argument("--retriever", help="Comma-separated retriever names.")
    parser.add_argument("--versions", help="Comma-separated version numbers.")
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
