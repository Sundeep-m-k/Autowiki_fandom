"""Step 1: Build query dataset.

This script:
  1. Loads internal links from the articles JSONL
  2. Filters to links whose source article is in the Task 1 test split
  3. Samples up to n_sample queries (stratified by source article)
  4. Generates all 24 query variation texts per link
  5. Saves the query dataset to JSONL

Run:
  python scripts/03_Article_retrieval/01_build_query_dataset.py
  python scripts/03_Article_retrieval/01_build_query_dataset.py --domain money-heist --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import article_retrieval.config_utils as cu
from article_retrieval.query_builder import build_query_dataset
from article_retrieval.logging_utils import setup_logger


def build_for_domain(config: dict, domain: str, force: bool) -> None:
    log_dir = cu.get_log_dir(config, domain)
    log     = setup_logger(log_dir, script_name="01_build_query_dataset")

    output_path = cu.get_query_dataset_path(config, domain)
    if output_path.exists() and not force:
        log.info("[01] query dataset exists — skip (use --force to rebuild): %s", output_path)
        return

    raw_path = cu.get_articles_page_granularity_path(config, domain)
    if not raw_path.exists():
        log.error("Articles JSONL not found: %s — run data scraping first.", raw_path)
        return

    split = config.get("queries", {}).get("split", "test")
    # Temporarily update domain in config for reuse of Task 1 split helper
    test_ids = cu.get_task1_split_article_ids(config, domain, split)
    if test_ids:
        log.info("[01] restricting queries to %d test-split source articles", len(test_ids))
    else:
        log.warning(
            "[01] No Task 1 split found for domain=%s split=%s — using all articles",
            domain, split,
        )

    build_query_dataset(raw_path, config, test_ids, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build query dataset for retrieval.")
    parser.add_argument("--config", default="configs/article_retrieval/article_retrieval.yaml")
    parser.add_argument("--domain", help="Override domain list (single domain).")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached.")
    args = parser.parse_args()

    config  = cu.resolve_config(cu.load_config(ROOT / args.config))
    domains = [args.domain] if args.domain else config.get("domains", [])

    for domain in domains:
        build_for_domain(config, domain, force=args.force)


if __name__ == "__main__":
    main()
