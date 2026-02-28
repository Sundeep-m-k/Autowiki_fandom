"""Step 4: Fine-tune a cross-encoder re-ranker on retrieval results.

Training data is mined from step 02 (retrieval) outputs:
  - Positive: (query_text, gold_article_text)
  - Hard negatives: top retrieved articles that are NOT the gold article,
    drawn from the configured source retriever.

Only training-split queries are used (source articles in the Task 1 train split)
so the test set is never touched during training.

After fine-tuning, add the checkpoint path to reranking.models in your config
to include it as an additional Exp 6 re-ranker variant.

Skip logic: if the fine-tuned model directory already exists and --force is not
given, the step is skipped.

Run:
  python scripts/03_Article_retrieval/04_train_reranker.py
  python scripts/03_Article_retrieval/04_train_reranker.py --domain money-heist
  python scripts/03_Article_retrieval/04_train_reranker.py --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import article_retrieval.config_utils as cu
from article_retrieval.reranker import build_article_lookup
from article_retrieval.query_builder import load_query_dataset
from article_retrieval.reranker_trainer import (
    build_training_examples,
    load_training_examples,
    save_training_examples,
    train_reranker,
)
from article_retrieval.logging_utils import setup_logger

log = logging.getLogger("article_retrieval")


def train_for_domain(config: dict, domain: str, force: bool) -> None:
    rt_cfg = config.get("reranker_training", {})
    if not rt_cfg.get("enabled", False):
        log.info("[04_train] reranker_training.enabled=false — skipping domain: %s", domain)
        return

    log_dir = cu.get_log_dir(config, domain)
    setup_logger(log_dir, script_name="04_train_reranker")

    source_retriever  = rt_cfg["source_retriever"]
    source_version    = rt_cfg.get("source_version", 6)
    n_hard_negatives  = rt_cfg.get("n_hard_negatives", 5)
    base_model        = rt_cfg.get("base_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    output_dir        = cu.get_reranker_checkpoint_dir(config)
    train_split       = rt_cfg.get("split", "train")

    # ── Skip if already trained ────────────────────────────────────────────────
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        log.info("[04_train] checkpoint exists — skip (use --force to retrain): %s", output_dir)
        return

    # ── Load prerequisite artifacts ────────────────────────────────────────────
    articles_path = cu.get_articles_jsonl_path(config, domain)
    if not articles_path.exists():
        log.error("[04_train] article JSONL not found: %s — run step 00 first.", articles_path)
        return

    retrieval_path = cu.get_retrieval_path(
        config, domain, source_retriever, source_version,
        config.get("retrieval", {}).get("top_k", 100),
    )
    if not retrieval_path.exists():
        log.error(
            "[04_train] retrieval results not found: %s — run step 02 first.", retrieval_path,
        )
        return

    query_path = cu.get_query_dataset_path(config, domain)
    if not query_path.exists():
        log.error("[04_train] query dataset not found: %s — run step 01 first.", query_path)
        return

    # ── Load train-split article IDs ───────────────────────────────────────────
    train_source_ids = cu.get_task1_split_article_ids(config, domain, train_split)
    if not train_source_ids:
        log.warning(
            "[04_train] No Task 1 '%s' split found for domain=%s — "
            "using ALL queries (test leakage risk!).",
            train_split, domain,
        )

    # ── Build or load training examples ───────────────────────────────────────
    training_data_path = cu.get_reranker_training_data_path(config, domain)

    if training_data_path.exists() and not force:
        log.info("[04_train] loading cached training examples: %s", training_data_path)
        examples = load_training_examples(training_data_path)
    else:
        log.info("[04_train] building training examples from retrieval results...")
        article_lookup      = build_article_lookup(articles_path)
        query_records       = load_query_dataset(query_path)
        query_records_by_id = {r["query_id"]: r for r in query_records}

        examples = build_training_examples(
            retrieval_path=retrieval_path,
            query_records_by_id=query_records_by_id,
            article_lookup=article_lookup,
            version=source_version,
            n_hard_negatives=n_hard_negatives,
            train_source_ids=train_source_ids,
        )
        if not examples:
            log.error("[04_train] no training examples built — check retrieval results and splits.")
            return
        save_training_examples(examples, training_data_path)

    log.info("[04_train] %d training triples ready", len(examples))

    # ── Fine-tune ──────────────────────────────────────────────────────────────
    train_reranker(
        examples=examples,
        base_model=base_model,
        output_dir=output_dir,
        epochs=rt_cfg.get("epochs", 3),
        batch_size=rt_cfg.get("batch_size", 16),
        learning_rate=rt_cfg.get("learning_rate", 2e-5),
        warmup_ratio=rt_cfg.get("warmup_ratio", 0.1),
        max_length=rt_cfg.get("max_length", 512),
    )

    log.info(
        "[04_train] done — fine-tuned reranker saved to %s\n"
        "  To use it, add the path to reranking.models in your config:\n"
        "    - \"%s\"",
        output_dir, output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a cross-encoder re-ranker.")
    parser.add_argument("--config", default="configs/article_retrieval/article_retrieval.yaml")
    parser.add_argument("--domain", help="Single domain override.")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if checkpoint already exists.")
    args = parser.parse_args()

    config  = cu.resolve_config(cu.load_config(ROOT / args.config))
    domains = [args.domain] if args.domain else config.get("domains", [])

    for domain in domains:
        train_for_domain(config, domain, force=args.force)


if __name__ == "__main__":
    main()
