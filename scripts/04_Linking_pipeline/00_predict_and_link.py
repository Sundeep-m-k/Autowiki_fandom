"""Step 0: Link spans using pre-computed Task 2 retrieval results.

For each article in the Task 1 test split:
  1. Load gold spans (anchor_text, char_start, char_end, gold_article_id)
  2. For each span, look up the top-1 retrieved article from Task 2 results
     using key (source_article_id, char_start, char_end) — falls back to
     (source_article_id, anchor_text) for legacy query datasets
  3. Apply NIL threshold — set linked=False if score < threshold
  4. Look up page_name from article index for URL construction
  5. Save linking results JSONL (one record per article)

No model inference is performed — everything is a fast disk lookup.

Run:
  python scripts/04_Linking_pipeline/00_predict_and_link.py
  python scripts/04_Linking_pipeline/00_predict_and_link.py --domain beverlyhillscop
  python scripts/04_Linking_pipeline/00_predict_and_link.py --force
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import linking_pipeline.config_utils as cu
from linking_pipeline.logging_utils import setup_logger
from linking_pipeline.span_predictor import load_gold_spans
from linking_pipeline.span_to_query import build_lookup, lookup_span
from linking_pipeline.nil_detector import apply_nil_filter

log = logging.getLogger("linking")


def _build_article_page_lookup(articles_jsonl_path: Path) -> dict[int, dict]:
    """Build article_id → {page_name, title} lookup from Task 2 article index."""
    lookup: dict[int, dict] = {}
    if not articles_jsonl_path.exists():
        log.warning("[00] article index JSONL not found: %s", articles_jsonl_path)
        return lookup
    with open(articles_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            aid = rec.get("article_id")
            if aid is not None:
                lookup[int(aid)] = {
                    "page_name": rec.get("page_name", ""),
                    "title":     rec.get("title", ""),
                }
    return lookup


def run_for_domain(config: dict, domain: str, force: bool) -> None:
    log_dir = cu.get_log_dir(config, domain)
    setup_logger(log_dir, script_name="00_predict_and_link")

    out_path = cu.get_linking_results_path(config, domain)
    if out_path.exists() and not force:
        log.info("[00] linking results exist — skip (use --force to rebuild): %s", out_path)
        return

    # ── Load Task 1 gold spans ─────────────────────────────────────────────────
    t1_cfg       = config.get("task1", {})
    split_path   = cu.get_task1_split_path(config, domain)
    articles     = load_gold_spans(split_path, granularity=t1_cfg.get("granularity", "article"))
    if not articles:
        log.error("[00] no articles loaded — aborting")
        return

    # ── Load Task 2 lookup ─────────────────────────────────────────────────────
    query_path = cu.get_task2_query_dataset_path(config, domain)
    t2_cfg     = config.get("task2", {})
    stage      = t2_cfg.get("stage", "reranking")

    if stage == "reranking":
        results_path = cu.get_task2_reranking_path(config, domain)
    else:
        results_path = cu.get_task2_retrieval_path(config, domain)

    lookup = build_lookup(query_path, results_path)
    if not lookup:
        log.error("[00] empty lookup — check Task 2 results exist for this config")
        return

    # ── Article page_name lookup (for URL construction) ────────────────────────
    articles_jsonl = cu.get_task2_articles_jsonl_path(config, domain)
    page_lookup    = _build_article_page_lookup(articles_jsonl)
    fandom_base    = cu.get_fandom_base_url(config, domain)
    nil_threshold  = config.get("nil_detection", {}).get("threshold", 0.0)

    # ── Link each article ──────────────────────────────────────────────────────
    results: list[dict] = []
    n_spans_total    = 0
    n_linked         = 0
    n_not_found      = 0

    for article in articles:
        article_id = article["article_id"]
        predicted_links: list[dict] = []

        for span in article["gold_spans"]:
            n_spans_total += 1
            hit = lookup_span(
                lookup,
                article_id,
                span["anchor_text"],
                char_start=span.get("char_start"),
                char_end=span.get("char_end"),
            )

            if hit is None:
                # No Task 2 result for this (article_id, anchor) pair
                n_not_found += 1
                predicted_links.append({
                    "char_start":      span["char_start"],
                    "char_end":        span["char_end"],
                    "anchor_text":     span["anchor_text"],
                    "gold_article_id": span["gold_article_id"],
                    "article_id":      None,
                    "page_name":       None,
                    "fandom_url":      None,
                    "retrieval_score": 0.0,
                    "linked":          False,
                })
                continue

            pred_aid   = hit["article_id"]
            page_info  = page_lookup.get(pred_aid, {})
            page_name  = page_info.get("page_name", "")
            fandom_url = f"{fandom_base}/{page_name.replace(' ', '_')}" if page_name else None

            predicted_links.append({
                "char_start":      span["char_start"],
                "char_end":        span["char_end"],
                "anchor_text":     span["anchor_text"],
                "gold_article_id": span["gold_article_id"],
                "article_id":      pred_aid,
                "page_name":       page_name,
                "fandom_url":      fandom_url,
                "retrieval_score": hit["score"],
                "linked":          True,   # will be updated by nil filter below
            })

        # Apply NIL filter
        apply_nil_filter(predicted_links, nil_threshold)
        n_linked += sum(1 for p in predicted_links if p.get("linked"))

        results.append({
            "article_id":      article_id,
            "text":            article["text"],
            "page_name":       article["page_name"],
            "gold_spans":      article["gold_spans"],
            "predicted_links": predicted_links,
        })

    log.info(
        "[00] %d articles | %d spans | %d linked | %d not found in Task 2 | "
        "%d NIL (threshold=%.2f)",
        len(results), n_spans_total, n_linked, n_not_found,
        n_spans_total - n_linked - n_not_found, nil_threshold,
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("[00] linking results saved → %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Link spans using Task 2 results.")
    parser.add_argument("--config", default="configs/linking/linking.yaml")
    parser.add_argument("--domain", help="Single domain override.")
    parser.add_argument("--force", action="store_true", help="Recompute even if cached.")
    args = parser.parse_args()

    config  = cu.load_config(ROOT / args.config)
    domains = [args.domain] if args.domain else config.get("domains", [])

    for domain in domains:
        run_for_domain(config, domain, force=args.force)


if __name__ == "__main__":
    main()
