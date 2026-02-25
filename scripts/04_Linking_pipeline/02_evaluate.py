"""Step 2: Evaluate linking results.

Reads linking_results.jsonl and computes:
  - Linking F1   (span boundary correct AND article_id correct)
  - Span F1      (boundary only)
  - Entity Accuracy (article correct given correct span)
  - NIL Rate     (fraction of spans assigned NIL)
  - Coverage     (fraction of spans with a Task 2 lookup hit)

Saves per-run metrics JSON and appends a row to the research CSV.

Run:
  python scripts/04_Linking_pipeline/02_evaluate.py
  python scripts/04_Linking_pipeline/02_evaluate.py --domain beverlyhillscop
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
from linking_pipeline.evaluator import (
    evaluate_article,
    aggregate_metrics,
    save_metrics_json,
    append_to_research_csv,
)

log = logging.getLogger("linking")


def run_for_domain(config: dict, domain: str, force: bool) -> dict:
    log_dir = cu.get_log_dir(config, domain)
    setup_logger(log_dir, script_name="02_evaluate")

    results_path = cu.get_linking_results_path(config, domain)
    if not results_path.exists():
        log.error("[02] linking results not found: %s — run step 00 first", results_path)
        return {}

    metrics_path = cu.get_metrics_path(config, domain)
    if metrics_path.exists() and not force:
        log.info("[02] metrics exist — skip (use --force to recompute): %s", metrics_path)
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)

    article_metrics: list[dict] = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            article_metrics.append(evaluate_article(rec))

    agg = aggregate_metrics(article_metrics)

    log.info(
        "[02] domain=%s | Linking F1=%.3f | Span F1=%.3f | "
        "Entity Acc=%.3f | NIL Rate=%.3f | Coverage=%.3f | n=%d articles",
        domain,
        agg.get("linking_f1", 0),
        agg.get("span_f1", 0),
        agg.get("entity_accuracy", 0),
        agg.get("nil_rate", 0),
        agg.get("coverage", 0),
        agg.get("n_articles", 0),
    )

    save_metrics_json(agg, metrics_path)

    csv_path = cu.get_research_csv_path(config)
    append_to_research_csv(csv_path, domain, agg, config)

    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate linking results.")
    parser.add_argument("--config", default="configs/linking.yaml")
    parser.add_argument("--domain", help="Single domain override.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config  = cu.load_config(ROOT / args.config)
    domains = [args.domain] if args.domain else config.get("domains", [])

    for domain in domains:
        run_for_domain(config, domain, force=args.force)


if __name__ == "__main__":
    main()
