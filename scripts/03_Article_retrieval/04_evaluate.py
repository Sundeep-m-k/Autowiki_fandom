"""Step 4: Evaluate retrieval and reranking results.

For each (domain, retriever, query version) combination:
  - Computes Recall@1,3,5,10,20,50,100 and MRR
  - Saves per-version metrics JSON
  - Appends a row to the research CSV

Also processes reranking results (stage="reranking") using the same metrics.

Run:
  python scripts/03_Article_retrieval/04_evaluate.py
  python scripts/03_Article_retrieval/04_evaluate.py --domain money-heist
  python scripts/03_Article_retrieval/04_evaluate.py --stage retrieval  # skip reranking
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
from article_retrieval.reranker import load_reranking_results
from article_retrieval.evaluator import compute_metrics, save_metrics_json, append_to_research_csv
from article_retrieval.logging_utils import setup_logger

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from src.utils.stats_utils import update_article_retrieval_stats

log = logging.getLogger("article_retrieval")


def get_n_articles(config: dict, domain: str) -> int:
    """Read number of articles from index metadata, or 0 if not available."""
    import json
    meta_path = cu.get_index_meta_path(config, domain)
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f).get("n_articles", 0)
    return 0


def evaluate_for_domain(
    config: dict,
    domain: str,
    retrievers: list[str],
    versions: list[int],
    stages: list[str],
    force: bool,
) -> list[dict]:
    """Evaluate all (retriever, version, stage) combinations. Returns summary rows."""
    log_dir = cu.get_log_dir(config, domain)
    setup_logger(log_dir, script_name="04_evaluate")

    recall_at_k  = config.get("evaluation", {}).get("recall_at_k", [1, 3, 5, 10, 20, 50, 100])
    csv_path     = cu.get_research_csv_path(config)
    n_articles   = get_n_articles(config, domain)
    reranker_name = config.get("reranking", {}).get("model", "")
    top_k         = config.get("retrieval", {}).get("top_k", 100)
    summary_rows: list[dict] = []

    for retriever in retrievers:
        for version in versions:
            if "retrieval" in stages:
                ret_path = cu.get_retrieval_path(config, domain, retriever, version, top_k)
                if not ret_path.exists():
                    log.warning("[04] retrieval results not found: %s", ret_path)
                else:
                    met_path = cu.get_metrics_path(config, domain, retriever, version, "retrieval")
                    if not met_path.exists() or force:
                        results  = load_retrieval_results(ret_path)
                        metrics  = compute_metrics(results, recall_at_k)
                        save_metrics_json(metrics, met_path)
                        append_to_research_csv(
                            csv_path, domain, retriever, metrics, config,
                            stage="retrieval", version=version,
                            n_articles=n_articles,
                        )
                        log.info(
                            "[04] retrieval v%d | %s | R@10=%.3f | MRR=%.3f",
                            version, retriever,
                            metrics.get("recall_at_10", 0),
                            metrics.get("mrr", 0),
                        )
                        summary_rows.append({
                            "domain": domain, "stage": "retrieval",
                            "retriever": retriever, "version": version,
                            **metrics,
                        })
                    else:
                        log.info("[04] skip (cached): %s", met_path)

            if "reranking" in stages and config.get("reranking", {}).get("enabled", True):
                rer_path = cu.get_reranking_path(config, domain, retriever, reranker_name, version)
                if not rer_path.exists():
                    log.warning("[04] reranking results not found: %s", rer_path)
                else:
                    met_path = cu.get_metrics_path(
                        config, domain, retriever, version, "reranking", reranker_name,
                    )
                    if not met_path.exists() or force:
                        results = load_reranking_results(rer_path)
                        metrics = compute_metrics(results, recall_at_k)
                        save_metrics_json(metrics, met_path)
                        append_to_research_csv(
                            csv_path, domain, retriever, metrics, config,
                            stage="reranking", reranker=reranker_name,
                            version=version, n_articles=n_articles,
                        )
                        log.info(
                            "[04] reranking v%d | %s+%s | R@10=%.3f | MRR=%.3f",
                            version, retriever, reranker_name,
                            metrics.get("recall_at_10", 0),
                            metrics.get("mrr", 0),
                        )
                        summary_rows.append({
                            "domain": domain, "stage": "reranking",
                            "retriever": retriever, "reranker": reranker_name,
                            "version": version, **metrics,
                        })
                    else:
                        log.info("[04] skip (cached): %s", met_path)

    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and reranking results.")
    parser.add_argument("--config", default="configs/article_retrieval.yaml")
    parser.add_argument("--domain", help="Single domain override.")
    parser.add_argument("--retriever", help="Comma-separated retriever names.")
    parser.add_argument("--versions", help="Comma-separated version numbers.")
    parser.add_argument(
        "--stage",
        choices=["retrieval", "reranking", "all"],
        default="all",
        help="Which stage to evaluate.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config  = cu.load_config(ROOT / args.config)
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

    stages = ["retrieval", "reranking"] if args.stage == "all" else [args.stage]

    csv_path = cu.get_research_csv_path(config)
    for domain in domains:
        evaluate_for_domain(config, domain, retrievers, versions, stages, force=args.force)
        update_article_retrieval_stats(domain, csv_path)


if __name__ == "__main__":
    main()
