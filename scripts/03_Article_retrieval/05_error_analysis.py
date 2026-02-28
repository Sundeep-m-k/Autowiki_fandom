"""Step 5: Error analysis for article retrieval and reranking results.

By default, error analysis is run only on the best-performing query version
per (domain, retriever, stage), as determined by the highest MRR recorded in
the experiments CSV. Pass --versions to override and run on specific versions.

Output layout
-------------
data/article_retrieval/<domain>/error_analysis/
  retrieval_<retriever>_v<N>/
    errors_summary.json
    miss_samples.jsonl
  reranking_<retriever>_<reranker>_v<N>/
    errors_summary.json
    miss_samples.jsonl
    rank_change_summary.json

Run
---
  python scripts/03_Article_retrieval/05_error_analysis.py
  python scripts/03_Article_retrieval/05_error_analysis.py --domain money-heist
  python scripts/03_Article_retrieval/05_error_analysis.py --stage retrieval
  python scripts/03_Article_retrieval/05_error_analysis.py --retriever bm25
  python scripts/03_Article_retrieval/05_error_analysis.py --versions 1,2,3  # override best-version selection
  python scripts/03_Article_retrieval/05_error_analysis.py --force
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import article_retrieval.config_utils as cu
from article_retrieval.retriever import load_retrieval_results
from article_retrieval.reranker import load_reranking_results
from article_retrieval.error_analysis import (
    aggregate_errors,
    compare_retrieval_reranking,
    sample_misses,
    save_error_analysis,
)
from article_retrieval.logging_utils import setup_logger

log = logging.getLogger("article_retrieval")


# ── Best-version selection ────────────────────────────────────────────────────

def _find_best_version(
    config: dict,
    domain: str,
    retriever: str,
    stage: str,
) -> int | None:
    """
    Read the experiments CSV and return the query version with the highest
    Recall@1 for the given (domain, retriever, stage) combination.

    For the reranking stage the reranker name must also match the configured one.
    Returns None if no matching rows are found.
    """
    csv_path = cu.get_research_csv_path(config, domain)
    if not csv_path.exists():
        return None

    reranker_name = config.get("reranking", {}).get("model", "") if stage == "reranking" else ""

    best_version: int | None = None
    best_r1: float = -1.0

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("domain") != domain:
                continue
            if row.get("retriever") != retriever:
                continue
            if row.get("stage") != stage:
                continue
            if stage == "reranking" and row.get("reranker") != reranker_name:
                continue
            try:
                r1 = float(row["recall_at_1"])
                version = int(row["version"])
            except (ValueError, KeyError):
                continue
            if r1 > best_r1:
                best_r1 = r1
                best_version = version

    return best_version


# ── Query anchor lookup ────────────────────────────────────────────────────────

def _load_anchor_lookup(config: dict, domain: str) -> dict[str, str]:
    """Load query_id → anchor_text mapping from the query dataset JSONL."""
    qpath = cu.get_query_dataset_path(config, domain)
    if not qpath.exists():
        log.warning("[05] query dataset not found, anchor_text will be empty: %s", qpath)
        return {}
    lookup: dict[str, str] = {}
    with open(qpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("query_id")
            anchor = rec.get("anchor_text", "")
            if qid:
                lookup[qid] = anchor
    log.info("[05] loaded %d anchor texts from %s", len(lookup), qpath)
    return lookup


# ── Retrieval error analysis ───────────────────────────────────────────────────

def _run_retrieval_ea(
    config: dict,
    domain: str,
    retriever: str,
    version: int,
    anchor_lookup: dict[str, str],
    ea_cfg: dict,
    force: bool,
) -> None:
    top_k = config.get("retrieval", {}).get("top_k", 100)
    ret_path = cu.get_retrieval_path(config, domain, retriever, version, top_k)
    if not ret_path.exists():
        log.warning("[05] retrieval results not found, skipping: %s", ret_path)
        return

    out_dir = cu.get_error_analysis_path(config, domain, retriever, version, stage="retrieval")
    if (out_dir / "errors_summary.json").exists() and not force:
        log.info("[05] skip (cached): %s", out_dir)
        return

    results = load_retrieval_results(ret_path)
    summary = aggregate_errors(results, hit_k=ea_cfg.get("hit_k", 10))
    misses = sample_misses(
        results,
        max_samples=ea_cfg.get("max_miss_samples", 50),
        seed=ea_cfg.get("seed", 42),
        hit_k=ea_cfg.get("hit_k", 10),
        anchor_lookup=anchor_lookup,
    )
    save_error_analysis(out_dir, summary, misses)
    log.info(
        "[05] retrieval v%d | %s | not_retrieved=%d low_rank=%d top10_hit=%d MRR=%.3f → %s",
        version, retriever,
        summary["not_retrieved"], summary["low_rank"], summary["top10_hit"],
        summary["mrr"], out_dir,
    )


# ── Reranking error analysis ───────────────────────────────────────────────────

def _run_reranking_ea(
    config: dict,
    domain: str,
    retriever: str,
    version: int,
    anchor_lookup: dict[str, str],
    ea_cfg: dict,
    force: bool,
) -> None:
    reranker_name = config.get("reranking", {}).get("model", "")
    if not reranker_name:
        log.warning("[05] no reranker configured, skipping reranking error analysis")
        return

    rer_path = cu.get_reranking_path(config, domain, retriever, reranker_name, version)
    if not rer_path.exists():
        log.warning("[05] reranking results not found, skipping: %s", rer_path)
        return

    out_dir = cu.get_error_analysis_path(
        config, domain, retriever, version, stage="reranking", reranker=reranker_name,
    )
    if (out_dir / "errors_summary.json").exists() and not force:
        log.info("[05] skip (cached): %s", out_dir)
        return

    rer_results = load_reranking_results(rer_path)
    summary = aggregate_errors(rer_results, hit_k=ea_cfg.get("hit_k", 10))
    misses = sample_misses(
        rer_results,
        max_samples=ea_cfg.get("max_miss_samples", 50),
        seed=ea_cfg.get("seed", 42),
        hit_k=ea_cfg.get("hit_k", 10),
        anchor_lookup=anchor_lookup,
    )

    # Rank-change comparison requires the paired retrieval results.
    top_k = config.get("retrieval", {}).get("top_k", 100)
    ret_path = cu.get_retrieval_path(config, domain, retriever, version, top_k)
    rank_change: dict | None = None
    if ret_path.exists():
        ret_results = load_retrieval_results(ret_path)
        rank_change = compare_retrieval_reranking(ret_results, rer_results)
    else:
        log.warning("[05] retrieval results missing for rank-change comparison: %s", ret_path)

    save_error_analysis(out_dir, summary, misses, rank_change_summary=rank_change)
    log.info(
        "[05] reranking v%d | %s+%s | not_retrieved=%d low_rank=%d top10_hit=%d MRR=%.3f → %s",
        version, retriever, reranker_name,
        summary["not_retrieved"], summary["low_rank"], summary["top10_hit"],
        summary["mrr"], out_dir,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Error analysis for article retrieval and reranking results.",
    )
    p.add_argument("--config", default="configs/article_retrieval/article_retrieval.yaml",
                   help="Config file path (relative to project root).")
    p.add_argument("--domain", help="Single domain override.")
    p.add_argument("--retriever", help="Comma-separated retriever names.")
    p.add_argument("--versions", help="Comma-separated version numbers.")
    p.add_argument("--stage", choices=["retrieval", "reranking", "all"], default="all",
                   help="Which stage to analyse (default: all).")
    p.add_argument("--force", action="store_true",
                   help="Rerun even if output already exists.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    config = cu.resolve_config(cu.load_config(ROOT / args.config))

    domains = [args.domain] if args.domain else config.get("domains", [])

    cfg_sparse = config.get("retrievers", {}).get("sparse", [])
    cfg_dense  = config.get("retrievers", {}).get("dense", [])
    all_retrievers = cfg_sparse + cfg_dense
    retrievers = (
        [r.strip() for r in args.retriever.split(",")] if args.retriever else all_retrievers
    )

    # When --versions is given explicitly, use those versions for every retriever/stage.
    # Otherwise, best-version selection is done per (domain, retriever, stage) below.
    explicit_versions: list[int] | None = (
        [int(v.strip()) for v in args.versions.split(",")] if args.versions else None
    )

    run_retrieval_ea = args.stage in ("retrieval", "all")
    run_reranking_ea = args.stage in ("reranking", "all")

    # Error analysis knobs — can be added to the config under an "error_analysis" key.
    ea_cfg = config.get("error_analysis", {
        "hit_k": 10,
        "max_miss_samples": 50,
        "seed": 42,
    })

    for domain in domains:
        log_dir = cu.get_log_dir(config, domain)
        setup_logger(log_dir, script_name="05_error_analysis")
        log.info(
            "[main] domain=%s  retrievers=%s  stage=%s  explicit_versions=%s",
            domain, retrievers, args.stage, explicit_versions,
        )

        anchor_lookup = _load_anchor_lookup(config, domain)

        for retriever in retrievers:
            if run_retrieval_ea:
                if explicit_versions is not None:
                    versions = explicit_versions
                else:
                    best = _find_best_version(config, domain, retriever, "retrieval")
                    if best is None:
                        log.warning(
                            "[main] no retrieval results in CSV for %s/%s — skipping",
                            domain, retriever,
                        )
                        versions = []
                    else:
                        log.info(
                            "[main] best retrieval version for %s/%s → v%d",
                            domain, retriever, best,
                        )
                        versions = [best]
                for version in versions:
                    _run_retrieval_ea(
                        config, domain, retriever, version,
                        anchor_lookup, ea_cfg, args.force,
                    )

            if run_reranking_ea and config.get("reranking", {}).get("enabled", True):
                if explicit_versions is not None:
                    versions = explicit_versions
                else:
                    best = _find_best_version(config, domain, retriever, "reranking")
                    if best is None:
                        log.warning(
                            "[main] no reranking results in CSV for %s/%s — skipping",
                            domain, retriever,
                        )
                        versions = []
                    else:
                        log.info(
                            "[main] best reranking version for %s/%s → v%d",
                            domain, retriever, best,
                        )
                        versions = [best]
                for version in versions:
                    _run_reranking_ea(
                        config, domain, retriever, version,
                        anchor_lookup, ea_cfg, args.force,
                    )

    log.info("[main] error analysis complete.")


if __name__ == "__main__":
    main()
