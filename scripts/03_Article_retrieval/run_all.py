"""Master orchestration script for the Article Retrieval pipeline.

Runs all pipeline steps in order with skip logic at each step.
Individual steps can be skipped with --skip-* flags or re-forced with --force.

Usage:
  # Default run (all steps, all domains in config):
  python scripts/03_Article_retrieval/run_all.py

  # Machine-specific config:
  python scripts/03_Article_retrieval/run_all.py --config configs/article_retrieval/kudremukh.yaml

  # Single domain:
  python scripts/03_Article_retrieval/run_all.py --domain money-heist

  # Ablation: only one retriever, specific query versions:
  python scripts/03_Article_retrieval/run_all.py --retriever bm25 --versions 1,2,3,22,23

  # Force rebuild of everything:
  python scripts/03_Article_retrieval/run_all.py --force

  # Skip indexing (e.g. if already built), only run retrieval + eval:
  python scripts/03_Article_retrieval/run_all.py --skip-index --skip-queries

Pipeline steps:
  00  build_article_index   — BM25 / TF-IDF / FAISS indexes
  01  build_query_dataset   — generate 24 query variations per link
  02  run_retrieval         — retrieve top-K articles per (retriever, version)
  03  run_reranking         — re-rank with cross-encoder (Exp 6 & 7)
  04  evaluate              — compute Recall@K and MRR, write research CSV
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).parent

PYTHON = sys.executable

# Auto-select config based on hostname; can always be overridden with --config.
_HOSTNAME_CONFIG: dict[str, str] = {
    "kudremukh": "configs/article_retrieval/kudremukh.yaml",
}
_DEFAULT_CONFIG = _HOSTNAME_CONFIG.get(
    socket.gethostname().lower(),
    "configs/article_retrieval/base.yaml",
)


def run_step(script: str, extra_args: list[str], step_name: str) -> None:
    cmd = [PYTHON, str(SCRIPTS_DIR / script)] + extra_args
    print(f"\n{'=' * 70}")
    print(f"  STEP: {step_name}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{'=' * 70}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [ERROR] Step '{step_name}' failed (exit {result.returncode}) after {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"  [OK] Step '{step_name}' completed in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full article retrieval pipeline.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG,
                        help=f"Config file (auto-detected: {_DEFAULT_CONFIG}).")
    parser.add_argument("--domain", help="Single domain override.")
    parser.add_argument("--retriever", help="Comma-separated retriever names.")
    parser.add_argument("--versions", help="Comma-separated version numbers.")
    parser.add_argument("--stage", choices=["retrieval", "reranking", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Force rebuild of all cached artifacts.")
    parser.add_argument("--skip-index",   action="store_true", help="Skip step 00 (article indexing).")
    parser.add_argument("--skip-queries", action="store_true", help="Skip step 01 (query building).")
    parser.add_argument("--skip-retrieval", action="store_true", help="Skip step 02 (retrieval).")
    parser.add_argument("--skip-reranking", action="store_true", help="Skip step 03 (reranking).")
    parser.add_argument("--skip-eval", action="store_true", help="Skip step 04 (evaluation).")
    args = parser.parse_args()

    # ── Shared args to pass to all sub-scripts ──────────────────────────────────
    shared = ["--config", args.config]
    if args.domain:
        shared += ["--domain", args.domain]
    if args.force:
        shared += ["--force"]

    retrieval_extra = []
    if args.retriever:
        retrieval_extra += ["--retriever", args.retriever]
    if args.versions:
        retrieval_extra += ["--versions", args.versions]

    eval_extra = list(retrieval_extra)
    if args.stage != "all":
        eval_extra += ["--stage", args.stage]

    t_start = time.time()
    auto = " (auto-detected)" if args.config == _DEFAULT_CONFIG else " (user-supplied)"
    print(f"\nArticle Retrieval Pipeline — config: {args.config}{auto}")
    print(f"domain={'(all)' if not args.domain else args.domain}  "
          f"force={args.force}  stage={args.stage}")

    # ── Step 00: Build article index ────────────────────────────────────────────
    if not args.skip_index:
        run_step("00_build_article_index.py", shared, "00 — Build Article Index")

    # ── Step 01: Build query dataset ────────────────────────────────────────────
    if not args.skip_queries:
        run_step("01_build_query_dataset.py", shared, "01 — Build Query Dataset")

    # ── Step 02: Run retrieval ──────────────────────────────────────────────────
    if not args.skip_retrieval and args.stage in ("retrieval", "all"):
        run_step("02_run_retrieval.py", shared + retrieval_extra, "02 — Run Retrieval")

    # ── Step 03: Run reranking ──────────────────────────────────────────────────
    if not args.skip_reranking and args.stage in ("reranking", "all"):
        run_step("03_run_reranking.py", shared + retrieval_extra, "03 — Run Reranking")

    # ── Step 04: Evaluate ───────────────────────────────────────────────────────
    if not args.skip_eval:
        run_step("04_evaluate.py", shared + eval_extra, "04 — Evaluate")

    # ── Summary ─────────────────────────────────────────────────────────────────
    total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete in {total:.1f}s")
    print(f"{'=' * 70}")

    # Print results summary
    run_step("aggregate_results.py", ["--config", args.config] + (["--domain", args.domain] if args.domain else []),
             "Summary")


if __name__ == "__main__":
    main()
