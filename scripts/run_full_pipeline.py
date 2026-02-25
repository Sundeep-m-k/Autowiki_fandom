"""Master end-to-end pipeline runner.

Runs every stage in order, from raw HTML scraping to the linking pipeline:

  Stage 0  — Scrape Fandom wiki HTML
  Stage 1  — Build ground truth (paragraphs, sentences, articles, links)
  Stage 2  — Span identification (baselines + model experiments)
  Stage 3  — Article retrieval  (index → queries → retrieval → reranking → eval)
  Stage 4  — Linking pipeline   (predict → HTML → evaluate)

Each stage has a corresponding --skip-* flag so you can re-enter mid-pipeline
without redoing expensive earlier work (e.g. scraping or GPU training).

Usage examples
--------------
  # Full pipeline, default domain from each config:
  python scripts/run_full_pipeline.py

  # Single domain end-to-end:
  python scripts/run_full_pipeline.py --domain beverlyhillscop

  # Skip scraping (HTML already in data/raw/):
  python scripts/run_full_pipeline.py --skip-scrape

  # Skip scraping + ground-truth + span-id (Task 1 already done):
  python scripts/run_full_pipeline.py --skip-scrape --skip-gt --skip-span-id

  # Start from article retrieval onwards:
  python scripts/run_full_pipeline.py --skip-scrape --skip-gt --skip-span-id

  # Span ID only (skip everything else):
  python scripts/run_full_pipeline.py --skip-scrape --skip-gt \\
      --skip-retrieval --skip-linking

  # Force rebuild of all cached artifacts (except scraping):
  python scripts/run_full_pipeline.py --skip-scrape --force

  # Machine-specific configs:
  python scripts/run_full_pipeline.py \\
      --span-id-config   configs/span_id/kudremukh.yaml \\
      --retrieval-config configs/article_retrieval/kudremukh.yaml \\
      --linking-config   configs/linking/linking.yaml

  # NIL threshold ablation in the linking stage:
  python scripts/run_full_pipeline.py --skip-scrape --skip-gt --skip-span-id \\
      --skip-retrieval --nil-thresholds 0.0,0.1,0.2,0.3,0.5

Configs
-------
  Scraping:          configs/data_processing/scraping.yaml
  Ground truth:      configs/data_processing/ground_truth.yaml
  Span ID:           configs/span_id/span_id.yaml          (override: --span-id-config)
  Article retrieval: configs/article_retrieval/article_retrieval.yaml (override: --retrieval-config)
  Linking:           configs/linking/linking.yaml           (override: --linking-config)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# ── Stage script locations ────────────────────────────────────────────────────
DATA_PROC   = ROOT / "scripts" / "01_Data_processing"
SPAN_ID     = ROOT / "scripts" / "02_Span_identification"
RETRIEVAL   = ROOT / "scripts" / "03_Article_retrieval"
LINKING     = ROOT / "scripts" / "04_Linking_pipeline"


# ── Step runner ───────────────────────────────────────────────────────────────

def run_step(script: Path, extra_args: list[str], step_name: str) -> None:
    cmd = [PYTHON, str(script)] + extra_args
    width = 72
    print(f"\n{'=' * width}")
    print(f"  STEP : {step_name}")
    print(f"  CMD  : {' '.join(cmd)}")
    print(f"{'=' * width}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(
            f"\n  [ERROR] '{step_name}' failed "
            f"(exit {result.returncode}) after {elapsed:.1f}s"
        )
        print("  Aborting pipeline — fix the error above and re-run with "
              "the appropriate --skip-* flags to resume.")
        sys.exit(result.returncode)
    print(f"  [OK] '{step_name}' completed in {elapsed:.1f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end Autowiki pipeline: scrape → ground truth → "
                    "span ID → article retrieval → linking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Skip flags ──────────────────────────────────────────────────────────
    parser.add_argument("--skip-scrape",    action="store_true",
                        help="Skip Stage 0: scraping.")
    parser.add_argument("--skip-gt",        action="store_true",
                        help="Skip Stage 1: ground truth build.")
    parser.add_argument("--skip-span-id",   action="store_true",
                        help="Skip Stage 2: span identification.")
    parser.add_argument("--skip-retrieval", action="store_true",
                        help="Skip Stage 3: article retrieval.")
    parser.add_argument("--skip-linking",   action="store_true",
                        help="Skip Stage 4: linking pipeline.")

    # ── Fine-grained skip flags inside stages ────────────────────────────────
    parser.add_argument("--skip-index",    action="store_true",
                        help="(Stage 3) Skip article index building.")
    parser.add_argument("--skip-queries",  action="store_true",
                        help="(Stage 3) Skip query dataset building.")
    parser.add_argument("--skip-html",     action="store_true",
                        help="(Stage 4) Skip HTML rendering.")

    # ── Domain ───────────────────────────────────────────────────────────────
    parser.add_argument("--domain",
                        help="Override the domain for all tasks that accept it. "
                             "Each config's default domain is used otherwise.")

    # ── Configs ───────────────────────────────────────────────────────────────
    parser.add_argument("--span-id-config",   default="configs/span_id/span_id.yaml",
                        help="Span ID config (default: configs/span_id/span_id.yaml)")
    parser.add_argument("--retrieval-config", default="configs/article_retrieval/article_retrieval.yaml",
                        help="Article retrieval config.")
    parser.add_argument("--linking-config",   default="configs/linking/linking.yaml",
                        help="Linking pipeline config.")

    # ── Force ────────────────────────────────────────────────────────────────
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild of all cached artifacts in Stages 3 & 4.")

    # ── NIL ablation ─────────────────────────────────────────────────────────
    parser.add_argument("--nil-thresholds",
                        help="(Stage 4) Comma-separated NIL thresholds, e.g. 0.0,0.1,0.3")

    args = parser.parse_args()

    # ── Build shared domain arg fragments ────────────────────────────────────
    domain_args   = ["--domain", args.domain] if args.domain else []
    force_args    = ["--force"] if args.force else []

    t_pipeline = time.time()
    print("\n" + "=" * 72)
    print("  Autowiki End-to-End Pipeline")
    if args.domain:
        print(f"  Domain : {args.domain}")
    print(f"  Stages : "
          f"{'scrape ' if not args.skip_scrape else ''}"
          f"{'gt ' if not args.skip_gt else ''}"
          f"{'span-id ' if not args.skip_span_id else ''}"
          f"{'retrieval ' if not args.skip_retrieval else ''}"
          f"{'linking' if not args.skip_linking else ''}")
    print("=" * 72)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 0 — Scraping
    # ──────────────────────────────────────────────────────────────────────────
    if not args.skip_scrape:
        # Domain is read from configs/data_processing/scraping.yaml; no CLI domain arg supported
        run_step(
            DATA_PROC / "00_scrape_fandom.py",
            [],
            "Stage 0 — Scrape Fandom wiki",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1 — Ground Truth
    # ──────────────────────────────────────────────────────────────────────────
    if not args.skip_gt:
        # 01_build_ground_truth.py accepts optional domain as first positional arg
        gt_args = [args.domain] if args.domain else []
        run_step(
            DATA_PROC / "01_build_ground_truth.py",
            gt_args,
            "Stage 1 — Build Ground Truth",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2 — Span Identification
    # ──────────────────────────────────────────────────────────────────────────
    if not args.skip_span_id:
        # 01_run_span_id.py reads domain(s) from its config; no --domain flag
        run_step(
            SPAN_ID / "01_run_span_id.py",
            ["--config", args.span_id_config],
            "Stage 2 — Span Identification (baselines + models)",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 3 — Article Retrieval
    # ──────────────────────────────────────────────────────────────────────────
    if not args.skip_retrieval:
        retrieval_args = (
            ["--config", args.retrieval_config]
            + domain_args
            + force_args
        )
        if args.skip_index:
            retrieval_args += ["--skip-index"]
        if args.skip_queries:
            retrieval_args += ["--skip-queries"]

        run_step(
            RETRIEVAL / "run_all.py",
            retrieval_args,
            "Stage 3 — Article Retrieval (index → queries → retrieval → reranking → eval)",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 4 — Linking Pipeline
    # ──────────────────────────────────────────────────────────────────────────
    if not args.skip_linking:
        linking_args = (
            ["--config", args.linking_config]
            + domain_args
            + force_args
        )
        if args.skip_html:
            linking_args += ["--skip-html"]
        if args.nil_thresholds:
            linking_args += ["--nil-thresholds", args.nil_thresholds]

        run_step(
            LINKING / "run_all.py",
            linking_args,
            "Stage 4 — Linking Pipeline (predict → HTML → evaluate)",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Done
    # ──────────────────────────────────────────────────────────────────────────
    total = time.time() - t_pipeline
    print(f"\n{'=' * 72}")
    print(f"  ALL STAGES COMPLETE in {total / 60:.1f} min ({total:.0f}s)")
    print(f"  Research CSVs  : data/research/")
    print(f"  Domain stats   : data/stats/<domain>.json")
    print(f"  Linking HTML   : data/linking/<domain>/html/")
    print(f"  Retrieval plots: data/article_retrieval/<domain>/plots/")
    print(f"  Linking plots  : data/linking/<domain>/plots/")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
