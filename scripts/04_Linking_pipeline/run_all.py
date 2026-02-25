"""Master orchestration script for the Linking Pipeline (Task 3).

Runs all pipeline steps in order with skip logic at each step.

Usage:
  # Default run:
  python scripts/04_Linking_pipeline/run_all.py

  # Single domain:
  python scripts/04_Linking_pipeline/run_all.py --domain beverlyhillscop

  # Force rebuild of everything:
  python scripts/04_Linking_pipeline/run_all.py --force

  # Skip HTML rendering (just link + evaluate):
  python scripts/04_Linking_pipeline/run_all.py --skip-html

  # NIL threshold ablation (run step 00 + 02 for multiple thresholds):
  python scripts/04_Linking_pipeline/run_all.py --nil-thresholds 0.0,0.1,0.2,0.5

Pipeline steps:
  00  predict_and_link  — look up Task 2 results, assign article_ids, apply NIL
  01  render_html       — write HTML files with <a> tags
  02  evaluate          — compute Linking F1, Span F1, Entity Accuracy
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).parent
PYTHON      = sys.executable


def run_step(script: str, extra_args: list[str], step_name: str) -> None:
    cmd = [PYTHON, str(SCRIPTS_DIR / script)] + extra_args
    print(f"\n{'=' * 70}")
    print(f"  STEP: {step_name}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{'=' * 70}")
    t0     = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [ERROR] '{step_name}' failed (exit {result.returncode}) after {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"  [OK] '{step_name}' completed in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full linking pipeline.")
    parser.add_argument("--config",  default="configs/linking.yaml")
    parser.add_argument("--domain",  help="Single domain override.")
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--skip-html", action="store_true", help="Skip HTML rendering.")
    parser.add_argument(
        "--nil-thresholds",
        help="Comma-separated NIL thresholds for ablation. "
             "Temporarily overrides config nil_detection.threshold for each value.",
    )
    args = parser.parse_args()

    shared = ["--config", args.config]
    if args.domain:
        shared += ["--domain", args.domain]
    if args.force:
        shared += ["--force"]

    nil_thresholds = (
        [float(t.strip()) for t in args.nil_thresholds.split(",")]
        if args.nil_thresholds else [None]
    )

    t_start = time.time()
    print(f"\nLinking Pipeline — config: {args.config}")

    for nil_thr in nil_thresholds:
        if nil_thr is not None:
            print(f"\n--- NIL threshold ablation: {nil_thr} ---")
            # Write a temporary override config for this threshold
            import yaml, tempfile, os
            sys.path.insert(0, str(ROOT / "src"))
            import linking_pipeline.config_utils as cu
            cfg = cu.load_config(ROOT / args.config)
            cfg.setdefault("nil_detection", {})["threshold"] = nil_thr
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False,
                dir=str(ROOT / "configs"), prefix="linking_nil_tmp_"
            ) as tf:
                yaml.dump(cfg, tf)
                tmp_config = tf.name
            run_shared = ["--config", tmp_config] + (["--domain", args.domain] if args.domain else []) + (["--force"] if args.force else [])
        else:
            run_shared = shared
            tmp_config = None

        run_step("00_predict_and_link.py", run_shared, "00 — Predict and Link")

        if not args.skip_html and nil_thr is None:
            run_step("01_render_html.py", run_shared, "01 — Render HTML")

        run_step("02_evaluate.py", run_shared, "02 — Evaluate")

        if tmp_config:
            import os
            os.unlink(tmp_config)

    # Summary
    run_step(
        "visualise_linking.py",
        ["--config", args.config, "--no-show"]
        + (["--domain", args.domain] if args.domain else []),
        "Summary",
    )

    total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Linking pipeline complete in {total:.1f}s")
    print(f"  HTML output: data/linking/<domain>/html/")
    print(f"  Metrics:     data/linking/<domain>/metrics/")
    print(f"  Research CSV: data/research/linking_experiments.csv")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
