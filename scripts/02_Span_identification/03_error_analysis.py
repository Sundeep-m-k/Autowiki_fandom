#!/usr/bin/env python3
"""Run error analysis on validation predictions and save to research/error_analysis/."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.span_identification.config_utils import load_config
from src.span_identification.dataset import ensure_splits
from src.span_identification.baselines import run_baseline
from src.span_identification.error_analysis import categorize_errors, sample_errors, save_error_analysis


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "span_id.yaml"
    config = load_config(config_path)

    research_dir = Path(config["research_dir"])
    error_dir = research_dir / "error_analysis"

    domains = config.get("domains", ["beverlyhillscop"])
    granularities = config.get("granularities", ["sentence", "paragraph"])
    baselines = config.get("baselines", ["rule_capitalized", "heuristic_anchor"])

    for domain in domains:
        for granularity in granularities:
            try:
                _, val_ex, _ = ensure_splits(config, domain, granularity)
            except FileNotFoundError as e:
                print(f"Skipping {domain}/{granularity}: {e}")
                continue
            for name in baselines:
                pred_val = run_baseline(name, val_ex)
                for ex in pred_val:
                    ex.setdefault("pred_spans", [])
                summary = {"domain": domain, "granularity": granularity, "baseline": name}
                all_fp = all_fn = 0
                for ex in pred_val:
                    cat = categorize_errors(ex["gold_spans"], ex.get("pred_spans", []))
                    all_fp += cat["fp_count"]
                    all_fn += cat["fn_count"]
                summary["total_fp"] = all_fp
                summary["total_fn"] = all_fn
                fp_samp, fn_samp = sample_errors(pred_val, max_fp=20, max_fn=20)
                out = error_dir / f"{domain}_{granularity}_{name}"
                save_error_analysis(out, summary, fp_samp, fn_samp)
                print(f"Saved error analysis to {out}")


if __name__ == "__main__":
    main()
