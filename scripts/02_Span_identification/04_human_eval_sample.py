#!/usr/bin/env python3
"""Sample predictions for human evaluation and save to research/human_eval/."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.span_identification.config_utils import load_config
from src.span_identification.dataset import ensure_splits
from src.span_identification.baselines import run_baseline


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "span_id.yaml"
    config = load_config(config_path)

    research_dir = Path(config["research_dir"])
    human_dir = research_dir / "human_eval"
    human_dir.mkdir(parents=True, exist_ok=True)

    domains = config.get("domains", ["beverlyhillscop"])
    granularities = config.get("granularities", ["sentence", "paragraph"])
    n_samples = 50
    seed = 42
    rng = random.Random(seed)

    for domain in domains:
        for granularity in granularities:
            try:
                _, val_ex, _ = ensure_splits(config, domain, granularity)
            except FileNotFoundError as e:
                print(f"Skipping {domain}/{granularity}: {e}")
                continue
            pred_val = run_baseline("heuristic_anchor", val_ex)
            sampled = rng.sample(pred_val, min(n_samples, len(pred_val)))
            samples = [
                {
                    "unit_id": ex.get("unit_id"),
                    "text": ex["text"],
                    "gold_spans": ex["gold_spans"],
                    "pred_spans": ex["pred_spans"],
                }
                for ex in sampled
            ]
            out_path = human_dir / f"{domain}_{granularity}_samples.jsonl"
            with open(out_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
            with open(human_dir / "annotation_schema.md", "w") as f:
                f.write("# Annotation schema\n\n")
                f.write("For each sample, mark: correct / partially_correct / wrong\n")
            print(f"Saved {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
