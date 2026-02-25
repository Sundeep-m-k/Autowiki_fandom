#!/usr/bin/env python3
"""Run baseline span predictors and append results to research CSV."""
import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.span_identification.config_utils import get_research_csv_path, get_span_id_log_dir, load_config
from src.span_identification.logging_utils import setup_span_id_logger
from src.span_identification.dataset import ensure_splits
from src.span_identification.baselines import run_baseline
from src.span_identification.evaluator import evaluate_example, aggregate_metrics


def main():
    config_path = PROJECT_ROOT / "configs" / "span_id" / "span_id.yaml"
    config = load_config(config_path)

    domains = config.get("domains", ["beverlyhillscop"])
    log = setup_span_id_logger(log_dir=str(get_span_id_log_dir(config, domains[0])), script_name="02_run_baselines")
    log.info("[main] config loaded run_baselines")

    research_csv = get_research_csv_path(config)
    research_csv.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    fieldnames = [
        "run_id", "timestamp", "seed", "experiment_type",
        "granularity", "domain", "model", "label_scheme", "data_fraction",
        "train_size", "val_size", "span_f1", "span_precision", "span_recall",
        "char_f1", "exact_match_pct", "wall_time_sec", "checkpoint_path", "notes",
    ]

    def append_row(row):
        write_header = not research_csv.exists() or research_csv.stat().st_size == 0
        with open(research_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)

    granularities = config.get("granularities", ["sentence", "paragraph"])
    baselines = config.get("baselines", ["rule_capitalized", "heuristic_anchor", "random"])

    for domain in domains:
        log = setup_span_id_logger(log_dir=str(get_span_id_log_dir(config, domain)), script_name="02_run_baselines")
        for granularity in granularities:
            log.info("[main] domain=%s granularity=%s", domain, granularity)
            try:
                train_ex, val_ex, _ = ensure_splits(config, domain, granularity)
            except FileNotFoundError as e:
                log.warning("[main] skipping %s/%s: %s", domain, granularity, e)
                continue
            for name in baselines:
                pred_val = run_baseline(name, val_ex)
                metrics_list = [
                    evaluate_example(ex["gold_spans"], ex["pred_spans"], len(ex["text"]))
                    for ex in pred_val
                ]
                m = aggregate_metrics(metrics_list)
                log.info("[main] baseline %s %s/%s -> span_f1=%.4f", name, domain, granularity, m.get("span_f1", 0))
                append_row({
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                    "seed": -1,
                    "experiment_type": "baseline",
                    "granularity": granularity,
                    "domain": domain,
                    "model": name,
                    "label_scheme": "",
                    "data_fraction": 1.0,
                    "train_size": len(train_ex),
                    "val_size": len(val_ex),
                    "span_f1": m.get("span_f1", 0),
                    "span_precision": m.get("span_precision", 0),
                    "span_recall": m.get("span_recall", 0),
                    "char_f1": m.get("char_f1", 0),
                    "exact_match_pct": m.get("exact_match_pct", 0),
                    "wall_time_sec": 0,
                    "checkpoint_path": "",
                    "notes": "",
                })

    log.info("[main] done. Results appended to %s", research_csv)


if __name__ == "__main__":
    main()
