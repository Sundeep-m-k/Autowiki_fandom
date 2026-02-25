#!/usr/bin/env python3
"""Run span identification on Skanda (1× A6000). Uses configs/span_id_skanda.yaml."""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.span_identification.logging_utils import setup_span_id_logger
from src.span_identification.config_utils import (
    get_checkpoint_dir,
    get_research_csv_path,
    get_span_id_log_dir,
    get_token_data_path,
    load_config,
)
from src.span_identification.dataset import ensure_splits
from src.span_identification.preprocess import build_token_dataset
from src.span_identification.hf_trainer import train_and_evaluate
from src.span_identification.evaluator import evaluate_example, aggregate_metrics
from src.span_identification.baselines import run_baseline


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "span_id_skanda.yaml"
    config = load_config(config_path)

    domains = config.get("domains", ["beverlyhillscop"])
    log_dir = str(get_span_id_log_dir(config, domains[0]))
    log = setup_span_id_logger(log_dir=log_dir, script_name="01_run_span_id_skanda")
    log.info("[main] Skanda config loaded from %s (1 GPU)", config_path)

    if config.get("fix_random_seeds"):
        import random
        import numpy as np
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info("[main] run_id=%s", run_id)

    research_csv = get_research_csv_path(config)
    research_csv.parent.mkdir(parents=True, exist_ok=True)
    log.info("[main] research_csv=%s", research_csv)

    fieldnames = [
        "run_id", "timestamp", "seed", "experiment_type",
        "granularity", "domain", "model", "label_scheme", "data_fraction",
        "train_size", "val_size", "val_span_f1", "span_f1", "span_precision", "span_recall",
        "char_f1", "exact_match_pct", "wall_time_sec", "checkpoint_path", "notes",
    ]

    def append_row(row: dict) -> None:
        write_header = not research_csv.exists() or research_csv.stat().st_size == 0
        with open(research_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)

    granularities = config.get("granularities", ["sentence", "paragraph", "article"])
    models = config.get("models", ["bert-base-uncased"])
    seeds = config.get("seeds", [42])
    data_fractions = config.get("data_fractions", [1.0])

    if config.get("run_baselines"):
        log.info("[main] starting baselines sweep domains=%s granularities=%s baselines=%s",
                 domains, granularities, config.get("baselines", []))
        for domain in domains:
            log = setup_span_id_logger(log_dir=str(get_span_id_log_dir(config, domain)), script_name="01_run_span_id_skanda")
            for granularity in granularities:
                log.info("[main] baseline domain=%s granularity=%s", domain, granularity)
                try:
                    train_ex, val_ex, test_ex = ensure_splits(config, domain, granularity, internal_only=True)
                    for baseline_name in config.get("baselines", []):
                        pred_val = run_baseline(baseline_name, val_ex)
                        pred_test = run_baseline(baseline_name, test_ex)
                        m_val = aggregate_metrics([
                            evaluate_example(ex["gold_spans"], ex["pred_spans"], len(ex["text"]))
                            for ex in pred_val
                        ])
                        m_test = aggregate_metrics([
                            evaluate_example(ex["gold_spans"], ex["pred_spans"], len(ex["text"]))
                            for ex in pred_test
                        ])
                        log.info("[main] baseline %s %s/%s -> val_f1=%.4f test_f1=%.4f",
                                 baseline_name, domain, granularity, m_val.get("span_f1", 0), m_test.get("span_f1", 0))
                        append_row({
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(),
                            "seed": -1,
                            "experiment_type": "baseline",
                            "granularity": granularity,
                            "domain": domain,
                            "model": baseline_name,
                            "label_scheme": "",
                            "data_fraction": 1.0,
                            "train_size": len(train_ex),
                            "val_size": len(val_ex),
                            "val_span_f1": m_val.get("span_f1", 0),
                            "span_f1": m_test.get("span_f1", 0),
                            "span_precision": m_test.get("span_precision", 0),
                            "span_recall": m_test.get("span_recall", 0),
                            "char_f1": m_test.get("char_f1", 0),
                            "exact_match_pct": m_test.get("exact_match_pct", 0),
                            "wall_time_sec": 0,
                            "checkpoint_path": "",
                            "notes": "skanda",
                        })
                except FileNotFoundError as e:
                    log.warning("[main] skipping baseline %s/%s: %s", domain, granularity, e)

    log.info("[main] starting model experiments models=%s seeds=%s data_fractions=%s", models, seeds, data_fractions)
    for domain in domains:
        log = setup_span_id_logger(log_dir=str(get_span_id_log_dir(config, domain)), script_name="01_run_span_id_skanda")
        for granularity in granularities:
            log.info("[main] model domain=%s granularity=%s", domain, granularity)

            for model_name in models:
                train_path = get_token_data_path(config, domain, granularity, model_name, "train")
                dev_path = get_token_data_path(config, domain, granularity, model_name, "dev")
                test_path = get_token_data_path(config, domain, granularity, model_name, "test")
                if not train_path.exists():
                    try:
                        log.info("[main] building token dataset domain=%s gran=%s model=%s", domain, granularity, model_name)
                        build_token_dataset(config, domain, granularity, model_name)
                    except FileNotFoundError as e:
                        log.warning("[main] skipping %s/%s: %s", domain, granularity, e)
                        continue

                for frac in data_fractions:
                    for seed in seeds:
                        ckpt_dir = get_checkpoint_dir(config, run_id, domain)
                        ckpt_sub = ckpt_dir / f"{granularity}_{domain}_{model_name.replace('/', '_')}_seed{seed}_frac{frac}"
                        ckpt_sub.mkdir(parents=True, exist_ok=True)
                        log.info("[main] CHECKPOINT model=%s gran=%s seed=%s frac=%.2f", model_name, granularity, seed, frac)

                        result = train_and_evaluate(
                            config=config,
                            train_path=train_path,
                            dev_path=dev_path,
                            test_path=test_path,
                            output_dir=ckpt_sub,
                            model_name=model_name,
                            seed=seed,
                            data_fraction=frac,
                        )

                        with open(train_path) as f:
                            train_size = sum(1 for _ in f)
                        with open(dev_path) as f:
                            val_size = sum(1 for _ in f)
                        if frac < 1.0:
                            train_size = max(1, int(train_size * frac))

                        log.info("[main] training done val_f1=%.4f test_f1=%.4f wall_time=%.1fs",
                                 result.get("val_span_f1", 0), result.get("span_f1", 0), result.get("wall_time_sec", 0))
                        append_row({
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(),
                            "seed": seed,
                            "experiment_type": "model",
                            "granularity": granularity,
                            "domain": domain,
                            "model": model_name,
                            "label_scheme": "BILOU",
                            "data_fraction": frac,
                            "train_size": train_size,
                            "val_size": val_size,
                            "val_span_f1": result.get("val_span_f1", 0),
                            "span_f1": result.get("span_f1", 0),
                            "span_precision": result.get("span_precision", 0),
                            "span_recall": result.get("span_recall", 0),
                            "char_f1": result.get("char_f1", 0),
                            "exact_match_pct": result.get("exact_match_pct", 0),
                            "wall_time_sec": result.get("wall_time_sec", 0),
                            "checkpoint_path": str(ckpt_sub),
                            "notes": "skanda",
                        })
                        log.info("[main] done: %s %s %s seed=%s -> F1=%.4f", domain, granularity, model_name, seed, result.get("span_f1", 0))

    log.info("[main] run complete. Results appended to %s", research_csv)


if __name__ == "__main__":
    main()
