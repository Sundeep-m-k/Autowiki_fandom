"""Integration test: run baseline and ensure splits."""
import pytest
from pathlib import Path

from src.span_identification.config_utils import load_config
from src.span_identification.dataset import ensure_splits
from src.span_identification.baselines import run_baseline
from src.span_identification.evaluator import evaluate_example, aggregate_metrics


def test_ensure_splits_with_real_data():
    """Ensure splits on real processed data if available."""
    root = Path(__file__).parents[2]
    config = load_config(root / "configs" / "span_id.yaml")
    proc_path = Path(config["processed_dir"]) / "beverlyhillscop" / "paragraphs_beverlyhillscop.jsonl"
    if not proc_path.exists():
        pytest.skip("Processed data not found, run 01_build_ground_truth first")
    train, val, test = ensure_splits(config, "beverlyhillscop", "paragraph", seed=42)
    assert len(train) + len(val) + len(test) >= 1


def test_baseline_produces_metrics():
    """Run baseline on minimal data."""
    examples = [
        {"text": "Axel Foley.", "gold_spans": [(0, 10)]},
        {"text": "No links here.", "gold_spans": []},
    ]
    pred = run_baseline("rule_capitalized", examples)
    m = aggregate_metrics([
        evaluate_example(ex["gold_spans"], ex["pred_spans"], len(ex["text"]))
        for ex in pred
    ])
    assert "span_f1" in m
