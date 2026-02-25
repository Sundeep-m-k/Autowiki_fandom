"""Error analysis: categorize FP/FN, sample for review.

Works for both baselines (which already expose per-example gold/pred spans)
and trained HF models (via ``predict_from_checkpoint`` in hf_trainer.py,
which decodes token-level predictions back to char-level spans).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def categorize_errors(
    gold_spans: list[tuple[int, int]],
    pred_spans: list[tuple[int, int]],
) -> dict[str, Any]:
    """Categorize false positives and false negatives."""
    gold_set = {tuple(s) for s in gold_spans}
    pred_set = {tuple(s) for s in pred_spans}
    tp = gold_set & pred_set
    fp = pred_set - gold_set
    fn = gold_set - pred_set

    def span_len(s: tuple[int, int]) -> int:
        return s[1] - s[0]

    fp_lengths = [span_len(s) for s in fp]
    fn_lengths = [span_len(s) for s in fn]

    return {
        "tp_count": len(tp),
        "fp_count": len(fp),
        "fn_count": len(fn),
        "fp_avg_len": sum(fp_lengths) / len(fp_lengths) if fp_lengths else 0,
        "fn_avg_len": sum(fn_lengths) / len(fn_lengths) if fn_lengths else 0,
        "fp_short": sum(1 for s in fp if span_len(s) <= 5),
        "fp_long": sum(1 for s in fp if span_len(s) > 20),
        "fn_short": sum(1 for s in fn if span_len(s) <= 5),
        "fn_long": sum(1 for s in fn if span_len(s) > 20),
    }


def sample_errors(
    examples: list[dict],
    max_fp: int = 20,
    max_fn: int = 20,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Sample FP and FN examples for human review."""
    import random
    rng = random.Random(seed)
    fp_samples = []
    fn_samples = []
    for ex in examples:
        gold_set = {tuple(s) for s in ex["gold_spans"]}
        pred_set = {tuple(s) for s in ex.get("pred_spans", [])}
        fp = pred_set - gold_set
        fn = gold_set - pred_set
        for s in fp:
            if len(fp_samples) < max_fp:
                fp_samples.append({
                    "text": ex["text"],
                    "span": list(s),
                    "span_text": ex["text"][s[0]:s[1]],
                    "gold_spans": ex["gold_spans"],
                    "pred_spans": ex.get("pred_spans", []),
                    "unit_id": ex.get("unit_id"),
                })
        for s in fn:
            if len(fn_samples) < max_fn:
                fn_samples.append({
                    "text": ex["text"],
                    "span": list(s),
                    "span_text": ex["text"][s[0]:s[1]],
                    "gold_spans": ex["gold_spans"],
                    "pred_spans": ex.get("pred_spans", []),
                    "unit_id": ex.get("unit_id"),
                })
        if len(fp_samples) >= max_fp and len(fn_samples) >= max_fn:
            break

    fp_samples = rng.sample(fp_samples, min(len(fp_samples), max_fp))
    fn_samples = rng.sample(fn_samples, min(len(fn_samples), max_fn))
    return fp_samples, fn_samples


def save_error_analysis(
    output_dir: Path,
    summary: dict,
    fp_samples: list[dict],
    fn_samples: list[dict],
) -> None:
    """Save error analysis to JSON/JSONL files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "errors_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "fp_samples.jsonl", "w") as f:
        for s in fp_samples:
            f.write(json.dumps(s) + "\n")
    with open(output_dir / "fn_samples.jsonl", "w") as f:
        for s in fn_samples:
            f.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# Model-level error analysis
# ---------------------------------------------------------------------------

def run_model_error_analysis(
    checkpoint_dir: Path,
    test_jsonl_path: Path,
    raw_split_jsonl_path: Path,
    output_dir: Path,
    label_scheme: str = "BILOU",
    max_seq_length: int = 512,
    batch_size: int = 32,
    max_fp: int = 50,
    max_fn: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run full error analysis for a single trained checkpoint.

    Loads the model, predicts on the test split, decodes to char spans,
    aggregates summary statistics, samples FP/FN examples, and saves everything
    under ``output_dir``.

    Returns the summary dict.
    """
    from src.span_identification.hf_trainer import predict_from_checkpoint

    log = logging.getLogger("span_id")
    log.info("[error_analysis] checkpoint=%s  output=%s", checkpoint_dir, output_dir)

    examples = predict_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        test_jsonl_path=test_jsonl_path,
        raw_split_jsonl_path=raw_split_jsonl_path,
        label_scheme=label_scheme,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
    )

    # Aggregate statistics across all examples
    total: dict[str, Any] = {
        "tp_count": 0, "fp_count": 0, "fn_count": 0,
        "fp_avg_len": 0.0, "fn_avg_len": 0.0,
        "fp_short": 0, "fp_long": 0,
        "fn_short": 0, "fn_long": 0,
        "num_examples": len(examples),
        "num_examples_with_gold": 0,
    }
    all_fp_lens: list[int] = []
    all_fn_lens: list[int] = []

    for ex in examples:
        stats = categorize_errors(ex["gold_spans"], ex["pred_spans"])
        for key in ("tp_count", "fp_count", "fn_count",
                    "fp_short", "fp_long", "fn_short", "fn_long"):
            total[key] += stats[key]
        all_fp_lens.extend([s[1] - s[0] for s in ex["pred_spans"]
                             if list(s) not in ex["gold_spans"]])
        all_fn_lens.extend([s[1] - s[0] for s in ex["gold_spans"]
                             if list(s) not in ex["pred_spans"]])
        if ex["gold_spans"]:
            total["num_examples_with_gold"] += 1

    total["fp_avg_len"] = sum(all_fp_lens) / len(all_fp_lens) if all_fp_lens else 0.0
    total["fn_avg_len"] = sum(all_fn_lens) / len(all_fn_lens) if all_fn_lens else 0.0

    tp = total["tp_count"]
    fp = total["fp_count"]
    fn = total["fn_count"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total["precision"] = precision
    total["recall"]    = recall
    total["span_f1"]   = f1

    fp_samples, fn_samples = sample_errors(examples, max_fp=max_fp, max_fn=max_fn, seed=seed)
    save_error_analysis(output_dir, total, fp_samples, fn_samples)

    log.info(
        "[error_analysis] P=%.3f R=%.3f F1=%.3f  FP=%d FN=%d  saved to %s",
        precision, recall, f1, fp, fn, output_dir,
    )
    return total


def find_checkpoints(
    ckpt_root: Path,
    domain: str,
    granularity: str,
    model_name: str,
    label_scheme: str,
    seed: int,
    frac: float,
) -> Path | None:
    """
    Locate the best-model checkpoint directory produced by train_and_evaluate.

    The naming convention mirrors what the run scripts write:
      <ckpt_root>/<granularity>_<domain>_<model>_<scheme>_seed<seed>_frac<frac>/

    Returns the path if it exists, otherwise None.
    """
    safe_model = model_name.replace("/", "_")
    frac_str = f"{frac:.1f}" if frac == int(frac) else str(frac)
    sub = ckpt_root / f"{granularity}_{domain}_{safe_model}_{label_scheme}_seed{seed}_frac{frac_str}"
    if sub.exists():
        return sub
    # Also try without trailing .0 on frac
    alt = ckpt_root / f"{granularity}_{domain}_{safe_model}_{label_scheme}_seed{seed}_frac{frac}"
    if alt.exists():
        return alt
    return None
