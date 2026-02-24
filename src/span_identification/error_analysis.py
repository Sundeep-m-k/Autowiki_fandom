"""Error analysis: categorize FP/FN, sample for review."""
from __future__ import annotations

import json
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
