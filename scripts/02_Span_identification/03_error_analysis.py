#!/usr/bin/env python3
"""
Error analysis for Span Identification — baselines and trained models.

Usage
-----
# Default config (auto-detects hostname)
python scripts/02_Span_identification/03_error_analysis.py

# Override config path
python scripts/02_Span_identification/03_error_analysis.py --config configs/span_id/kudremukh.yaml

# Limit to one domain / granularity
python scripts/02_Span_identification/03_error_analysis.py --domain beverlyhillscop --granularity sentence

# Only run baseline or model error analysis
python scripts/02_Span_identification/03_error_analysis.py --type baseline
python scripts/02_Span_identification/03_error_analysis.py --type model

Output layout
-------------
data/span_id/<domain>/error_analysis/<granularity>/baseline_<name>/
    errors_summary.json
    fp_samples.jsonl
    fn_samples.jsonl

data/span_id/<domain>/error_analysis/<granularity>/model_<safe_model>_<scheme>_seed<s>/
    errors_summary.json
    fp_samples.jsonl
    fn_samples.jsonl
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.span_identification.config_utils import (
    get_checkpoint_dir,
    get_research_csv_path,
    get_span_id_log_dir,
    get_token_data_path,
    load_config,
)
from src.span_identification.dataset import ensure_splits
from src.span_identification.baselines import run_baseline
from src.span_identification.error_analysis import (
    categorize_errors,
    find_checkpoints,
    run_model_error_analysis,
    sample_errors,
    save_error_analysis,
)
from src.span_identification.logging_utils import setup_span_id_logger


# Error analysis always uses its own dedicated config.
# The file inherits from span_id_base.yaml (via the `base:` key) and only
# overrides the analysis-specific knobs, keeping training configs untouched.
_DEFAULT_CONFIG = "configs/span_id/error_analysis.yaml"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Span-ID error analysis for baselines and trained models.")
    p.add_argument("--config",      default=_DEFAULT_CONFIG,
                   help=f"Config file (auto-detected: {_DEFAULT_CONFIG})")
    p.add_argument("--domain",      help="Limit to one domain.")
    p.add_argument("--granularity", help="Limit to one granularity.")
    p.add_argument("--type",        choices=["baseline", "model", "all"], default="all",
                   help="Which experiment types to analyse (default: all).")
    p.add_argument("--model",       help="Limit model error analysis to this model name.")
    p.add_argument("--label_scheme", choices=["BIO", "BILOU"],
                   help="Limit model error analysis to this label scheme.")
    return p.parse_args()


# ── Baseline error analysis ───────────────────────────────────────────────────

def _run_baseline_analysis(
    config: dict,
    domain: str,
    granularity: str,
    error_root: Path,
    ea_cfg: dict,
    log,
) -> None:
    """Run error analysis for all configured baselines on the validation split."""
    baselines_list = config.get("baselines", ["rule_capitalized", "heuristic_anchor", "random"])
    try:
        _, val_ex, _ = ensure_splits(config, domain, granularity)
    except FileNotFoundError as e:
        log.warning("[baseline_ea] skipping %s/%s: %s", domain, granularity, e)
        return

    for name in baselines_list:
        pred_val = run_baseline(name, val_ex)
        for ex in pred_val:
            ex.setdefault("pred_spans", [])

        total_fp = total_fn = total_tp = 0
        all_fp_lens: list[int] = []
        all_fn_lens: list[int] = []
        fp_short = fp_long = fn_short = fn_long = 0

        for ex in pred_val:
            cat = categorize_errors(ex["gold_spans"], ex.get("pred_spans", []))
            total_tp += cat["tp_count"]
            total_fp += cat["fp_count"]
            total_fn += cat["fn_count"]
            fp_short  += cat["fp_short"]
            fp_long   += cat["fp_long"]
            fn_short  += cat["fn_short"]
            fn_long   += cat["fn_long"]
            gold_set = set(tuple(s) for s in ex["gold_spans"])
            pred_set = set(tuple(s) for s in ex.get("pred_spans", []))
            all_fp_lens.extend(s[1] - s[0] for s in pred_set - gold_set)
            all_fn_lens.extend(s[1] - s[0] for s in gold_set - pred_set)

        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        summary = {
            "domain": domain, "granularity": granularity,
            "experiment_type": "baseline", "model": name,
            "num_examples": len(pred_val),
            "tp_count": total_tp, "fp_count": total_fp, "fn_count": total_fn,
            "fp_short": fp_short, "fp_long": fp_long,
            "fn_short": fn_short, "fn_long": fn_long,
            "fp_avg_len": sum(all_fp_lens) / len(all_fp_lens) if all_fp_lens else 0.0,
            "fn_avg_len": sum(all_fn_lens) / len(all_fn_lens) if all_fn_lens else 0.0,
            "precision": prec, "recall": rec, "span_f1": f1,
        }

        fp_samp, fn_samp = sample_errors(
            pred_val,
            max_fp=ea_cfg.get("max_fp_samples", 50),
            max_fn=ea_cfg.get("max_fn_samples", 50),
            seed=ea_cfg.get("seed", 42),
        )
        out = error_root / domain / "error_analysis" / granularity / f"baseline_{name}"
        save_error_analysis(out, summary, fp_samp, fn_samp)
        log.info(
            "[baseline_ea] %s/%s/%s  P=%.3f R=%.3f F1=%.3f  saved → %s",
            domain, granularity, name, prec, rec, f1, out,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_best_model(
    config: dict,
    domain: str,
    granularity: str,
    label_scheme: str,
) -> tuple[str, int] | None:
    """
    Read the experiments CSV and return (model_name, best_seed) for the model
    with the highest mean val_span_f1 across all seeds for the given
    (domain, granularity, label_scheme) combination.

    "Best seed" is the individual seed row that achieved the highest val_span_f1
    for that winning model — this is the checkpoint we run error analysis on.

    Returns None if no matching rows are found.
    """
    csv_path = get_research_csv_path(config, domain)
    if not csv_path.exists():
        return None

    # model → list of (val_span_f1, seed) across all seeds
    from collections import defaultdict
    model_seed_f1: dict[str, list[tuple[float, int]]] = defaultdict(list)

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if (
                row.get("experiment_type") == "model"
                and row.get("domain") == domain
                and row.get("granularity") == granularity
                and row.get("label_scheme") == label_scheme
            ):
                try:
                    f1 = float(row["val_span_f1"])
                    seed = int(row["seed"])
                    model_seed_f1[row["model"]].append((f1, seed))
                except (ValueError, KeyError):
                    continue

    if not model_seed_f1:
        return None

    # Pick the model with the highest mean val_span_f1 across seeds
    best_model = max(
        model_seed_f1,
        key=lambda m: sum(f1 for f1, _ in model_seed_f1[m]) / len(model_seed_f1[m]),
    )
    # Within that model, use the seed that achieved the highest val_span_f1
    best_seed = max(model_seed_f1[best_model], key=lambda t: t[0])[1]
    return best_model, best_seed


def _lookup_run_id(
    config: dict,
    domain: str,
    granularity: str,
    model_name: str,
    label_scheme: str,
    seed: int,
) -> str | None:
    """
    Find the most recent run_id in the experiments CSV that matches the given
    (domain, granularity, model, label_scheme, seed) combination.

    Returns the run_id string (e.g. "20260225_095245"), or None if not found.
    Iterates all rows so the last (most recent) matching run_id wins.
    """
    csv_path = get_research_csv_path(config, domain)
    if not csv_path.exists():
        return None

    match: str | None = None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if (
                row.get("experiment_type") == "model"
                and row.get("domain") == domain
                and row.get("granularity") == granularity
                and row.get("model") == model_name
                and row.get("label_scheme") == label_scheme
                and row.get("seed") == str(seed)
            ):
                match = row["run_id"]
    return match


# ── Model error analysis ──────────────────────────────────────────────────────

def _run_model_analysis(
    config: dict,
    domain: str,
    granularity: str,
    model_name: str,
    label_scheme: str,
    error_root: Path,
    ea_cfg: dict,
    log,
    seed: int | None = None,
) -> None:
    """Run error analysis for a single trained-model checkpoint."""
    max_seq = config.get("model", {}).get("max_length", 512)
    fracs   = config.get("data_fractions", [1.0])

    if seed is None:
        seed = config.get("seeds", [42])[0]
    frac = fracs[0]

    # Resolve the run_id from the experiments CSV so we point at the right
    # checkpoint directory (data/span_id/<domain>/checkpoints/<run_id>/).
    run_id = _lookup_run_id(config, domain, granularity, model_name, label_scheme, seed)
    if run_id is None:
        log.warning(
            "[model_ea] no matching run in experiments CSV for %s/%s/%s/%s seed=%d — skipping",
            domain, granularity, model_name, label_scheme, seed,
        )
        return

    ckpt_root = PROJECT_ROOT / get_checkpoint_dir(config, run_id, domain)

    ckpt_dir = find_checkpoints(ckpt_root, domain, granularity, model_name, label_scheme, seed, frac)
    if ckpt_dir is None:
        log.warning(
            "[model_ea] checkpoint not found for %s/%s/%s/%s seed=%d frac=%s run_id=%s — skipping",
            domain, granularity, model_name, label_scheme, seed, frac, run_id,
        )
        return

    # Test tokenised JSONL
    try:
        test_token_path = get_token_data_path(config, domain, granularity, model_name, "test", label_scheme)
    except Exception as e:
        log.warning("[model_ea] cannot resolve token test path: %s — skipping", e)
        return

    if not test_token_path.exists():
        log.warning("[model_ea] token test file missing: %s — skipping", test_token_path)
        return

    # Raw (un-tokenised) test split for recovering text / unit_id
    span_id_dir = PROJECT_ROOT / config.get("span_id_dir", "data/span_id")
    raw_test_path = span_id_dir / domain / "splits" / f"test_{granularity}.jsonl"
    if not raw_test_path.exists():
        log.warning("[model_ea] raw test split missing: %s — skipping", raw_test_path)
        return

    safe_model = model_name.replace("/", "_")
    out = error_root / domain / "error_analysis" / granularity / f"model_{safe_model}_{label_scheme}_seed{seed}"

    run_model_error_analysis(
        checkpoint_dir=ckpt_dir,
        test_jsonl_path=test_token_path,
        raw_split_jsonl_path=raw_test_path,
        output_dir=out,
        label_scheme=label_scheme,
        max_seq_length=max_seq,
        batch_size=config.get("training", {}).get("batch_size", 32),
        max_fp=ea_cfg.get("max_fp_samples", 50),
        max_fn=ea_cfg.get("max_fn_samples", 50),
        seed=ea_cfg.get("seed", 42),
    )
    log.info("[model_ea] saved → %s", out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    domains       = [args.domain]      if args.domain      else config.get("domains",       ["beverlyhillscop"])
    granularities = [args.granularity] if args.granularity else config.get("granularities", ["sentence", "paragraph", "article"])
    label_schemes = config.get("label_schemes", ["BIO", "BILOU"])
    ea_cfg        = config.get("error_analysis", {})

    run_baseline_ea = args.type in ("baseline", "all")
    run_model_ea    = args.type in ("model", "all")

    if args.label_scheme:
        label_schemes = [args.label_scheme]

    span_id_dir = PROJECT_ROOT / config.get("span_id_dir", "data/span_id")
    error_root  = span_id_dir

    log = setup_span_id_logger(
        log_dir=str(PROJECT_ROOT / config.get("log_dir", "data/logs") / "error_analysis"),
        script_name="03_error_analysis",
    )
    log.info("[main] config=%s  domains=%s  granularities=%s", args.config, domains, granularities)
    log.info("[main] run_baseline_ea=%s  run_model_ea=%s", run_baseline_ea, run_model_ea)

    for domain in domains:
        for granularity in granularities:

            if run_baseline_ea:
                _run_baseline_analysis(config, domain, granularity, error_root, ea_cfg, log)

            if run_model_ea:
                for label_scheme in label_schemes:
                    if args.model:
                        # Manual override: use the specified model with seeds[0]
                        _run_model_analysis(
                            config, domain, granularity,
                            args.model, label_scheme,
                            error_root, ea_cfg, log,
                        )
                    else:
                        # Auto-select the best model from the experiments CSV
                        result = _find_best_model(config, domain, granularity, label_scheme)
                        if result is None:
                            log.warning(
                                "[main] no model results in CSV for %s/%s/%s — skipping",
                                domain, granularity, label_scheme,
                            )
                            continue
                        best_model, best_seed = result
                        log.info(
                            "[main] best model for %s/%s/%s → %s (seed=%d)",
                            domain, granularity, label_scheme, best_model, best_seed,
                        )
                        _run_model_analysis(
                            config, domain, granularity,
                            best_model, label_scheme,
                            error_root, ea_cfg, log,
                            seed=best_seed,
                        )

    log.info("[main] error analysis complete.  Output root: %s", error_root)


if __name__ == "__main__":
    main()
