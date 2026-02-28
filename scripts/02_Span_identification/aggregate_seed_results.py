#!/usr/bin/env python3
"""Aggregate model results across seeds: compute mean ± std for research reporting."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate span_id results across seeds")
    parser.add_argument("--csv", type=str, default="data/research/span_id_experiments.csv",
                        help="Path to CSV, or parent research dir to auto-discover domain subdirs")
    parser.add_argument("--out", type=str, default="data/research/span_id_experiments_aggregated.csv")
    parser.add_argument("--domain", type=str, help="Filter to a single domain")
    parser.add_argument("--run-id", type=str, help="Filter to this run_id only")
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    out_path = PROJECT_ROOT / args.out

    # Collect rows from domain-scoped subdirectories when flat file doesn't exist
    rows = []
    if csv_path.exists():
        candidate_csvs = [csv_path]
    else:
        csv_name = csv_path.name
        parent = csv_path.parent
        candidate_csvs = sorted(
            d / csv_name for d in parent.iterdir()
            if d.is_dir() and (d / csv_name).exists()
        )
        if not candidate_csvs:
            print(f"Input CSV not found: {csv_path}")
            return

    for cpath in candidate_csvs:
        with open(cpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if args.domain and row.get("domain") != args.domain:
                    continue
                if args.run_id and row.get("run_id") != args.run_id:
                    continue
                if row.get("experiment_type") != "model":
                    continue
                rows.append(row)

    # Group by (domain, granularity, model, data_fraction)
    groups = defaultdict(list)
    for r in rows:
        key = (r["domain"], r["granularity"], r["model"], str(r.get("data_fraction", "1.0")))
        groups[key].append(r)

    metric_keys = ["val_span_f1", "span_f1", "span_precision", "span_recall", "char_f1", "exact_match_pct", "wall_time_sec"]

    out_rows = []
    for (domain, gran, model, frac), group in sorted(groups.items()):
        n = len(group)
        agg = {
            "domain": domain,
            "granularity": gran,
            "model": model,
            "data_fraction": frac,
            "n_seeds": n,
            "run_id": group[0]["run_id"],
        }
        for k in metric_keys:
            vals = [float(r.get(k, 0) or 0) for r in group if r.get(k) != ""]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((x - mean) ** 2 for x in vals) / len(vals)
                std = var ** 0.5
                agg[k] = round(mean, 4)
                agg[f"{k}_std"] = round(std, 4)
            else:
                agg[k] = 0
                agg[f"{k}_std"] = 0
        agg["train_size"] = group[0].get("train_size", "")
        agg["val_size"] = group[0].get("val_size", "")
        out_rows.append(agg)

    fieldnames = ["domain", "granularity", "model", "data_fraction", "n_seeds", "run_id"]
    for k in metric_keys:
        fieldnames.extend([k, f"{k}_std"])
    fieldnames.extend(["train_size", "val_size"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    print(f"Aggregated {len(out_rows)} configs (from {len(rows)} seed rows) -> {out_path}")


if __name__ == "__main__":
    main()
