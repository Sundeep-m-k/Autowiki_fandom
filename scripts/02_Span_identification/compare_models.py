#!/usr/bin/env python3
"""Compare two systems (e.g. model vs baseline) with bootstrap significance test."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys_path = str(PROJECT_ROOT)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from src.span_identification.stats import bootstrap_significance


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare span_id systems with significance testing")
    parser.add_argument("--csv", type=str, default="data/research/span_id_experiments.csv")
    parser.add_argument("--run-id", type=str, help="Filter to this run_id")
    parser.add_argument("--baseline", type=str, default="rule_capitalized")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--metric", type=str, default="span_f1")
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if args.run_id and r.get("run_id") != args.run_id:
                continue
            rows.append(r)

    # Group by (domain, granularity)
    def key(r):
        return (r["domain"], r["granularity"])

    baseline_vals = {}  # (domain, gran) -> list of metric values
    model_vals = {}

    for r in rows:
        k = key(r)
        v = float(r.get(args.metric, 0) or 0)
        if r.get("model") == args.baseline and r.get("experiment_type") == "baseline":
            baseline_vals.setdefault(k, []).append(v)
        if r.get("model") == args.model and r.get("experiment_type") == "model":
            model_vals.setdefault(k, []).append(v)

    print(f"Comparing {args.model} vs {args.baseline} on {args.metric}")
    print(f"{'domain':<20} {'gran':<12} {'baseline':>10} {'model':>10} {'model_std':>10} {'p-value':>10}")
    print("-" * 75)

    for k in sorted(set(baseline_vals) | set(model_vals)):
        b_vals = baseline_vals.get(k, [0])
        m_vals = model_vals.get(k, [])
        b_mean = sum(b_vals) / len(b_vals)
        m_mean = sum(m_vals) / len(m_vals) if m_vals else 0
        m_std = (sum((x - m_mean) ** 2 for x in m_vals) / len(m_vals)) ** 0.5 if len(m_vals) > 1 else 0
        # Bootstrap p-value: compare model (over seeds) vs baseline
        # H0: no difference. Pad baseline to match model count for paired bootstrap.
        if len(m_vals) >= 2 and len(b_vals) >= 1:
            b_padded = (b_vals * ((len(m_vals) // len(b_vals)) + 1))[:len(m_vals)]
            p = bootstrap_significance(b_padded, m_vals)
        else:
            p = float("nan")
        print(f"{k[0]:<20} {k[1]:<12} {b_mean:>10.4f} {m_mean:>10.4f} {m_std:>10.4f} {p:>10.4f}")


if __name__ == "__main__":
    main()
