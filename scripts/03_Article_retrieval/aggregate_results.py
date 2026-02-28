"""Aggregate and display article retrieval experiment results.

Reads the research CSV and prints a summary table grouped by
(domain, stage, retriever, version) ranked by MRR.

Run:
  python scripts/03_Article_retrieval/aggregate_results.py
  python scripts/03_Article_retrieval/aggregate_results.py --stage retrieval --top 20
  python scripts/03_Article_retrieval/aggregate_results.py --domain money-heist
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import article_retrieval.config_utils as cu


def load_research_csv(path: Path) -> list[dict]:
    """Load rows from a single CSV, or from all domain subdirectories if path doesn't exist."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    # Try loading from per-domain subdirectories
    csv_name = path.name
    parent = path.parent
    rows: list[dict] = []
    if parent.exists():
        for domain_dir in sorted(parent.iterdir()):
            domain_csv = domain_dir / csv_name
            if domain_dir.is_dir() and domain_csv.exists():
                with open(domain_csv, encoding="utf-8") as f:
                    rows.extend(csv.DictReader(f))
    if not rows:
        print(f"Research CSV not found: {path}")
    return rows


def print_table(rows: list[dict], title: str, top: int = 30) -> None:
    if not rows:
        print(f"\n{title}: (no results)")
        return

    rows = sorted(rows, key=lambda r: float(r.get("mrr", 0)), reverse=True)[:top]
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    header = (
        f"{'domain':<18} {'stage':<12} {'retriever':<45} "
        f"{'v':>3}  {'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@100':>7} {'MRR':>7}"
    )
    print(header)
    print("-" * 100)
    for r in rows:
        retriever = r.get("retriever", "")[:44]
        print(
            f"{r.get('domain',''):<18} {r.get('stage',''):<12} {retriever:<45} "
            f"{r.get('version','?'):>3}  "
            f"{float(r.get('recall_at_1', 0)):>6.3f} "
            f"{float(r.get('recall_at_5', 0)):>6.3f} "
            f"{float(r.get('recall_at_10', 0)):>6.3f} "
            f"{float(r.get('recall_at_100', 0)):>7.3f} "
            f"{float(r.get('mrr', 0)):>7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate article retrieval results.")
    parser.add_argument("--config", default="configs/article_retrieval/article_retrieval.yaml")
    parser.add_argument("--domain", help="Filter by domain.")
    parser.add_argument(
        "--stage",
        choices=["retrieval", "reranking", "all"],
        default="all",
    )
    parser.add_argument("--top", type=int, default=30, help="Number of top rows to print.")
    args = parser.parse_args()

    config   = cu.resolve_config(cu.load_config(ROOT / args.config))
    csv_path = cu.get_research_csv_path(config)
    rows     = load_research_csv(csv_path)

    if not rows:
        sys.exit(0)

    if args.domain:
        rows = [r for r in rows if r.get("domain") == args.domain]

    retrieval_rows  = [r for r in rows if r.get("stage") == "retrieval"]
    reranking_rows  = [r for r in rows if r.get("stage") == "reranking"]

    if args.stage in ("retrieval", "all"):
        print_table(retrieval_rows, "Retrieval Results (ranked by MRR)", top=args.top)

    if args.stage in ("reranking", "all"):
        print_table(reranking_rows, "Reranking Results (ranked by MRR)", top=args.top)

    # Best per retriever summary
    if args.stage == "all":
        print(f"\n{'=' * 100}")
        print("  Best Result Per Retriever (across all versions, ranked by MRR)")
        print(f"{'=' * 100}")
        best: dict[str, dict] = {}
        for r in rows:
            key = (r.get("domain"), r.get("stage"), r.get("retriever"))
            if key not in best or float(r.get("mrr", 0)) > float(best[key].get("mrr", 0)):
                best[key] = r
        print_table(list(best.values()), "Best Per Retriever", top=args.top)


if __name__ == "__main__":
    main()
