"""Visualise linking pipeline results.

Reads data/research/linking_experiments.csv and produces:
  Plot 1 — Linking F1 vs Span F1 vs Entity Accuracy bar chart
  Plot 2 — Precision / Recall / F1 breakdown (linking stage)
  Plot 3 — NIL threshold vs Linking F1 curve (if multiple thresholds run)
  Plot 4 — Coverage and NIL rate bar chart

Run:
  python scripts/04_Linking_pipeline/visualise_linking.py
  python scripts/04_Linking_pipeline/visualise_linking.py --no-show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

import linking_pipeline.config_utils as cu

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("tab10")


def load_data(csv_path: Path, domain: str | None) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"Research CSV not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    if domain:
        df = df[df["domain"] == domain]
    if df.empty:
        print("No data found.")
        sys.exit(1)
    for col in ["linking_f1", "span_f1", "entity_accuracy",
                "linking_precision", "linking_recall",
                "nil_rate", "coverage", "nil_threshold"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save(fig, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def plot_metric_comparison(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    """Bar chart: Linking F1 vs Span F1 vs Entity Accuracy per run."""
    metrics = ["linking_f1", "span_f1", "entity_accuracy"]
    labels  = ["Linking F1", "Span F1", "Entity Accuracy"]
    row     = df.iloc[0]  # most recent run if multiple

    fig, ax = plt.subplots(figsize=(7, 4))
    vals   = [row.get(m, 0) for m in metrics]
    colors = [PALETTE[i] for i in range(len(metrics))]
    bars   = ax.bar(labels, vals, color=colors, width=0.5)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11,
        )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.set_ylim(0, 1.1)
    ax.set_title(
        f"Task 3 Linking — {row.get('domain', '')} | "
        f"retriever: {str(row.get('retriever', '')).split('/')[-1]} | "
        f"stage: {row.get('stage', '')} | v{row.get('query_version', '')}",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out_dir / "01_metric_comparison.png", show)


def plot_precision_recall(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    """Grouped bar: Precision / Recall / F1 for linking and span."""
    row = df.iloc[0]
    categories = ["Linking", "Span"]
    precision  = [row.get("linking_precision", 0), row.get("span_precision", 0)]
    recall     = [row.get("linking_recall", 0),    row.get("span_recall", 0)]
    f1         = [row.get("linking_f1", 0),        row.get("span_f1", 0)]

    x     = range(len(categories))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width for i in x], precision, width, label="Precision", color=PALETTE[0])
    ax.bar([i          for i in x], recall,   width, label="Recall",    color=PALETTE[1])
    ax.bar([i + width  for i in x], f1,       width, label="F1",        color=PALETTE[2])

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_title("Precision / Recall / F1 Breakdown")
    fig.tight_layout()
    save(fig, out_dir / "02_precision_recall.png", show)


def plot_nil_threshold_curve(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    """Linking F1 vs NIL threshold (only useful when multiple thresholds were run)."""
    if df["nil_threshold"].nunique() < 2:
        print("  [skip] only one NIL threshold in data — run ablation to see curve")
        return

    grouped = df.groupby("nil_threshold")[["linking_f1", "linking_precision", "linking_recall"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grouped.index, grouped["linking_f1"],        marker="o", label="F1",        color=PALETTE[2])
    ax.plot(grouped.index, grouped["linking_precision"], marker="s", label="Precision", color=PALETTE[0])
    ax.plot(grouped.index, grouped["linking_recall"],    marker="^", label="Recall",    color=PALETTE[1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.set_xlabel("NIL Threshold")
    ax.set_title("NIL Threshold vs Linking Metrics")
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir / "03_nil_threshold_curve.png", show)


def plot_coverage_nil(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    """Coverage and NIL rate bar chart."""
    row = df.iloc[0]
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics = ["coverage", "nil_rate"]
    labels  = ["Coverage\n(Task 2 hit rate)", "NIL Rate\n(below threshold)"]
    vals    = [row.get(m, 0) for m in metrics]
    ax.bar(labels, vals, color=[PALETTE[2], PALETTE[3]], width=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.set_ylim(0, 1.1)
    ax.set_title("Coverage and NIL Rate")
    fig.tight_layout()
    save(fig, out_dir / "04_coverage_nil_rate.png", show)


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("  LINKING PIPELINE RESULTS")
    print("=" * 80)
    print(f"{'domain':<18} {'retriever':<30} {'stage':<12} {'v':>3} {'nil':>5}  "
          f"{'Link F1':>8} {'Span F1':>8} {'EntAcc':>8} {'Cov':>6}")
    print("-" * 80)
    for _, row in df.iterrows():
        ret = str(row.get("retriever", "")).split("/")[-1][:29]
        print(
            f"{str(row.get('domain','')):<18} {ret:<30} "
            f"{str(row.get('stage','')):<12} {str(row.get('query_version','?')):>3} "
            f"{float(row.get('nil_threshold', 0)):>5.2f}  "
            f"{float(row.get('linking_f1', 0)):>8.3f} "
            f"{float(row.get('span_f1', 0)):>8.3f} "
            f"{float(row.get('entity_accuracy', 0)):>8.3f} "
            f"{float(row.get('coverage', 0)):>6.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise linking pipeline results.")
    parser.add_argument("--config", default="configs/linking.yaml")
    parser.add_argument("--domain", help="Filter to a single domain.")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    config   = cu.load_config(ROOT / args.config)
    csv_path = cu.get_research_csv_path(config)
    domain   = args.domain or config.get("domains", [None])[0]
    out_dir  = Path(config.get("linking_dir", "data/linking")) / "plots"
    show     = not args.no_show

    df = load_data(csv_path, domain)
    print_summary(df)

    print(f"\nGenerating plots → {out_dir}/")
    plot_metric_comparison(df, out_dir, show)
    plot_precision_recall(df, out_dir, show)
    plot_nil_threshold_curve(df, out_dir, show)
    plot_coverage_nil(df, out_dir, show)
    print(f"\nAll plots saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
