"""Visualise article retrieval experiment results.

Reads data/research/article_retrieval_experiments.csv and produces:
  Plot 1  — Recall@K curves per retriever (retrieval stage, best version each)
  Plot 2  — Retrieval vs Reranking comparison (Recall@K curves, best config each)
  Plot 3  — MRR per retriever × stage (bar chart)
  Plot 4  — Per-version MRR heatmap (retriever × version, retrieval stage)
  Plot 5  — Reranking gain: reranking MRR − retrieval MRR per retriever

All plots are saved to data/article_retrieval/plots/ and also shown interactively
if a display is available (falls back to file-only on headless servers).

Run:
  python scripts/03_Article_retrieval/visualise_results.py
  python scripts/03_Article_retrieval/visualise_results.py --domain money-heist
  python scripts/03_Article_retrieval/visualise_results.py --no-show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")           # headless-safe; overridden below if display present
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

import article_retrieval.config_utils as cu

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("tab10")

K_COLS    = ["recall_at_1", "recall_at_3", "recall_at_5",
             "recall_at_10", "recall_at_20", "recall_at_50", "recall_at_100"]
K_LABELS  = [1, 3, 5, 10, 20, 50, 100]

# Short display names for long model paths
RETRIEVER_LABELS = {
    "bm25":                                             "BM25",
    "tfidf":                                            "TF-IDF",
    "sentence-transformers/all-mpnet-base-v2":          "all-mpnet-v2",
    "sentence-transformers/all-MiniLM-L6-v2":           "all-MiniLM-L6",
    "sentence-transformers/msmarco-distilbert-base-v4": "msmarco-distilbert",
    "roberta-base":                                     "RoBERTa-base",
}


def short(name: str) -> str:
    return RETRIEVER_LABELS.get(name, name.split("/")[-1])


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(csv_path: Path, domain: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if domain:
        df = df[df["domain"] == domain]
    if df.empty:
        print(f"No data found in {csv_path}" + (f" for domain={domain}" if domain else ""))
        sys.exit(1)
    df["retriever_short"] = df["retriever"].map(lambda x: short(x))
    for col in K_COLS + ["mrr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def best_version_per_retriever(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """For each retriever, pick the version with the highest MRR."""
    sub = df[df["stage"] == stage]
    idx = sub.groupby("retriever")["mrr"].idxmax()
    return sub.loc[idx].reset_index(drop=True)


# ── Plot helpers ───────────────────────────────────────────────────────────────

def save(fig: plt.Figure, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


# ── Plot 1: Recall@K curves — retrieval stage, best version per retriever ──────

def plot_recall_curves_retrieval(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    best = best_version_per_retriever(df, "retrieval")
    if best.empty:
        print("  [skip] no retrieval stage rows"); return

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (_, row) in enumerate(best.iterrows()):
        vals = [row[c] for c in K_COLS]
        ax.plot(K_LABELS, vals, marker="o", label=f"{row['retriever_short']} (v{int(row['version'])})",
                color=PALETTE[i % len(PALETTE)], linewidth=2)

    ax.set_xscale("log")
    ax.set_xticks(K_LABELS)
    ax.set_xticklabels(K_LABELS)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title("Recall@K — Retrieval Stage (best query version per retriever)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save(fig, out_dir / "01_recall_curves_retrieval.png", show)


# ── Plot 2: Retrieval vs Reranking — Recall@K curves ──────────────────────────

def plot_retrieval_vs_reranking(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    ret_best = best_version_per_retriever(df, "retrieval")
    rer_best = best_version_per_retriever(df, "reranking")
    if ret_best.empty:
        print("  [skip] no data for retrieval vs reranking"); return

    retrievers = ret_best["retriever"].tolist()
    n = len(retrievers)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), sharey=True)
    axes = np.array(axes).flatten()

    for i, retriever in enumerate(retrievers):
        ax = axes[i]
        ret_row = ret_best[ret_best["retriever"] == retriever]
        rer_row = rer_best[rer_best["retriever"] == retriever]

        if not ret_row.empty:
            vals = [ret_row.iloc[0][c] for c in K_COLS]
            ax.plot(K_LABELS, vals, marker="o", label=f"Retrieval (v{int(ret_row.iloc[0]['version'])})",
                    color=PALETTE[0], linewidth=2)
        if not rer_row.empty:
            vals = [rer_row.iloc[0][c] for c in K_COLS]
            ax.plot(K_LABELS, vals, marker="s", label=f"+ Reranking (v{int(rer_row.iloc[0]['version'])})",
                    color=PALETTE[1], linewidth=2, linestyle="--")

        ax.set_xscale("log")
        ax.set_xticks(K_LABELS)
        ax.set_xticklabels(K_LABELS, fontsize=8)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(short(retriever), fontsize=10)
        ax.set_xlabel("K")
        ax.set_ylabel("Recall@K")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Retrieval vs Reranking — Recall@K (best version per retriever)", fontsize=12)
    fig.tight_layout()
    save(fig, out_dir / "02_retrieval_vs_reranking.png", show)


# ── Plot 3: MRR bar chart — retriever × stage ──────────────────────────────────

def plot_mrr_bar(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    ret_best = best_version_per_retriever(df, "retrieval")[["retriever_short", "mrr"]].copy()
    ret_best["stage"] = "Retrieval"
    rer_best = best_version_per_retriever(df, "reranking")[["retriever_short", "mrr"]].copy()
    rer_best["stage"] = "Reranking"
    combined = pd.concat([ret_best, rer_best], ignore_index=True)
    if combined.empty:
        print("  [skip] no MRR data"); return

    order = ret_best.sort_values("mrr", ascending=False)["retriever_short"].tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=combined, x="retriever_short", y="mrr", hue="stage",
        order=order, palette=[PALETTE[0], PALETTE[1]], ax=ax,
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.set_xlabel("Retriever")
    ax.set_ylabel("MRR (best version)")
    ax.set_title("MRR — Best Version per Retriever × Stage")
    ax.set_ylim(0, min(1.05, combined["mrr"].max() * 1.3))
    ax.legend(title="Stage")

    # Annotate bar values
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2, h + 0.005),
                        ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save(fig, out_dir / "03_mrr_bar.png", show)


# ── Plot 4: Per-version MRR heatmap (retrieval stage) ─────────────────────────

def plot_version_heatmap(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    sub = df[df["stage"] == "retrieval"]
    if sub.empty:
        print("  [skip] no retrieval rows for heatmap"); return

    pivot = sub.pivot_table(index="retriever_short", columns="version", values="mrr", aggfunc="mean")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 0.7), max(4, len(pivot) * 0.8)))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.4, ax=ax, cbar_kws={"label": "MRR"},
        vmin=0, vmax=pivot.values.max(),
    )
    ax.set_xlabel("Query Version")
    ax.set_ylabel("Retriever")
    ax.set_title("MRR per Query Version × Retriever (Retrieval Stage)")
    fig.tight_layout()
    save(fig, out_dir / "04_version_mrr_heatmap.png", show)


# ── Plot 5: Reranking gain ─────────────────────────────────────────────────────

def plot_reranking_gain(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    ret = best_version_per_retriever(df, "retrieval").set_index("retriever")[K_COLS + ["mrr"]]
    rer = best_version_per_retriever(df, "reranking").set_index("retriever")[K_COLS + ["mrr"]]
    common = ret.index.intersection(rer.index)
    if len(common) == 0:
        print("  [skip] no common retrievers for gain plot"); return

    gain_mrr = (rer.loc[common, "mrr"] - ret.loc[common, "mrr"]).reset_index()
    gain_mrr.columns = ["retriever", "mrr_gain"]
    gain_mrr["retriever_short"] = gain_mrr["retriever"].map(lambda x: short(x))
    gain_mrr = gain_mrr.sort_values("mrr_gain", ascending=False)

    # Also compute Recall@K gain
    gain_k = pd.DataFrame(index=common)
    for c in K_COLS:
        gain_k[c] = rer.loc[common, c] - ret.loc[common, c]
    gain_k.index = [short(r) for r in gain_k.index]
    gain_k.columns = K_LABELS

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar: MRR gain
    colors = [PALETTE[2] if g >= 0 else PALETTE[3] for g in gain_mrr["mrr_gain"]]
    ax1.barh(gain_mrr["retriever_short"], gain_mrr["mrr_gain"], color=colors)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax1.set_xlabel("MRR Gain (Reranking − Retrieval)")
    ax1.set_title("MRR Gain from Cross-Encoder Reranking")
    for i, (_, row) in enumerate(gain_mrr.iterrows()):
        ax1.annotate(f"{row['mrr_gain']:+.3f}",
                     (row["mrr_gain"] + (0.002 if row["mrr_gain"] >= 0 else -0.002), i),
                     va="center", ha="left" if row["mrr_gain"] >= 0 else "right", fontsize=9)

    # Heatmap: Recall@K gain
    sns.heatmap(
        gain_k, annot=True, fmt="+.2f", cmap="RdYlGn",
        center=0, linewidths=0.4, ax=ax2,
        cbar_kws={"label": "Recall gain"},
    )
    ax2.set_xlabel("K")
    ax2.set_ylabel("Retriever")
    ax2.set_title("Recall@K Gain from Reranking")

    fig.suptitle("Reranking Gain (Cross-Encoder over Best Retrieval)", fontsize=12)
    fig.tight_layout()
    save(fig, out_dir / "05_reranking_gain.png", show)


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("  BEST RESULT PER RETRIEVER (across all versions)")
    print("=" * 90)
    print(f"{'Retriever':<28} {'Stage':<12} {'Best V':>6}  "
          f"{'R@1':>6} {'R@5':>6} {'R@10':>7} {'R@100':>7} {'MRR':>7}")
    print("-" * 90)

    for stage in ["retrieval", "reranking"]:
        best = best_version_per_retriever(df, stage)
        for _, row in best.sort_values("mrr", ascending=False).iterrows():
            print(
                f"{row['retriever_short']:<28} {stage:<12} {int(row['version']):>6}  "
                f"{row['recall_at_1']:>6.3f} {row['recall_at_5']:>6.3f} "
                f"{row['recall_at_10']:>7.3f} {row['recall_at_100']:>7.3f} {row['mrr']:>7.3f}"
            )
        print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise article retrieval results.")
    parser.add_argument("--config", default="configs/article_retrieval/article_retrieval.yaml")
    parser.add_argument("--domain", help="Filter to a single domain.")
    parser.add_argument("--no-show", action="store_true", help="Save only, do not show plots.")
    args = parser.parse_args()

    config   = cu.resolve_config(cu.load_config(ROOT / args.config))
    domain   = args.domain or config.get("domains", [None])[0]
    csv_path = cu.get_research_csv_path(config, domain)
    out_dir  = Path(config.get("article_retrieval_dir", "data/article_retrieval")) / domain / "plots"
    show     = not args.no_show

    print(f"\nLoading results: {csv_path}")
    df = load_data(csv_path, domain)
    print(f"Rows: {len(df)} | Domain: {domain} | "
          f"Retrievers: {df['retriever_short'].nunique()} | "
          f"Versions: {df['version'].nunique()}")

    print_summary(df)

    print(f"\nGenerating plots → {out_dir}/")
    plot_recall_curves_retrieval(df, out_dir, show)
    plot_retrieval_vs_reranking(df, out_dir, show)
    plot_mrr_bar(df, out_dir, show)
    plot_version_heatmap(df, out_dir, show)
    plot_reranking_gain(df, out_dir, show)

    print(f"\nAll plots saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
