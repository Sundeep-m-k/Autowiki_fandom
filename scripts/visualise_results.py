"""
Visualise Money-Heist experiment results for a research slide.
Dark-theme figure optimised for PowerPoint / conference slides.
  Left  – Span Identification: baselines vs BERT-base (sentence granularity)
  Right – Article Retrieval:   Recall@k curves by retriever
"""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ── Load data ──────────────────────────────────────────────────────────────────
df_span = pd.read_csv("/home/sundeep/Autowiki_fandom/data/research/wikipedia/span_id_experiments.csv")
df_ret  = pd.read_csv("/home/sundeep/Autowiki_fandom/data/research/wikipedia/article_retrieval_experiments.csv")

# ── Palette (Tailwind-inspired, all vibrant on dark) ──────────────────────────
BG        = "#0f172a"   # slate-900
PANEL_BG  = "#1e293b"   # slate-800
GRID_C    = "#334155"   # slate-700
TEXT      = "#f1f5f9"   # slate-100
SUBTEXT   = "#94a3b8"   # slate-400

C_BASELINE = "#60a5fa"  # blue-400
C_BERT     = "#fb923c"  # orange-400

RETRIEVER_COLORS = {
    "MPNet-base-v2":       "#a78bfa",   # violet-400
    "MiniLM-L6-v2":        "#34d399",   # emerald-400
    "MS-MARCO DistilBERT": "#38bdf8",   # sky-400
    "TF-IDF":              "#fbbf24",   # amber-400
    "BM25":                "#f87171",   # red-400
    "RoBERTa-base":        "#94a3b8",   # slate-400
}

RETRIEVER_MAP = {
    "bm25":                                             "BM25",
    "tfidf":                                            "TF-IDF",
    "roberta-base":                                     "RoBERTa-base",
    "sentence-transformers/all-MiniLM-L6-v2":           "MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2":          "MPNet-base-v2",
    "sentence-transformers/msmarco-distilbert-base-v4": "MS-MARCO DistilBERT",
}

RETRIEVER_ORDER = [
    "MPNet-base-v2", "MiniLM-L6-v2", "MS-MARCO DistilBERT",
    "TF-IDF", "BM25", "RoBERTa-base",
]

# ── Prepare span-ID data ───────────────────────────────────────────────────────
# Best heuristic baseline per granularity
baselines = (
    df_span[df_span["experiment_type"] == "baseline"]
    .groupby(["granularity", "model"])["span_f1"].mean().reset_index()
)
best_bl = baselines.groupby("granularity")["span_f1"].max()

# All three sentence-level baselines (for the focused sentence panel)
GRAN_ORDER  = ["sentence", "paragraph", "article"]
GRAN_LABELS = ["Sentence", "Paragraph", "Article"]

best_baseline = best_bl.reindex(GRAN_ORDER).values

# BERT – sentence only
bert = df_span[
    (df_span["experiment_type"] == "model") &
    (df_span["granularity"] == "sentence")
]
bert_mean = bert["span_f1"].mean()
bert_std  = bert["span_f1"].std()

# ── Prepare retrieval data ─────────────────────────────────────────────────────
ret = df_ret[df_ret["stage"] == "retrieval"].copy()
ret["label"] = ret["retriever"].map(RETRIEVER_MAP)

recall_cols = ["recall_at_1","recall_at_3","recall_at_5",
               "recall_at_10","recall_at_20","recall_at_50","recall_at_100"]
k_values = [1, 3, 5, 10, 20, 50, 100]

ret_mean = ret.groupby("label")[recall_cols].mean()

# ── Global matplotlib style ────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     PANEL_BG,
    "axes.edgecolor":     PANEL_BG,
    "axes.labelcolor":    TEXT,
    "axes.titlecolor":    TEXT,
    "xtick.color":        SUBTEXT,
    "ytick.color":        SUBTEXT,
    "text.color":         TEXT,
    "grid.color":         GRID_C,
    "grid.linewidth":     0.8,
    "grid.linestyle":     "--",
    "legend.facecolor":   PANEL_BG,
    "legend.edgecolor":   GRID_C,
    "legend.labelcolor":  TEXT,
    "font.family":        "DejaVu Sans",
    "font.size":          12,
})

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 6.2), facecolor=BG)

# Slight padding and shared title band
ax1 = fig.add_axes([0.06, 0.13, 0.38, 0.72])
ax2 = fig.add_axes([0.57, 0.13, 0.40, 0.72])

# ── Helper: style an axis ──────────────────────────────────────────────────────
def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=12)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=8, color=SUBTEXT)
    ax.set_ylabel(ylabel, fontsize=13, labelpad=8)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14, color=TEXT)
    ax.yaxis.grid(True, color=GRID_C, linestyle="--", linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 – Span Identification
# ══════════════════════════════════════════════════════════════════════════════
style_ax(ax1, "Span Identification", "", "Span F1")

x     = np.arange(len(GRAN_ORDER))
BAR_W = 0.32
GAP   = 0.04

# Baseline bars
b1 = ax1.bar(
    x - BAR_W/2 - GAP/2, best_baseline, BAR_W,
    color=C_BASELINE, alpha=0.9, zorder=3,
    label="Best heuristic baseline"
)

# BERT bar (sentence only)
b2 = ax1.bar(
    x[0] + BAR_W/2 + GAP/2, bert_mean, BAR_W,
    color=C_BERT, alpha=0.95, zorder=3,
    label="BERT-base-uncased  (BIO/BILOU, 3 seeds)"
)

# Error bar on BERT sentence
ax1.errorbar(
    x[0] + BAR_W/2 + GAP/2, bert_mean, yerr=bert_std,
    fmt="none", color="white", capsize=5, linewidth=1.8, zorder=5
)

# Value labels
for bar in b1:
    h = bar.get_height()
    if h > 0.001:
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.018,
                 f"{h:.1%}", ha="center", va="bottom",
                 fontsize=11, color=C_BASELINE, fontweight="bold")

ax1.text(
    x[0] + BAR_W/2 + GAP/2,
    bert_mean + bert_std + 0.022,
    f"{bert_mean:.1%}",
    ha="center", va="bottom",
    fontsize=11, color=C_BERT, fontweight="bold"
)

ax1.set_xticks(x)
ax1.set_xticklabels(GRAN_LABELS, fontsize=13)
ax1.set_ylim(0, 0.92)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))

leg1 = ax1.legend(
    fontsize=10.5, loc="upper right",
    framealpha=0.0, handlelength=1.2,
    borderpad=0.6, labelspacing=0.5
)

# ── Callout annotation: "10× improvement" ─────────────────────────────────────
ax1.annotate(
    "~10× improvement\nover best heuristic",
    xy=(x[0] + BAR_W/2 + GAP/2, bert_mean),
    xytext=(x[0] - 0.55, bert_mean + 0.08),
    fontsize=10, color=TEXT,
    arrowprops=dict(arrowstyle="->", color=SUBTEXT, lw=1.4),
    ha="center",
    bbox=dict(boxstyle="round,pad=0.35", fc=BG, ec=GRID_C, alpha=0.85)
)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 – Article Retrieval Recall@k
# ══════════════════════════════════════════════════════════════════════════════
style_ax(ax2, "Article Retrieval", "k  (log scale)", "Recall@k")

MARKERS     = ["o", "s", "^", "D", "v", "P"]
LINE_STYLES = ["-", "-", "-", "--", "--", ":"]
LW_MAP      = [2.6, 2.6, 2.6, 1.8, 1.8, 1.6]

for i, label in enumerate(RETRIEVER_ORDER):
    if label not in ret_mean.index:
        continue
    vals  = ret_mean.loc[label, recall_cols].values
    color = RETRIEVER_COLORS[label]
    ax2.plot(
        k_values, vals,
        label=label,
        color=color,
        linestyle=LINE_STYLES[i],
        marker=MARKERS[i],
        markersize=7,
        linewidth=LW_MAP[i],
        markeredgewidth=0,
        zorder=4
    )
    # End-of-line label at k=100
    ax2.text(
        105, vals[-1],
        f" {vals[-1]:.0%}",
        va="center", fontsize=9.5, color=color
    )

ax2.set_xscale("log")
ax2.set_xlim(0.8, 160)
ax2.set_xticks(k_values)
ax2.set_xticklabels([str(k) for k in k_values], fontsize=12)
ax2.set_ylim(0, 1.05)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))

leg2 = ax2.legend(
    fontsize=10.5, loc="upper left",
    framealpha=0.0, handlelength=1.6,
    borderpad=0.6, labelspacing=0.55
)

# ── Divider between dense and sparse in legend ────────────────────────────────
# (visual grouping via subtle colour already handles it)

# ── Save ───────────────────────────────────────────────────────────────────────
out = "data/research/money-heist/results_slide.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
plt.show()
