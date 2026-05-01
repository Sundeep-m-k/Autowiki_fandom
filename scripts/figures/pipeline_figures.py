"""
Generate 4 publication-quality pipeline figures for ICWSM paper.
Outputs PDF + PNG at 300 DPI for each figure.

Usage:
    python scripts/figures/pipeline_figures.py
    python scripts/figures/pipeline_figures.py --outdir docs/figures
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Shared style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})

# Palette (colorblind-friendly, prints well in greyscale)
C = {
    "blue":    "#2166AC",
    "lblue":   "#74ADD1",
    "green":   "#1A9850",
    "lgreen":  "#A6D96A",
    "orange":  "#F46D43",
    "lorange": "#FDAE61",
    "purple":  "#762A83",
    "lpurple": "#C2A5CF",
    "teal":    "#006D77",
    "lteal":   "#83C5BE",
    "red":     "#D73027",
    "grey":    "#BABABA",
    "dgrey":   "#4D4D4D",
    "bg":      "#F7F7F7",
    "white":   "#FFFFFF",
}

ARROW_KW = dict(
    arrowstyle="-|>",
    color=C["dgrey"],
    lw=1.2,
    mutation_scale=10,
    connectionstyle="arc3,rad=0.0",
)


# ── Drawing helpers ────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, sublabel=None,
        fc=C["lblue"], ec=C["blue"], lw=1.2,
        fontsize=7.5, bold=False, radius=0.04):
    """Draw a rounded rectangle with centered label (and optional sublabel)."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    yo = 0 if sublabel is None else h * 0.13
    ax.text(x, y + yo, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=C["dgrey"], zorder=4)
    if sublabel:
        ax.text(x, y - yo * 1.6, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color="#555555", zorder=4,
                style="italic")


def data_box(ax, x, y, w, h, label, sublabel=None,
             fc=C["bg"], ec=C["grey"], fontsize=7):
    """A cylinder-like data store box (just a rectangle with dashed border)."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=1.0,
        linestyle="--", zorder=3,
    )
    ax.add_patch(patch)
    yo = 0 if sublabel is None else h * 0.14
    ax.text(x, y + yo, label, ha="center", va="center",
            fontsize=fontsize, color=C["dgrey"], zorder=4)
    if sublabel:
        ax.text(x, y - yo * 1.6, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color="#777777", zorder=4,
                style="italic")


def arrow(ax, x0, y0, x1, y1, label=None, color=C["dgrey"], rad=0.0):
    ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color, lw=1.1,
                    mutation_scale=9,
                    connectionstyle=f"arc3,rad={rad}",
                ),
                zorder=5)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.02, my, label, fontsize=6, color="#555555",
                ha="left", va="center", zorder=6)


def section_label(ax, x, y, text):
    ax.text(x, y, text, fontsize=7, color=C["dgrey"],
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=C["bg"],
                      ec=C["grey"], lw=0.8))


def save(fig, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Dataset Processing Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def fig_dataset_processing(outdir):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.set_axis_off()
    ax.set_facecolor(C["white"])
    fig.patch.set_facecolor(C["white"])

    # ── Title ──────────────────────────────────────────────────────────────
    ax.text(5, 5.7, "Dataset Processing Pipeline",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=C["dgrey"])

    # ── Source nodes (left column) ─────────────────────────────────────────
    # Fandom source
    box(ax, 1.4, 4.6, 2.0, 0.55, "Fandom Wikis",
        sublabel="MediaWiki API",
        fc=C["lteal"], ec=C["teal"], bold=True, fontsize=7.5)
    # Wikipedia source
    box(ax, 1.4, 3.3, 2.0, 0.55, "Wikipedia",
        sublabel="XML Dump (bz2)",
        fc=C["lteal"], ec=C["teal"], bold=True, fontsize=7.5)

    # ── Scraping / Parsing ─────────────────────────────────────────────────
    box(ax, 4.1, 4.6, 2.2, 0.55, "Web Scraper",
        sublabel="00_scrape_fandom.py",
        fc=C["lorange"], ec=C["orange"], fontsize=7.5)
    box(ax, 4.1, 3.3, 2.2, 0.55, "XML Dump Parser",
        sublabel="00_parse_wikipedia_dump.py",
        fc=C["lorange"], ec=C["orange"], fontsize=7.5)

    # ── Raw data store ─────────────────────────────────────────────────────
    data_box(ax, 6.8, 4.6, 1.8, 0.50,
             "data/raw/<domain>/",
             sublabel="HTML + plain text")
    data_box(ax, 6.8, 3.3, 1.8, 0.50,
             "data/raw/wikipedia/",
             sublabel="parsed JSONL")

    # ── Ground truth builder ───────────────────────────────────────────────
    box(ax, 5.1, 2.05, 2.4, 0.55, "Ground Truth Builder",
        sublabel="01_build_ground_truth.py",
        fc=C["lgreen"], ec=C["green"], bold=True, fontsize=7.5)

    # ── Output JSONL files ─────────────────────────────────────────────────
    y_out = 0.85
    xs = [2.2, 5.0, 7.8]
    labels  = ["articles_\npage_granularity", "paragraphs_\n<domain>", "sentences_\n<domain>"]
    for xo, lb in zip(xs, labels):
        data_box(ax, xo, y_out, 2.3, 0.65, lb + "\n.jsonl",
                 fc="#EEF5EA", ec=C["lgreen"])

    # ── Arrows: sources → scrapers ─────────────────────────────────────────
    arrow(ax, 2.4, 4.6, 3.0, 4.6)
    arrow(ax, 2.4, 3.3, 3.0, 3.3)

    # ── Arrows: scrapers → raw stores ──────────────────────────────────────
    arrow(ax, 5.2, 4.6, 5.9, 4.6)
    arrow(ax, 5.2, 3.3, 5.9, 3.3)

    # ── Arrows: raw stores → ground truth builder ─────────────────────────
    arrow(ax, 6.8, 4.35, 6.4, 2.35, rad=-0.25)
    arrow(ax, 6.8, 3.05, 6.4, 2.35, rad=0.1)

    # ── Arrows: GT builder → output files ─────────────────────────────────
    for xo in xs:
        arrow(ax, 5.1, 1.77, xo, 1.17)

    # ── Annotation: link extraction ────────────────────────────────────────
    ax.text(5.1, 1.55, "extract links · character offsets · split granularity",
            ha="center", va="center", fontsize=6.2, color="#555555", style="italic")

    # ── Legend note ────────────────────────────────────────────────────────
    legend_elems = [
        mpatches.Patch(fc=C["lteal"],   ec=C["teal"],   label="Data Source"),
        mpatches.Patch(fc=C["lorange"], ec=C["orange"], label="Processing Script"),
        mpatches.Patch(fc=C["lgreen"],  ec=C["green"],  label="Key Step"),
        mpatches.Patch(fc=C["bg"],      ec=C["grey"],   label="Data Store",
                       linestyle="--"),
    ]
    ax.legend(handles=legend_elems, loc="lower right",
              fontsize=6.5, framealpha=0.85,
              edgecolor=C["grey"], handlelength=1.2)

    save(fig, "fig1_dataset_processing", outdir)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Span Identification Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def fig_span_identification(outdir):
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.set_axis_off()
    fig.patch.set_facecolor(C["white"])

    ax.text(5, 6.65, "Span Identification Pipeline (Task 1)",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=C["dgrey"])

    # ── Input ──────────────────────────────────────────────────────────────
    data_box(ax, 5, 6.1, 4.0, 0.52,
             "data/processed/<domain>/  ·  paragraphs / sentences / articles .jsonl",
             fc="#EEF5EA", ec=C["lgreen"])

    # ── Preprocessing ──────────────────────────────────────────────────────
    box(ax, 2.5, 5.1, 2.6, 0.55, "Data Splits",
        sublabel="train 70% · val 15% · test 15%\n(stratified by article_id)",
        fc=C["lorange"], ec=C["orange"], fontsize=7)
    box(ax, 7.5, 5.1, 2.6, 0.55, "Tokenisation & Labelling",
        sublabel="BIO / BILOU encoding\nHuggingFace tokenizer",
        fc=C["lorange"], ec=C["orange"], fontsize=7)

    arrow(ax, 5, 5.84, 2.5, 5.38)    # input → splits
    arrow(ax, 5, 5.84, 7.5, 5.38)    # input → tokenise

    # ── Two parallel tracks ────────────────────────────────────────────────
    # Left: baselines
    ax.text(2.5, 4.6, "Rule-based Baselines", ha="center", fontsize=7.5,
            fontweight="bold", color=C["blue"])
    blines = [
        ("Capitalised-word Rule", 4.1),
        ("Heuristic Anchor Dict", 3.5),
        ("Random Span", 2.9),
    ]
    for lbl, yb in blines:
        box(ax, 2.5, yb, 2.3, 0.43, lbl,
            fc=C["lpurple"], ec=C["purple"], fontsize=7)
        arrow(ax, 2.5, 4.55 if yb == 4.1 else yb + 0.44, 2.5, yb + 0.22)

    # Right: neural models
    ax.text(7.5, 4.6, "Neural Sequence Labellers", ha="center", fontsize=7.5,
            fontweight="bold", color=C["blue"])
    models = [
        ("BERT-base-uncased",        4.1),
        ("DeBERTa-v3-base",          3.5),
        ("RoBERTa-base",             2.9),
        ("DistilBERT-base-uncased",  2.3),
    ]
    for lbl, ym in models:
        box(ax, 7.5, ym, 2.5, 0.43, lbl,
            fc=C["lblue"], ec=C["blue"], fontsize=7)
        arrow(ax, 7.5, 4.55 if ym == 4.1 else ym + 0.44, 7.5, ym + 0.22)

    arrow(ax, 2.5, 4.82, 2.5, 4.32)   # splits → baselines
    arrow(ax, 7.5, 4.82, 7.5, 4.32)   # tokenise → models

    # ── Evaluation ────────────────────────────────────────────────────────
    box(ax, 5, 1.65, 3.4, 0.52, "Evaluation",
        sublabel="Span F1 · Char F1 · Exact Match %",
        fc=C["lgreen"], ec=C["green"], bold=True, fontsize=7.5)

    arrow(ax, 2.5, 2.68, 3.3, 1.92)
    arrow(ax, 7.5, 2.08, 6.7, 1.92)

    # ── Output ────────────────────────────────────────────────────────────
    data_box(ax, 3.2, 0.72, 2.5, 0.52,
             "span_id_experiments.csv",
             sublabel="data/research/<domain>/",
             fc=C["bg"], ec=C["grey"])
    data_box(ax, 7.0, 0.72, 2.5, 0.52,
             "test split JSONL",
             sublabel="data/span_id/<domain>/splits/",
             fc=C["bg"], ec=C["grey"])

    arrow(ax, 5, 1.39, 3.2, 1.0)
    arrow(ax, 5, 1.39, 7.0, 1.0)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elems = [
        mpatches.Patch(fc=C["lorange"], ec=C["orange"], label="Data Prep"),
        mpatches.Patch(fc=C["lpurple"], ec=C["purple"], label="Baseline"),
        mpatches.Patch(fc=C["lblue"],   ec=C["blue"],   label="Neural Model"),
        mpatches.Patch(fc=C["lgreen"],  ec=C["green"],  label="Evaluation"),
        mpatches.Patch(fc=C["bg"],      ec=C["grey"],   label="Data Store",
                       linestyle="--"),
    ]
    ax.legend(handles=legend_elems, loc="lower right",
              fontsize=6.5, framealpha=0.85,
              edgecolor=C["grey"], handlelength=1.2)

    save(fig, "fig2_span_identification", outdir)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Article Retrieval Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def fig_article_retrieval(outdir):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.set_axis_off()
    fig.patch.set_facecolor(C["white"])

    ax.text(5, 7.65, "Article Retrieval Pipeline (Task 2)",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=C["dgrey"])

    # ── Input ──────────────────────────────────────────────────────────────
    data_box(ax, 5, 7.15, 5.5, 0.48,
             "data/processed/<domain>/articles_page_granularity_<domain>.jsonl",
             fc="#EEF5EA", ec=C["lgreen"])

    # ── Step 0: Build Article Index ────────────────────────────────────────
    box(ax, 5, 6.3, 4.0, 0.60, "Step 0 — Build Article Index",
        sublabel="BM25 (rank_bm25)  ·  TF-IDF (sklearn)  ·  FAISS + SentenceTransformers",
        fc=C["lorange"], ec=C["orange"], fontsize=7.5)
    arrow(ax, 5, 6.91, 5, 6.60)

    # ── Step 1: Query Dataset ──────────────────────────────────────────────
    box(ax, 5, 5.35, 4.2, 0.65, "Step 1 — Build Query Dataset",
        sublabel="24 query templates per anchor  ·  3 context modes\n"
                 "3 anchor preprocessings  ·  test-split anchors only",
        fc=C["lorange"], ec=C["orange"], fontsize=7.5)
    arrow(ax, 5, 6.0, 5, 5.67)

    # ── Step 2: Retrieval — three parallel columns ─────────────────────────
    ax.text(5, 4.75, "Step 2 — Retrieval", ha="center", fontsize=8,
            fontweight="bold", color=C["blue"])

    ret_items = [
        ("BM25",             C["lpurple"], C["purple"], 2.0),
        ("TF-IDF",           C["lpurple"], C["purple"], 4.2),
        ("Dense (FAISS)\n×4 models", C["lblue"],   C["blue"],   6.5),
        ("Source article\nexcluded",  "#E8E8E8",   C["grey"],   8.6),
    ]
    for lbl, fc, ec, xr in ret_items:
        box(ax, xr, 4.2, 1.65, 0.62, lbl,
            fc=fc, ec=ec, fontsize=7)
        arrow(ax, 5, 4.70, xr, 4.52)

    # ── Step 3: Re-ranking ─────────────────────────────────────────────────
    box(ax, 5, 3.2, 5.0, 0.68, "Step 3 — Re-ranking (Zero-shot)",
        sublabel="Cross-encoder  ·  top-K input: {5, 10, 20, 50}\n"
                 "ms-marco-MiniLM-L-6  ·  ms-marco-MiniLM-L-12  ·  deberta-v3-base",
        fc=C["lteal"], ec=C["teal"], fontsize=7.5)
    for _, _, _, xr in ret_items[:3]:
        arrow(ax, xr, 3.89, 4.6, 3.54)

    # ── Step 4: Optional fine-tune ─────────────────────────────────────────
    box(ax, 5, 2.18, 4.0, 0.60, "Step 4 — Fine-tune Reranker  (optional)",
        sublabel="mine (query, positive, hard-negative) triples\n"
                 "binary cross-entropy  ·  train split only",
        fc="#F0E6F6", ec=C["purple"], fontsize=7.5)
    arrow(ax, 5, 2.86, 5, 2.48)

    # ── Step 5: Evaluate ───────────────────────────────────────────────────
    box(ax, 5, 1.25, 3.6, 0.58, "Step 5 — Evaluate",
        sublabel="Recall@1/3/5/10/20/50/100  ·  MRR",
        fc=C["lgreen"], ec=C["green"], bold=True, fontsize=7.5)
    arrow(ax, 5, 1.88, 5, 1.54)

    # ── Output ────────────────────────────────────────────────────────────
    data_box(ax, 5, 0.4, 4.8, 0.52,
             "article_retrieval_experiments.csv",
             sublabel="data/research/<domain>/",
             fc=C["bg"], ec=C["grey"])
    arrow(ax, 5, 0.96, 5, 0.67)

    # ── Ablation bracket ──────────────────────────────────────────────────
    ax.annotate("", xy=(9.35, 5.0), xytext=(9.35, 1.0),
                arrowprops=dict(arrowstyle="-", color=C["grey"], lw=1.0))
    ax.text(9.6, 3.0, "11-dim\nablation\nsweep",
            ha="center", va="center", fontsize=6.2, color="#555555",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc=C["bg"], ec=C["grey"], lw=0.7))

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elems = [
        mpatches.Patch(fc=C["lorange"], ec=C["orange"],  label="Index / Query Build"),
        mpatches.Patch(fc=C["lpurple"], ec=C["purple"],  label="Sparse Retriever"),
        mpatches.Patch(fc=C["lblue"],   ec=C["blue"],    label="Dense Retriever"),
        mpatches.Patch(fc=C["lteal"],   ec=C["teal"],    label="Re-ranking"),
        mpatches.Patch(fc=C["lgreen"],  ec=C["green"],   label="Evaluation"),
        mpatches.Patch(fc=C["bg"],      ec=C["grey"],    label="Data Store",
                       linestyle="--"),
    ]
    ax.legend(handles=legend_elems, loc="lower left",
              fontsize=6.5, framealpha=0.85,
              edgecolor=C["grey"], handlelength=1.2)

    save(fig, "fig3_article_retrieval", outdir)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Linking Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def fig_linking_pipeline(outdir):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.2)
    ax.set_axis_off()
    fig.patch.set_facecolor(C["white"])

    ax.text(5, 6.9, "Linking Pipeline (Task 3)",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=C["dgrey"])

    # ── Two inputs at top ─────────────────────────────────────────────────
    data_box(ax, 2.2, 6.35, 3.4, 0.52,
             "Task 1 — Gold Spans",
             sublabel="data/span_id/<domain>/splits/test_*.jsonl",
             fc="#EEF5EA", ec=C["lgreen"])
    data_box(ax, 7.8, 6.35, 3.4, 0.52,
             "Task 2 — Retrieval Results",
             sublabel="data/article_retrieval/<domain>/reranking/…jsonl",
             fc="#EAF2FB", ec=C["blue"])

    # ── Span predictor ────────────────────────────────────────────────────
    box(ax, 2.2, 5.42, 3.0, 0.55, "Span Predictor",
        sublabel="load gold spans\nchar_start · char_end · anchor_text",
        fc=C["lorange"], ec=C["orange"], fontsize=7)
    arrow(ax, 2.2, 6.09, 2.2, 5.7)

    # ── Span-to-Query lookup ───────────────────────────────────────────────
    box(ax, 7.8, 5.42, 3.0, 0.55, "Span → Query Lookup",
        sublabel="match (article_id, anchor_text)\nto Task 2 top-1 result",
        fc=C["lorange"], ec=C["orange"], fontsize=7)
    arrow(ax, 7.8, 6.09, 7.8, 5.7)

    # ── Merge ─────────────────────────────────────────────────────────────
    box(ax, 5, 4.45, 3.2, 0.55, "Merge Spans + Predictions",
        sublabel="(source_article_id, anchor_text) key join",
        fc=C["lorange"], ec=C["orange"], fontsize=7.5)
    arrow(ax, 2.2, 5.15, 3.6, 4.72)
    arrow(ax, 7.8, 5.15, 6.4, 4.72)

    # ── NIL detector ─────────────────────────────────────────────────────
    box(ax, 5, 3.5, 3.0, 0.55, "NIL Detector",
        sublabel="score < threshold → linked = False\nthreshold sweep: {0.0, 0.1, 0.2, 0.5}",
        fc=C["lteal"], ec=C["teal"], fontsize=7)
    arrow(ax, 5, 4.17, 5, 3.77)

    # ── Overlap resolution ────────────────────────────────────────────────
    box(ax, 5, 2.57, 3.0, 0.55, "Overlap Resolution",
        sublabel="longest-span wins strategy",
        fc=C["lteal"], ec=C["teal"], fontsize=7)
    arrow(ax, 5, 3.22, 5, 2.84)

    # ── HTML Renderer ─────────────────────────────────────────────────────
    box(ax, 5, 1.63, 3.2, 0.56, "HTML Renderer",
        sublabel='inject  <a href="…fandom.com/wiki/<Page>">anchor</a>',
        fc=C["lorange"], ec=C["orange"], fontsize=7)
    arrow(ax, 5, 2.29, 5, 1.91)

    # ── Outputs ───────────────────────────────────────────────────────────
    data_box(ax, 2.1, 0.62, 2.8, 0.65,
             "Linked HTML",
             sublabel="data/linking/<domain>/html/",
             fc="#FFF8E1", ec="#F9A825")
    data_box(ax, 5.4, 0.62, 2.5, 0.65,
             "Predictions JSONL",
             sublabel="linking_results.jsonl",
             fc=C["bg"], ec=C["grey"])
    data_box(ax, 8.4, 0.62, 2.5, 0.65,
             "Metrics CSV",
             sublabel="linking_experiments.csv",
             fc=C["bg"], ec=C["grey"])

    arrow(ax, 5, 1.35, 2.1, 0.95)
    arrow(ax, 5, 1.35, 5.4, 0.95)
    arrow(ax, 5, 1.35, 8.4, 0.95)

    # ── Metrics bubble ────────────────────────────────────────────────────
    ax.text(8.4, 0.23,
            "Linking F1 · Span F1 · Entity Acc · NIL Rate · Coverage",
            ha="center", va="center", fontsize=6.0, color="#555555",
            style="italic")

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elems = [
        mpatches.Patch(fc=C["lorange"], ec=C["orange"],  label="Processing Step"),
        mpatches.Patch(fc=C["lteal"],   ec=C["teal"],    label="Filtering Step"),
        mpatches.Patch(fc=C["bg"],      ec=C["grey"],    label="Data Store",
                       linestyle="--"),
    ]
    ax.legend(handles=legend_elems, loc="lower left",
              fontsize=6.5, framealpha=0.85,
              edgecolor=C["grey"], handlelength=1.2)

    save(fig, "fig4_linking_pipeline", outdir)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="docs/figures",
                        help="Output directory (default: docs/figures)")
    args = parser.parse_args()
    outdir = args.outdir

    print(f"\nGenerating pipeline figures → {outdir}/\n")
    fig_dataset_processing(outdir)
    fig_span_identification(outdir)
    fig_article_retrieval(outdir)
    fig_linking_pipeline(outdir)
    print("\nDone. 4 figures × 2 formats = 8 files.")


if __name__ == "__main__":
    main()
