# Autowiki Fandom

Research pipeline for automated hyperlinking of Fandom wiki articles.
Given a plain-text wiki article, the system identifies which text spans should become
hyperlinks and retrieves the correct target article for each span — producing HTML
output with injected `<a href="…">` tags.

The pipeline is split into three independent, composable tasks:

| Task | Description | Entry Point |
|------|-------------|-------------|
| **Task 1 — Span Identification** | Token classification to find anchor spans | `scripts/02_Span_identification/` |
| **Task 2 — Article Retrieval** | IR pipeline to rank candidate target articles | `scripts/03_Article_retrieval/` |
| **Task 3 — Linking Pipeline** | End-to-end: spans → links → HTML output | `scripts/04_Linking_pipeline/` |

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Data Processing](#data-processing)
4. [Task 1 — Span Identification](#task-1--span-identification)
5. [Task 2 — Article Retrieval](#task-2--article-retrieval)
6. [Task 3 — Linking Pipeline](#task-3--linking-pipeline)
7. [Full Pipeline Orchestration](#full-pipeline-orchestration)
8. [Results and Statistics](#results-and-statistics)
9. [Configuration System](#configuration-system)
10. [Running Tests](#running-tests)

---

## Project Structure

```
Autowiki_fandom/
├── configs/                        # All YAML configuration files
│   ├── scraping.yaml               # Fandom scraper settings
│   ├── ground_truth.yaml           # Ground truth builder settings
│   ├── span_id_base.yaml           # Task 1 base config (all defaults)
│   ├── span_id.yaml                # Task 1 local/default run
│   ├── span_id_kudremukh.yaml      # Task 1 overrides for Kudremukh GPU server
│   ├── article_retrieval_base.yaml # Task 2 base config (all experiment dims)
│   ├── article_retrieval.yaml      # Task 2 local/default run
│   ├── article_retrieval_kudremukh.yaml  # Task 2 overrides for Kudremukh
│   ├── linking_base.yaml           # Task 3 base config
│   └── linking.yaml                # Task 3 local/default run
│
├── scripts/
│   ├── 01_Data_processing/
│   │   ├── 00_scrape_fandom.py     # Scrape HTML from Fandom
│   │   └── 01_build_ground_truth.py  # Build JSONL corpus with link annotations
│   ├── 02_Span_identification/
│   │   ├── 01_run_span_id.py       # Full sweep: baselines + models
│   │   └── 01_run_span_id_kudremukh.py  # Same, GPU-optimised
│   ├── 03_Article_retrieval/
│   │   ├── run_all.py              # Master script (auto-detects machine config)
│   │   ├── 00_build_article_index.py
│   │   ├── 01_build_query_dataset.py
│   │   ├── 02_run_retrieval.py
│   │   ├── 03_run_reranking.py
│   │   ├── 04_evaluate.py
│   │   └── visualise_results.py
│   ├── 04_Linking_pipeline/
│   │   ├── run_all.py
│   │   ├── 00_predict_and_link.py
│   │   ├── 01_render_html.py
│   │   ├── 02_evaluate.py
│   │   └── visualise_linking.py
│   ├── run_full_pipeline.py        # Master script for all 5 stages
│   └── summarise_corpus.py         # Cross-domain stats dashboard
│
├── src/
│   ├── data_scraping/              # Fandom HTML scraper
│   ├── span_identification/        # Task 1 library code
│   │   ├── preprocess.py           # Tokenisation and BIO/BILOU labelling
│   │   ├── hf_trainer.py           # HuggingFace Trainer wrapper
│   │   ├── span_metrics.py         # Span + seqeval metrics
│   │   ├── evaluator.py            # Character-level and exact-match metrics
│   │   ├── baselines.py            # Rule-based baselines
│   │   └── dataset.py              # Article-ID-based train/val/test splits
│   ├── article_retrieval/          # Task 2 library code
│   │   ├── query_builder.py        # 24 query variations per anchor span
│   │   ├── embedder.py             # Dense encoding (SentenceTransformers + FAISS)
│   │   ├── retriever.py            # BM25, TF-IDF, dense retrieval
│   │   └── reranker.py             # Cross-encoder re-ranking
│   ├── linking_pipeline/           # Task 3 library code
│   │   ├── span_predictor.py       # Load gold spans from Task 1
│   │   ├── span_to_query.py        # Match spans to Task 2 results
│   │   ├── nil_detector.py         # Threshold-based NIL filtering
│   │   └── html_renderer.py        # Inject <a> tags into HTML
│   └── utils/
│       └── stats_utils.py          # Centralized stats tracking per domain
│
├── data/
│   ├── processed/<domain>/         # Scraped + parsed JSONL files
│   ├── span_id/<domain>/splits/    # Train/val/test splits (article-ID-based)
│   ├── article_retrieval/<domain>/ # FAISS indexes, retrieval results, plots
│   ├── linking/<domain>/           # Linking predictions, HTML output, plots
│   ├── research/                   # Experiment result CSVs
│   │   ├── span_id_experiments.csv
│   │   ├── article_retrieval_experiments.csv
│   │   └── linking_experiments.csv
│   └── stats/<domain>.json         # Aggregated best results per domain
│
├── docs/
│   ├── architecture_span_id.md
│   ├── architecture_article_retrieval.md
│   └── architecture_linking_pipeline.md
│
├── tests/                          # Automated tests
├── requirements-span-id.txt
└── requirements-article-retrieval.txt
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd Autowiki_fandom
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
# Task 1 — Span Identification
pip install -r requirements-span-id.txt

# Task 2 — Article Retrieval (includes FAISS, sentence-transformers)
pip install -r requirements-article-retrieval.txt
```

Key dependencies:
- Python 3.10+, PyTorch 2.0+, Transformers 4.30+
- `sentence-transformers==3.4.1`, `faiss-cpu`, `rank_bm25`
- `seqeval`, `datasets`, `sentencepiece`, `protobuf`
- `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `pandas`

---

## Data Processing

### Step 0 — Scrape Fandom

```bash
python scripts/01_Data_processing/00_scrape_fandom.py \
    --config configs/scraping.yaml
```

Scrapes HTML from `scraping.yaml → start_url` (currently `beverlyhillscop.fandom.com`).
Saves raw HTML to `data/raw/<domain>/`.

### Step 1 — Build Ground Truth

```bash
python scripts/01_Data_processing/01_build_ground_truth.py \
    --config configs/ground_truth.yaml
```

Parses HTML, extracts internal links with character offsets, and writes:

| File | Description |
|------|-------------|
| `data/processed/<domain>/articles_page_granularity_<domain>.jsonl` | Full articles with link spans |
| `data/processed/<domain>/paragraphs_<domain>.jsonl` | Paragraph-level examples |
| `data/processed/<domain>/sentences_<domain>.jsonl` | Sentence-level examples |

Also updates `data/stats/<domain>.json` with dataset statistics (article count,
paragraph count, sentence count, link type breakdown, split sizes).

---

## Task 1 — Span Identification

**Goal:** Given a text unit (sentence / paragraph / article), predict which character
spans should become hyperlinks.

**Models:** `bert-base-uncased`, `microsoft/deberta-v3-base`, `SpanBERT/spanbert-base-cased`,
`roberta-base`, `distilbert-base-uncased`

**Labelling schemes:** BIO and BILOU (configured via `label_schemes` in config)

**Baselines:** `rule_capitalized`, `heuristic_anchor`, `random`

### Run (local)

```bash
python scripts/02_Span_identification/01_run_span_id.py
```

### Run (Kudremukh GPU server)

```bash
python scripts/02_Span_identification/01_run_span_id_kudremukh.py
```

### Configuration

```yaml
# configs/span_id_base.yaml (excerpt)
models: ["bert-base-uncased", "microsoft/deberta-v3-base", ...]
label_schemes: ["BIO", "BILOU"]
granularities: ["sentence", "paragraph", "article"]
seeds: [42, 123, 456]
training:
  epochs: 20
  learning_rate: 5.0e-5
  batch_size: 32
```

### Metrics (written to `data/research/span_id_experiments.csv`)

| Column | Description |
|--------|-------------|
| `span_f1` | Exact-boundary span F1 (primary metric) |
| `span_precision` | Exact-boundary span precision |
| `span_recall` | Exact-boundary span recall |
| `char_f1` | Overlap/relaxed span F1 (token-level proxy for character F1) |
| `exact_match_pct` | Fraction of gold spans exactly recalled, averaged over examples with ≥1 gold span |
| `val_span_f1` | Validation span F1 (used for model selection) |

### Architecture

See [`docs/architecture_span_id.md`](docs/architecture_span_id.md) for full details.

---

## Task 2 — Article Retrieval

**Goal:** Given an anchor span and its surrounding context, retrieve the correct
target article from the wiki corpus.

**Retrievers:**
- Sparse baselines: `BM25`, `TF-IDF`
- Dense bi-encoders: `all-mpnet-base-v2`, `all-MiniLM-L6-v2`,
  `msmarco-distilbert-base-v4`, `roberta-base`
- Re-ranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Query variations:** 24 query templates per anchor span (v1–v24), covering different
formulations of the anchor text and surrounding context.

### Run

```bash
# Auto-detects machine config (kudremukh → kudremukh yaml, else base yaml)
python scripts/03_Article_retrieval/run_all.py

# Force specific config
python scripts/03_Article_retrieval/run_all.py \
    --config configs/article_retrieval_kudremukh.yaml

# Single domain, skip indexing if already built
python scripts/03_Article_retrieval/run_all.py \
    --domain beverlyhillscop \
    --skip-index --skip-queries
```

### Pipeline steps

```
00_build_article_index.py   → BM25 / TF-IDF / FAISS indexes
01_build_query_dataset.py   → 24 query variations per anchor link
02_run_retrieval.py         → Top-K candidates per (retriever, version)
03_run_reranking.py         → Re-rank with cross-encoder
04_evaluate.py              → Recall@K and MRR, write research CSV
visualise_results.py        → Plots saved to data/article_retrieval/<domain>/plots/
```

### Experiment dimensions (11 total)

| # | Dimension | Values |
|---|-----------|--------|
| 1 | Query versions | v1–v24 |
| 2 | Retriever model | BM25, TF-IDF, 4 dense models |
| 3 | Corpus representation | title_only / title_lead / title_full |
| 4 | Query-side context | anchor_only / anchor_sentence / anchor_paragraph |
| 5 | Corpus granularity | article / paragraph / sentence |
| 6 | Re-ranker model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| 7 | Re-ranker input K | top_k_input (default 20) |
| 8 | Domain | beverlyhillscop, money-heist, … |
| 9 | Query sample size | n_sample (default 1000) |
| 10 | FAISS index type | flat / ivf / hnsw (Phase 2) |
| 11 | Anchor preprocessing | raw / lowercase / stopword_removed (Phase 2) |

### Metrics (written to `data/research/article_retrieval_experiments.csv`)

`Recall@1`, `Recall@3`, `Recall@5`, `Recall@10`, `Recall@20`, `Recall@50`,
`Recall@100`, `MRR`

### Architecture

See [`docs/architecture_article_retrieval.md`](docs/architecture_article_retrieval.md).

---

## Task 3 — Linking Pipeline

**Goal:** End-to-end pipeline. Input: plain text. Output: HTML with injected
`<a href="https://<domain>.fandom.com/wiki/<page>">` hyperlinks.

### How it works

```
Plain text
    │
    ▼ (Phase 1) span_predictor.py
Gold spans from Task 1 test split
    │
    ▼ (Phase 2) span_to_query.py
Match each span → pre-computed Task 2 result
using key: (source_article_id, char_start, char_end)
    │
    ▼ (Phase 3) nil_detector.py
Filter spans whose top-1 score < nil_threshold
    │
    ▼ (Phase 4) html_renderer.py
Inject <a href="…"> tags, resolve overlapping spans
    │
    ▼
Linked HTML output
```

Task 2 results are reused from disk — **no GPU required at linking time**.

### Run

```bash
python scripts/04_Linking_pipeline/run_all.py \
    --config configs/linking.yaml

# Ablation: sweep NIL thresholds
python scripts/04_Linking_pipeline/run_all.py \
    --nil-thresholds 0.0,0.1,0.2,0.5
```

### Pipeline steps

```
00_predict_and_link.py  → load spans, match to Task 2 results, apply NIL filter
01_render_html.py       → write HTML files with injected links
02_evaluate.py          → compute linking metrics, write research CSV
visualise_linking.py    → plots saved to data/linking/<domain>/plots/
```

### Metrics (written to `data/research/linking_experiments.csv`)

| Metric | Description |
|--------|-------------|
| `linking_f1` | Span boundary correct AND target article correct (primary) |
| `span_f1` | Span boundary correct (Task 1 contribution) |
| `entity_accuracy` | Target article correct, given span is correct (Task 2 contribution) |
| `nil_rate` | Fraction of spans filtered as NIL |
| `coverage` | Fraction of gold links that received any prediction |

### Architecture

See [`docs/architecture_linking_pipeline.md`](docs/architecture_linking_pipeline.md).

---

## Full Pipeline Orchestration

Run all 5 stages end-to-end:

```bash
python scripts/run_full_pipeline.py \
    --domain beverlyhillscop

# Skip stages already completed
python scripts/run_full_pipeline.py \
    --skip-scraping \
    --skip-ground-truth \
    --domain beverlyhillscop

# Force rebuild everything
python scripts/run_full_pipeline.py --force
```

---

## Results and Statistics

Each domain accumulates best results in `data/stats/<domain>.json`:

```json
{
  "domain": "beverlyhillscop",
  "dataset_stats": {
    "num_articles": 129,
    "num_paragraphs": 855,
    "num_sentences": 3064,
    "link_type_counts": { "internal": 974, "external": 9, "file": 117 },
    "avg_article_length_chars": 3702,
    "avg_internal_links_per_article": 7.55,
    "split_sizes": {
      "sentence":  { "train": 1874, "val": 742, "test": 448 },
      "paragraph": { "train": 572,  "val": 157, "test": 126 },
      "article":   { "train": 90,   "val": 19,  "test": 20  }
    }
  },
  "span_id": { "overall_best": { ... } },
  "article_retrieval": { "best_retrieval": { ... }, "best_reranking": { ... } },
  "linking_pipeline": { "best": { ... } }
}
```

Print a cross-domain dashboard:

```bash
# Human-readable tables
python scripts/summarise_corpus.py

# Single domain
python scripts/summarise_corpus.py --domain beverlyhillscop

# Save machine-readable summary
python scripts/summarise_corpus.py --save
```

---

## Configuration System

All configs use YAML inheritance via a `base:` key:

```yaml
# configs/span_id_kudremukh.yaml
base: "span_id_base.yaml"    # inherits all defaults
domains: ["beverlyhillscop", "money-heist"]
training:
  batch_size: 64             # override only what differs
```

Machine-specific configs:

| Config | Machine | Notes |
|--------|---------|-------|
| `span_id_base.yaml` | any | default, 3 seeds, all models |
| `span_id_kudremukh.yaml` | kudremukh | 4× RTX 6000 Ada, larger batches |
| `article_retrieval_base.yaml` | any | default, all retrievers |
| `article_retrieval_kudremukh.yaml` | kudremukh | batch_size=512, n_workers=16 |

The Article Retrieval `run_all.py` auto-detects the machine hostname and
selects the appropriate config automatically — no `--config` flag needed
on Kudremukh.

---

## Running Tests

```bash
# All tests
.venv/bin/python -m pytest tests/ -v

# Skip tests with external dependencies
.venv/bin/python -m pytest tests/ -v \
    --ignore=tests/test_scrape_pipeline.py
```

Test coverage includes: span metrics, evaluator, baselines, dataset splits,
tokenization, stats utilities, span-to-query lookup bridge, and an integration test.
