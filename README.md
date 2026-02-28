# Autowiki Fandom

Research pipeline for automated hyperlinking of wiki articles.
Given a plain-text wiki article, the system identifies which text spans should become
hyperlinks and retrieves the correct target article for each span — producing HTML
output with injected `<a href="…">` tags.

The pipeline supports two data sources:

| Data Source | Acquisition method | Domain examples |
|---|---|---|
| **Fandom wikis** | Live web scraping via MediaWiki API | `harrypotter`, `beverlyhillscop`, `money-heist`, `disney` |
| **Wikipedia** | Offline XML dump parsing (`mwparserfromhell`) | `wikipedia` (chunk 1: 21,009 articles) |

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
3. [Data Processing — Fandom](#data-processing--fandom)
4. [Data Processing — Wikipedia](#data-processing--wikipedia)
5. [Task 1 — Span Identification](#task-1--span-identification)
6. [Task 2 — Article Retrieval](#task-2--article-retrieval)
7. [Task 3 — Linking Pipeline](#task-3--linking-pipeline)
8. [Full Pipeline Orchestration](#full-pipeline-orchestration)
9. [Results and Statistics](#results-and-statistics)
10. [Configuration System](#configuration-system)
11. [Running Tests](#running-tests)

---

## Project Structure

```
Autowiki_fandom/
├── configs/
│   ├── data_processing/
│   │   ├── scraping.yaml               # Fandom scraper settings
│   │   ├── ground_truth.yaml           # Fandom ground truth builder settings
│   │   └── wikipedia_ground_truth.yaml # Wikipedia dump parser settings
│   ├── span_id/
│   │   ├── base.yaml                   # Task 1 base config (all defaults)
│   │   ├── span_id.yaml                # Task 1 local/default run
│   │   ├── kudremukh.yaml              # Task 1 overrides for Kudremukh GPU server
│   │   ├── skanda.yaml                 # Task 1 overrides for Skanda GPU server
│   │   └── error_analysis.yaml         # Task 1 error analysis settings
│   ├── article_retrieval/
│   │   ├── base.yaml                   # Task 2 base config (all 11 experiment dims as lists)
│   │   ├── article_retrieval.yaml      # Task 2 local/default run
│   │   └── kudremukh.yaml              # Task 2 compute overrides for Kudremukh
│   └── linking/
│       ├── base.yaml                   # Task 3 base config
│       └── linking.yaml                # Task 3 local/default run
│
├── scripts/
│   ├── 01_Data_processing/
│   │   ├── 00_scrape_fandom.py         # Scrape HTML from Fandom wikis
│   │   ├── 00_parse_wikipedia_dump.py  # Parse Wikipedia XML dump (bz2) → same JSONL schema
│   │   └── 01_build_ground_truth.py    # Build JSONL corpus from Fandom HTML
│   ├── 02_Span_identification/
│   │   ├── 01_run_span_id.py           # Full sweep: baselines + models (local)
│   │   ├── 01_run_span_id_kudremukh.py # Same, optimised for Kudremukh (4× GPU)
│   │   ├── 01_run_span_id_skanda.py    # Same, optimised for Skanda (1× GPU)
│   │   ├── 02_run_baselines.py         # Rule-based baselines only
│   │   ├── 03_error_analysis.py        # Per-model error analysis
│   │   ├── 04_human_eval_sample.py     # Sample predictions for human evaluation
│   │   ├── aggregate_seed_results.py   # Mean ± std across seeds
│   │   └── compare_models.py           # Bootstrap significance testing
│   ├── 03_Article_retrieval/
│   │   ├── run_all.py                  # Master script — runs all 11 experiments
│   │   ├── 00_build_article_index.py
│   │   ├── 01_build_query_dataset.py
│   │   ├── 02_run_retrieval.py
│   │   ├── 03_run_reranking.py
│   │   ├── 04_train_reranker.py        # Fine-tune cross-encoder on retrieval results
│   │   ├── 05_evaluate.py
│   │   ├── aggregate_results.py
│   │   └── visualise_results.py
│   ├── 04_Linking_pipeline/
│   │   ├── run_all.py
│   │   ├── 00_predict_and_link.py
│   │   ├── 01_render_html.py
│   │   ├── 02_evaluate.py
│   │   └── visualise_linking.py
│   ├── run_full_pipeline.py            # Master script for all 5 stages
│   └── summarise_corpus.py             # Cross-domain stats dashboard
│
├── src/
│   ├── data_scraping/                  # Fandom HTML scraper
│   ├── data_processing/
│   │   ├── ground_truth.py             # Fandom HTML → JSONL ground truth
│   │   └── wikipedia_ground_truth.py   # Wikipedia XML dump → same JSONL schema
│   ├── span_identification/            # Task 1 library code
│   │   ├── preprocess.py               # Tokenisation and BIO/BILOU labelling
│   │   ├── hf_trainer.py               # HuggingFace Trainer wrapper
│   │   ├── span_metrics.py             # Span + seqeval metrics
│   │   ├── evaluator.py                # Character-level and exact-match metrics
│   │   ├── baselines.py                # Rule-based baselines
│   │   └── dataset.py                  # Article-ID-based train/val/test splits
│   ├── article_retrieval/              # Task 2 library code
│   │   ├── config_utils.py             # Path helpers + ablation config expansion
│   │   ├── query_builder.py            # 24 query variations per anchor span
│   │   ├── embedder.py                 # Dense encoding (SentenceTransformers + FAISS)
│   │   ├── retriever.py                # BM25, TF-IDF, dense retrieval
│   │   ├── reranker.py                 # Zero-shot cross-encoder re-ranking
│   │   └── reranker_trainer.py         # Fine-tuning cross-encoder on retrieval data
│   ├── linking_pipeline/               # Task 3 library code
│   │   ├── span_predictor.py           # Load gold spans from Task 1
│   │   ├── span_to_query.py            # Match spans to Task 2 results
│   │   ├── nil_detector.py             # Threshold-based NIL filtering
│   │   └── html_renderer.py            # Inject <a> tags into HTML
│   └── utils/
│       └── stats_utils.py              # Centralised stats tracking per domain
│
├── data/
│   ├── raw/wikipedia/                  # Downloaded Wikipedia XML dump (bz2)
│   ├── processed/<domain>/             # Parsed JSONL files (Fandom or Wikipedia)
│   ├── span_id/<domain>/splits/        # Train/val/test splits (article-ID-based)
│   ├── article_retrieval/<domain>/     # FAISS indexes, retrieval results, plots
│   │   └── reranker_training/          # Mined training data for fine-tuned reranker
│   ├── article_retrieval/checkpoints/  # Fine-tuned reranker model
│   ├── linking/<domain>/               # Linking predictions, HTML output, plots
│   ├── research/<domain>/              # Experiment result CSVs (one dir per domain)
│   │   ├── span_id_experiments.csv
│   │   ├── article_retrieval_experiments.csv
│   │   └── linking_experiments.csv
│   └── stats/<domain>.json             # Aggregated best results per domain
│
├── docs/
│   ├── architecture_span_id.md
│   ├── architecture_article_retrieval.md
│   ├── architecture_linking_pipeline.md
│   └── data_sources_wikipedia.md       # Wikipedia dump ingestion: design + usage
│
├── tests/                              # Automated tests
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
- `mwparserfromhell>=0.6` (Wikipedia XML dump parsing only)

---

## Data Processing — Fandom

### Step 0 — Scrape Fandom

```bash
python scripts/01_Data_processing/00_scrape_fandom.py \
    --config configs/data_processing/scraping.yaml
```

Scrapes HTML from `scraping.yaml → start_url`. Saves raw HTML to `data/raw/<domain>/`.

### Step 1 — Build Ground Truth

```bash
python scripts/01_Data_processing/01_build_ground_truth.py \
    --config configs/data_processing/ground_truth.yaml
```

Parses HTML, extracts internal links with character offsets, and writes:

| File | Description |
|------|-------------|
| `data/processed/<domain>/articles_page_granularity_<domain>.jsonl` | Full articles with link spans |
| `data/processed/<domain>/paragraphs_<domain>.jsonl` | Paragraph-level examples |
| `data/processed/<domain>/sentences_<domain>.jsonl` | Sentence-level examples |

Also updates `data/stats/<domain>.json` with dataset statistics.

---

## Data Processing — Wikipedia

Wikipedia data is acquired from **offline XML dumps** (no crawling, no rate limits) and
parsed with `mwparserfromhell`. The output schema is identical to the Fandom pipeline,
so Tasks 1–3 work against Wikipedia data without any code changes.

### Step 0 — Download a dump chunk

Chunk 1 covers pages 1–41,242 (~280 MB compressed, ~21,000 true articles after filtering redirects):

```bash
mkdir -p data/raw/wikipedia
wget -c "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p41242.bz2" \
     -O data/raw/wikipedia/enwiki-latest-pages-articles1.xml-p1p41242.bz2
```

The dump URL is already set in `configs/data_processing/wikipedia_ground_truth.yaml`.
For other chunks, browse `https://dumps.wikimedia.org/enwiki/latest/` and update `dump_path`.

### Step 1 — Parse and build ground truth

```bash
# Full chunk (takes ~43 minutes for 21,009 articles)
python3 scripts/01_Data_processing/00_parse_wikipedia_dump.py

# Quick prototype (500 articles, ~2 minutes)
python3 scripts/01_Data_processing/00_parse_wikipedia_dump.py --max-articles 500

# Different domain name or dump file
python3 scripts/01_Data_processing/00_parse_wikipedia_dump.py \
    --dump data/raw/wikipedia/enwiki-latest-pages-articles2.xml-p41243p151573.bz2 \
    --domain wikipedia_chunk2
```

Writes the same files as the Fandom pipeline — everything downstream is identical:

| File | Description |
|------|-------------|
| `data/processed/wikipedia/articles_page_granularity_wikipedia.jsonl` | Full articles with link spans |
| `data/processed/wikipedia/paragraphs_wikipedia.jsonl` | Paragraph-level examples |
| `data/processed/wikipedia/sentences_wikipedia.jsonl` | Sentence-level examples |
| `data/processed/wikipedia/articles_wikipedia.jsonl` | Article index (id, title, url) |
| `data/processed/wikipedia/paragraphs_wikipedia.csv` | Flat CSV for quick inspection |
| `data/processed/wikipedia/paragraph_links_wikipedia.csv` | Internal links only, flat CSV |

### Chunk 1 statistics (as of Feb 2026)

| Metric | Value |
|--------|-------|
| Articles | 21,009 |
| Paragraphs | 912,357 |
| Sentences | 3,148,806 |
| Internal links | 3,636,342 |
| Avg internal links per article | 173 |
| Avg article length | 24,142 chars |
| Max article length | 608,208 chars |

### How it works

The parser does **two passes** over the dump (streaming — never loads the full file into RAM):

1. **Pass 1** — builds the complete `page_name → article_id` mapping from all 41k entries
   (including redirects) so internal links can be resolved.
2. **Pass 2** — parses each article's wikitext with `mwparserfromhell`:
   - Removes noise templates (infoboxes, navboxes, `<ref>` tags, citation templates)
   - Walks the wikitext node tree to collect plain text and `[[link]]` nodes simultaneously,
     recording exact character offsets for each anchor
   - Splits the result into paragraphs (on `\n\n`), then sentences (NLTK Punkt tokenizer)
   - Skips non-article namespaces (`File:`, `Category:`, `Template:`, etc.)
   - Skips redirect pages

See [`docs/data_sources_wikipedia.md`](docs/data_sources_wikipedia.md) for full design details.

### Configuration (`configs/data_processing/wikipedia_ground_truth.yaml`)

```yaml
domain: "wikipedia"
dump_path: "data/raw/wikipedia/enwiki-latest-pages-articles1.xml-p1p41242.bz2"
processed_dir: "data/processed"
max_articles: 0        # 0 = no limit; set a positive int to limit for prototyping
min_paragraph_chars: 30
outputs:
  paragraphs_jsonl: true
  sentences_jsonl: true
  articles_page_granularity_jsonl: true
  articles_index_jsonl: true
  paragraphs_csv: true
  paragraph_links_csv: true
```

---

## Task 1 — Span Identification

**Goal:** Given a text unit (sentence / paragraph / article), predict which character
spans should become hyperlinks.

**Models:** `bert-base-uncased`, `microsoft/deberta-v3-base`, `roberta-base`, `distilbert-base-uncased`

**Labelling schemes:** BIO and BILOU (configured via `label_schemes` in config)

**Baselines:** `rule_capitalized`, `heuristic_anchor`, `random`

### Run

```bash
# Local
python scripts/02_Span_identification/01_run_span_id.py

# Kudremukh (4× RTX 6000 Ada)
python scripts/02_Span_identification/01_run_span_id_kudremukh.py

# Skanda (1× RTX A6000)
python scripts/02_Span_identification/01_run_span_id_skanda.py
```

### Configuration

```yaml
# configs/span_id/base.yaml (excerpt)
granularities: ["sentence", "paragraph", "article"]
label_schemes: ["BIO", "BILOU"]
seeds: [42, 123, 456]
training:
  epochs: 20
  learning_rate: 5.0e-5
  batch_size: 32
```

Each machine-specific config (`kudremukh.yaml`, `skanda.yaml`) inherits `base.yaml` via
`base: "base.yaml"` and overrides only `domains` and `models`.

### Metrics (written to `data/research/<domain>/span_id_experiments.csv`)

| Column | Description |
|--------|-------------|
| `span_f1` | Exact-boundary span F1 (primary metric) |
| `span_precision` / `span_recall` | Exact-boundary precision / recall |
| `char_f1` | Overlap/relaxed span F1 |
| `exact_match_pct` | Fraction of gold spans exactly recalled |
| `val_span_f1` | Validation span F1 (used for model selection / early stopping) |

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
- Zero-shot re-ranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (and variants)
- Fine-tuned re-ranker: trained on hard negatives mined from retrieval results (optional)

**Query variations:** 24 query templates per anchor span (v1–v24).

### Run

```bash
# Auto-detects machine config (kudremukh hostname → kudremukh.yaml, else article_retrieval.yaml)
python scripts/03_Article_retrieval/run_all.py

# Force specific config
python scripts/03_Article_retrieval/run_all.py \
    --config configs/article_retrieval/kudremukh.yaml

# Single domain, skip indexing if already built
python scripts/03_Article_retrieval/run_all.py \
    --domain money-heist \
    --skip-index --skip-queries

# With reranker fine-tuning enabled (set reranker_training.enabled: true in config first)
python scripts/03_Article_retrieval/run_all.py
```

### Pipeline steps

```
00_build_article_index.py   → BM25 / TF-IDF / FAISS indexes
01_build_query_dataset.py   → 24 query variations per anchor link
02_run_retrieval.py         → Top-K candidates per (retriever, version)
03_run_reranking.py         → Zero-shot re-rank with cross-encoder
04_train_reranker.py        → Fine-tune cross-encoder on retrieval results (optional)
05_evaluate.py              → Recall@K and MRR, write research CSV
visualise_results.py        → Plots saved to data/article_retrieval/<domain>/plots/
```

### All 11 experiments in a single run

`run_all.py` iterates the **Cartesian product** of all ablation dimension lists declared
in `base.yaml`. Each combination produces fully independent artifacts on disk (via
dimension-encoded filenames) and appends a row to the research CSV.

| # | Dimension | Config key | Values |
|---|-----------|------------|--------|
| 1 | Query versions | `queries.versions` | v1–v24 |
| 2 | Retriever model | `retrievers.sparse/dense` | BM25, TF-IDF, 4 dense models |
| 3 | Corpus representation | `article_index.corpus_representations` | `title_full`, `title_only`, `title_lead` |
| 4 | Query-side context | `queries.query_context_modes` | `anchor_sentence`, `anchor_only`, `anchor_paragraph` |
| 5 | Corpus granularity | `article_index.corpus_granularities` | `article`, `paragraph`, `sentence` |
| 6 | Re-ranker model | `reranking.models` | 3 cross-encoder variants |
| 7 | Re-ranker input K | `reranking.top_k_inputs` | `5`, `10`, `20`, `50` |
| 8 | Domain | `domains` | `money-heist`, … |
| 9 | Query sample size | `queries.n_samples` | `1000`, `null` (all) |
| 10 | FAISS index type | `faiss_index_type` | `flat` (ivf/hnsw — future) |
| 11 | Anchor preprocessing | `queries.anchor_preprocessings` | `raw`, `lowercase`, `stopword_removed` |

### Reranker Fine-tuning (optional)

Enable in `configs/article_retrieval/base.yaml`:

```yaml
reranker_training:
  enabled: true
  source_retriever: "sentence-transformers/all-mpnet-base-v2"
  source_version: 6
  base_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  n_hard_negatives: 5
  epochs: 3
  output_dir: "data/article_retrieval/checkpoints/reranker_finetuned"
```

Training data is mined directly from step 02 retrieval results — no separate
negative mining step needed. Only **training-split** queries are used (source articles
from the Task 1 train split), ensuring the test set is never seen during training.

After training, add the checkpoint path to `reranking.models` to evaluate it as an
additional Exp 6 variant.

### Metrics (written to `data/research/<domain>/article_retrieval_experiments.csv`)

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
    ▼ span_predictor.py
Gold spans from Task 1 test split
    │
    ▼ span_to_query.py
Match each span → pre-computed Task 2 result
using key: (source_article_id, char_start, char_end)
    │
    ▼ nil_detector.py
Filter spans whose top-1 score < nil_threshold
    │
    ▼ html_renderer.py
Inject <a href="…"> tags, resolve overlapping spans (longest-wins)
    │
    ▼
Linked HTML output
```

Task 2 results are reused from disk — **no GPU required at linking time**.

### Run

```bash
python scripts/04_Linking_pipeline/run_all.py \
    --config configs/linking/linking.yaml

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

### Metrics (written to `data/research/<domain>/linking_experiments.csv`)

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
    --domain money-heist

# Skip stages already completed
python scripts/run_full_pipeline.py \
    --skip-scraping \
    --skip-ground-truth

# Force rebuild everything
python scripts/run_full_pipeline.py --force
```

---

## Results and Statistics

Research outputs are saved **per domain** under `data/research/<domain>/`:

```
data/research/
  money-heist/
    span_id_experiments.csv
    article_retrieval_experiments.csv
    linking_experiments.csv
  beverlyhillscop/
    span_id_experiments.csv
    ...
```

Aggregate cross-domain scripts (`aggregate_results.py`, `compare_models.py`) automatically
discover all domain subdirectories and merge results for display.

Each domain also accumulates best results in `data/stats/<domain>.json`:

```json
{
  "domain": "money-heist",
  "dataset_stats": { "num_articles": 129, "num_paragraphs": 855, ... },
  "span_id": { "overall_best": { "model": "deberta-v3-base", "span_f1": 0.83, ... } },
  "article_retrieval": { "best_retrieval": { ... }, "best_reranking": { ... } },
  "linking_pipeline": { "best": { ... } }
}
```

Print a cross-domain dashboard:

```bash
python scripts/summarise_corpus.py

# Single domain
python scripts/summarise_corpus.py --domain money-heist

# Save machine-readable summary
python scripts/summarise_corpus.py --save
```

---

## Configuration System

All configs use YAML inheritance via a `base:` key — machine-specific files override
only what differs:

```yaml
# configs/article_retrieval/kudremukh.yaml
base: "base.yaml"
domains: ["money-heist"]
parallel:
  n_workers: 16
  embedding_batch_size: 512
```

### Span Identification configs

| Config | Purpose |
|--------|---------|
| `span_id/base.yaml` | All defaults: training hyperparams, early stopping, seeds, baselines |
| `span_id/span_id.yaml` | Local run: sets `domains` and `models` |
| `span_id/kudremukh.yaml` | Kudremukh (4× RTX 6000 Ada): same as `span_id.yaml` |
| `span_id/skanda.yaml` | Skanda (1× RTX A6000): same as `span_id.yaml` |
| `span_id/error_analysis.yaml` | Error analysis settings only (`domains` + `error_analysis` block) |

### Article Retrieval configs

| Config | Purpose |
|--------|---------|
| `article_retrieval/base.yaml` | All 11 experiment dimensions as **lists** for full sweep |
| `article_retrieval/article_retrieval.yaml` | Local run: sets `domains` |
| `article_retrieval/kudremukh.yaml` | Kudremukh: larger `n_workers` and `embedding_batch_size` |

The Article Retrieval `run_all.py` auto-detects the machine hostname and selects the
appropriate config automatically — no `--config` flag needed on Kudremukh.

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
