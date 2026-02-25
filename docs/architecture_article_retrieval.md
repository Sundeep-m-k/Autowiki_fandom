# Architecture: Article Retrieval (Task 2)

## Overview

Task 2 — **Article Retrieval** — is an independent research pipeline that, given a hyperlink
anchor from a Fandom wiki article, retrieves the correct target article from the wiki corpus.

It sits between:

```
Task 1: Span Identification → finds hyperlink spans in article text
Task 2: Article Retrieval   → retrieves the correct target article for a span      ← THIS TASK
Task 3: Linking Pipeline    → combines Task 1 + Task 2 end-to-end
```

The pipeline is designed so that Task 3 can consume Task 2's outputs with minimal glue code.

---

## Motivation and Design Principles

1. **Research-first**: Every run writes to the research CSV so that all results — BM25 baseline
   through cross-encoder re-ranking — are directly comparable.
2. **Cache everything**: Indexes, embeddings, query datasets, retrieval and reranking results are
   all persisted to disk. Restarting a run resumes from the last completed step.
3. **Clean interfaces**: Each pipeline stage reads JSONL written by the previous stage. Changing a
   retriever does not require changing the evaluator. Changing the query format does not require
   rebuilding the article index.
4. **Config-driven**: All 11 experiment dimensions are expressed in YAML. A script only knows
   what the config tells it.
5. **Artifact naming**: Every output file encodes the experiment dimensions that produced it
   (see *Artifact Naming Convention* below) so that multiple runs coexist on disk.

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           data/processed/<domain>/                           │
│            articles_page_granularity_<domain>.jsonl  (Task 1 output)        │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │
             ┌─────────────▼─────────────────────────────────────┐
             │         Step 00: Build Article Index               │
             │  ┌────────────────────────────────────────────┐   │
             │  │  BM25 index  (rank_bm25)                   │   │
             │  │  TF-IDF index (sklearn)                    │   │
             │  │  Dense embeddings + FAISS index per model  │   │
             │  │  articles_<repr>_<gran>.jsonl  (for lookup)│   │
             │  └────────────────────────────────────────────┘   │
             └─────────────┬─────────────────────────────────────┘
                           │
             ┌─────────────▼─────────────────────────────────────┐
             │        Step 01: Build Query Dataset                │
             │  Filter to test-split source articles (Task 1)     │
             │  Sample n_sample links (stratified)                │
             │  Generate 24 query variation texts per link        │
             │  → query_dataset_<ctx>_<preproc>_n<N>.jsonl       │
             └─────────────┬─────────────────────────────────────┘
                           │
             ┌─────────────▼─────────────────────────────────────┐
             │         Step 02: Run Retrieval                     │
             │  For each (retriever, query version):              │
             │    BM25, TF-IDF  → keyword scoring                 │
             │    Dense models  → FAISS cosine search             │
             │  → <retriever>_..._v<N>_top<K>.jsonl per combo    │
             └─────────────┬─────────────────────────────────────┘
                           │
             ┌─────────────▼─────────────────────────────────────┐
             │         Step 03: Run Reranking                     │
             │  Cross-encoder scores top-K_input candidates       │
             │  → <retriever>_<reranker>_..._v<N>.jsonl          │
             └─────────────┬─────────────────────────────────────┘
                           │
             ┌─────────────▼─────────────────────────────────────┐
             │         Step 04: Evaluate                          │
             │  Recall@1,3,5,10,20,50,100 and MRR                │
             │  → per-version metrics JSON                        │
             │  → article_retrieval_experiments.csv               │
             └────────────────────────────────────────────────────┘
```

---

## Directory Layout

```
data/article_retrieval/<domain>/
├── article_index/
│   ├── articles_<repr>_<gran>.jsonl          # persisted article text records
│   ├── bm25_<repr>_<gran>_<preproc>.pkl      # BM25 index
│   ├── tfidf_<repr>_<gran>_<preproc>.pkl     # TF-IDF vectorizer
│   ├── tfidf_matrix_<repr>_<gran>_<preproc>.npz
│   ├── embeddings_<model>_<repr>_<gran>.npy  # article embeddings
│   ├── embeddings_<model>_<repr>_<gran>_ids.json
│   ├── faiss_<model>_<repr>_<gran>_flat.index
│   ├── faiss_<model>_<repr>_<gran>_flat_meta.json
│   └── index_meta.json
├── queries/
│   ├── query_dataset_<ctx>_<preproc>_n<N>.jsonl
│   ├── query_embeddings_<model>_<ctx>_<preproc>_n<N>_v<V>.npy
│   └── query_embeddings_<model>_<ctx>_<preproc>_n<N>_v<V>_ids.json
├── retrieval/
│   └── <retriever>_<repr>_<gran>_<ctx>_<preproc>_n<N>_v<V>_top<K>.jsonl
├── reranking/
│   └── <retriever>_<reranker>_topk<K>_<repr>_<gran>_<ctx>_<preproc>_n<N>_v<V>.jsonl
└── metrics/
    ├── retrieval_<retriever>_..._v<V>.json
    ├── reranking_<retriever>_<reranker>_..._v<V>.json
    └── summary_<domain>.csv

data/research/
└── article_retrieval_experiments.csv          # global research log

data/logs/<domain>/article_retrieval/
└── <timestamp>_<step>.log
```

---

## Module Reference

### `src/article_retrieval/config_utils.py`

- `load_config(path)` — YAML loader with `base:` inheritance (same pattern as Task 1).
- `get_*_path(config, domain, ...)` — all artifact path resolution functions.
- Path naming encodes active experiment dimensions so multiple configurations coexist.

### `src/article_retrieval/logging_utils.py`

- `setup_logger(log_dir, script_name)` — file + console logger, safe to call multiple times.
- Same pattern as `span_identification/logging_utils.py`.

### `src/article_retrieval/article_index.py`

Builds and persists all article indexes.

| Function | Purpose |
|---|---|
| `load_articles()` | Load + clean articles from processed JSONL. Handles Exp 3 & 5. |
| `save_articles_jsonl()` | Persist clean records for reranker text lookup. |
| `build_bm25_index()` | Build BM25Okapi index. Handles Exp 11 (preprocessing). |
| `build_tfidf_index()` | Build TF-IDF vectorizer + sparse matrix. |
| `build_faiss_index()` | Build FAISS index. Handles Exp 10 (index type, flat only active). |
| `save_*/load_*` | Persistence helpers for all index types. |

### `src/article_retrieval/query_builder.py`

Generates query dataset from ground truth links.

| Constant/Function | Purpose |
|---|---|
| `QUERY_TEMPLATES` | 24 query variation templates. |
| `CONTEXT_VERSIONS` | Versions that use `{paragraph_text}`. |
| `generate_queries_for_link()` | Apply templates to one anchor + context. |
| `build_query_dataset()` | End-to-end builder: filter → sample → generate → save. |
| `load_query_dataset()` | Load persisted query JSONL. |

### `src/article_retrieval/embedder.py`

Dense text encoding with caching.

| Function | Purpose |
|---|---|
| `embed_articles()` | Encode article texts; load from disk if cached. |
| `embed_queries()` | Encode query texts for one (model, version) pair; cached. |

### `src/article_retrieval/retriever.py`

Retrieval logic for all retriever types.

| Function | Purpose |
|---|---|
| `retrieve_bm25()` | BM25 retrieval for one query version. |
| `retrieve_tfidf()` | TF-IDF retrieval for one query version. |
| `retrieve_dense()` | FAISS inner product search for one (model, version) pair. |
| `save/load_retrieval_results()` | JSONL I/O. |

**Output record format:**
```json
{
  "query_id": "beverlyhillscop_q_000001",
  "gold_article_id": 42,
  "source_article_id": 17,
  "version": 3,
  "retriever": "bm25",
  "retrieved": [
    {"article_id": 42, "score": 12.3, "rank": 1},
    {"article_id": 99, "score": 11.1, "rank": 2}
  ]
}
```

### `src/article_retrieval/reranker.py`

Zero-shot cross-encoder re-ranking.

| Function | Purpose |
|---|---|
| `rerank()` | Score (query, article) pairs; re-sort top-K_input candidates. |
| `build_article_lookup()` | Build `article_id → text` dict for scoring. |
| `save/load_reranking_results()` | JSONL I/O; same format as retriever output + `"reranker"` field. |

### `src/article_retrieval/evaluator.py`

Metric computation and research CSV management.

| Function | Purpose |
|---|---|
| `reciprocal_rank()` | 1/rank of first correct hit (per query). |
| `is_hit_at_k()` | Boolean hit at rank K (per query). |
| `compute_metrics()` | Aggregate Recall@K and MRR; K capped at corpus size. |
| `append_to_research_csv()` | Append one row to `article_retrieval_experiments.csv`. |
| `save_summary_csv()` | Write flat summary CSV for a domain. |

---

## Experiment Dimensions

All 11 experiment dimensions are engineered into the config. Default values represent
the best practice choice for Phase 1 (default run). Phase 2 ablations are commented out
but ready to activate by changing one config key.

| # | Dimension | Config key | Default | Phase 2 options |
|---|---|---|---|---|
| 1 | Query versions | `queries.versions` | 1–24 | subset, e.g. `[1,3,22,23]` |
| 2 | Retriever model | `retrievers.sparse/dense` | BM25, TF-IDF, 5 dense models | add/remove models |
| 3 | Corpus representation | `article_index.corpus_representation` | `title_full` | `title_only`, `title_lead` |
| 4 | Query-side context | `queries.query_context_mode` | `anchor_sentence` | `anchor_only`, `anchor_paragraph` |
| 5 | Corpus granularity | `article_index.corpus_granularity` | `article` | `paragraph`, `sentence` |
| 6 | Re-ranker model | `reranking.model` | `ms-marco-MiniLM-L-6-v2` | larger CE models, BAAI/bge-reranker |
| 7 | Re-ranker input K | `reranking.top_k_input` | `20` | `5`, `10`, `50` |
| 8 | Domain | `domains` | `beverlyhillscop` | `money-heist`, multi-domain |
| 9 | Query sample size | `queries.n_sample` | `1000` | `500`, `null` (all) |
| 10 | FAISS index type | `faiss_index_type` | `flat` | `ivf`, `hnsw` (Future) |
| 11 | Anchor preprocessing | `queries.anchor_preprocessing` | `raw` | `lowercase`, `stopword_removed` |

### Execution Strategy

Rather than a full combinatorial sweep (11 dimensions × all values = thousands of runs), we use
a **default + ablation** strategy:

1. **Default run**: Run all retrievers, all 24 query versions, with all other dimensions at their
   default values. This produces 24 × (2 sparse + N dense) retrieval results plus re-ranking.
2. **Ablation**: To test one dimension, change only that config key and rerun. Artifact naming
   ensures results coexist and the research CSV accumulates all rows.
3. **Multi-domain**: Run with `article_retrieval_kudremukh.yaml` which sets multiple domains.

---

## Data Flow

### Input (from Task 1 / scraping pipeline)

```
data/processed/<domain>/articles_page_granularity_<domain>.jsonl
```

Each line is one article with:
- `article_id`, `title`, `page_name`, `article_plain_text`
- `links`: list of internal links with `anchor_text`, `article_id_of_internal_link`

### Query Dataset Record

```json
{
  "query_id": "beverlyhillscop_q_000001",
  "anchor_text": "Axel Foley",
  "gold_article_id": 42,
  "source_article_id": 17,
  "paragraph_text": "Axel Foley is a Detroit detective who travels to Beverly Hills.",
  "queries": {
    "v1": "Retrieve documents for the term 'Axel Foley', the context is: ...",
    "v2": "Find an article that defines and explains 'Axel Foley'.",
    "...": "..."
  }
}
```

### Retrieval / Reranking Result Record

```json
{
  "query_id": "beverlyhillscop_q_000001",
  "gold_article_id": 42,
  "source_article_id": 17,
  "version": 3,
  "retriever": "bm25",
  "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "retrieved": [
    {"article_id": 42, "score": 3.21, "rank": 1},
    {"article_id": 11, "score": 2.87, "rank": 2}
  ]
}
```

### Research CSV Columns

| Column | Description |
|---|---|
| `timestamp` | When the run was recorded |
| `domain` | Fandom wiki domain |
| `retriever` | Retriever name (BM25, TF-IDF, model path) |
| `reranker` | Re-ranker model name (empty for retrieval stage) |
| `stage` | `retrieval` or `reranking` |
| `version` | Query version number (1–24) |
| `corpus_representation` | Exp 3: `title_only`, `title_lead`, `title_full` |
| `corpus_granularity` | Exp 5: `article`, `paragraph`, `sentence` |
| `query_context_mode` | Exp 4: `anchor_only`, `anchor_sentence`, `anchor_paragraph` |
| `anchor_preprocessing` | Exp 11: `raw`, `lowercase`, `stopword_removed` |
| `n_queries` | Number of queries evaluated |
| `n_articles` | Corpus size |
| `recall_at_1` … `recall_at_100` | Recall@K values |
| `mrr` | Mean Reciprocal Rank |
| `notes` | Free-text notes |

---

## 24 Query Versions

Versions using `{paragraph_text}` (context-dependent): **1, 3, 4, 22, 23**

| Version | Template |
|---|---|
| 1 | Retrieve documents for the term '{word}', the context is: {paragraph_text}. |
| 2 | Find an article that defines and explains '{word}'. |
| 3 | Given the following paragraph: {paragraph_text}, which article best explains '{word}'? |
| 4 | Find the best article that can explain the term '{word}' given this context: {paragraph_text}. |
| 5 | Which article provides the best information about '{word}'? |
| 6 | Retrieve the topic discussing '{word}'. |
| 7 | Find an article that summarizes the concept of '{word}'. |
| 8 | Which paragraph of text elaborates on the topic of '{word}'? |
| 9 | Locate an article that gives a comprehensive definition of '{word}'. |
| 10 | What is '{word}'? Find the best article explaining it. |
| 11 | Find paragraphs of texts related to '{word}'. |
| 12 | Which article provides background knowledge about '{word}'? |
| 13 | Retrieve articles covering the technical aspects of '{word}'. |
| 14 | Which page gives examples related to '{word}'? |
| 15 | Find an article detailing the history of '{word}'. |
| 16 | Which paragraph of texts gives an in-depth explanation of '{word}'? |
| 17 | Retrieve texts that discuss the fundamentals of '{word}'. |
| 18 | Which article discusses the real-world application of '{word}'? |
| 19 | Find texts that includes research studies on '{word}'. |
| 20 | Locate an article that introduces the concept of '{word}'. |
| 21 | Find texts with an educational overview of '{word}'. |
| 22 | As a domain expert, explain the meaning of '{word}' in this context: {paragraph_text}. |
| 23 | First define '{word}', then explain how it applies in the context: {paragraph_text}. |
| 24 | Which article maps multiple perspectives or connections related to '{word}'? |

---

## Scripts Reference

| Script | Step | Description |
|---|---|---|
| `00_build_article_index.py` | 0 | Build BM25, TF-IDF, FAISS indexes |
| `01_build_query_dataset.py` | 1 | Generate 24 query versions per link |
| `02_run_retrieval.py` | 2 | Retrieve top-K articles per (retriever, version) |
| `03_run_reranking.py` | 3 | Re-rank with cross-encoder |
| `04_evaluate.py` | 4 | Compute metrics, write research CSV |
| `aggregate_results.py` | — | Print summary table from research CSV |
| `run_all.py` | — | Master orchestration with skip logic |

All scripts accept `--config`, `--domain`, `--force`, and specific filtering flags.

---

## How to Run

### Quick Start (single domain)

```bash
python scripts/03_Article_retrieval/run_all.py \
    --config configs/article_retrieval.yaml \
    --domain beverlyhillscop
```

### Multi-domain on Kudremukh

```bash
python scripts/03_Article_retrieval/run_all.py \
    --config configs/article_retrieval_kudremukh.yaml
```

### Ablation: BM25 baseline only, versions 1–5

```bash
python scripts/03_Article_retrieval/run_all.py \
    --retriever bm25 \
    --versions 1,2,3,4,5
```

### Resume interrupted run (skip already-done steps)

```bash
python scripts/03_Article_retrieval/run_all.py \
    --skip-index \
    --skip-queries
```

### View results

```bash
python scripts/03_Article_retrieval/aggregate_results.py --top 20
```

---

## Dependencies

Add to `requirements-article-retrieval.txt`:

```
sentence-transformers>=2.2
faiss-cpu>=1.7.4
rank-bm25>=0.2.2
scikit-learn>=1.3
scipy>=1.11
nltk>=3.8
numpy>=1.24
pyyaml>=6.0
```

For GPU: replace `faiss-cpu` with `faiss-gpu`.

---

## Future Work (Stage 3)

### Negative Mining and Dataset Generation

After retrieval results are available, hard negatives (top-K retrieved but wrong articles)
and easy negatives (random articles) can be mined to create a training dataset in
`query_doc_score.csv` format for bi-encoder or cross-encoder fine-tuning.

Key config keys (commented out in `article_retrieval_base.yaml`):
```yaml
negative_mining:
  enabled: false
  n_hard_negatives: 5
  n_easy_negatives: 5
```

### Re-ranker Training

If zero-shot re-ranking results are insufficient, the mined dataset can be used to
fine-tune a cross-encoder on Fandom wiki data.

Key config keys (commented out):
```yaml
reranker_training:
  enabled: false
  base_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  epochs: 5
```

### FAISS Approximate Search (Exp 10)

For corpora of 100k+ articles, IVF and HNSW indexes reduce search time significantly.
The infrastructure (`build_faiss_index` with `index_type` parameter) is already in place —
only the implementation inside the `ivf`/`hnsw` branches needs to be uncommented.

### Task 3: Linking Pipeline

Task 3 will consume:
- Task 1 output: predicted hyperlink spans from `hf_trainer.train_and_evaluate`
- Task 2 output: top-K retrieved articles from retrieval/reranking results JSONL

The clean JSONL interface between stages is designed explicitly for this integration.
