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

1. **Research-first**: Every run appends to the research CSV so that all results — BM25
   baseline through fine-tuned cross-encoder re-ranking — are directly comparable.
2. **All 11 experiments in one run**: `run_all.py` iterates the Cartesian product of all
   ablation dimension lists in the config. Each combination produces independent artifacts
   via dimension-encoded filenames — no manual re-running needed.
3. **Cache everything**: Indexes, embeddings, query datasets, retrieval and reranking results
   are all persisted to disk. Restarting a run resumes from the last completed step.
4. **Clean interfaces**: Each pipeline stage reads JSONL written by the previous stage.
   Changing a retriever does not require changing the evaluator.
5. **Config-driven**: All 11 experiment dimensions are expressed in YAML as lists.
   `get_ablation_configs()` expands them into single-valued configs for each combination.
6. **Artifact naming**: Every output file encodes the experiment dimensions that produced it
   so that multiple runs coexist on disk without collision.
7. **Per-domain research CSVs**: Results are saved to `data/research/<domain>/` rather than
   a single shared file, enabling clean per-domain analysis and cross-domain aggregation.

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
             │         Step 03: Run Reranking (zero-shot)         │
             │  Cross-encoder scores top-K_input candidates       │
             │  One model load per reranker — all versions batched│
             │  → <retriever>_<reranker>_topk<K>_..._v<N>.jsonl  │
             └─────────────┬─────────────────────────────────────┘
                           │
             ┌─────────────▼─────────────────────────────────────┐
             │    Step 04: Train Reranker (optional)              │
             │  Mine (query, positive, hard negative) triples     │
             │  from step 02 retrieval results                    │
             │  Fine-tune CrossEncoder with binary cross-entropy  │
             │  Only train-split queries used (no test leakage)   │
             │  → data/article_retrieval/checkpoints/reranker_*/  │
             └─────────────┬─────────────────────────────────────┘
                           │
             ┌─────────────▼─────────────────────────────────────┐
             │         Step 05: Evaluate                          │
             │  Recall@1,3,5,10,20,50,100 and MRR                │
             │  → per-version metrics JSON                        │
             │  → data/research/<domain>/                         │
             │      article_retrieval_experiments.csv             │
             └────────────────────────────────────────────────────┘
```

---

## Ablation Sweep: All 11 Experiments in One Run

`run_all.py` calls `get_ablation_configs(config)` which expands all plural config keys
into a list of single-valued configs via Cartesian product. The pipeline loops over every
combination, with skip logic (cached artifacts are not recomputed).

| # | Dimension | Config key | Values |
|---|-----------|------------|--------|
| 1 | Query versions | `queries.versions` | v1–v24 (inner loop in step 02) |
| 2 | Retriever model | `retrievers.sparse/dense` | BM25, TF-IDF, 4 dense models |
| 3 | Corpus representation | `article_index.corpus_representations` | `title_full`, `title_only`, `title_lead` |
| 4 | Query-side context | `queries.query_context_modes` | `anchor_sentence`, `anchor_only`, `anchor_paragraph` |
| 5 | Corpus granularity | `article_index.corpus_granularities` | `article`, `paragraph`, `sentence` |
| 6 | Re-ranker model | `reranking.models` | 3 cross-encoder variants |
| 7 | Re-ranker input K | `reranking.top_k_inputs` | `5`, `10`, `20`, `50` |
| 8 | Domain | `domains` | one or more Fandom wikis |
| 9 | Query sample size | `queries.n_samples` | `1000`, `null` (all) |
| 10 | FAISS index type | `faiss_index_type` | `flat` (ivf/hnsw — future) |
| 11 | Anchor preprocessing | `queries.anchor_preprocessings` | `raw`, `lowercase`, `stopword_removed` |

Standalone sub-scripts (`02_run_retrieval.py`, etc.) call `cu.resolve_config()` which
picks the first ablation combination as the default — so they work correctly when run
directly with a base config.

---

## Directory Layout

```
data/article_retrieval/<domain>/
├── article_index/
│   ├── articles_<repr>_<gran>.jsonl             # persisted article text records
│   ├── bm25_<repr>_<gran>_<preproc>.pkl         # BM25 index
│   ├── tfidf_<repr>_<gran>_<preproc>.pkl        # TF-IDF vectorizer
│   ├── tfidf_matrix_<repr>_<gran>_<preproc>.npz
│   ├── embeddings_<model>_<repr>_<gran>.npy     # article embeddings
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
├── reranker_training/
│   └── reranker_train_<retriever>_<repr>_<gran>_<ctx>_<preproc>_n<N>_v<V>_neg<N>.jsonl
└── metrics/
    ├── retrieval_<retriever>_..._v<V>.json
    ├── reranking_<retriever>_<reranker>_..._v<V>.json
    └── summary_<domain>.csv

data/article_retrieval/checkpoints/
└── reranker_finetuned/                          # fine-tuned CrossEncoder (if trained)

data/research/<domain>/
└── article_retrieval_experiments.csv            # per-domain research log

data/logs/<domain>/article_retrieval/
└── <timestamp>_<step>.log
```

---

## Module Reference

### `src/article_retrieval/config_utils.py`

- `load_config(path)` — YAML loader with `base:` inheritance.
- `get_ablation_configs(config)` — expand plural ablation keys into a list of
  single-valued configs via Cartesian product (Exps 3, 4, 5, 6, 7, 9, 11).
- `resolve_config(config)` — for standalone script usage: returns the first ablation
  config so path helpers always receive well-defined singular values.
- `ablation_label(config)` — short human-readable label for logging.
- `get_*_path(config, domain, ...)` — all artifact path resolution functions; naming
  encodes active experiment dimensions.
- `get_reranker_training_data_path(config, domain)` — path for mined training JSONL.
- `get_reranker_checkpoint_dir(config)` — path for the fine-tuned model.
- `get_research_csv_path(config, domain)` — `data/research/<domain>/article_retrieval_experiments.csv`.

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
| `CONTEXT_VERSIONS` | Versions that use `{paragraph_text}` (1, 3, 4, 22, 23). |
| `generate_queries_for_link()` | Apply templates to one anchor + context. |
| `build_query_dataset()` | End-to-end builder: filter → sample → generate → save. |
| `load_query_dataset()` | Load persisted query JSONL. |

### `src/article_retrieval/embedder.py`

Dense text encoding with caching.

| Function | Purpose |
|---|---|
| `embed_articles()` | Encode article texts; load from disk if cached. |
| `embed_queries_all_versions()` | Encode all query versions in one model-load pass. |

### `src/article_retrieval/retriever.py`

Retrieval logic for all retriever types.

| Function | Purpose |
|---|---|
| `retrieve_bm25()` | BM25 retrieval for one query version. |
| `retrieve_tfidf()` | TF-IDF retrieval for one query version. |
| `retrieve_dense()` | FAISS inner product search for one (model, version) pair. |
| `save/load_retrieval_results()` | JSONL I/O. |

**Source article exclusion:** All three retrievers exclude `source_article_id` from
returned results. Every query is built from anchor text inside the source article, so
that article always scores highest without this filter — making metrics meaningless.

**Output record format:**
```json
{
  "query_id": "money-heist_q_000001",
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
| `rerank_all_versions()` | Load cross-encoder ONCE; process all (retriever × version) jobs in batch. |
| `build_article_lookup()` | Build `article_id → text` dict for scoring. |
| `save/load_reranking_results()` | JSONL I/O; same format as retriever output + `"reranker"` field. |

The reranker also filters `source_article_id` from candidates before calling the
cross-encoder, serving as a second line of defence.

### `src/article_retrieval/reranker_trainer.py`

Fine-tuning a cross-encoder on Fandom-specific retrieval data.

| Function | Purpose |
|---|---|
| `build_training_examples()` | Mine (query, positive, hard negative) triples from retrieval results. Only uses training-split queries. |
| `save/load_training_examples()` | JSONL I/O for mined training data (cached to disk). |
| `train_reranker()` | Fine-tune a `CrossEncoder` with binary cross-entropy. Positive pairs → 1.0, negatives → 0.0. |

Training data is mined directly from step 02 outputs — no separate negative mining
pipeline needed. The gold article is the positive; top retrieved non-gold articles are
the hard negatives.

### `src/article_retrieval/evaluator.py`

Metric computation and research CSV management.

| Function | Purpose |
|---|---|
| `compute_metrics()` | Aggregate Recall@K and MRR over a list of results. |
| `append_to_research_csv()` | Append one row to the domain-scoped research CSV. |

`_filter_source()` is applied inside metric functions as a defensive guard — even
previously generated JSONL files produce correct metrics when re-evaluated.

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
  "query_id": "money-heist_q_000001",
  "anchor_text": "Berlin",
  "gold_article_id": 42,
  "source_article_id": 17,
  "paragraph_text": "Berlin is Álvaro Morte's character in the heist.",
  "queries": {
    "v1": "Retrieve documents for the term 'Berlin', the context is: ...",
    "v2": "Find an article that defines and explains 'Berlin'.",
    "...": "..."
  }
}
```

### Retrieval / Reranking Result Record

```json
{
  "query_id": "money-heist_q_000001",
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

### Reranker Training Example Record

```json
{
  "query": "Retrieve the topic discussing 'Berlin'.",
  "positive": "Berlin is the alias of Andrés de Fonollosa, older brother of...",
  "negative": "Nairobi is a character known for her role in the gold melting..."
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
| `03_run_reranking.py` | 3 | Zero-shot re-rank with cross-encoder |
| `04_train_reranker.py` | 4 | Fine-tune cross-encoder on retrieval results (optional) |
| `05_evaluate.py` | 5 | Compute metrics, write domain-scoped research CSV |
| `aggregate_results.py` | — | Print summary table; auto-discovers all domain CSVs |
| `visualise_results.py` | — | Plots saved to `data/article_retrieval/<domain>/plots/` |
| `run_all.py` | — | Master orchestration: Cartesian sweep of all ablation combos |

All scripts accept `--config`, `--domain`, `--force`, and specific filtering flags.
All sub-scripts call `cu.resolve_config()` so they work correctly when run standalone.

---

## How to Run

### Quick start (single domain, all experiments)

```bash
python scripts/03_Article_retrieval/run_all.py \
    --config configs/article_retrieval/article_retrieval.yaml \
    --domain money-heist
```

### Multi-domain on Kudremukh

```bash
python scripts/03_Article_retrieval/run_all.py \
    --config configs/article_retrieval/kudremukh.yaml
```

### With reranker fine-tuning

```yaml
# In configs/article_retrieval/article_retrieval.yaml
reranker_training:
  enabled: true
```

```bash
python scripts/03_Article_retrieval/run_all.py
```

### Ablation: BM25 baseline only, versions 1–5

```bash
python scripts/03_Article_retrieval/run_all.py \
    --retriever bm25 \
    --versions 1,2,3,4,5
```

### Resume interrupted run

```bash
python scripts/03_Article_retrieval/run_all.py \
    --skip-index \
    --skip-queries
```

### View results

```bash
python scripts/03_Article_retrieval/aggregate_results.py --top 20
python scripts/03_Article_retrieval/aggregate_results.py --domain money-heist
```

---

## Dependencies

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

## Future Work

### FAISS Approximate Search (Exp 10)

For corpora of 100k+ articles, IVF and HNSW indexes reduce search time significantly.
The infrastructure (`build_faiss_index` with `index_type` parameter) is already in place.
