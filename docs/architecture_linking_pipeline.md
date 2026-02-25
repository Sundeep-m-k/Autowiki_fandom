# Architecture: Linking Pipeline (Task 3)

## Overview

Task 3 — **Linking Pipeline** — is the end-to-end combination of Task 1 and Task 2.

```
Input:  Plain text of a wiki article (no existing links)
Output: HTML version of the article with internal <a href="..."> hyperlinks injected
```

It sits at the top of the research stack:

```
Task 1: Span Identification  → find hyperlink spans in plain text
Task 2: Article Retrieval    → retrieve the correct target article per span
Task 3: Linking Pipeline     → combine Tasks 1 & 2 → produce linked HTML     ← THIS TASK
```

---

## Design Principles

1. **No model inference at linking time.** Task 2 retrieval and reranking results are pre-computed and stored as JSONL files. Task 3 is a pure lookup + rendering step — it runs in seconds with no GPU required.

2. **Lookup key: `(source_article_id, anchor_text)`.** Every gold span from Task 1 can be matched to its Task 2 query record via this key. The Task 2 query dataset was built from the same ground truth links, so the key always exists for test-split spans.

3. **Phase 1: gold spans.** Task 3 Phase 1 uses gold spans from the Task 1 test split rather than model-predicted spans. This isolates the linking component's quality from Task 1 prediction errors and lets you evaluate each component independently. Phase 2 (live Task 1 inference) is straightforward to add once both components are validated.

4. **Config-driven.** Which Task 2 retriever, reranker, and query version to use for linking is configured in `configs/linking.yaml`. Swapping to a different Task 2 result requires only a config change — no code changes.

---

## Pipeline Architecture

```
data/span_id/<domain>/splits/test_article.jsonl  (Task 1 gold spans)
        │
        ▼  span_predictor.py
List of articles with gold spans:
  [{article_id, text, gold_spans: [{char_start, char_end, anchor_text, gold_article_id}]}]
        │
        ▼  span_to_query.py  (lookup, no inference)
Build (source_article_id, anchor_text) → {article_id, score}
from Task 2 pre-computed JSONL:
  data/article_retrieval/<domain>/reranking/<retriever>_<reranker>_..._v<N>.jsonl
        │
        ▼  nil_detector.py
For each span: score < threshold → linked=False (NIL)
        │
        ▼  html_renderer.py
Build HTML with <a href="https://<domain>.fandom.com/wiki/<PageName>">anchor</a>
        │
        ▼  evaluator.py
Linking F1, Span F1, Entity Accuracy, NIL Rate, Coverage
        │
        ▼
data/linking/<domain>/
  ├── linking_results.jsonl
  ├── html/<article_id>.html
  ├── metrics/metrics_*.json
  └── plots/

data/research/linking_experiments.csv
```

---

## Module Reference

### `src/linking_pipeline/config_utils.py`

- `load_config(path)` — YAML loader with `base:` inheritance.
- `get_task1_split_path(config, domain)` — path to Task 1 test JSONL.
- `get_task2_query_dataset_path(config, domain)` — Task 2 query dataset.
- `get_task2_retrieval_path(config, domain)` — pre-computed retrieval JSONL.
- `get_task2_reranking_path(config, domain)` — pre-computed reranking JSONL.
- `get_task2_articles_jsonl_path(config, domain)` — article index for page_name lookup.
- `get_linking_results_path(config, domain)` — output JSONL (encodes config in filename).
- `get_fandom_base_url(config, domain)` — Fandom wiki base URL.

### `src/linking_pipeline/span_predictor.py`

Loads gold spans from the Task 1 test split JSONL. Returns one dict per article with `gold_spans` containing only internal link spans with valid char offsets.

### `src/linking_pipeline/span_to_query.py`

Builds the lookup table from Task 2 pre-computed results.

| Function | Purpose |
|---|---|
| `build_lookup(query_dataset_path, results_path)` | Returns `(source_article_id, anchor_text_lower) → {article_id, score}` |
| `lookup_span(lookup, source_article_id, anchor_text)` | Look up one span |

Anchor text is lowercased for case-insensitive matching. If the same anchor appears multiple times in a source article, the hit with the highest score is kept.

### `src/linking_pipeline/nil_detector.py`

- `should_link(score, threshold)` — `True` if score ≥ threshold.
- `apply_nil_filter(predicted_links, threshold)` — sets `linked` field in-place.

### `src/linking_pipeline/html_renderer.py`

- `render_html(text, confirmed_links, ...)` — builds HTML from plain text + confirmed links.
- Spans are processed front-to-back; overlapping spans resolved by `longest` strategy.
- `save_html(html_str, path)` — writes a self-contained HTML file with minimal CSS.

### `src/linking_pipeline/evaluator.py`

| Metric | Definition |
|---|---|
| Linking F1 | span boundary correct AND predicted article_id == gold article_id |
| Span F1 | span boundary correct (boundary only, ignores article_id) |
| Entity Accuracy | article_id correct, given span boundary is correct |
| NIL Rate | fraction of gold spans assigned NIL by the threshold |
| Coverage | fraction of gold spans that had a Task 2 lookup hit |

---

## Scripts Reference

| Script | Step | Description |
|---|---|---|
| `00_predict_and_link.py` | 0 | Load gold spans → lookup Task 2 → apply NIL → save results JSONL |
| `01_render_html.py` | 1 | Render one HTML file per article with `<a>` tags |
| `02_evaluate.py` | 2 | Compute metrics, write research CSV |
| `visualise_linking.py` | — | Generate plots from research CSV |
| `run_all.py` | — | Master orchestrator with skip logic and NIL ablation |

---

## Data Formats

### `linking_results.jsonl` (one line per article)

```json
{
  "article_id": 2061,
  "text": "The plain article text...",
  "page_name": "Beverly_Hills_Cop",
  "gold_spans": [
    {
      "char_start": 12, "char_end": 21,
      "anchor_text": "Axel Foley",
      "gold_article_id": 2170
    }
  ],
  "predicted_links": [
    {
      "char_start": 12, "char_end": 21,
      "anchor_text": "Axel Foley",
      "gold_article_id": 2170,
      "article_id": 2170,
      "page_name": "Axel_Foley",
      "fandom_url": "https://beverlyhillscop.fandom.com/wiki/Axel_Foley",
      "retrieval_score": 3.21,
      "linked": true
    }
  ]
}
```

### HTML output (per article)

```html
<!DOCTYPE html>
<html lang='en'>
<head><meta charset='utf-8'> ... </head>
<body>
<div class="wiki-article">
<p>... <a href="https://beverlyhillscop.fandom.com/wiki/Axel_Foley"
         class="wiki-link">Axel Foley</a> is a Detroit detective ...</p>
</div>
</body>
</html>
```

### `linking_experiments.csv` columns

| Column | Description |
|---|---|
| `timestamp` | Run timestamp |
| `domain` | Fandom wiki domain |
| `retriever` | Task 2 retriever used |
| `reranker` | Task 2 reranker (empty if stage=retrieval) |
| `stage` | `retrieval` or `reranking` |
| `query_version` | Task 2 query version (1–24) |
| `nil_threshold` | NIL detection threshold |
| `n_articles` | Number of test articles |
| `n_gold_total` | Total gold spans |
| `n_pred_total` | Total predicted links (linked=True) |
| `linking_f1` | Primary end-to-end metric |
| `span_f1` | Task 1 component quality |
| `entity_accuracy` | Task 2 component quality |
| `nil_rate` | Fraction of spans suppressed |
| `coverage` | Fraction of spans with Task 2 hit |

---

## Config Dimensions

| Key | Default | Ablation |
|---|---|---|
| `task2.stage` | `reranking` | `retrieval` |
| `task2.retriever` | `msmarco-distilbert-base-v4` | `tfidf`, `bm25`, `all-MiniLM-L6-v2` |
| `task2.query_version` | `6` | `10`, `19` (best from Task 2 experiments) |
| `nil_detection.threshold` | `0.0` | `0.1`, `0.2`, `0.5`, `1.0` |
| `task1.granularity` | `article` | `paragraph`, `sentence` |
| `domains` | `beverlyhillscop` | `money-heist`, multi-domain |

---

## How to Run

### Full pipeline (default config)

```bash
python scripts/04_Linking_pipeline/run_all.py
```

### Single domain, skip HTML rendering

```bash
python scripts/04_Linking_pipeline/run_all.py --domain beverlyhillscop --skip-html
```

### NIL threshold ablation (sweep thresholds 0.0 → 0.5)

```bash
python scripts/04_Linking_pipeline/run_all.py --nil-thresholds 0.0,0.1,0.2,0.5
```

### View results

```bash
python scripts/04_Linking_pipeline/visualise_linking.py
```

---

## Interpreting Results

| Metric | What it tells you |
|---|---|
| **Linking F1** | End-to-end quality: correct span AND correct article |
| **Span F1 > Linking F1** | Task 2 (retrieval) is the bottleneck |
| **Span F1 ≈ Linking F1** | Task 1 is the bottleneck (span prediction errors) |
| **Entity Accuracy** | Given a correct span, how often is the right article found |
| **Coverage < 1.0** | Some spans from the test split were not in Task 2 query dataset (e.g. not sampled) |
| **NIL Rate > 0** | Threshold is filtering links — trade-off between precision and recall |

---

## Phase 2: Live Task 1 Inference (Future Work)

Phase 1 uses gold spans to isolate Task 3 evaluation. Phase 2 would replace `span_predictor.py` with a live inference call to the Task 1 trained model:

1. Load the best Task 1 checkpoint from `data/span_id/<domain>/checkpoints/`
2. Run `AutoModelForTokenClassification` inference on raw article text
3. Decode BILOU/BIO labels to char spans
4. Feed predicted spans into the same Task 2 lookup + HTML rendering pipeline

The `span_to_query.py` bridge handles both gold and predicted spans identically — the anchor text and source article ID are all that is needed for the Task 2 lookup.
