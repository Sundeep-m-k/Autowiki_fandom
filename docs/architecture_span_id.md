# Span Identification Architecture

Technical architecture and experimental design for the hyperlink span identification task on Fandom wikis. Use this document for implementation reference and paper writing.

---

## 1. Task Definition

### 1.1 Problem

Given a text segment (sentence, paragraph, or full article) extracted from a Fandom wiki article, **identify character spans that correspond to hyperlinks**. We predict *where* links appear in the text, not their target URLs or link types.

### 1.2 Input/Output

| | Description |
|---|---|
| **Input** | Plain text of a sentence, paragraph, or article (e.g., "Axel Foley is a Detroit cop who goes to Beverly Hills.") |
| **Output** | Set of character-level spans \((s, e)\) indicating link locations (e.g., \(\{(0, 10), (29, 41)\}\) for "Axel Foley" and "Beverly Hills") |

### 1.3 Scope

- **In scope:** Span boundaries (start, end character offsets)
- **Out of scope:** Link target prediction, link type classification, entity typing
- **Internal links only:** For training and evaluation we use only **internal** links (same-wiki targets). External and other link types are excluded to focus on in-wiki entity mentions.

---

## 2. End-to-End Pipeline

### 2.1 Overview

```
Fandom wiki (e.g. beverlyhillscop.fandom.com)
    │
    ▼  Step 1: Scraping
    │  scripts/01_Data_processing/00_scrape_fandom.py
    │  configs/scraping.yaml
    │
    ▼  data/raw/<domain>/           (HTML files: <article_id>.html)
    │  data/raw/url_lists/<domain>_urls.txt
    │
    ▼  Step 2: Ground truth build
    │  scripts/01_Data_processing/01_build_ground_truth.py <domain>
    │  configs/ground_truth.yaml
    │
    ▼  data/processed/<domain>/     (paragraphs, sentences, articles JSONL/CSV)
    │
    ▼  Step 3: Span identification
    │  scripts/02_Span_identification/01_run_span_id.py
    │  configs/span_id.yaml
    │
    ▼  data/span_id/<domain>/splits/     (train/val/test)
    │  data/span_id/<domain>/token_data/ (pre-tokenized BILOU)
    │  data/span_id/<domain>/checkpoints/
    │
    ▼  data/research/span_id_experiments.csv
```

### 2.2 Directory Layout

| Path | Contents |
|------|----------|
| `data/raw/<domain>/` | Raw HTML (`<article_id>.html`) and plain text (`.txt`) from scraping |
| `data/raw/url_lists/<domain>_urls.txt` | URL list used for scraping |
| `data/processed/<domain>/` | Paragraphs, sentences, articles JSONL/CSV with link spans |
| `data/span_id/<domain>/splits/` | Train/val/test split JSONL per granularity |
| `data/span_id/<domain>/token_data/<gran>_<model>/` | Pre-tokenized BILOU train/dev/test JSONL |
| `data/span_id/<domain>/checkpoints/<run_id>/` | Model checkpoints |
| `data/research/` | Experiment CSV (raw and aggregated) |
| `data/logs/<domain>/scraping/` | Scraping logs |
| `data/logs/<domain>/ground_truth/` | Ground truth build logs |
| `data/logs/<domain>/span_id/` | Span identification run logs |

---

## 3. Step 1: Scraping

### 3.1 Config

`configs/scraping.yaml`:

- **base_url:** Fandom wiki URL (e.g. `https://beverlyhillscop.fandom.com/`)
- **start_url:** AllPages or category URL for discovering articles
- **delay_seconds:** Delay between requests
- **use_api_fallback:** Fall back to MediaWiki API on 403

### 3.2 Output

- **HTML:** `data/raw/<domain>/<article_id>.html` (or URL-derived filename if no article ID)
- **Plain text:** `data/raw/<domain>/<article_id>.txt`
- **URL list:** `data/raw/url_lists/<domain>_urls.txt`
- **Logs:** `data/logs/<domain>/scraping/`

### 3.3 Command

```bash
python scripts/01_Data_processing/00_scrape_fandom.py
```

Domain is inferred from `base_url` (e.g. `beverlyhillscop.fandom.com` → `beverlyhillscop`).

---

## 4. Step 2: Ground Truth Build

### 4.1 Config

`configs/ground_truth.yaml`:

- **domain:** Wiki domain (overridable via CLI)
- **raw_dir**, **processed_dir:** Paths for input/output
- **outputs:** Toggles for paragraphs/sentences JSONL, CSV, etc.
- **paragraph:** Splitting constraints (max_chars, min_chars, etc.)

### 4.2 Processing

- Reads HTML from `data/raw/<domain>/`
- Parses `mw-parser-output`, extracts paragraphs (by `<p>`) and sentences
- Extracts links with character offsets; classifies as internal/external/category/file/other
- **Internal** = links to other articles on the **same wiki** (same subdomain only)
- Writes JSONL and CSV to `data/processed/<domain>/`

### 4.3 Output Files

| File | Description |
|------|-------------|
| `paragraphs_<domain>.jsonl` | Paragraph units with `paragraph_text`, `links` (rel char offsets) |
| `sentences_<domain>.jsonl` | Sentence units with `sentence_text`, `links` |
| `articles_page_granularity_<domain>.jsonl` | Full-article units with `article_plain_text`, `links` (abs char offsets) |
| `articles_<domain>.jsonl` | Article index |
| `*_links_*.csv` | Link tables |
| `manifest.json` | Manifest of outputs |

### 4.4 Processed Data Format

| Granularity | Text field | Span fields in links | Unit ID |
|-------------|------------|----------------------|---------|
| Sentence | `sentence_text` | `plain_text_rel_char_start`, `plain_text_rel_char_end` | `sentence_id` |
| Paragraph | `paragraph_text` | `plain_text_rel_char_start`, `plain_text_rel_char_end` | `paragraph_id` |
| Article | `article_plain_text` | `plain_text_char_start`, `plain_text_char_end` | `article_record_id` |

**Example:**
```json
{
  "granularity": "paragraph",
  "article_id": 1461,
  "paragraph_text": "...documenting the heroics of Axel Foley...the Beverly Hills Cop.",
  "links": [
    {"plain_text_rel_char_start": 219, "plain_text_rel_char_end": 230, "link_type": "internal", "anchor_text": "Axel Foley"},
    {"plain_text_rel_char_start": 301, "plain_text_rel_char_end": 319, "link_type": "internal", "anchor_text": "Beverly Hills Cop"}
  ]
}
```

### 4.5 Command

```bash
python scripts/01_Data_processing/01_build_ground_truth.py <domain>
# e.g. python scripts/01_Data_processing/01_build_ground_truth.py beverlyhillscop
```

---

## 5. Step 3: Span Identification

### 5.1 Config

`configs/span_id.yaml` (main); `span_id_skanda.yaml` and `span_id_kudremukh.yaml` for machine-specific runs.

Key parameters:

| Parameter | Purpose |
|-----------|---------|
| `domains` | Wiki domains to run on |
| `granularities` | `sentence`, `paragraph`, `article` |
| `models` | HuggingFace model names (e.g. `bert-base-uncased`) |
| `label_schemes` | `BIO`, `BILOU` (pipeline uses BILOU for training) |
| `seeds` | Random seeds for reproducibility |
| `data_fractions` | For learning curves (1.0 = full data) |
| `run_baselines` | Run rule/heuristic baselines first |
| `baselines` | `rule_capitalized`, `heuristic_anchor`, `random` |

### 5.2 Data Split Strategy

- **Split by:** `article_id` (no article overlap between train/val/test)
- **Ratios:** 70% train, 15% val, 15% test (config-driven)
- **Location:** `data/span_id/<domain>/splits/`  
  - `train_<granularity>.jsonl`, `val_<granularity>.jsonl`, `test_<granularity>.jsonl`

### 5.3 Preprocessing (BILOU)

- **Internal links only:** Gold spans taken only from `link_type == "internal"`
- **Label scheme:** BILOU (O, B-SPAN, I-SPAN, L-SPAN, U-SPAN)
- **Overlap logic:** A token is labeled if it overlaps a gold span (any overlap)
- **Output:** Pre-tokenized JSONL (`train.jsonl`, `dev.jsonl`, `test.jsonl`) per `(granularity, model)`

### 5.4 Model Architecture

- **Encoder:** HuggingFace `AutoModelForTokenClassification` (e.g. BERT-base-uncased)
- **Training:** HuggingFace Trainer; early stopping on validation span F1
- **Decoding:** Predicted labels → token spans → character spans via tokenizer offset mapping

### 5.5 Checkpointing

- **Best:** Best validation span F1
- **Last:** Final epoch
- **Path:** `data/span_id/<domain>/checkpoints/<run_id>/<gran>_<domain>_<model>_seed<seed>_frac<frac>/`

### 5.6 Metrics

| Metric | Description |
|--------|-------------|
| **val_span_f1** | Validation span F1 (for model selection) |
| **span_f1** | Test span F1 (primary reporting metric) |
| **span_precision**, **span_recall** | Test span-level |
| **token_f1** | seqeval token F1 |
| **exact_match_pct** | Fraction of gold spans exactly predicted |

### 5.7 Commands

```bash
# Main (uses configs/span_id.yaml)
python scripts/02_Span_identification/01_run_span_id.py

# Skanda (1 GPU)
python scripts/02_Span_identification/01_run_span_id_skanda.py

# Kudremukh (4 GPUs)
python scripts/02_Span_identification/01_run_span_id_kudremukh.py
```

---

## 6. Baselines

| Baseline | Strategy |
|----------|----------|
| **rule_capitalized** | Regex: spans matching Title Case patterns |
| **heuristic_anchor** | Regex: capitalized phrases and wiki-like anchor patterns |
| **random** | Random spans with similar count/length distribution to gold |

Baselines run on the same splits as models; results are written to the same research CSV.

---

## 7. Experiments

### 7.1 Sweep Dimensions

| Dimension | Example values | Purpose |
|-----------|----------------|---------|
| **Domain** | beverlyhillscop, money-heist | Wiki/fandom |
| **Granularity** | sentence, paragraph, article | Text unit size |
| **Model** | bert-base-uncased | Encoder |
| **Label scheme** | BILOU | (currently fixed in pipeline) |
| **Seed** | 42, 123, 456 | Reproducibility and variance |
| **Data fraction** | 1.0, 0.5 | Learning curves |

### 7.2 Research CSV

**Path:** `data/research/span_id_experiments.csv`

**Columns:**

| Column | Description |
|--------|-------------|
| run_id | Timestamp-based run identifier |
| timestamp | ISO timestamp |
| seed | Random seed (-1 for baselines) |
| experiment_type | `baseline` or `model` |
| granularity | sentence, paragraph, article |
| domain | Wiki domain |
| model | Model name or baseline name |
| label_scheme | BILOU (or empty for baselines) |
| data_fraction | 1.0 = full data |
| train_size, val_size | Example counts |
| val_span_f1 | Validation span F1 |
| span_f1, span_precision, span_recall | Test metrics |
| token_f1 | Test token F1 |
| exact_match_pct | Test exact match |
| wall_time_sec | Runtime |
| checkpoint_path | Path to best checkpoint |
| notes | Optional (e.g. skanda, kudremukh) |

One row per (run_id, granularity, domain, model/baseline, seed).

### 7.3 Experiment Workflow

#### 1. Run full sweep

```bash
python scripts/02_Span_identification/01_run_span_id.py
```

Runs baselines first (if enabled), then models across domains, granularities, seeds. Appends to `span_id_experiments.csv`.

#### 2. Aggregate across seeds

```bash
python scripts/02_Span_identification/aggregate_seed_results.py
# Output: data/research/span_id_experiments_aggregated.csv
```

Computes mean ± std for each (domain, granularity, model, data_fraction) across seeds.

Filter by run:
```bash
python scripts/02_Span_identification/aggregate_seed_results.py --run-id 20260223_120000
```

#### 3. Compare model vs baseline (significance)

```bash
python scripts/02_Span_identification/compare_models.py \
  --csv data/research/span_id_experiments.csv \
  --baseline rule_capitalized \
  --model bert-base-uncased \
  --metric span_f1
```

Reports mean ± std and bootstrap p-value (paired permutation test) when multiple seeds exist.

#### 4. Error analysis

```bash
python scripts/02_Span_identification/03_error_analysis.py
```

#### 5. Human eval sample

```bash
python scripts/02_Span_identification/04_human_eval_sample.py
```

### 7.4 Ablations (Config-Driven)

- **Label scheme:** Change `label_schemes` (BIO vs BILOU)
- **Data fraction:** `data_fractions: [0.1, 0.25, 0.5, 1.0]` for learning curves
- **Max length:** `model.max_length` (64, 128, 256, 512)

### 7.5 Cross-Domain Transfer (Optional)

Set `transfer_experiments: true` in config to train on one domain and evaluate on another.

---

## 8. Reproducibility

- **Seeds:** `fix_random_seeds: true` sets Python, NumPy, PyTorch seeds
- **Config-driven:** Paths, hyperparameters, sweep options in YAML configs
- **Split persistence:** `split.recreate_if_exists: false` reuses existing splits

---

## 9. Data Flow Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │  data/processed/<domain>/                   │
                    │  paragraphs_*.jsonl, sentences_*.jsonl      │
                    └──────────────────┬──────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │  Split by article_id (70/15/15)             │
                    │  data/span_id/<domain>/splits/              │
                    └──────────────────┬──────────────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
    ┌──────────┐              ┌──────────────┐              ┌──────────┐
    │  Train   │              │  Validation  │              │   Test   │
    └────┬─────┘              └──────┬───────┘              └────┬─────┘
         │                           │                           │
         │    Pre-tokenize: text → BILOU labels, token_ids       │
         │    data/span_id/<domain>/token_data/                  │
         ▼                           │                           │
    ┌──────────┐                     │                           │
    │  HF      │◄────────────────────┘                           │
    │  Trainer │   Early stop on val_span_f1                     │
    └────┬─────┘                                                  │
         │                                                        │
         ▼                                                        │
    data/span_id/<domain>/checkpoints/                            │
         │                                                        │
         └────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │  Evaluate: span F1, token F1, exact match   │
                    │  Append to span_id_experiments.csv          │
                    └─────────────────────────────────────────────┘
```

---

## 10. File Reference

| Path | Purpose |
|------|---------|
| `configs/scraping.yaml` | Scraping config |
| `configs/ground_truth.yaml` | Ground truth build config |
| `configs/span_id.yaml` | Span ID main config |
| `configs/span_id_skanda.yaml` | Skanda (1 GPU) config |
| `configs/span_id_kudremukh.yaml` | Kudremukh (4 GPU) config |
| `src/data_scraping/scrape_pipeline.py` | Scraping logic |
| `src/data_processing/ground_truth.py` | Ground truth parsing |
| `src/span_identification/dataset.py` | Data loading, splits |
| `src/span_identification/preprocess.py` | BILOU encoding, token datasets |
| `src/span_identification/hf_trainer.py` | HF Trainer integration |
| `src/span_identification/span_metrics.py` | seqeval + span metrics |
| `src/span_identification/stats.py` | Bootstrap significance |
| `scripts/01_Data_processing/00_scrape_fandom.py` | Scraping entry point |
| `scripts/01_Data_processing/01_build_ground_truth.py` | Ground truth entry point |
| `scripts/02_Span_identification/01_run_span_id.py` | Main experiment entry point |
| `scripts/02_Span_identification/aggregate_seed_results.py` | Seed aggregation |
| `scripts/02_Span_identification/compare_models.py` | Model vs baseline comparison |

---

## 11. Paper Sections Mapping

| Paper section | Use this document |
|---------------|-------------------|
| Task definition | §1 |
| Data pipeline | §2–4 |
| Model | §5 |
| Evaluation | §5.6 |
| Baselines | §6 |
| Experiments | §7 |
| Reproducibility | §8 |
