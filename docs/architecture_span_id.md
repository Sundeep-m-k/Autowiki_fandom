# Span Identification Architecture

**Document purpose:** Technical architecture and experimental design for the hyperlink span identification task on Fandom wikis. Use this document for clear understanding when writing the paper.

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
- **Out of scope:** Link target prediction, link type (internal/external), entity typing

---

## 2. Data Pipeline

### 2.1 Upstream Processing

```
Fandom HTML pages
    → Scraping (configs/scraping.yaml)
    → data/raw/<domain>/
    → Ground truth build (configs/ground_truth.yaml)
    → data/processed/<domain>/
```

### 2.2 Processed Data Format

Units are stored as JSON Lines per granularity:

| Granularity | Text field | Span fields in links | Unit ID |
|-------------|------------|----------------------|---------|
| Sentence | `sentence_text` | `plain_text_rel_char_start`, `plain_text_rel_char_end` | `sentence_id` |
| Paragraph | `paragraph_text` | `plain_text_rel_char_start`, `plain_text_rel_char_end` | `paragraph_id` |
| Article | `article_plain_text` | `plain_text_char_start`, `plain_text_char_end` | `article_record_id` |

Articles use absolute character offsets; sentences and paragraphs use relative offsets within the unit.

**Example:** A paragraph with links "Axel Foley" (chars 219–230) and "Beverly Hills Cop" (chars 301–319) becomes:

```json
{
  "paragraph_text": "...documenting the heroics of Axel Foley...the Beverly Hills Cop.",
  "links": [
    {"plain_text_rel_char_start": 219, "plain_text_rel_char_end": 230, "anchor_text": "Axel Foley"},
    {"plain_text_rel_char_start": 301, "plain_text_rel_char_end": 319, "anchor_text": "Beverly Hills Cop"}
  ]
}
```

Gold spans are extracted as \((start, end)\) tuples from `links`.

### 2.3 Data Split Strategy

- **Split by:** `article_id` (no overlap between train/val/test)
- **Ratios:** 70% train, 15% validation, 15% test (config-driven)
- **Location:** `data/span_id/<domain>/splits/`  
  - `train_<granularity>.jsonl`, `val_<granularity>.jsonl`, `test_<granularity>.jsonl`

---

## 3. Model Architecture

### 3.1 Formulation as Sequence Labeling

Span identification is framed as **token classification** over a pretrained encoder:

1. Tokenize text (BERT/RoBERTa tokenizer)
2. Map gold character spans → token spans → label sequence (BIO/BIEOS/BILOU/IO)
3. Predict label per token
4. Map predicted labels → token spans → character spans

### 3.2 Label Schemes

| Scheme | Labels | Example for span "Axel Foley" |
|--------|--------|------------------------------|
| **BIO** | O, B, I | `B I I O O O ...` |
| **BIEOS** | O, B, I, E, S | `B I E O O O ...` or `S O O ...` for single-token spans |
| **BILOU** | O, B, I, L, U | `B I L O O O ...` or `U O O ...` for single-token spans |
| **IO** | O, I | `I I O O O ...` |

### 3.3 Model Components

```
Input: token_ids, attention_mask
    → Pretrained encoder (BERT-base, RoBERTa, etc.)
    → Sequence representation [batch, seq_len, hidden]
    → Dropout
    → Linear classifier [hidden → num_labels]
    → Logits [batch, seq_len, num_labels]
Output: predicted labels per token → decoded spans
```

- **Encoder:** HuggingFace `AutoModel` (BERT-base-uncased, RoBERTa-base, etc.)
- **Classifier:** Single linear layer over encoder hidden states
- **Inference:** Decode BIO/BIEOS/IO labels → token spans → character spans via tokenizer offset mapping

### 3.4 Subword Alignment

Character spans are mapped to token spans using the tokenizer's `offset_mapping`. Only tokens with non-zero offsets (content tokens) are labeled; special tokens ([CLS], [SEP]) and padding are masked.

---

## 4. Training

### 4.1 Objective

Cross-entropy loss over token labels. Padding and special tokens use `ignore_index = -100`.

### 4.2 Hyperparameters (Config-Driven)

| Parameter | Default | Config path |
|-----------|---------|-------------|
| Epochs | 10 | `training.epochs` |
| Learning rate | 5e-5 | `training.learning_rate` |
| Batch size | 32 | `training.batch_size` |
| Weight decay | 0.01 | `training.weight_decay` |
| Max sequence length | 512 | `model.max_length` |
| Dropout | 0.1 | `model.dropout` |
| Optimizer | AdamW | `optimizer` |
| Scheduler | Linear | `scheduler` |

### 4.3 Checkpointing

- **Best:** Best validation span F1
- **Last:** Final epoch
- **Per-epoch:** Optional (config: `checkpoint.save_every_epoch`)
- **Path:** `data/checkpoints/<run_id>/<granularity>_<domain>_<model>_seed<seed>/`

---

## 5. Evaluation

### 5.1 Metrics

| Metric | Description |
|--------|-------------|
| **Span F1** | Span-level precision, recall, F1 (exact span match) |
| **Token F1** | Character-position-level precision, recall, F1 |
| **Exact Match %** | Fraction of gold spans exactly predicted |
| **Overlap F1** | Span F1 with overlap-based matching (any overlap counts as match) |

### 5.2 Aggregation

- **Micro-averaging:** Average metrics over all examples in the evaluation set
- **Per-seed:** Metrics for each random seed; report mean ± std across seeds

---

## 6. Baselines

| Baseline | Strategy |
|----------|----------|
| **rule_capitalized** | Regex: link spans that match Title Case patterns (e.g., "Axel Foley") |
| **heuristic_anchor** | Regex: capitalized phrases and wiki-like anchor patterns |
| **random** | Random spans with similar count and length distribution to gold |

---

## 7. Experimental Design

### 7.1 Sweep Dimensions

| Dimension | Values | Purpose |
|-----------|--------|---------|
| **Domain** | beverlyhillscop, ... | Wiki/fandom |
| **Granularity** | sentence, paragraph, article | Text unit size |
| **Model** | bert-base-uncased, roberta-base, ... | Encoder |
| **Label scheme** | BIO, BIEOS, IO | Sequence labeling format |
| **Seed** | 42, 123, 456 | Reproducibility, variance |

### 7.2 Ablations (Optional)

- Label scheme: BIO vs BIEOS vs IO
- Context window: max_length ∈ {64, 128, 256, 512}
- Data fraction: 10%, 25%, 50%, 100% (learning curves)

### 7.3 Cross-Domain Transfer (Optional)

Train on domain A, evaluate on domain B.

### 7.4 Output

- **Research CSV:** `data/research/span_id_experiments.csv`  
  - One row per (domain, granularity, model, label_scheme, seed, ...)  
  - Columns: span_f1, span_precision, span_recall, token_f1, exact_match_pct, wall_time_sec, checkpoint_path, etc.
- **Checkpoints:** Saved for every run for later analysis and inference.

---

## 8. Reproducibility

- **Fixed seeds:** PyTorch, NumPy, Python `random` (config: `fix_random_seeds`)
- **Config-driven:** All paths, hyperparameters, and sweep options in `configs/span_id.yaml`
- **Split persistence:** Train/val/test splits saved; reuse with `split.recreate_if_exists: false`

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
                    │  data/span_id/<domain>/splits/             │
                    └──────────────────┬──────────────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
    ┌──────────┐              ┌──────────────┐              ┌──────────┐
    │  Train   │              │  Validation  │              │   Test   │
    └────┬─────┘              └──────┬───────┘              └────┬─────┘
         │                           │                           │
         │    Encode: text → BIO labels, token_ids               │
         ▼                           │                           │
    ┌──────────┐                     │                           │
    │  Model   │◄────────────────────┘                           │
    │  Train   │   Early stop on val F1                           │
    └────┬─────┘                                                  │
         │                                                        │
         ▼                                                        │
    Checkpoints                                                   │
    data/checkpoints/                                             │
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

## 10. Paper Sections Mapping

| Paper section | Use this document |
|---------------|-------------------|
| **Task definition** | §1 |
| **Data** | §2 |
| **Model** | §3 |
| **Training** | §4 |
| **Evaluation** | §5 |
| **Baselines** | §6 |
| **Experimental setup** | §7 |
| **Reproducibility** | §8 |

---

## 11. File Reference

| Path | Purpose |
|------|---------|
| `configs/span_id.yaml` | All configuration |
| `src/span_identification/dataset.py` | Data loading, splits |
| `src/span_identification/tokenization.py` | BIO/BIEOS encoding |
| `src/span_identification/models/token_classifier.py` | Model |
| `src/span_identification/trainer.py` | Training loop |
| `src/span_identification/evaluator.py` | Metrics |
| `scripts/02_Span_identification/01_run_span_id.py` | Main sweep entry point |
