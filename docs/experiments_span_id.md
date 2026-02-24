# Span Identification: Running Experiments

This guide explains how to run span identification experiments for research.

---

## Quick Start

```bash
# Full sweep (baselines + models) from project root
python scripts/02_Span_identification/01_run_span_id.py
```

Results are appended to `data/research/span_id_experiments.csv`.

---

## Configuration

Edit `configs/span_id.yaml` to change:

| Parameter | Purpose |
|-----------|---------|
| `domains` | Wiki domains (e.g. `["beverlyhillscop"]`) |
| `granularities` | `sentence`, `paragraph`, `article` |
| `models` | HuggingFace model names |
| `seeds` | Random seeds for reproducibility |
| `data_fractions` | Learning curves (1.0 = full data) |
| `run_baselines` | Run rule/heuristic baselines first |
| `baselines` | `rule_capitalized`, `heuristic_anchor`, `random` |

---

## Output Layout

| Path | Contents |
|------|----------|
| `data/research/span_id_experiments.csv` | Per-run metrics (one row per experiment) |
| `data/<domain>/span_id/` | Logs |
| `data/span_id/<domain>/checkpoints/<run_id>/` | Model checkpoints |
| `data/span_id/<domain>/token_data/<gran>_<model>/` | Pre-tokenized train/dev/test JSONL |

---

## CSV Columns

- **val_span_f1** – Validation F1 (model selection)
- **span_f1** – Test F1 (final reporting metric)
- **span_precision**, **span_recall** – Test split
- **token_f1** – Token-level seqeval F1

---

## Research Workflow

### 1. Run full sweep

```bash
python scripts/02_Span_identification/01_run_span_id.py
```

### 2. Aggregate across seeds

```bash
python scripts/02_Span_identification/aggregate_seed_results.py
# Output: data/research/span_id_experiments_aggregated.csv
```

Filter by run:

```bash
python scripts/02_Span_identification/aggregate_seed_results.py --run-id 20260223_120000
```

### 3. Compare model vs baseline

```bash
python scripts/02_Span_identification/compare_models.py \
  --csv data/research/span_id_experiments.csv \
  --baseline rule_capitalized \
  --model bert-base-uncased \
  --metric span_f1
```

Reports mean ± std and bootstrap p-value (when multiple seeds).

---

## Single-Config Run

To run one configuration (e.g. for debugging), edit `configs/span_id.yaml`:

```yaml
domains: ["beverlyhillscop"]
granularities: ["sentence"]
models: ["bert-base-uncased"]
seeds: [42]
data_fractions: [1.0]
run_baselines: false  # optional
```

---

## Reproducibility

- `fix_random_seeds: true` sets Python/NumPy/PyTorch seeds
- Each model run uses the seed from config
- `run_id` (timestamp) tags each sweep for traceability
