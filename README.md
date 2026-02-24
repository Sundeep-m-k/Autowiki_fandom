# Autowiki Fandom

Pipeline for span identification on Fandom wiki articles.

## Setup

```bash
pip install -r requirements-span-id.txt  # or install from pyproject.toml
```

## Pipeline

1. **Scrape**: `python scripts/01_Data_processing/00_scrape_fandom.py` — scrapes HTML from Fandom
2. **Ground truth**: `python scripts/01_Data_processing/01_build_ground_truth.py <domain>` — builds processed paragraphs/sentences with link spans
3. **Span identification**: `python scripts/02_Span_identification/01_run_span_id.py` — trains/evaluates span detection models
