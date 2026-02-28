"""Fine-tuning a cross-encoder re-ranker on Fandom wiki retrieval data.

Training data is mined directly from the retrieval results already on disk:
  - Positive: (query_text, gold_article_text) — the correct target article
  - Hard negatives: (query_text, retrieved_article_text) for the top-N retrieved
    articles that are NOT the gold, drawn from a chosen retriever's results.

This avoids a separate negative-mining step — the retrieval results (step 02)
are the natural source of informative hard negatives.

Only training-split queries are used (source articles from the Task 1 train split)
so the fine-tuned model is never contaminated by test examples.

Output
------
  data/article_retrieval/<domain>/reranker_training/reranker_train_*.jsonl
      — one JSONL per (query, positive, negative) triple
  data/article_retrieval/checkpoints/reranker_finetuned/
      — saved CrossEncoder model, ready to load with CrossEncoder(path)
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Callable

log = logging.getLogger("article_retrieval")


# ── Training data builder ──────────────────────────────────────────────────────

def build_training_examples(
    retrieval_path: Path,
    query_records_by_id: dict[str, dict],
    article_lookup: dict[int, str],
    version: int,
    n_hard_negatives: int,
    train_source_ids: set[int],
) -> list[dict]:
    """
    Convert retrieval results into (query, positive, negative) training triples.

    Args:
        retrieval_path:     JSONL of retrieval results (step 02 output).
        query_records_by_id: query_id → query record (for query text lookup).
        article_lookup:     article_id → article text.
        version:            Query version integer to use for query text.
        n_hard_negatives:   Max number of hard negatives to mine per query.
        train_source_ids:   Set of source article IDs in the train split.
                            Only queries whose source_article_id is in this set
                            are used — ensures no test leakage.

    Returns:
        List of example dicts with keys: query, positive, negative.
    """
    version_key = f"v{version}"
    examples: list[dict] = []

    with open(retrieval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # Only use training-split queries
            if rec.get("source_article_id") not in train_source_ids:
                continue

            query_id     = rec["query_id"]
            gold_id      = rec.get("gold_article_id")
            qrec         = query_records_by_id.get(query_id)
            if qrec is None or gold_id is None:
                continue

            query_text = qrec.get("queries", {}).get(version_key, "")
            if not query_text:
                continue

            positive_text = article_lookup.get(gold_id, "")
            if not positive_text:
                continue

            # Mine hard negatives: top retrieved articles that are NOT the gold
            negatives = [
                article_lookup[c["article_id"]]
                for c in rec.get("retrieved", [])
                if c.get("article_id") != gold_id
                and c.get("article_id") != rec.get("source_article_id")
                and c.get("article_id") in article_lookup
            ][:n_hard_negatives]

            for negative_text in negatives:
                examples.append({
                    "query":    query_text,
                    "positive": positive_text,
                    "negative": negative_text,
                })

    log.info(
        "[reranker_trainer] mined %d training examples from %s",
        len(examples), retrieval_path.name,
    )
    return examples


def save_training_examples(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info("[reranker_trainer] saved %d training examples → %s", len(examples), path)


def load_training_examples(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ── Fine-tuning ────────────────────────────────────────────────────────────────

def train_reranker(
    examples: list[dict],
    base_model: str,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 512,
    seed: int = 42,
) -> None:
    """
    Fine-tune a CrossEncoder on (query, positive, negative) triples.

    Uses sentence-transformers' CrossEncoder with a standard binary
    cross-entropy loss: positive pairs are labelled 1.0, negative pairs 0.0.

    Args:
        examples:       List of dicts with keys: query, positive, negative.
        base_model:     HuggingFace model name or local path to initialise from.
        output_dir:     Where to save the fine-tuned CrossEncoder.
        epochs:         Number of training epochs.
        batch_size:     Per-device training batch size.
        learning_rate:  AdamW learning rate.
        warmup_ratio:   Fraction of total steps used for linear warmup.
        max_length:     Maximum token length for the cross-encoder.
        seed:           Random seed for reproducibility.
    """
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers and torch are required for reranker training. "
            "Install with: pip install sentence-transformers torch"
        ) from exc

    import torch

    random.seed(seed)
    torch.manual_seed(seed)

    log.info(
        "[reranker_trainer] fine-tuning %s on %d examples  "
        "epochs=%d  batch_size=%d  lr=%.2e",
        base_model, len(examples), epochs, batch_size, learning_rate,
    )

    model = CrossEncoder(base_model, max_length=max_length)

    # Build flat list of (text_pair, label) — one positive + one negative per example
    train_samples: list[tuple[list[str], float]] = []
    for ex in examples:
        train_samples.append(([ex["query"], ex["positive"]], 1.0))
        train_samples.append(([ex["query"], ex["negative"]], 0.0))

    random.shuffle(train_samples)

    # sentence-transformers CrossEncoder.fit() accepts InputExample objects
    from sentence_transformers import InputExample
    st_examples = [
        InputExample(texts=pair, label=label)
        for pair, label in train_samples
    ]

    train_dataloader = DataLoader(st_examples, shuffle=True, batch_size=batch_size)
    warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        show_progress_bar=True,
    )

    log.info("[reranker_trainer] fine-tuned model saved → %s", output_dir)
