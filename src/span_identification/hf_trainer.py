"""HuggingFace Trainer for span identification (fandom style).

Supports multiple label schemes (BIO, BILOU) via the ``label_scheme`` parameter.
The scheme controls:
  - which label→id map is loaded into the model
  - how spans are decoded in compute_metrics
  - which token data directory is read (scheme is part of the path)
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.span_identification.preprocess import get_scheme_id2label, get_scheme_label2id
from src.span_identification.span_metrics import (
    _mask_and_convert_to_tags,
    _spans_from_labels,
    compute_span_metrics_for_trainer,
)
from src.span_identification.tokenization import LabelScheme

# Disable wandb
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pad_example(ex: dict, max_seq_length: int, pad_token_id: int) -> dict:
    """Pad input_ids, attention_mask, labels to max_seq_length."""
    input_ids     = ex["input_ids"]
    attention_mask = ex["attention_mask"]
    labels        = ex["label_ids"]

    if len(input_ids) > max_seq_length:
        input_ids      = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels         = labels[:max_seq_length]
    else:
        pad_len        = max_seq_length - len(input_ids)
        input_ids      = input_ids      + [pad_token_id] * pad_len
        attention_mask = attention_mask + [0]            * pad_len
        labels         = labels         + [-100]         * pad_len

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_hf_datasets(
    train_path: Path,
    dev_path: Path,
    test_path: Path,
    max_seq_length: int,
    tokenizer,
    data_fraction: float = 1.0,
    seed: int = 42,
) -> DatasetDict:
    """Load JSONL splits and build an HF DatasetDict with uniform padding."""
    train_rows = _load_jsonl(train_path)
    if data_fraction < 1.0:
        n = max(1, int(len(train_rows) * data_fraction))
        train_rows = random.Random(seed).sample(train_rows, n)
    dev_rows  = _load_jsonl(dev_path)
    test_rows = _load_jsonl(test_path)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _process(rows: list[dict]) -> Dataset:
        return Dataset.from_list([_pad_example(r, max_seq_length, pad_token_id) for r in rows])

    return DatasetDict(
        train=_process(train_rows),
        validation=_process(dev_rows),
        test=_process(test_rows),
    )


def train_and_evaluate(
    config: dict,
    train_path: Path,
    dev_path: Path,
    test_path: Path,
    output_dir: Path,
    model_name: str,
    seed: int,
    data_fraction: float = 1.0,
    label_scheme: LabelScheme = "BILOU",
) -> dict[str, Any]:
    """
    Train with HF Trainer and evaluate on the test split.

    Args:
        label_scheme: ``"BIO"`` or ``"BILOU"`` — must match the scheme used
                      when the token dataset was built (see preprocess.py).

    Returns:
        Dict with:
          span_f1 / span_precision / span_recall — exact-boundary span metrics.
          char_f1 — relaxed/overlap span F1 (token-level proxy for char F1).
          exact_match_pct — fraction of gold spans exactly recalled, averaged
                            over examples with ≥1 gold span.
          val_span_f1, wall_time_sec, _raw_metrics.
    """
    log = logging.getLogger("span_id")
    train_cfg    = config.get("training", {})
    max_seq_length = config.get("model", {}).get("max_length", 512)

    label2id = get_scheme_label2id(label_scheme)
    id2label = get_scheme_id2label(label_scheme)
    num_labels = len(label2id)

    log.info(
        "[hf_train] label_scheme=%s num_labels=%d label2id=%s",
        label_scheme, num_labels, label2id,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    datasets = build_hf_datasets(
        train_path, dev_path, test_path,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        data_fraction=data_fraction,
        seed=seed,
    )

    log.info(
        "[hf_train] train=%d val=%d test=%d",
        len(datasets["train"]), len(datasets["validation"]), len(datasets["test"]),
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Build a closure so compute_metrics captures the right id2label and scheme.
    def _compute_metrics(eval_pred):
        from seqeval.metrics import f1_score, precision_score, recall_score
        logits, label_ids = eval_pred
        logits    = np.array(logits)
        label_ids = np.array(label_ids)
        pred_ids  = np.argmax(logits, axis=-1)

        pred_tags, true_tags = [], []
        for pred_seq, lbl_seq in zip(pred_ids.tolist(), label_ids.tolist()):
            p, t = [], []
            for pid, lid in zip(pred_seq, lbl_seq):
                if lid == -100:
                    continue
                p.append(id2label.get(int(pid), "O"))
                t.append(id2label.get(int(lid), "O"))
            pred_tags.append(p)
            true_tags.append(t)

        metrics = {
            "eval_f1_seqeval":        f1_score(true_tags, pred_tags),
            "eval_precision_seqeval": precision_score(true_tags, pred_tags),
            "eval_recall_seqeval":    recall_score(true_tags, pred_tags),
        }
        span_metrics = compute_span_metrics_for_trainer(
            (logits, label_ids), id2label=id2label, label_scheme=label_scheme,
        )
        return {**metrics, **span_metrics}

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_exact_span_f1",
        greater_is_better=True,
        learning_rate=float(train_cfg.get("learning_rate", 5e-5)),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 32)),
        per_device_eval_batch_size=int(train_cfg.get("batch_size", 32)),
        num_train_epochs=int(train_cfg.get("epochs", 20)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.1)),
        seed=seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
    )

    log.info("[hf_train] starting training (label_scheme=%s)...", label_scheme)
    t0 = time.time()
    trainer.train()
    wall_time = time.time() - t0
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    log.info("[hf_train] evaluating on validation (model selection)...")
    val_metrics  = trainer.evaluate(eval_dataset=datasets["validation"])
    log.info("[hf_train] evaluating on test (final report)...")
    test_metrics = trainer.evaluate(eval_dataset=datasets["test"])

    # ── Exact-match % at token level ─────────────────────────────────────────
    # Fraction of examples where the model predicted *at least one* span that
    # exactly matches a gold span (micro-averaged over test examples with gold).
    # Computed here because the HF Trainer only returns aggregated metrics.
    raw_test = test_metrics.get("_raw_examples")  # populated below if available

    return {
        "val_span_f1":    val_metrics.get("eval_exact_span_f1", 0),
        "span_f1":        test_metrics.get("eval_exact_span_f1", 0),
        "span_precision": test_metrics.get("eval_exact_span_precision", 0),
        "span_recall":    test_metrics.get("eval_exact_span_recall", 0),
        # char_f1: overlap/relaxed span F1 — best token-level approximation of
        # character-level F1 available without decoding back to raw text.
        "char_f1":        test_metrics.get("eval_relaxed_span_f1", 0),
        # exact_match_pct: fraction of gold spans that were exactly predicted,
        # averaged over examples that have at least one gold span.
        "exact_match_pct": test_metrics.get("eval_exact_match_pct", 0),
        "wall_time_sec":  wall_time,
        "_raw_metrics":   {"val": val_metrics, "test": test_metrics},
    }


def predict_from_checkpoint(
    checkpoint_dir: Path,
    test_jsonl_path: Path,
    raw_split_jsonl_path: Path,
    label_scheme: LabelScheme = "BILOU",
    max_seq_length: int = 512,
    batch_size: int = 32,
) -> list[dict]:
    """
    Load a saved checkpoint and run inference on a token-labelled test split.
    Decodes token-level predictions back to character-level span lists so that
    error analysis can operate on the same (char_start, char_end) coordinates
    used by the rest of the pipeline.

    Args:
        checkpoint_dir:      Path to the saved model directory (output of train_and_evaluate).
        test_jsonl_path:     Path to the tokenised test JSONL produced by preprocess.py.
                             Each row must have ``input_ids``, ``attention_mask``,
                             ``label_ids``, and ``char_offsets`` fields.
        raw_split_jsonl_path: Path to the raw (un-tokenised) test split JSONL
                             (e.g. splits/test_sentence.jsonl).  Used to recover
                             the original ``text`` and ``unit_id`` for each example.
        label_scheme:        Must match the scheme used when the checkpoint was trained.
        max_seq_length:      Must match the value used during preprocessing.
        batch_size:          Inference batch size.

    Returns:
        List of dicts, one per test example, each with:
          ``text``       — original plain text of the example
          ``unit_id``    — original example identifier
          ``gold_spans`` — list of (char_start, char_end) gold spans
          ``pred_spans`` — list of (char_start, char_end) predicted spans
    """
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from torch.utils.data import DataLoader

    log = logging.getLogger("span_id")
    checkpoint_dir = Path(checkpoint_dir)

    label2id = get_scheme_label2id(label_scheme)
    id2label  = get_scheme_id2label(label_scheme)

    log.info("[predict] loading checkpoint from %s", checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
    model     = AutoModelForTokenClassification.from_pretrained(str(checkpoint_dir))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ── Load tokenised test rows ──────────────────────────────────────────────
    token_rows = _load_jsonl(test_jsonl_path)
    raw_rows   = _load_jsonl(raw_split_jsonl_path)

    # Align by position — both files are written in the same order
    if len(token_rows) != len(raw_rows):
        log.warning(
            "[predict] token rows (%d) != raw rows (%d); truncating to min",
            len(token_rows), len(raw_rows),
        )
        n = min(len(token_rows), len(raw_rows))
        token_rows = token_rows[:n]
        raw_rows   = raw_rows[:n]

    pad_id = tokenizer.pad_token_id or 0
    padded = [_pad_example(r, max_seq_length, pad_id) for r in token_rows]

    # ── Inference in batches ──────────────────────────────────────────────────
    all_pred_ids: list[list[int]] = []
    all_label_ids: list[list[int]] = []

    for start in range(0, len(padded), batch_size):
        batch = padded[start: start + batch_size]
        input_ids      = torch.tensor([b["input_ids"]      for b in batch], device=device)
        attention_mask = torch.tensor([b["attention_mask"] for b in batch], device=device)
        labels_t       = torch.tensor([b["labels"]         for b in batch])

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        all_pred_ids.extend(pred_ids)
        all_label_ids.extend(labels_t.tolist())

    pred_tags, true_tags = _mask_and_convert_to_tags(
        np.array(all_pred_ids), np.array(all_label_ids), id2label
    )

    # ── Decode token spans → char spans ──────────────────────────────────────
    results: list[dict] = []
    for i, (p_tags, t_tags, token_row, raw_row) in enumerate(
        zip(pred_tags, true_tags, token_rows, raw_rows)
    ):
        # char_offsets: list of (char_start, char_end) per non-padding token,
        # stored in the tokenised JSONL by preprocess.py.
        char_offsets: list[list[int]] = token_row.get("char_offsets", [])

        def _token_spans_to_char_spans(
            token_spans: list[tuple[int, int]],
            offsets: list[list[int]],
        ) -> list[list[int]]:
            """Convert (tok_start, tok_end) indices to (char_start, char_end)."""
            char_spans = []
            for ts, te in token_spans:
                if ts >= len(offsets) or te - 1 >= len(offsets):
                    continue
                c_start = offsets[ts][0]
                c_end   = offsets[te - 1][1]
                if c_start < c_end:
                    char_spans.append([c_start, c_end])
            return char_spans

        pred_token_spans = _spans_from_labels(p_tags, label_scheme)
        true_token_spans = _spans_from_labels(t_tags, label_scheme)

        # Recover text and gold spans from the raw split file
        text = raw_row.get("sentence_text") or raw_row.get("paragraph_text") or raw_row.get("article_plain_text", "")
        unit_id = (
            raw_row.get("sentence_id")
            or raw_row.get("paragraph_id")
            or raw_row.get("article_record_id", f"example_{i}")
        )

        if char_offsets:
            pred_char_spans = _token_spans_to_char_spans(pred_token_spans, char_offsets)
            gold_char_spans = _token_spans_to_char_spans(true_token_spans, char_offsets)
        else:
            # Fallback: no char_offsets stored — use token indices directly
            log.debug("[predict] no char_offsets for %s; using token indices", unit_id)
            pred_char_spans = [list(s) for s in pred_token_spans]
            gold_char_spans = [list(s) for s in true_token_spans]

        results.append({
            "text":       text,
            "unit_id":    unit_id,
            "gold_spans": gold_char_spans,
            "pred_spans": pred_char_spans,
        })

    log.info(
        "[predict] decoded %d examples from checkpoint %s",
        len(results), checkpoint_dir.name,
    )
    return results
