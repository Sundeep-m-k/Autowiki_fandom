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
from src.span_identification.span_metrics import compute_span_metrics_for_trainer
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
        Dict with span_f1, span_precision, span_recall, char_f1,
        exact_match_pct, val_span_f1, wall_time_sec, and _raw_metrics.
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

    return {
        "val_span_f1":    val_metrics.get("eval_exact_span_f1", 0),
        "span_f1":        test_metrics.get("eval_exact_span_f1", 0),
        "span_precision": test_metrics.get("eval_exact_span_precision", 0),
        "span_recall":    test_metrics.get("eval_exact_span_recall", 0),
        "char_f1":        test_metrics.get("eval_f1_seqeval", 0),
        "exact_match_pct": test_metrics.get("eval_exact_span_f1", 0),
        "wall_time_sec":  wall_time,
        "_raw_metrics":   {"val": val_metrics, "test": test_metrics},
    }
