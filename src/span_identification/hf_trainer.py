"""HuggingFace Trainer for span identification (fandom style)."""
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

from src.span_identification.preprocess import ID2LABEL, LABEL2ID
from src.span_identification.span_metrics import compute_span_metrics_for_trainer

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


def _pad_example(
    ex: dict,
    max_seq_length: int,
    pad_token_id: int,
) -> dict:
    """Pad input_ids, attention_mask, labels to max_seq_length. Use -100 for label padding."""
    input_ids = ex["input_ids"]
    attention_mask = ex["attention_mask"]
    labels = ex["label_ids"]

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = labels[:max_seq_length]
    else:
        pad_len = max_seq_length - len(input_ids)
        input_ids = input_ids + [pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_hf_datasets(
    train_path: Path,
    dev_path: Path,
    test_path: Path,
    max_seq_length: int,
    tokenizer,
    data_fraction: float = 1.0,
    seed: int = 42,
) -> DatasetDict:
    """Load JSONL and build HF DatasetDict with padding."""
    train_rows = _load_jsonl(train_path)
    if data_fraction < 1.0:
        n = max(1, int(len(train_rows) * data_fraction))
        rng = random.Random(seed)
        train_rows = rng.sample(train_rows, n)
    dev_rows = _load_jsonl(dev_path)
    test_rows = _load_jsonl(test_path)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _process(rows: list[dict]) -> Dataset:
        padded = [
            _pad_example(r, max_seq_length, pad_token_id)
            for r in rows
        ]
        return Dataset.from_list(padded)

    return DatasetDict(
        train=_process(train_rows),
        validation=_process(dev_rows),
        test=_process(test_rows),
    )


def _compute_metrics(eval_pred) -> dict[str, float]:
    """HF Trainer compute_metrics callback."""
    logits, labels = eval_pred
    logits = np.array(logits)
    labels = np.array(labels)
    pred_ids = np.argmax(logits, axis=-1)

    pred_tags = []
    true_tags = []
    for pred_seq, label_seq in zip(pred_ids.tolist(), labels.tolist()):
        p_tags = []
        t_tags = []
        for pid, lid in zip(pred_seq, label_seq):
            if lid == -100:
                continue
            p_tags.append(ID2LABEL.get(int(pid), "O"))
            t_tags.append(ID2LABEL.get(int(lid), "O"))
        pred_tags.append(p_tags)
        true_tags.append(t_tags)

    from seqeval.metrics import f1_score, precision_score, recall_score
    metrics = {
        "eval_f1_seqeval": f1_score(true_tags, pred_tags),
        "eval_precision_seqeval": precision_score(true_tags, pred_tags),
        "eval_recall_seqeval": recall_score(true_tags, pred_tags),
    }
    span_metrics = compute_span_metrics_for_trainer((logits, labels), id2label=ID2LABEL)
    return {**metrics, **span_metrics}


def train_and_evaluate(
    config: dict,
    train_path: Path,
    dev_path: Path,
    test_path: Path,
    output_dir: Path,
    model_name: str,
    seed: int,
    data_fraction: float = 1.0,
) -> dict[str, Any]:
    """
    Train with HF Trainer, evaluate on test, return metrics.
    """
    log = logging.getLogger("span_id")
    train_cfg = config.get("training", {})
    max_seq_length = config.get("model", {}).get("max_length", 512)
    num_labels = len(LABEL2ID)

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
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

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
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    log.info("[hf_train] starting training...")
    t0 = time.time()
    trainer.train()
    wall_time = time.time() - t0
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Final eval on val (model-selection metric) and test (reporting metric)
    log.info("[hf_train] evaluating on validation (model selection)...")
    val_metrics = trainer.evaluate(eval_dataset=datasets["validation"])
    log.info("[hf_train] evaluating on test (final report)...")
    test_metrics = trainer.evaluate(eval_dataset=datasets["test"])

    return {
        "val_span_f1": val_metrics.get("eval_exact_span_f1", 0),
        "span_f1": test_metrics.get("eval_exact_span_f1", 0),
        "span_precision": test_metrics.get("eval_exact_span_precision", 0),
        "span_recall": test_metrics.get("eval_exact_span_recall", 0),
        "token_f1": test_metrics.get("eval_f1_seqeval", 0),
        "exact_match_pct": test_metrics.get("eval_exact_span_f1", 0),
        "wall_time_sec": wall_time,
        "_raw_metrics": {"val": val_metrics, "test": test_metrics},
    }
