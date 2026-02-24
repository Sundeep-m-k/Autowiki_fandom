"""Training loop with checkpointing for span identification."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.span_identification.evaluator import evaluate_example, aggregate_metrics
from src.span_identification.tokenization import (
    LabelScheme,
    char_spans_to_token_spans,
    get_label2id,
    labels_to_spans,
    spans_to_bio_labels,
    spans_to_bieos_labels,
    spans_to_bilou_labels,
    spans_to_io_labels,
    token_spans_to_char_spans,
)


def encode_example(
    example: dict,
    tokenizer,
    label_scheme: LabelScheme,
    max_length: int,
) -> dict[str, Any]:
    """Encode one example for model input."""
    text = example["text"]
    gold_spans = example["gold_spans"]
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    offset_mapping = encoding["offset_mapping"][0].tolist()
    # Filter [CLS]/[SEP] and padding
    token_offsets = [
        (o[0], o[1]) for i, o in enumerate(offset_mapping)
        if o != (0, 0) and encoding["input_ids"][0][i].item() not in (tokenizer.cls_token_id, tokenizer.sep_token_id)
    ]

    # Map char spans to token spans
    token_spans = char_spans_to_token_spans(text, gold_spans, token_offsets)
    if label_scheme == "BIO":
        labels = spans_to_bio_labels(len(token_offsets), token_spans)
    elif label_scheme == "BIEOS":
        labels = spans_to_bieos_labels(len(token_offsets), token_spans)
    elif label_scheme == "BILOU":
        labels = spans_to_bilou_labels(len(token_offsets), token_spans)
    else:
        labels = spans_to_io_labels(len(token_offsets), token_spans)

    label2id = get_label2id(label_scheme)
    content_label_ids = [label2id.get(l, 0) for l in labels]
    # Build full-sequence labels aligned with input_ids (CLS, tokens, SEP, PAD)
    pad_id = -100
    full_label_ids = []
    content_idx = 0
    for i, o in enumerate(offset_mapping):
        if o == (0, 0) or encoding["input_ids"][0][i].item() in (tokenizer.cls_token_id, tokenizer.sep_token_id):
            full_label_ids.append(pad_id)
        else:
            full_label_ids.append(content_label_ids[content_idx])
            content_idx += 1
    while len(full_label_ids) < max_length:
        full_label_ids.append(pad_id)

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": torch.tensor(full_label_ids[:max_length], dtype=torch.long),
        "offset_mapping": offset_mapping,
        "text": text,
        "gold_spans": gold_spans,
    }


def decode_predictions(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    offset_mapping: list[tuple[int, int]],
    id2label: dict[int, str],
    label_scheme: LabelScheme,
) -> list[tuple[int, int]]:
    """Decode model logits to character spans. offset_mapping from tokenizer (full seq)."""
    pred_ids = logits.argmax(-1).squeeze(0).tolist()
    # Use only positions with non-zero offset (real tokens)
    labels = []
    for i, pid in enumerate(pred_ids):
        if i < len(offset_mapping) and offset_mapping[i] != (0, 0):
            labels.append(id2label.get(pid, "O"))
    token_spans = labels_to_spans(labels, scheme=label_scheme)
    # Map token indices back to char spans using offset_mapping for content tokens only
    content_offsets = [o for o in offset_mapping if o != (0, 0)]
    return token_spans_to_char_spans(token_spans, content_offsets)


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    epoch: int | None = None,
) -> float:
    """Train one epoch, return average loss."""
    log = logging.getLogger("span_id")
    model.train()
    total_loss = 0.0
    n = 0
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    avg_loss = total_loss / n if n else 0.0
    if epoch is not None:
        log.info("[train_epoch] epoch=%d batches=%d avg_loss=%.6f", epoch + 1, n, avg_loss)
    return avg_loss


def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    id2label: dict[int, str],
    label_scheme: LabelScheme,
) -> dict[str, float]:
    """Evaluate model on dataloader."""
    log = logging.getLogger("span_id")
    n_batches = len(dataloader)
    log.info("[evaluate] start num_batches=%d", n_batches)
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits"]
            for i in range(logits.size(0)):
                text = batch["text"][i]
                gold = batch["gold_spans"][i]
                offset_mapping = batch["offset_mapping"][i]
                pred_spans = decode_predictions(
                    logits[i:i+1],
                    attention_mask[i:i+1],
                    offset_mapping,
                    id2label,
                    label_scheme,
                )
                m = evaluate_example(gold, pred_spans, len(text))
                all_metrics.append(m)
    m_agg = aggregate_metrics(all_metrics)
    log.info("[evaluate] done span_f1=%.4f num_examples=%d", m_agg.get("span_f1", 0), len(all_metrics))
    return m_agg


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    checkpoint_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Full training loop with checkpointing."""
    log = logging.getLogger("span_id")
    training = config.get("training", {})
    epochs = training.get("epochs", 10)
    lr = training.get("learning_rate", 5e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=training.get("weight_decay", 0.01))

    log.info("[train] start checkpoint_dir=%s epochs=%d lr=%s", checkpoint_dir, epochs, lr)
    best_f1 = 0.0
    best_metrics = {}
    start_time = time.time()

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch=epoch)
        log.info("[train] epoch=%d train_loss=%.6f evaluating...", epoch + 1, train_loss)
        val_metrics = evaluate(model, val_loader, device, model.id2label, model.label_scheme)
        val_f1 = val_metrics.get("span_f1", 0.0)
        log.info("[train] epoch=%d val_span_f1=%.4f val_precision=%.4f val_recall=%.4f", epoch + 1, val_f1, val_metrics.get("span_precision", 0), val_metrics.get("span_recall", 0))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = val_metrics
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_path = checkpoint_dir / "best_val_f1.pt"
            model.save_pretrained(str(best_path))
            log.info("[train] checkpoint saved best_val_f1.pt (val_f1=%.4f) -> %s", val_f1, best_path)

        if config.get("checkpoint", {}).get("save_every_epoch", False):
            epoch_path = checkpoint_dir / f"epoch_{epoch}.pt"
            model.save_pretrained(str(epoch_path))
            log.info("[train] checkpoint saved epoch_%d.pt -> %s", epoch, epoch_path)

    last_path = checkpoint_dir / "last.pt"
    model.save_pretrained(str(last_path))
    elapsed = time.time() - start_time
    log.info("[train] done checkpoint saved last.pt -> %s | wall_time=%.1fs best_span_f1=%.4f", last_path, elapsed, best_f1)
    return {
        **best_metrics,
        "wall_time_sec": elapsed,
        "best_span_f1": best_f1,
    }
