"""Fandom-style preprocessing: build token-level BILOU datasets from raw units."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from src.span_identification.config_utils import get_processed_path, get_token_data_path
from src.span_identification.dataset import create_splits, load_units

# Fandom-style BILOU labels (single entity type: SPAN)
BILOU_LABELS = ["O", "B-SPAN", "I-SPAN", "L-SPAN", "U-SPAN"]
LABEL2ID = {label: i for i, label in enumerate(BILOU_LABELS)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def _spans_from_links_internal_only(
    links: list[dict],
    use_char_offsets: bool = False,
) -> list[dict[str, int]]:
    """Extract spans from internal links only. Returns list of {start, end}."""
    spans = []
    for link in links:
        if link.get("link_type") != "internal":
            continue
        if use_char_offsets:
            start = link.get("plain_text_char_start")
            end = link.get("plain_text_char_end")
        else:
            start = link.get("plain_text_rel_char_start")
            end = link.get("plain_text_rel_char_end")
        if start is not None and end is not None and end > start:
            spans.append({"start": int(start), "end": int(end)})
    return spans


def assign_bilou_labels(
    text: str,
    spans: list[dict[str, int]],
    tokenizer,
    max_seq_length: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Assign BILOU labels to token sequence (fandom style).
    Labels aligned with full input_ids (CLS, tokens, SEP).
    Uses overlap logic for span-to-token mapping.
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=True,
    )
    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    labels = [LABEL2ID["O"]] * len(input_ids)

    for sp in spans:
        start_char = int(sp["start"])
        end_char = int(sp["end"])
        if end_char <= start_char:
            continue

        token_indices = []
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end == 0:  # special tokens
                continue
            if tok_end <= start_char or tok_start >= end_char:  # no overlap
                continue
            token_indices.append(i)

        if not token_indices:
            continue

        if len(token_indices) == 1:
            labels[token_indices[0]] = LABEL2ID["U-SPAN"]
        else:
            labels[token_indices[0]] = LABEL2ID["B-SPAN"]
            for ti in token_indices[1:-1]:
                labels[ti] = LABEL2ID["I-SPAN"]
            labels[token_indices[-1]] = LABEL2ID["L-SPAN"]

    return input_ids, attention_mask, labels


def _unit_to_token_example(
    unit: dict,
    granularity: str,
    tokenizer,
    max_seq_length: int,
) -> dict[str, Any] | None:
    """Convert unit to token-level example with BILOU labels (internal links only)."""
    if granularity == "article":
        text = unit.get("article_plain_text", "")
    elif granularity == "paragraph":
        text = unit.get("paragraph_text", "")
    else:
        text = unit.get("sentence_text", "")
    if not text.strip():
        return None

    # Get spans from internal links only
    links = unit.get("links", [])
    use_char = granularity == "article"
    spans = _spans_from_links_internal_only(links, use_char_offsets=use_char)

    input_ids, attention_mask, labels = assign_bilou_labels(
        text=text,
        spans=spans,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    return {
        "article_id": unit.get("article_id"),
        "text": text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": labels,
    }


def build_token_dataset(
    config: dict,
    domain: str,
    granularity: str,
    model_name: str,
    seed: int | None = None,
) -> tuple[Path, Path, Path]:
    """
    Build token-level BILOU train/dev/test JSONL (fandom style).
    Uses internal links only, article-based splits, overlap logic.
    Returns (train_path, dev_path, test_path).
    """
    log = logging.getLogger("span_id")
    processed_path = get_processed_path(config, domain, granularity)
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_path}")

    max_seq_length = config.get("model", {}).get("max_length", 512)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    log.info(
        "[build_token_dataset] domain=%s granularity=%s model=%s",
        domain, granularity, model_name,
    )

    units = load_units(processed_path)
    log.info("[build_token_dataset] loaded %d units from %s", len(units), processed_path)

    train_units, val_units, test_units = create_splits(units, config, domain, granularity, seed)

    def _convert(unit_list: list[dict]) -> list[dict]:
        out = []
        for u in unit_list:
            ex = _unit_to_token_example(u, granularity, tokenizer, max_seq_length)
            if ex is not None:
                out.append(ex)
        return out

    train_ex = _convert(train_units)
    val_ex = _convert(val_units)
    test_ex = _convert(test_units)

    out_dir = get_token_data_path(config, domain, granularity, model_name).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = get_token_data_path(config, domain, granularity, model_name, "train")
    dev_path = get_token_data_path(config, domain, granularity, model_name, "dev")
    test_path = get_token_data_path(config, domain, granularity, model_name, "test")

    for path, examples in [(train_path, train_ex), (dev_path, val_ex), (test_path, test_ex)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info("[build_token_dataset] wrote %s (%d examples)", path, len(examples))

    return train_path, dev_path, test_path
