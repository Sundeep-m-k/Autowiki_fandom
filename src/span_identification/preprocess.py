"""Fandom-style preprocessing: build token-level datasets from raw units.

Supports BIO and BILOU label schemes (and any scheme from tokenization.py).
The label scheme is a first-class parameter throughout; BILOU is kept as the
default to preserve backward compatibility with existing datasets on disk.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from src.span_identification.config_utils import get_processed_path, get_token_data_path
from src.span_identification.dataset import create_splits, load_units
from src.span_identification.tokenization import LabelScheme, get_label2id, get_id2label

# ---------------------------------------------------------------------------
# Scheme-aware label/id maps (built on demand via helpers below)
# ---------------------------------------------------------------------------

# Legacy BILOU globals kept for any external code that imported them directly.
_BILOU_LABEL2ID = get_label2id("BILOU")
_BILOU_ID2LABEL = get_id2label("BILOU")

# These are still exported under the old names so existing imports don't break,
# but callers that need a specific scheme should use get_label2id / get_id2label.
BILOU_LABELS = ["O", "B-SPAN", "I-SPAN", "L-SPAN", "U-SPAN"]
LABEL2ID = _BILOU_LABEL2ID
ID2LABEL = _BILOU_ID2LABEL


# ---------------------------------------------------------------------------
# Scheme-aware label maps — use these everywhere new code is written
# ---------------------------------------------------------------------------

def get_scheme_label2id(label_scheme: LabelScheme) -> dict[str, int]:
    """Return label→id mapping for a scheme, using SPAN-suffixed keys for seqeval."""
    raw = get_label2id(label_scheme)
    if label_scheme == "BILOU":
        # Map raw single-char keys to seqeval-compatible B-SPAN / I-SPAN etc.
        _seqeval = {
            "O": 0,
            "B-SPAN": raw["B"],
            "I-SPAN": raw["I"],
            "L-SPAN": raw["L"],
            "U-SPAN": raw["U"],
        }
        return _seqeval
    if label_scheme == "BIO":
        return {"O": 0, "B-SPAN": raw["B"], "I-SPAN": raw["I"]}
    # Fall back to raw keys for IO / BIEOS
    return raw


def get_scheme_id2label(label_scheme: LabelScheme) -> dict[int, str]:
    return {v: k for k, v in get_scheme_label2id(label_scheme).items()}


# ---------------------------------------------------------------------------
# Span extraction helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core label assignment (scheme-aware)
# ---------------------------------------------------------------------------

def assign_labels(
    text: str,
    spans: list[dict[str, int]],
    tokenizer,
    max_seq_length: int,
    label_scheme: LabelScheme = "BILOU",
) -> tuple[list[int], list[int], list[int], list[list[int]]]:
    """
    Assign token-level labels for a given label scheme.

    Returns (input_ids, attention_mask, label_ids, char_offsets) all aligned
    with the full tokenized sequence (CLS, real tokens, SEP).
    Special tokens get label O and char_offset [0, 0].

    char_offsets is a list of [char_start, char_end] per non-padding token,
    stored alongside the tokenised example so that predicted token spans can
    be decoded back to character-level spans for error analysis.
    """
    label2id = get_scheme_label2id(label_scheme)
    o_id = label2id["O"]

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

    labels = [o_id] * len(input_ids)
    # Store char offsets as plain lists for JSON serialisation
    char_offsets = [list(o) for o in offsets]

    for sp in spans:
        start_char = int(sp["start"])
        end_char = int(sp["end"])
        if end_char <= start_char:
            continue

        token_indices = [
            i for i, (tok_start, tok_end) in enumerate(offsets)
            if not (tok_start == tok_end == 0)          # skip special tokens
            and not (tok_end <= start_char or tok_start >= end_char)  # skip non-overlapping
        ]

        if not token_indices:
            continue

        if label_scheme == "BILOU":
            if len(token_indices) == 1:
                labels[token_indices[0]] = label2id["U-SPAN"]
            else:
                labels[token_indices[0]] = label2id["B-SPAN"]
                for ti in token_indices[1:-1]:
                    labels[ti] = label2id["I-SPAN"]
                labels[token_indices[-1]] = label2id["L-SPAN"]

        elif label_scheme == "BIO":
            labels[token_indices[0]] = label2id["B-SPAN"]
            for ti in token_indices[1:]:
                labels[ti] = label2id["I-SPAN"]

        elif label_scheme == "IO":
            for ti in token_indices:
                labels[ti] = label2id.get("I", label2id.get("I-SPAN", 1))

        else:
            raise ValueError(f"Unsupported label_scheme in preprocess: {label_scheme!r}")

    return input_ids, attention_mask, labels, char_offsets


# Keep the old name as an alias so existing call-sites still work.
def assign_bilou_labels(
    text: str,
    spans: list[dict[str, int]],
    tokenizer,
    max_seq_length: int,
) -> tuple[list[int], list[int], list[int]]:
    """Backward-compatible wrapper: always uses BILOU scheme. Drops char_offsets."""
    input_ids, attention_mask, label_ids, _offsets = assign_labels(
        text, spans, tokenizer, max_seq_length, label_scheme="BILOU"
    )
    return input_ids, attention_mask, label_ids


# ---------------------------------------------------------------------------
# Unit → token example
# ---------------------------------------------------------------------------

def _unit_to_token_example(
    unit: dict,
    granularity: str,
    tokenizer,
    max_seq_length: int,
    label_scheme: LabelScheme = "BILOU",
) -> dict[str, Any] | None:
    """Convert a ground-truth unit to a token-level example with scheme labels."""
    if granularity == "article":
        text = unit.get("article_plain_text", "")
    elif granularity == "paragraph":
        text = unit.get("paragraph_text", "")
    else:
        text = unit.get("sentence_text", "")
    if not text.strip():
        return None

    links = unit.get("links", [])
    use_char = granularity == "article"
    spans = _spans_from_links_internal_only(links, use_char_offsets=use_char)

    input_ids, attention_mask, label_ids, char_offsets = assign_labels(
        text=text,
        spans=spans,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        label_scheme=label_scheme,
    )

    return {
        "article_id": unit.get("article_id"),
        "text": text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": label_ids,
        "char_offsets": char_offsets,
    }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_token_dataset(
    config: dict,
    domain: str,
    granularity: str,
    model_name: str,
    seed: int | None = None,
    label_scheme: LabelScheme = "BILOU",
) -> tuple[Path, Path, Path]:
    """
    Build token-level train/dev/test JSONL for a given label scheme.

    Token data is stored under:
      data/span_id/<domain>/token_data/<granularity>_<model>_<scheme>/

    so that BIO and BILOU datasets coexist on disk without overwriting each other.
    Returns (train_path, dev_path, test_path).
    """
    log = logging.getLogger("span_id")
    processed_path = get_processed_path(config, domain, granularity)
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_path}")

    max_seq_length = config.get("model", {}).get("max_length", 512)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    log.info(
        "[build_token_dataset] domain=%s granularity=%s model=%s label_scheme=%s",
        domain, granularity, model_name, label_scheme,
    )

    units = load_units(processed_path)
    log.info("[build_token_dataset] loaded %d units from %s", len(units), processed_path)

    train_units, val_units, test_units = create_splits(units, config, domain, granularity, seed)

    def _convert(unit_list: list[dict]) -> list[dict]:
        out = []
        for u in unit_list:
            ex = _unit_to_token_example(u, granularity, tokenizer, max_seq_length, label_scheme)
            if ex is not None:
                out.append(ex)
        return out

    train_ex = _convert(train_units)
    val_ex = _convert(val_units)
    test_ex = _convert(test_units)

    train_path = get_token_data_path(config, domain, granularity, model_name, "train", label_scheme)
    dev_path   = get_token_data_path(config, domain, granularity, model_name, "dev",   label_scheme)
    test_path  = get_token_data_path(config, domain, granularity, model_name, "test",  label_scheme)

    out_dir = train_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for path, examples in [(train_path, train_ex), (dev_path, val_ex), (test_path, test_ex)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info("[build_token_dataset] wrote %s (%d examples)", path, len(examples))

    return train_path, dev_path, test_path
