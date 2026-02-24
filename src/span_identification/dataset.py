"""Dataset loading and split management for span identification."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import torch

from src.span_identification.config_utils import (
    get_processed_path,
    get_split_meta_path,
    get_split_path,
    get_splits_root,
    load_config,
)


def extract_spans_from_links(
    links: list[dict],
    use_char_offsets: bool = False,
    internal_only: bool = False,
) -> list[tuple[int, int]]:
    """Extract (start, end) char spans from links.
    Article uses plain_text_char_start/end; paragraph/sentence use plain_text_rel_char_start/end.
    If internal_only, only include links with link_type=='internal' (fandom style)."""
    spans = []
    for link in links:
        if internal_only and link.get("link_type") != "internal":
            continue
        if use_char_offsets:
            start = link.get("plain_text_char_start")
            end = link.get("plain_text_char_end")
        else:
            start = link.get("plain_text_rel_char_start")
            end = link.get("plain_text_rel_char_end")
        if start is not None and end is not None:
            spans.append((int(start), int(end)))
    return sorted(spans)


def load_units(path: Path) -> list[dict]:
    """Load JSONL units (paragraphs or sentences) from file."""
    units = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            units.append(json.loads(line))
    return units


def unit_to_example(
    unit: dict,
    granularity: str = "paragraph",
    internal_only: bool = False,
) -> dict[str, Any]:
    """Convert a processed unit to (text, gold_spans) example.
    If internal_only, only include internal link spans (fandom style)."""
    if granularity == "article":
        text = unit.get("article_plain_text", "")
        unit_id = unit.get("article_record_id") or unit.get("article_id")
        gold_spans = extract_spans_from_links(
            unit.get("links", []), use_char_offsets=True, internal_only=internal_only
        )
    elif granularity == "paragraph":
        text = unit.get("paragraph_text", "")
        unit_id = unit.get("paragraph_id")
        gold_spans = extract_spans_from_links(
            unit.get("links", []), use_char_offsets=False, internal_only=internal_only
        )
    else:
        text = unit.get("sentence_text", "")
        unit_id = unit.get("sentence_id")
        gold_spans = extract_spans_from_links(
            unit.get("links", []), use_char_offsets=False, internal_only=internal_only
        )
    return {
        "text": text,
        "gold_spans": gold_spans,
        "unit_id": unit_id,
        "article_id": unit.get("article_id"),
    }


def create_splits(
    units: list[dict],
    config: dict,
    domain: str,
    granularity: str,
    seed: int | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split units into train/val/test by article_id."""
    split_cfg = config["split"]
    seed = seed or split_cfg.get("seed", 42)
    rng = random.Random(seed)

    # Group by article_id
    by_article: dict[int, list[dict]] = {}
    for u in units:
        aid = u.get("article_id")
        if aid is None:
            aid = -1
        by_article.setdefault(aid, []).append(u)

    articles = list(by_article.keys())
    rng.shuffle(articles)

    tr = split_cfg.get("train_ratio", 0.7)
    va = split_cfg.get("val_ratio", 0.15)
    te = split_cfg.get("test_ratio", 0.15)
    n = len(articles)
    n_train = int(n * tr)
    n_val = int(n * va)
    n_test = n - n_train - n_val

    train_articles = set(articles[:n_train])
    val_articles = set(articles[n_train : n_train + n_val])
    test_articles = set(articles[n_train + n_val :])

    train_units = []
    val_units = []
    test_units = []
    for aid, unit_list in by_article.items():
        if aid in train_articles:
            train_units.extend(unit_list)
        elif aid in val_articles:
            val_units.extend(unit_list)
        else:
            test_units.extend(unit_list)

    return train_units, val_units, test_units


def ensure_splits(
    config: dict,
    domain: str,
    granularity: str,
    config_path: str | Path | None = None,
    seed: int | None = None,
    internal_only: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load or create train/val/test splits.
    Returns (train_examples, val_examples, test_examples) as lists of example dicts.
    """
    log = logging.getLogger("span_id")
    processed_path = get_processed_path(config, domain, granularity)
    log.info("[ensure_splits] domain=%s granularity=%s processed_path=%s", domain, granularity, processed_path)
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_path}")

    units = load_units(processed_path)
    log.info("[ensure_splits] loaded %d units from %s", len(units), processed_path)
    split_cfg = config["split"]
    splits_root = get_splits_root(config, domain)
    train_path = get_split_path(config, domain, granularity, "train")
    meta_path = get_split_meta_path(config, domain)
    recreate = split_cfg.get("recreate_if_exists", False)
    log.info("[ensure_splits] splits_root=%s train_path=%s recreate=%s", splits_root, train_path, recreate)

    if train_path.exists() and not recreate:
        # Load existing splits
        log.info("[ensure_splits] loading existing splits from disk")
        train_units = load_units(get_split_path(config, domain, granularity, "train"))
        val_units = load_units(get_split_path(config, domain, granularity, "val"))
        test_units = load_units(get_split_path(config, domain, granularity, "test"))
        log.info("[ensure_splits] loaded train=%d val=%d test=%d", len(train_units), len(val_units), len(test_units))
    else:
        log.info("[ensure_splits] creating new splits (seed=%s)", seed or split_cfg.get("seed"))
        train_units, val_units, test_units = create_splits(units, config, domain, granularity, seed)
        log.info("[ensure_splits] created train=%d val=%d test=%d", len(train_units), len(val_units), len(test_units))
        splits_root.mkdir(parents=True, exist_ok=True)
        log.info("[ensure_splits] saving splits to %s", splits_root)

        def _save(us: list[dict], p: Path) -> None:
            with open(p, "w") as f:
                for u in us:
                    f.write(json.dumps(u) + "\n")

        _save(train_units, get_split_path(config, domain, granularity, "train"))
        log.info("[ensure_splits] saved train to %s", get_split_path(config, domain, granularity, "train"))
        _save(val_units, get_split_path(config, domain, granularity, "val"))
        _save(test_units, get_split_path(config, domain, granularity, "test"))

        seed_used = seed or split_cfg.get("seed", 42)
        log.info("[ensure_splits] saved split_meta.json to %s", meta_path)
        meta = {
            "domain": domain,
            "granularity": granularity,
            "seed": seed_used,
            "train_ratio": split_cfg.get("train_ratio"),
            "val_ratio": split_cfg.get("val_ratio"),
            "test_ratio": split_cfg.get("test_ratio"),
            "train_size": len(train_units),
            "val_size": len(val_units),
            "test_size": len(test_units),
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    log.info("[ensure_splits] converting units to examples (internal_only=%s)", internal_only)
    train_examples = [unit_to_example(u, granularity, internal_only=internal_only) for u in train_units]
    val_examples = [unit_to_example(u, granularity, internal_only=internal_only) for u in val_units]
    test_examples = [unit_to_example(u, granularity, internal_only=internal_only) for u in test_units]
    log.info("[ensure_splits] done: train=%d val=%d test=%d examples", len(train_examples), len(val_examples), len(test_examples))

    return train_examples, val_examples, test_examples


class SpanDataset:
    """PyTorch Dataset for span identification."""

    def __init__(self, encoded: list[dict]):
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, i: int) -> dict:
        return self.encoded[i]


def collate_span_batch(batch: list[dict]) -> dict:
    """Collate batch of encoded examples."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "text": [b["text"] for b in batch],
        "gold_spans": [b["gold_spans"] for b in batch],
        "offset_mapping": [b["offset_mapping"] for b in batch],
    }


def apply_data_fraction(examples: list[dict], fraction: float, seed: int = 42) -> list[dict]:
    """Subsample examples for learning curves."""
    if fraction >= 1.0:
        return examples
    n = max(1, int(len(examples) * fraction))
    rng = random.Random(seed)
    out = rng.sample(examples, n)
    log = logging.getLogger("span_id")
    log.info("[apply_data_fraction] fraction=%.2f original=%d subsampled=%d", fraction, len(examples), len(out))
    return out
