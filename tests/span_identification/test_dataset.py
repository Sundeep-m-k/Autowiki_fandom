"""Tests for dataset."""
from src.span_identification.dataset import extract_spans_from_links, unit_to_example, create_splits


def test_extract_spans_from_links():
    links = [{"plain_text_rel_char_start": 0, "plain_text_rel_char_end": 10}]
    assert extract_spans_from_links(links) == [(0, 10)]


def test_extract_spans_article_format():
    links = [{"plain_text_char_start": 100, "plain_text_char_end": 115}]
    assert extract_spans_from_links(links, use_char_offsets=True) == [(100, 115)]


def test_unit_to_example(sample_paragraph_unit):
    ex = unit_to_example(sample_paragraph_unit)
    assert "text" in ex
    assert ex["gold_spans"] == [(0, 10), (29, 41)]


def test_create_splits():
    config = {"split": {"train_ratio": 0.67, "val_ratio": 0.33, "test_ratio": 0.0, "seed": 42}}
    units = [{"article_id": 1, "paragraph_text": "a", "links": []}, {"article_id": 2, "paragraph_text": "b", "links": []}]
    train, val, test = create_splits(units, config, "test", "paragraph", seed=42)
    assert len(train) + len(val) + len(test) == 2
