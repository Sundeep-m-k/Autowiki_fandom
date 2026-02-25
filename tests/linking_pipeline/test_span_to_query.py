"""Tests for span_to_query lookup bridge."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.linking_pipeline.span_to_query import build_lookup, lookup_span


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ── build_lookup with char offsets ────────────────────────────────────────────

def test_build_lookup_offset_key(tmp_path):
    queries = [
        {
            "query_id": "q1",
            "source_article_id": 10,
            "anchor_text": "Axel",
            "char_start": 0,
            "char_end": 4,
            "paragraph_text": "Axel Foley",
            "gold_article_id": 99,
        },
    ]
    results = [
        {
            "query_id": "q1",
            "retrieved": [{"article_id": 99, "score": 0.9}],
        },
    ]
    qp = tmp_path / "queries.jsonl"
    rp = tmp_path / "results.jsonl"
    _write_jsonl(qp, queries)
    _write_jsonl(rp, results)

    lookup = build_lookup(qp, rp)
    assert (10, 0, 4) in lookup
    assert lookup[(10, 0, 4)]["article_id"] == 99
    assert lookup[(10, 0, 4)]["score"] == pytest.approx(0.9)


def test_build_lookup_legacy_anchor_key(tmp_path):
    """Records without char offsets use the anchor-text key."""
    queries = [
        {
            "query_id": "q1",
            "source_article_id": 10,
            "anchor_text": "Beverly Hills",
            "paragraph_text": "text",
            "gold_article_id": 42,
            # No char_start / char_end
        },
    ]
    results = [
        {
            "query_id": "q1",
            "retrieved": [{"article_id": 42, "score": 0.7}],
        },
    ]
    qp = tmp_path / "queries.jsonl"
    rp = tmp_path / "results.jsonl"
    _write_jsonl(qp, queries)
    _write_jsonl(rp, results)

    lookup = build_lookup(qp, rp)
    assert (10, "beverly hills") in lookup
    assert lookup[(10, "beverly hills")]["article_id"] == 42


def test_build_lookup_offset_key_disambiguates_duplicate_anchor(tmp_path):
    """Two spans with the same anchor text in one article map to different targets."""
    queries = [
        {
            "query_id": "q1",
            "source_article_id": 10,
            "anchor_text": "cop",
            "char_start": 5,
            "char_end": 8,
            "paragraph_text": "the cop is cool",
            "gold_article_id": 100,
        },
        {
            "query_id": "q2",
            "source_article_id": 10,
            "anchor_text": "cop",
            "char_start": 20,
            "char_end": 23,
            "paragraph_text": "another cop",
            "gold_article_id": 200,
        },
    ]
    results = [
        {"query_id": "q1", "retrieved": [{"article_id": 100, "score": 0.8}]},
        {"query_id": "q2", "retrieved": [{"article_id": 200, "score": 0.6}]},
    ]
    qp = tmp_path / "queries.jsonl"
    rp = tmp_path / "results.jsonl"
    _write_jsonl(qp, queries)
    _write_jsonl(rp, results)

    lookup = build_lookup(qp, rp)
    # Both spans coexist under distinct offset keys
    assert lookup[(10, 5, 8)]["article_id"] == 100
    assert lookup[(10, 20, 23)]["article_id"] == 200


def test_build_lookup_missing_files(tmp_path):
    assert build_lookup(tmp_path / "no_queries.jsonl", tmp_path / "no_results.jsonl") == {}


# ── lookup_span ────────────────────────────────────────────────────────────────

def test_lookup_span_offset_preferred(tmp_path):
    lookup = {(10, 0, 4): {"article_id": 99, "score": 0.9}}
    result = lookup_span(lookup, 10, "Axel", char_start=0, char_end=4)
    assert result is not None
    assert result["article_id"] == 99


def test_lookup_span_legacy_fallback(tmp_path):
    lookup = {(10, "beverly hills"): {"article_id": 42, "score": 0.7}}
    # No char offsets provided — falls back to anchor-text key
    result = lookup_span(lookup, 10, "Beverly Hills")
    assert result is not None
    assert result["article_id"] == 42


def test_lookup_span_not_found():
    lookup = {}
    result = lookup_span(lookup, 99, "unknown")
    assert result is None
