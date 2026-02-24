"""Tests for tokenization."""
from src.span_identification.tokenization import spans_to_bio_labels, labels_to_spans, get_label2id


def test_spans_to_bio():
    labels = spans_to_bio_labels(10, [(0, 3)])
    assert labels[0] == "B"
    assert labels[1] == "I"


def test_labels_to_spans():
    labels = ["B", "I", "I", "O"]
    assert labels_to_spans(labels, "BIO") == [(0, 3)]


def test_get_label2id():
    assert get_label2id("BIO")["O"] == 0


def test_bilou_labels():
    from src.span_identification.tokenization import spans_to_bilou_labels, labels_to_spans
    labels = spans_to_bilou_labels(10, [(0, 3), (5, 6)])
    assert labels[0] == "B"
    assert labels[2] == "L"
    assert labels[5] == "U"
    spans = labels_to_spans(labels, "BILOU")
    assert spans == [(0, 3), (5, 6)]
