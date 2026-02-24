"""Tokenization and label encoding (BIO/BIEOS) for span identification."""
from __future__ import annotations

from typing import Literal

LabelScheme = Literal["BIO", "BIEOS", "BILOU", "IO"]


def spans_to_bilou_labels(num_tokens: int, token_spans: list[tuple[int, int]]) -> list[str]:
    """Convert token-level spans to BILOU labels (B=Begin, I=Inside, L=Last, O=Outside, U=Unit)."""
    labels = ["O"] * num_tokens
    for start, end in token_spans:
        if start < num_tokens:
            if start == end - 1:
                labels[start] = "U"
            else:
                labels[start] = "B"
                for i in range(start + 1, min(end - 1, num_tokens)):
                    labels[i] = "I"
                if end - 1 < num_tokens and end - 1 > start:
                    labels[end - 1] = "L"
    return labels


def spans_to_bio_labels(num_tokens: int, token_spans: list[tuple[int, int]]) -> list[str]:
    """Convert token-level spans to BIO labels. O for outside, B for begin, I for inside."""
    labels = ["O"] * num_tokens
    for start, end in token_spans:
        if start < num_tokens:
            labels[start] = "B"
            for i in range(start + 1, min(end, num_tokens)):
                labels[i] = "I"
    return labels


def spans_to_bieos_labels(num_tokens: int, token_spans: list[tuple[int, int]]) -> list[str]:
    """Convert token-level spans to BIEOS labels."""
    labels = ["O"] * num_tokens
    for start, end in token_spans:
        if start < num_tokens:
            if start == end - 1:
                labels[start] = "S"
            else:
                labels[start] = "B"
                for i in range(start + 1, min(end - 1, num_tokens)):
                    labels[i] = "I"
                if end - 1 < num_tokens and end - 1 > start:
                    labels[end - 1] = "E"
    return labels


def spans_to_io_labels(num_tokens: int, token_spans: list[tuple[int, int]]) -> list[str]:
    """Convert token-level spans to IO labels (I for inside span, O for outside)."""
    labels = ["O"] * num_tokens
    for start, end in token_spans:
        for i in range(start, min(end, num_tokens)):
            labels[i] = "I"
    return labels


def char_spans_to_token_spans(
    text: str,
    char_spans: list[tuple[int, int]],
    tokenizer_offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Map character spans to token spans using tokenizer byte offsets.
    tokenizer_offsets: list of (start, end) for each token in char/byte space.
    Uses overlap logic: token overlaps span iff t_start < c_end and t_end > c_start
    (matches fandom_span_id_retrieval assign_bilou_labels).
    """
    token_spans = []
    for c_start, c_end in char_spans:
        if c_end <= c_start:
            continue
        token_indices = []
        for i, (t_start, t_end) in enumerate(tokenizer_offsets):
            if t_start == t_end == 0:  # special tokens
                continue
            if t_end <= c_start or t_start >= c_end:  # no overlap
                continue
            token_indices.append(i)
        if token_indices:
            token_spans.append((token_indices[0], token_indices[-1] + 1))
    return token_spans


def labels_to_spans(labels: list[str], scheme: LabelScheme = "BIO") -> list[tuple[int, int]]:
    """Convert BIO/BIEOS/IO labels back to token spans (start, end) exclusive end."""
    spans = []
    i = 0
    n = len(labels)
    while i < n:
        if scheme == "BIO":
            if labels[i] == "B":
                start = i
                i += 1
                while i < n and labels[i] == "I":
                    i += 1
                spans.append((start, i))
                continue
        elif scheme == "BIEOS":
            if labels[i] == "B":
                start = i
                i += 1
                while i < n and labels[i] in ("I", "E"):
                    i += 1
                spans.append((start, i))
                continue
            elif labels[i] == "S":
                spans.append((i, i + 1))
        elif scheme == "BILOU":
            if labels[i] == "B":
                start = i
                i += 1
                while i < n and labels[i] in ("I", "L"):
                    i += 1
                spans.append((start, i))
                continue
            elif labels[i] == "U":
                spans.append((i, i + 1))
        elif scheme == "IO":
            if labels[i] == "I":
                start = i
                i += 1
                while i < n and labels[i] == "I":
                    i += 1
                spans.append((start, i))
                continue
        i += 1
    return spans


def token_spans_to_char_spans(
    token_spans: list[tuple[int, int]],
    tokenizer_offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Map token spans back to character spans."""
    char_spans = []
    for t_start, t_end in token_spans:
        if t_start < len(tokenizer_offsets) and t_end > 0:
            c_start = tokenizer_offsets[t_start][0]
            c_end = tokenizer_offsets[t_end - 1][1] if t_end <= len(tokenizer_offsets) else tokenizer_offsets[-1][1]
            char_spans.append((c_start, c_end))
    return char_spans


def get_label2id(scheme: LabelScheme) -> dict[str, int]:
    """Get label to id mapping for a scheme."""
    if scheme == "BIO":
        return {"O": 0, "B": 1, "I": 2}
    if scheme == "BIEOS":
        return {"O": 0, "B": 1, "I": 2, "E": 3, "S": 4}
    if scheme == "BILOU":
        return {"O": 0, "B": 1, "I": 2, "L": 3, "U": 4}
    if scheme == "IO":
        return {"O": 0, "I": 1}
    raise ValueError(f"Unknown scheme: {scheme}")


def get_id2label(scheme: LabelScheme) -> dict[int, str]:
    """Get id to label mapping."""
    label2id = get_label2id(scheme)
    return {v: k for k, v in label2id.items()}
