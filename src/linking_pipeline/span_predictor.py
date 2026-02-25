"""Load ground-truth spans from Task 1 test split JSONL.

In Task 3 Phase 1 we use gold spans from Task 1 test data as the span source.
This isolates Task 3 linking quality from Task 1 span prediction errors,
letting us evaluate the retrieval + linking component independently.

Each article record from the split JSONL contains:
  - article_id
  - article_plain_text (or paragraph_text / sentence_text depending on granularity)
  - links: list of internal link dicts with anchor_text and char offsets

Output per article:
  {
    "article_id":    int,
    "text":          str,          # full plain text
    "page_name":     str,
    "gold_spans": [
      {
        "char_start":   int,
        "char_end":     int,
        "anchor_text":  str,
        "gold_article_id": int,
      },
      ...
    ]
  }
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("linking")


def load_gold_spans(split_path: Path, granularity: str = "article") -> list[dict]:
    """
    Load gold spans from a Task 1 split JSONL file.

    Returns a list of article dicts, each with 'gold_spans' from internal links only.
    Spans that lack char offsets or anchor text are skipped with a warning.
    """
    if not split_path.exists():
        log.error("[span_predictor] split file not found: %s", split_path)
        return []

    articles: list[dict] = []
    n_spans_total = 0

    with open(split_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            article_id = rec.get("article_id")
            if article_id is None:
                continue
            article_id = int(article_id)

            # Text field depends on granularity
            if granularity == "article":
                text = rec.get("article_plain_text", "").strip()
            elif granularity == "paragraph":
                text = rec.get("paragraph_text", "").strip()
            elif granularity == "sentence":
                text = rec.get("sentence_text", "").strip()
            else:
                text = rec.get("article_plain_text", rec.get("paragraph_text", "")).strip()

            page_name = rec.get("page_name", "")

            gold_spans: list[dict] = []
            for link in rec.get("links", []):
                if link.get("link_type") != "internal":
                    continue
                gold_aid = link.get("article_id_of_internal_link")
                if gold_aid is None:
                    continue

                anchor_text = link.get("anchor_text", "").strip()
                if not anchor_text:
                    continue

                # Try article-level offsets first, then relative offsets
                char_start = link.get("plain_text_char_start",
                             link.get("plain_text_rel_char_start"))
                char_end   = link.get("plain_text_char_end",
                             link.get("plain_text_rel_char_end"))

                if char_start is None or char_end is None:
                    log.debug(
                        "[span_predictor] span missing char offsets (article_id=%d anchor=%r) — skip",
                        article_id, anchor_text,
                    )
                    continue

                gold_spans.append({
                    "char_start":      int(char_start),
                    "char_end":        int(char_end),
                    "anchor_text":     anchor_text,
                    "gold_article_id": int(gold_aid),
                })

            n_spans_total += len(gold_spans)
            articles.append({
                "article_id": article_id,
                "text":       text,
                "page_name":  page_name,
                "gold_spans": gold_spans,
            })

    log.info(
        "[span_predictor] loaded %d articles, %d gold spans from %s",
        len(articles), n_spans_total, split_path,
    )
    return articles
