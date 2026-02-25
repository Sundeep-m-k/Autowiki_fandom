"""Query dataset builder for the article retrieval pipeline.

Responsibilities (I/O only):
  - Extract internal links from ground truth JSONL
  - Filter to test-set source articles (reusing Task 1 splits)
  - Sample up to n_sample queries stratified by source article
  - Generate all 24 query variation texts per link
  - Save query dataset to JSONL

Exp 1  (query versions):       all 24 templates are generated for every query
Exp 4  (query context mode):   anchor_only / anchor_sentence / anchor_paragraph
Exp 9  (query sample size):    n_sample cap with stratified sampling
Exp 11 (anchor preprocessing): raw / lowercase / stopword_removed applied to anchor text
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

log = logging.getLogger("article_retrieval")


# ── 24 Query templates ────────────────────────────────────────────────────────
# {word}           → replaced with (optionally preprocessed) anchor text
# {paragraph_text} → replaced with context string (sentence or paragraph)
# Templates without {paragraph_text} use anchor_only regardless of context mode.

QUERY_TEMPLATES: dict[int, str] = {
    1:  "Retrieve documents for the term '{word}', the context is: {paragraph_text}.",
    2:  "Find an article that defines and explains '{word}'.",
    3:  "Given the following paragraph: {paragraph_text}, which article best explains '{word}'?",
    4:  "Find the best article that can explain the term '{word}' given this context: {paragraph_text}.",
    5:  "Which article provides the best information about '{word}'?",
    6:  "Retrieve the topic discussing '{word}'.",
    7:  "Find an article that summarizes the concept of '{word}'.",
    8:  "Which paragraph of text elaborates on the topic of '{word}'?",
    9:  "Locate an article that gives a comprehensive definition of '{word}'.",
    10: "What is '{word}'? Find the best article explaining it.",
    11: "Find paragraphs of texts related to '{word}'.",
    12: "Which article provides background knowledge about '{word}'?",
    13: "Retrieve articles covering the technical aspects of '{word}'.",
    14: "Which page gives examples related to '{word}'?",
    15: "Find an article detailing the history of '{word}'.",
    16: "Which paragraph of texts gives an in-depth explanation of '{word}'?",
    17: "Retrieve texts that discuss the fundamentals of '{word}'.",
    18: "Which article discusses the real-world application of '{word}'?",
    19: "Find texts that includes research studies on '{word}'.",
    20: "Locate an article that introduces the concept of '{word}'.",
    21: "Find texts with an educational overview of '{word}'.",
    22: "As a domain expert, explain the meaning of '{word}' in this context: {paragraph_text}.",
    23: "First define '{word}', then explain how it applies in the context: {paragraph_text}.",
    24: "Which article maps multiple perspectives or connections related to '{word}'?",
}

# Versions that use {paragraph_text} — requires context_mode != anchor_only
CONTEXT_VERSIONS = {1, 3, 4, 22, 23}


# ── Text preprocessing for anchor (Exp 11) ────────────────────────────────────

def _preprocess_anchor(anchor: str, preprocessing: str) -> str:
    if preprocessing == "raw":
        return anchor
    anchor = anchor.lower()
    if preprocessing == "stopword_removed":
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            tokens = [w for w in anchor.split() if w not in ENGLISH_STOP_WORDS]
            anchor = " ".join(tokens) or anchor  # fallback if all words removed
        except ImportError:
            pass
    return anchor


# ── Context extraction (Exp 4) ─────────────────────────────────────────────────

def _build_context(
    anchor_text: str,
    paragraph_text: str,
    context_mode: str,
) -> str:
    """
    Return the context string used for {paragraph_text} substitution.

    Exp 4 — query_context_mode:
      anchor_only      : no context (paragraph_text placeholder = anchor text)
      anchor_sentence  : the sentence within the paragraph containing the anchor
      anchor_paragraph : the full paragraph text
    """
    if context_mode == "anchor_only":
        return anchor_text

    if context_mode == "anchor_paragraph":
        return paragraph_text.strip()

    # anchor_sentence: find the sentence containing the anchor
    if context_mode == "anchor_sentence":
        try:
            import nltk
            try:
                sentences = nltk.sent_tokenize(paragraph_text)
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
                sentences = nltk.sent_tokenize(paragraph_text)
            for sent in sentences:
                if anchor_text in sent:
                    return sent.strip()
        except ImportError:
            pass
        # Fallback: return full paragraph if sentence not found
        return paragraph_text.strip()

    return paragraph_text.strip()


# ── Query generation ──────────────────────────────────────────────────────────

def generate_queries_for_link(
    anchor_text: str,
    paragraph_text: str,
    context_mode: str,
    anchor_preprocessing: str,
    versions: list[int],
) -> dict[str, str]:
    """
    Generate all requested query variation texts for one link.
    Returns dict mapping "v<N>" → query string.
    """
    word    = _preprocess_anchor(anchor_text, anchor_preprocessing)
    context = _build_context(anchor_text, paragraph_text, context_mode)

    queries: dict[str, str] = {}
    for v in versions:
        template = QUERY_TEMPLATES.get(v, "")
        if not template:
            continue
        if "{paragraph_text}" in template:
            text = template.replace("{word}", word).replace("{paragraph_text}", context)
        else:
            text = template.replace("{word}", word)
        queries[f"v{v}"] = text
    return queries


# ── Sampling ──────────────────────────────────────────────────────────────────

def _stratified_sample(
    records: list[dict],
    n: int,
    key: str,
    seed: int = 42,
) -> list[dict]:
    """
    Sample n records stratified by the given key field.
    If len(records) <= n, returns all records.
    """
    if len(records) <= n:
        return records

    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in records:
        groups[r.get(key, "unknown")].append(r)

    rng = random.Random(seed)
    sampled: list[dict] = []
    group_keys = sorted(groups.keys())

    # Round-robin allocation across groups
    quota = {k: max(1, n // len(group_keys)) for k in group_keys}
    remainder = n - sum(quota.values())
    for k in group_keys[:remainder]:
        quota[k] += 1

    for k, items in groups.items():
        q = quota.get(k, 1)
        sampled.extend(rng.sample(items, min(q, len(items))))

    # Top up if we came up short (due to small groups)
    if len(sampled) < n:
        remaining = [r for r in records if r not in sampled]
        rng.shuffle(remaining)
        sampled.extend(remaining[:n - len(sampled)])

    return sampled[:n]


# ── Main builder ──────────────────────────────────────────────────────────────

def build_query_dataset(
    jsonl_path: Path,
    config: dict,
    test_article_ids: set[int],
    output_path: Path,
) -> list[dict]:
    """
    Extract internal links from the articles JSONL, apply filters and sampling,
    generate all 24 query variations, and save to output_path.

    Returns the list of query records written.
    """
    q_cfg        = config.get("queries", {})
    n_sample     = q_cfg.get("n_sample")          # None = use all
    versions     = q_cfg.get("versions", list(range(1, 25)))
    context_mode = q_cfg.get("query_context_mode", "anchor_sentence")
    preproc      = q_cfg.get("anchor_preprocessing", "raw")
    stratify_by  = q_cfg.get("stratify_by", "source_article_id")

    raw_links: list[dict] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            article_id = rec.get("article_id")
            if article_id is None:
                continue
            article_id = int(article_id)

            # Filter: only links whose source article is in the test split
            if test_article_ids and article_id not in test_article_ids:
                continue

            paragraph_text = rec.get("article_plain_text", "").strip()

            for link in rec.get("links", []):
                if link.get("link_type") != "internal":
                    continue
                gold_article_id = link.get("article_id_of_internal_link")
                if gold_article_id is None:
                    continue
                anchor_text = link.get("anchor_text", "").strip()
                if not anchor_text:
                    continue

                raw_links.append({
                    "anchor_text": anchor_text,
                    "gold_article_id": int(gold_article_id),
                    "source_article_id": article_id,
                    "paragraph_text": paragraph_text,
                    "char_start": link.get("plain_text_char_start"),
                    "char_end":   link.get("plain_text_char_end"),
                })

    log.info(
        "[query_builder] found %d internal links (split=test, domain filtered)",
        len(raw_links),
    )

    # Sample
    if n_sample and len(raw_links) > n_sample:
        raw_links = _stratified_sample(raw_links, n_sample, stratify_by)
        log.info("[query_builder] sampled %d queries (stratified by %s)", len(raw_links), stratify_by)
    else:
        log.info("[query_builder] using all %d queries (below n_sample threshold)", len(raw_links))

    # Build query records
    query_records: list[dict] = []
    domain = config.get("domains", ["unknown"])[0]
    for i, link in enumerate(raw_links):
        query_id = f"{domain}_q_{i + 1:06d}"
        queries  = generate_queries_for_link(
            anchor_text=link["anchor_text"],
            paragraph_text=link["paragraph_text"],
            context_mode=context_mode,
            anchor_preprocessing=preproc,
            versions=versions,
        )
        query_records.append({
            "query_id":         query_id,
            "anchor_text":      link["anchor_text"],
            "gold_article_id":  link["gold_article_id"],
            "source_article_id": link["source_article_id"],
            "paragraph_text":   link["paragraph_text"],
            "char_start":       link.get("char_start"),
            "char_end":         link.get("char_end"),
            "queries":          queries,
        })

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in query_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info("[query_builder] saved %d query records → %s", len(query_records), output_path)
    return query_records


def load_query_dataset(path: Path) -> list[dict]:
    """Load query dataset from JSONL."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
