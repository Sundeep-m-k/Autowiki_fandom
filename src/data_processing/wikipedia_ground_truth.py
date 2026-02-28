"""
wikipedia_ground_truth.py

Parse Wikipedia XML dump (bz2) into the same JSONL schema used by the
Fandom ground_truth pipeline.

Uses mwparserfromhell to extract:
  - plain text paragraphs
  - internal wikilinks with anchor text and character offsets
  - article-level text + link records

Output schema is identical to ground_truth.py so the rest of the pipeline
(span_id, article_retrieval, linking) works without any changes.
"""
from __future__ import annotations

import bz2
import csv
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mwparserfromhell

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Wikipedia namespaces to skip (not article content)
SKIP_PREFIXES = (
    "File:", "Image:", "Template:", "Category:", "Wikipedia:",
    "Talk:", "User:", "User_talk:", "Help:", "Special:", "Portal:",
    "WP:", "WT:", "MOS:", "MediaWiki:",
)

# Wikitext templates whose content should be dropped entirely before parsing
NOISE_TEMPLATES = {
    "reflist", "references", "notelist", "efn", "sfn", "harvnb",
    "cite", "citation", "infobox", "navbox", "sidebar", "succession box",
    "s-start", "s-end", "coord", "commonscat", "wikisource",
    "short description", "hatnote", "about", "redirect",
    "distinguished", "good article", "featured article",
}

BASE_WIKI_URL = "https://en.wikipedia.org/wiki/"


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class WikipediaGroundTruthConfig:
    dump_path: Path
    processed_dir: Path
    domain: str = "wikipedia"
    max_articles: int = 0        # 0 = no limit
    min_paragraph_chars: int = 30
    outputs: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, project_root: Path | None = None) -> "WikipediaGroundTruthConfig":
        import yaml
        pr = project_root or PROJECT_ROOT
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        dump = Path(cfg.get("dump_path", ""))
        if not dump.is_absolute():
            dump = pr / dump
        processed = Path(cfg.get("processed_dir", "data/processed"))
        if not processed.is_absolute():
            processed = pr / processed
        return cls(
            dump_path=dump,
            processed_dir=processed,
            domain=str(cfg.get("domain", "wikipedia")),
            max_articles=int(cfg.get("max_articles", 0)),
            min_paragraph_chars=int(cfg.get("min_paragraph_chars", 30)),
            outputs=cfg.get("outputs", {}),
        )


# ── XML streaming ─────────────────────────────────────────────────────────────

def _open_dump(dump_path: Path):
    """Open the dump file, handling .bz2 transparently."""
    if dump_path.suffix == ".bz2":
        return bz2.open(dump_path, "rt", encoding="utf-8")
    return open(dump_path, encoding="utf-8")


def iter_pages(dump_path: Path):
    """
    Stream (page_id, title, wikitext) tuples from a Wikipedia XML dump.
    Only yields main-namespace (ns=0) articles.
    """
    ns_map = {"mw": "http://www.mediawiki.org/xml/DTD/mediawiki"}

    with _open_dump(dump_path) as fh:
        context = ET.iterparse(fh, events=("end",))
        for event, elem in context:
            # Strip namespace prefix for tag matching
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag != "page":
                continue

            ns_el = elem.find(".//{http://www.mediawiki.org/xml/export-0.11/}ns")
            if ns_el is None:
                ns_el = elem.find(".//ns")
            ns = ns_el.text if ns_el is not None else "0"
            if ns != "0":
                elem.clear()
                continue

            title_el = elem.find(".//{http://www.mediawiki.org/xml/export-0.11/}title")
            if title_el is None:
                title_el = elem.find(".//title")
            title = title_el.text if title_el is not None else ""

            id_el = elem.find(".//{http://www.mediawiki.org/xml/export-0.11/}id")
            if id_el is None:
                id_el = elem.find(".//id")
            page_id = int(id_el.text) if id_el is not None else 0

            text_el = elem.find(".//{http://www.mediawiki.org/xml/export-0.11/}text")
            if text_el is None:
                text_el = elem.find(".//text")
            wikitext = text_el.text or "" if text_el is not None else ""

            # Skip redirects
            if wikitext.strip().lower().startswith("#redirect"):
                elem.clear()
                continue

            yield page_id, title, wikitext
            elem.clear()


# ── Wikitext parsing ──────────────────────────────────────────────────────────

def _is_skip_link(target: str) -> bool:
    """True if the link target should be excluded (non-article namespaces)."""
    t = target.strip()
    for prefix in SKIP_PREFIXES:
        if t.startswith(prefix):
            return True
    return False


def _normalise_target(target: str) -> str:
    """Convert link target to page_name format (spaces→underscores, strip fragment)."""
    t = target.strip()
    if "#" in t:
        t = t.split("#", 1)[0].strip()
    return t.replace(" ", "_")


def _drop_noise_templates(wikicode) -> None:
    """Remove noisy templates (infoboxes, navboxes, references) in-place."""
    for tpl in wikicode.filter_templates():
        name = tpl.name.strip().lower()
        for noise in NOISE_TEMPLATES:
            if name.startswith(noise):
                try:
                    wikicode.remove(tpl)
                except Exception:
                    pass
                break


def _extract_paragraphs_with_links(
    wikitext: str,
    page_name_to_article_id: dict[str, int],
    base_url: str,
) -> list[tuple[str, list[dict]]]:
    """
    Parse wikitext into (paragraph_text, links) pairs.

    Each link dict has:
        anchor_text, plain_text_rel_char_start, plain_text_rel_char_end,
        link_type, target_page_name, article_id_of_internal_link, resolved_url
    """
    try:
        wikicode = mwparserfromhell.parse(wikitext)
    except Exception as e:
        logger.warning("mwparserfromhell parse error: %s", e)
        return []

    _drop_noise_templates(wikicode)

    # Strip to plain text — preserving wikilinks so we can extract them
    # We do a manual pass: iterate nodes, build text+link list
    plain_parts: list[str] = []
    all_links: list[dict] = []
    offset = 0

    for node in wikicode.nodes:
        if isinstance(node, mwparserfromhell.nodes.Text):
            plain_parts.append(str(node))
            offset += len(str(node))
        elif isinstance(node, mwparserfromhell.nodes.Wikilink):
            target = str(node.title).strip()
            anchor = str(node.text).strip() if node.text else target
            # Remove any nested markup from anchor
            anchor = re.sub(r"\[\[.*?\]\]", "", anchor).strip()
            anchor = re.sub(r"'{2,}", "", anchor).strip()
            if not anchor:
                anchor = target

            if _is_skip_link(target):
                # Still include the display text, just not as a link
                plain_parts.append(anchor)
                offset += len(anchor)
            else:
                target_page = _normalise_target(target)
                aid = page_name_to_article_id.get(target_page)
                start = offset
                plain_parts.append(anchor)
                offset += len(anchor)
                all_links.append({
                    "anchor_text": anchor,
                    "plain_text_rel_char_start": start,
                    "plain_text_rel_char_end": offset,
                    "link_type": "internal",
                    "target_page_name": target_page,
                    "article_id_of_internal_link": aid,
                    "resolved_url": base_url + target_page,
                })
        elif isinstance(node, mwparserfromhell.nodes.Template):
            # Templates already removed above; remaining ones are minor — skip
            pass
        elif isinstance(node, mwparserfromhell.nodes.Tag):
            # <ref>, <br>, etc — extract inner text if it's meaningful
            inner = str(node.contents) if node.contents else ""
            # Skip reference tags
            if str(node.tag).lower() in ("ref", "references", "gallery", "math"):
                pass
            elif inner.strip():
                plain_parts.append(inner)
                offset += len(inner)
        else:
            # Headings, external links, etc — just get plain text
            txt = node.__strip__(normalize=True, collapse=True)
            if txt:
                plain_parts.append(str(txt))
                offset += len(str(txt))

    full_text = "".join(plain_parts)

    # Split into paragraphs by double newline
    paragraphs: list[tuple[str, list[dict]]] = []
    para_start = 0
    for block in re.split(r"\n{2,}", full_text):
        block = block.strip()
        # Skip section headers and very short blocks
        if not block or block.startswith("==") or len(block) < 30:
            para_start += len(block) + 2
            continue

        block_start_in_full = full_text.find(block, para_start)
        if block_start_in_full < 0:
            block_start_in_full = para_start
        block_end_in_full = block_start_in_full + len(block)

        para_links = []
        for lk in all_links:
            ls = lk["plain_text_rel_char_start"]
            le = lk["plain_text_rel_char_end"]
            if le <= block_start_in_full or ls >= block_end_in_full:
                continue
            para_links.append({
                "anchor_text": lk["anchor_text"],
                "plain_text_rel_char_start": max(0, ls - block_start_in_full),
                "plain_text_rel_char_end": min(len(block), le - block_start_in_full),
                "link_type": lk["link_type"],
                "target_page_name": lk["target_page_name"],
                "article_id_of_internal_link": lk["article_id_of_internal_link"],
                "resolved_url": lk["resolved_url"],
            })

        paragraphs.append((block, para_links))
        para_start = block_end_in_full

    return paragraphs


def _split_into_sentences(text: str) -> list[str]:
    """Split paragraph text into sentences."""
    text = text.strip()
    if not text:
        return []
    try:
        import nltk
        try:
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except ImportError:
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in parts if s.strip()]


def _links_in_segment(
    para_links: list[dict],
    seg_start: int,
    seg_end: int,
) -> list[dict]:
    out = []
    for lk in para_links:
        ls = lk["plain_text_rel_char_start"]
        le = lk["plain_text_rel_char_end"]
        if le <= seg_start or ls >= seg_end:
            continue
        out.append({
            "anchor_index": len(out) + 1,
            "anchor_text": lk["anchor_text"],
            "plain_text_rel_char_start": max(0, ls - seg_start),
            "plain_text_rel_char_end": min(seg_end - seg_start, le - seg_start),
            "link_type": lk["link_type"],
            "target_page_name": lk.get("target_page_name"),
            "article_id_of_internal_link": lk.get("article_id_of_internal_link"),
            "resolved_url": lk.get("resolved_url"),
        })
    return out


# ── Article ID mapping ────────────────────────────────────────────────────────

def build_page_name_to_article_id(dump_path: Path, max_articles: int = 0) -> dict[str, int]:
    """
    First pass over the dump: build {page_name: page_id} mapping.
    Required so internal links can be resolved to article_ids.
    """
    mapping: dict[str, int] = {}
    logger.info("First pass: building page_name → article_id mapping from %s", dump_path.name)
    count = 0
    for page_id, title, _ in iter_pages(dump_path):
        page_name = title.replace(" ", "_")
        mapping[page_name] = page_id
        count += 1
        if max_articles and count >= max_articles:
            break
        if count % 5000 == 0:
            logger.info("  First pass progress: %d articles indexed", count)
    logger.info("First pass done: %d articles indexed", len(mapping))
    return mapping


# ── Main build function ───────────────────────────────────────────────────────

def run_wikipedia_ground_truth_build(config: WikipediaGroundTruthConfig) -> dict[str, Path]:
    """
    Main entry point. Two-pass over the XML dump:
      Pass 1: build page_name → article_id mapping
      Pass 2: parse each article, emit JSONL records

    Output schema is identical to run_ground_truth_build() in ground_truth.py.
    """
    domain = config.domain
    out_dir = config.processed_dir / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    if not config.dump_path.exists():
        raise FileNotFoundError(f"Dump not found: {config.dump_path}")

    # Pass 1
    page_name_to_article_id = build_page_name_to_article_id(
        config.dump_path, config.max_articles
    )

    # Pass 2
    logger.info("Second pass: parsing articles and extracting ground truth")
    all_paragraphs: list[dict] = []
    all_sentences: list[dict] = []
    all_articles: list[dict] = []

    para_global = 0
    sent_global = 0
    count = 0

    for page_id, title, wikitext in iter_pages(config.dump_path):
        count += 1
        if config.max_articles and count > config.max_articles:
            break

        page_name = title.replace(" ", "_")
        article_url = BASE_WIKI_URL + page_name
        source_rel = f"data/raw/wikipedia/{page_id}.xml"

        para_list = _extract_paragraphs_with_links(
            wikitext, page_name_to_article_id, BASE_WIKI_URL
        )

        # Article plain text = join of all paragraph texts
        article_plain_text = "\n\n".join(p for p, _ in para_list)

        # Article-level links (all links across all paragraphs, with absolute offsets)
        art_links: list[dict] = []
        char_cursor = 0
        for para_text, para_links in para_list:
            for i, lk in enumerate(para_links):
                art_links.append({
                    "anchor_index": len(art_links) + 1,
                    "anchor_text": lk["anchor_text"],
                    "link_type": lk["link_type"],
                    "plain_text_char_start": char_cursor + lk["plain_text_rel_char_start"],
                    "plain_text_char_end": char_cursor + lk["plain_text_rel_char_end"],
                    "target_page_name": lk.get("target_page_name"),
                    "article_id_of_internal_link": lk.get("article_id_of_internal_link"),
                    "resolved_url": lk.get("resolved_url"),
                })
            char_cursor += len(para_text) + 2  # +2 for "\n\n"

        article_record = {
            "granularity": "article",
            "article_id": page_id,
            "article_record_id": f"{domain}_article_{page_id:07d}",
            "page_name": page_name,
            "title": title,
            "article_plain_text": article_plain_text,
            "url": article_url,
            "source_path": source_rel,
            "links": art_links,
        }
        all_articles.append(article_record)

        # Paragraph records
        for pi, (para_text, para_links) in enumerate(para_list):
            if len(para_text) < config.min_paragraph_chars:
                continue
            para_id = f"{domain}_paragraph_{para_global + 1:07d}"
            para_links_indexed = [
                {"anchor_index": i + 1, **lk} for i, lk in enumerate(para_links)
            ]
            all_paragraphs.append({
                "granularity": "paragraph",
                "article_id": page_id,
                "page_name": page_name,
                "title": title,
                "paragraph_id": para_id,
                "paragraph_index": pi,
                "paragraph_text": para_text,
                "url": article_url,
                "source_path": source_rel,
                "links": para_links_indexed,
            })

            # Sentence records
            sentences = _split_into_sentences(para_text)
            sent_char = 0
            for si, sent_text in enumerate(sentences):
                sent_id = f"{domain}_sentence_{sent_global + 1:07d}"
                seg_start = sent_char
                seg_end = sent_char + len(sent_text)
                sent_links = _links_in_segment(para_links, seg_start, seg_end)
                all_sentences.append({
                    "granularity": "sentence",
                    "article_id": page_id,
                    "page_name": page_name,
                    "title": title,
                    "sentence_id": sent_id,
                    "sentence_index": si,
                    "sentence_text": sent_text,
                    "url": article_url,
                    "source_path": source_rel,
                    "links": sent_links,
                })
                sent_char = seg_end + 1
                sent_global += 1

            para_global += 1

        if count % 500 == 0:
            logger.info(
                "[Progress] %d articles | paragraphs=%d | sentences=%d",
                count, len(all_paragraphs), len(all_sentences),
            )

    logger.info("=== Wikipedia ground truth summary ===")
    logger.info("Articles:   %d", len(all_articles))
    logger.info("Paragraphs: %d", len(all_paragraphs))
    logger.info("Sentences:  %d", len(all_sentences))
    total_internal = sum(
        1 for p in all_paragraphs for lk in p["links"]
        if lk.get("link_type") == "internal"
    )
    logger.info("Internal links (paragraph level): %d", total_internal)

    # Write outputs
    result_paths: dict[str, Path] = {}
    outputs = config.outputs

    para_jsonl = out_dir / f"paragraphs_{domain}.jsonl"
    if outputs.get("paragraphs_jsonl", True):
        with open(para_jsonl, "w", encoding="utf-8") as f:
            for r in all_paragraphs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        result_paths["paragraphs_jsonl"] = para_jsonl
        logger.info("Written: %s", para_jsonl)

    sent_jsonl = out_dir / f"sentences_{domain}.jsonl"
    if outputs.get("sentences_jsonl", True):
        with open(sent_jsonl, "w", encoding="utf-8") as f:
            for r in all_sentences:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        result_paths["sentences_jsonl"] = sent_jsonl
        logger.info("Written: %s", sent_jsonl)

    art_pg = out_dir / f"articles_page_granularity_{domain}.jsonl"
    if outputs.get("articles_page_granularity_jsonl", True):
        with open(art_pg, "w", encoding="utf-8") as f:
            for r in all_articles:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        result_paths["articles_page_granularity_jsonl"] = art_pg
        logger.info("Written: %s", art_pg)

    art_idx = out_dir / f"articles_{domain}.jsonl"
    if outputs.get("articles_index_jsonl", True):
        index_records = [
            {
                "article_id": r["article_id"],
                "page_name": r["page_name"],
                "title": r["title"],
                "url": r["url"],
            }
            for r in all_articles
        ]
        with open(art_idx, "w", encoding="utf-8") as f:
            for r in index_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        result_paths["articles_index_jsonl"] = art_idx
        logger.info("Written: %s", art_idx)

    if outputs.get("paragraphs_csv", False):
        csv_path = out_dir / f"paragraphs_{domain}.csv"
        if all_paragraphs:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["paragraph_id", "article_id", "paragraph_text"])
                w.writeheader()
                for r in all_paragraphs:
                    w.writerow({
                        "paragraph_id": r["paragraph_id"],
                        "article_id": r["article_id"],
                        "paragraph_text": r["paragraph_text"],
                    })
        result_paths["paragraphs_csv"] = csv_path

    if outputs.get("paragraph_links_csv", False):
        csv_path = out_dir / f"paragraph_links_{domain}.csv"
        rows = []
        for p in all_paragraphs:
            for lk in p["links"]:
                if lk.get("link_type") == "internal":
                    rows.append({
                        "paragraph_id": p["paragraph_id"],
                        "article_id": p["article_id"],
                        "anchor_index": lk["anchor_index"],
                        "anchor_text": lk["anchor_text"],
                        "target_page_name": lk.get("target_page_name", ""),
                        "article_id_of_internal_link": lk.get("article_id_of_internal_link", ""),
                        "char_start": lk["plain_text_rel_char_start"],
                        "char_end": lk["plain_text_rel_char_end"],
                    })
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        result_paths["paragraph_links_csv"] = csv_path

    return result_paths
