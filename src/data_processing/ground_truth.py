"""Ground truth building: parse Fandom HTML into paragraphs/sentences with link spans."""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString

logger = logging.getLogger()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Link type classification for Fandom/MediaWiki
LINK_SKIP_PREFIXES = (
    "File:",
    "User:",
    "User_blog:",
    "Template:",
    "Help:",
    "Special:",
    "Forum:",
    "Message_Wall:",
    "MediaWiki:",
    "Module:",
)


def _extract_page_name_from_html(html: str) -> str | None:
    """Extract wgPageName from Fandom HTML (e.g. Beverly_Hills_Cop_Wiki)."""
    m = re.search(r'"wgPageName"\s*:\s*"([^"]+)"', html)
    if m:
        return m.group(1)
    m = re.search(r'"wgRelevantPageName"\s*:\s*"([^"]+)"', html)
    if m:
        return m.group(1)
    return None


def _extract_title_from_html(html: str) -> str | None:
    """Extract human-readable title (wgTitle) from HTML."""
    m = re.search(r'"wgTitle"\s*:\s*"([^"]+)"', html)
    if m:
        return m.group(1)
    return None


def _classify_link(
    href: str,
    base_netloc: str,
    page_name_to_article_id: dict[str, int],
) -> tuple[str, str | None, int | None]:
    """
    Classify link and resolve internal targets.
    Returns (link_type, target_page_name, article_id_of_internal_link).
    link_type: internal | external | category | file | other
    """
    if not href or href.startswith("#"):
        return "other", None, None
    parsed = urlparse(href)
    path = unquote(parsed.path or "").strip()
    if not path.startswith("/wiki/"):
        # Relative or non-wiki
        if path.startswith("/"):
            full = urljoin(f"https://{base_netloc}", path)
            parsed = urlparse(full)
            path = unquote(parsed.path or "").strip()
        else:
            return "other", None, None
    page_part = path.split("/wiki/", 1)[1] if "/wiki/" in path else ""
    if not page_part:
        return "other", None, None
    target_page = page_part.replace(" ", "_")
    for prefix in LINK_SKIP_PREFIXES:
        if target_page.startswith(prefix):
            if prefix == "Category:" or "category" in prefix.lower():
                return "category", None, None
            if prefix == "File:":
                return "file", None, None
            return "other", None, None
    netloc = (parsed.netloc or "").lower()
    # Relative /wiki/ links (no netloc) are same-site internal
    if not netloc or base_netloc.lower() in netloc or netloc.endswith(".fandom.com"):
        aid = page_name_to_article_id.get(target_page)
        return "internal", target_page, aid
    return "external", None, None


def _get_resolved_url(href: str, base_url: str) -> str | None:
    """Resolve relative href to full URL."""
    if not href or href.startswith("#"):
        return None
    return urljoin(base_url, href)


def _extract_plain_text_and_links(
    soup: BeautifulSoup,
    base_url: str,
    base_netloc: str,
    page_name_to_article_id: dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Traverse mw-parser-output content to build plain text and link list with char offsets.
    Returns (plain_text, links) where links have anchor_text, plain_text_char_start, plain_text_char_end,
    link_type, target_page_name, article_id_of_internal_link, resolved_url.
    """
    plain_parts: list[str] = []
    links: list[dict[str, Any]] = []
    offset = 0

    def _process_element(el) -> None:
        nonlocal offset
        if isinstance(el, NavigableString):
            s = str(el)
            plain_parts.append(s)
            offset += len(s)
            return
        if el.name == "a":
            href = el.get("href") or ""
            anchor_text = el.get_text(strip=False)
            link_type, target_page, aid = _classify_link(
                href, base_netloc, page_name_to_article_id
            )
            resolved = _get_resolved_url(href, base_url)
            start = offset
            plain_parts.append(anchor_text)
            offset += len(anchor_text)
            end = offset
            links.append({
                "anchor_text": anchor_text,
                "plain_text_char_start": start,
                "plain_text_char_end": end,
                "link_type": link_type,
                "target_page_name": target_page,
                "article_id_of_internal_link": aid,
                "resolved_url": resolved,
            })
            return
        if el.name in ("script", "style", "noscript"):
            return
        for child in el.children:
            _process_element(child)

    content = soup.find("div", class_="mw-parser-output")
    if not content:
        content = soup
    for child in content.children:
        _process_element(child)

    plain_text = "".join(plain_parts)
    return plain_text, links


def _extract_paragraphs_with_links(
    soup: BeautifulSoup,
    base_url: str,
    base_netloc: str,
    page_name_to_article_id: dict[str, int],
) -> list[tuple[str, list[dict]]]:
    """
    Extract each <p> as (text, links) with link offsets relative to paragraph.
    """
    content = soup.find("div", class_="mw-parser-output")
    if not content:
        return []
    result = []
    for p_el in content.find_all("p"):
        plain_parts = []
        links = []
        offset = 0

        def _walk(el) -> None:
            nonlocal offset
            if isinstance(el, NavigableString):
                s = str(el)
                plain_parts.append(s)
                offset += len(s)
                return
            if el.name == "a":
                href = el.get("href") or ""
                anchor_text = el.get_text(strip=False)
                link_type, target_page, aid = _classify_link(
                    href, base_netloc, page_name_to_article_id
                )
                resolved = _get_resolved_url(href, base_url)
                start = offset
                plain_parts.append(anchor_text)
                offset += len(anchor_text)
                links.append({
                    "anchor_text": anchor_text,
                    "plain_text_rel_char_start": start,
                    "plain_text_rel_char_end": offset,
                    "link_type": link_type,
                    "target_page_name": target_page,
                    "article_id_of_internal_link": aid,
                    "resolved_url": resolved,
                })
                return
            for c in el.children:
                _walk(c)

        for c in p_el.children:
            _walk(c)
        text = "".join(plain_parts).strip()
        if text:
            result.append((text, links))
    if not result:
        plain_text, full_links = _extract_plain_text_and_links(
            soup, base_url, base_netloc, page_name_to_article_id
        )
        blocks = re.split(r"\n{2,}", plain_text)
        pos = 0
        for b in blocks:
            b = b.strip()
            if not b:
                continue
            seg_start = plain_text.find(b, pos)
            if seg_start < 0:
                seg_start = pos
            seg_end = seg_start + len(b)
            seg_links = []
            for lk in full_links:
                ls, le = lk["plain_text_char_start"], lk["plain_text_char_end"]
                if le <= seg_start or ls >= seg_end:
                    continue
                seg_links.append({
                    "anchor_text": lk["anchor_text"],
                    "plain_text_rel_char_start": max(0, ls - seg_start),
                    "plain_text_rel_char_end": min(len(b), le - seg_start),
                    "link_type": lk["link_type"],
                    "target_page_name": lk.get("target_page_name"),
                    "article_id_of_internal_link": lk.get("article_id_of_internal_link"),
                    "resolved_url": lk.get("resolved_url"),
                })
            result.append((b, seg_links))
            pos = seg_end
    return result


def _split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter (handles common abbreviations)."""
    text = text.strip()
    if not text:
        return []
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


@dataclass
class GroundTruthConfig:
    domain: str
    raw_dir: Path
    processed_dir: Path
    outputs: dict[str, bool] = field(default_factory=dict)
    paragraph: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, project_root: Path | None = None) -> "GroundTruthConfig":
        import yaml
        project_root = project_root or PROJECT_ROOT
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        raw = Path(cfg.get("raw_dir", "data/raw"))
        processed = Path(cfg.get("processed_dir", "data/processed"))
        if not raw.is_absolute():
            raw = project_root / raw
        if not processed.is_absolute():
            processed = project_root / processed
        return cls(
            domain=str(cfg.get("domain", "")),
            raw_dir=raw,
            processed_dir=processed,
            outputs=cfg.get("outputs", {}),
            paragraph=cfg.get("paragraph", {}),
        )


def build_page_name_to_article_id(
    html_dir: Path,
    domain: str,
) -> dict[str, int]:
    """Build mapping page_name -> article_id from HTML files."""
    mapping: dict[str, int] = {}
    for p in sorted(html_dir.glob("*.html")):
        stem = p.stem
        if stem.isdigit():
            aid = int(stem)
        else:
            continue
        try:
            html = p.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Could not read %s: %s", p, e)
            continue
        page_name = _extract_page_name_from_html(html)
        if page_name:
            mapping[page_name] = aid
    return mapping


def _links_in_text_segment(
    para_links: list[dict],
    para_text: str,
    seg_start: int,
    seg_end: int,
) -> list[dict]:
    """Filter links overlapping segment and make offsets relative to segment."""
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


def process_html_file(
    html_path: Path,
    article_id: int,
    page_name: str,
    title: str,
    base_url: str,
    page_name_to_article_id: dict[str, int],
    domain: str,
    source_rel: str,
) -> tuple[list[dict], list[dict], dict]:
    """
    Process one HTML file into paragraphs, sentences, and article-level record.
    Returns (paragraphs, sentences, article_record).
    """
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    base_netloc = urlparse(base_url).netloc

    para_list = _extract_paragraphs_with_links(
        soup, base_url, base_netloc, page_name_to_article_id
    )
    plain_text, full_links = _extract_plain_text_and_links(
        soup, base_url, base_netloc, page_name_to_article_id
    )

    para_records = []
    sent_records = []
    para_global = 0
    sent_global = 0

    for pi, (para_text, para_links) in enumerate(para_list):
        para_id = f"{domain}_paragraph_{para_global + 1:07d}"
        para_links_with_idx = [
            {"anchor_index": i + 1, **lk} for i, lk in enumerate(para_links)
        ]
        para_records.append({
            "granularity": "paragraph",
            "article_id": article_id,
            "page_name": page_name,
            "title": title,
            "paragraph_id": para_id,
            "paragraph_index": pi,
            "paragraph_text": para_text,
            "url": base_url,
            "source_path": source_rel,
            "links": para_links_with_idx,
        })
        sentences = _split_into_sentences(para_text)
        sent_char = 0
        for si, sent_text in enumerate(sentences):
            sent_id = f"{domain}_sentence_{sent_global + 1:07d}"
            seg_start = sent_char
            seg_end = sent_char + len(sent_text)
            sent_links = _links_in_text_segment(para_links, para_text, seg_start, seg_end)
            sent_records.append({
                "granularity": "sentence",
                "article_id": article_id,
                "page_name": page_name,
                "title": title,
                "sentence_id": sent_id,
                "sentence_index": si,
                "sentence_text": sent_text,
                "url": base_url,
                "source_path": source_rel,
                "links": sent_links,
            })
            sent_char = seg_end + 1
            sent_global += 1
        para_global += 1

    article_record = {
        "granularity": "article",
        "article_id": article_id,
        "article_record_id": f"{domain}_article_{article_id:07d}",
        "page_name": page_name,
        "title": title,
        "article_plain_text": plain_text,
        "url": base_url,
        "source_path": source_rel,
        "links": [
            {
                "anchor_index": i + 1,
                "anchor_text": l["anchor_text"],
                "link_type": l["link_type"],
                "plain_text_char_start": l["plain_text_char_start"],
                "plain_text_char_end": l["plain_text_char_end"],
                "target_page_name": l.get("target_page_name"),
                "article_id_of_internal_link": l.get("article_id_of_internal_link"),
                "resolved_url": l.get("resolved_url"),
            }
            for i, l in enumerate(full_links)
        ],
    }

    return para_records, sent_records, article_record


def run_ground_truth_build(config: GroundTruthConfig) -> dict[str, Path]:
    """
    Main entry: build ground truth for a domain.
    Returns dict of output paths.
    """
    domain = config.domain
    raw_dir = config.raw_dir / domain
    out_dir = config.processed_dir / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = config.outputs

    if not raw_dir.exists():
        raise FileNotFoundError(f"HTML dir not found: {raw_dir}")

    html_files = sorted(raw_dir.glob("*.html"))
    # Build page_name_to_article_id from HTML files
    logger.info("Building page_name_to_article_id from %d HTML files", len(html_files))
    page_name_to_article_id = build_page_name_to_article_id(raw_dir, domain)
    logger.info("[page_name_to_article_id] Built mapping for %d articles", len(page_name_to_article_id))

    base_url = f"https://{domain}.fandom.com/wiki/"
    all_paragraphs: list[dict] = []
    all_sentences: list[dict] = []
    all_articles: list[dict] = []

    for i, html_path in enumerate(html_files):
        stem = html_path.stem
        if not stem.isdigit():
            continue
        article_id = int(stem)
        html = html_path.read_text(encoding="utf-8")
        page_name = _extract_page_name_from_html(html) or f"Article_{article_id}"
        title = _extract_title_from_html(html) or page_name.replace("_", " ")
        article_url = base_url + page_name.replace(" ", "_")
        source_rel = str(html_path).replace(str(config.raw_dir), "data/raw").replace("\\", "/")

        para_recs, sent_recs, art_rec = process_html_file(
            html_path, article_id, page_name, title,
            article_url, page_name_to_article_id, domain, source_rel,
        )
        all_paragraphs.extend(para_recs)
        all_sentences.extend(sent_recs)
        all_articles.append(art_rec)

        if (i + 1) % 50 == 0:
            n_links = sum(len(p["links"]) for p in para_recs) + sum(len(s["links"]) for s in sent_recs)
            logger.info(
                "[Progress] Processed %d/%d HTML files; articles=%d, paragraphs=%d, sentences=%d, links=%d",
                i + 1, len(html_files), len(all_articles), len(all_paragraphs),
                len(all_sentences),
                sum(len(p["links"]) for p in all_paragraphs) + sum(len(s["links"]) for s in all_sentences),
            )

    link_type_counts: dict[str, int] = {}
    for p in all_paragraphs:
        for lk in p["links"]:
            t = lk.get("link_type", "other")
            link_type_counts[t] = link_type_counts.get(t, 0) + 1

    total_links = sum(len(p["links"]) for p in all_paragraphs)
    logger.info("=== Ground truth summary ===")
    logger.info("Articles:    %d", len(all_articles))
    logger.info("Paragraphs:  %d", len(all_paragraphs))
    logger.info("Sentences:   %d", len(all_sentences))
    logger.info("Links:       %d", total_links)
    logger.info("Link type counts: %s", link_type_counts)

    result_paths: dict[str, Path] = {}

    para_jsonl = out_dir / f"paragraphs_{domain}.jsonl"
    if outputs.get("paragraphs_jsonl", True):
        with open(para_jsonl, "w", encoding="utf-8") as f:
            for r in all_paragraphs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        result_paths["paragraphs_jsonl"] = para_jsonl

    sent_jsonl = out_dir / f"sentences_{domain}.jsonl"
    if outputs.get("sentences_jsonl", True):
        with open(sent_jsonl, "w", encoding="utf-8") as f:
            for r in all_sentences:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        result_paths["sentences_jsonl"] = sent_jsonl

    art_pg = out_dir / f"articles_page_granularity_{domain}.jsonl"
    if outputs.get("articles_page_granularity_jsonl", True):
        with open(art_pg, "w", encoding="utf-8") as f:
            for r in all_articles:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        result_paths["articles_page_granularity_jsonl"] = art_pg

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
        for r in all_paragraphs:
            for lk in r["links"]:
                rows.append({
                    "paragraph_id": r["paragraph_id"],
                    "anchor_text": lk["anchor_text"],
                    "link_type": lk["link_type"],
                })
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["paragraph_id", "anchor_text", "link_type"])
                w.writeheader()
                w.writerows(rows)
        result_paths["paragraph_links_csv"] = csv_path

    if outputs.get("sentences_csv", False):
        csv_path = out_dir / f"sentences_{domain}.csv"
        if all_sentences:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["sentence_id", "article_id", "sentence_text"])
                w.writeheader()
                for r in all_sentences:
                    w.writerow({
                        "sentence_id": r["sentence_id"],
                        "article_id": r["article_id"],
                        "sentence_text": r["sentence_text"],
                    })
        result_paths["sentences_csv"] = csv_path

    if outputs.get("sentence_links_csv", False):
        csv_path = out_dir / f"sentence_links_{domain}.csv"
        rows = []
        for r in all_sentences:
            for lk in r["links"]:
                rows.append({
                    "sentence_id": r["sentence_id"],
                    "anchor_text": lk["anchor_text"],
                    "link_type": lk["link_type"],
                })
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["sentence_id", "anchor_text", "link_type"])
                w.writeheader()
                w.writerows(rows)
        result_paths["sentence_links_csv"] = csv_path

    from datetime import datetime
    manifest = {
        "domain": domain,
        "created_at": datetime.now().isoformat(),
        "files": [{"path": p.name, "size_bytes": p.stat().st_size} for p in result_paths.values() if p.exists()],
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    result_paths["manifest"] = manifest_path

    for k, p in result_paths.items():
        if p.exists():
            logger.info("%s: %s", k, p)

    return result_paths
