"""HTML renderer for the linking pipeline.

Takes a plain-text article and a list of confirmed predicted links
(char_start, char_end, anchor_text, fandom_url) and produces an HTML string
with <a href="..."> tags injected at the correct positions.

Design:
  - Spans are processed in reverse char_start order so that inserting tags
    at later positions does not shift the offsets of earlier spans.
  - Overlapping spans are resolved before rendering according to the
    configured overlap_strategy (default: keep the longest span).
  - The output is a self-contained HTML snippet (not a full page) unless
    wrap_article=True, in which case a <div class="wiki-article"> wraps it.
"""
from __future__ import annotations

import html
import logging
from pathlib import Path

log = logging.getLogger("linking")


# ── Overlap resolution ────────────────────────────────────────────────────────

def _resolve_overlaps(links: list[dict], strategy: str = "longest") -> list[dict]:
    """
    Remove overlapping spans, keeping one per overlapping group.

    strategy="longest": keep the span with the largest (char_end - char_start).
    strategy="first":   keep the span with the smallest char_start (leftmost).
    """
    if not links:
        return links

    sorted_links = sorted(links, key=lambda x: (x["char_start"], -(x["char_end"] - x["char_start"])))
    kept: list[dict] = []
    last_end = -1

    for link in sorted_links:
        start = link["char_start"]
        end   = link["char_end"]
        if start >= last_end:
            kept.append(link)
            last_end = end
        else:
            # Overlap detected
            if strategy == "longest":
                if (end - start) > (kept[-1]["char_end"] - kept[-1]["char_start"]):
                    kept[-1] = link
                    last_end = end
            # strategy="first": do nothing, keep the existing one

    return kept


# ── HTML rendering ────────────────────────────────────────────────────────────

def render_html(
    text: str,
    confirmed_links: list[dict],
    wrap_article: bool = True,
    link_class: str = "wiki-link",
    overlap_strategy: str = "longest",
) -> str:
    """
    Render plain text as HTML with <a> tags at confirmed link positions.

    Args:
        text:            Plain article text.
        confirmed_links: List of dicts with keys:
                           char_start, char_end, anchor_text,
                           fandom_url, article_id, page_name.
                         Only links where linked=True should be passed.
        wrap_article:    If True, wrap output in <div class="wiki-article">.
        link_class:      CSS class for all injected <a> tags.
        overlap_strategy: How to resolve overlapping spans.

    Returns:
        HTML string.
    """
    if not confirmed_links:
        body = html.escape(text)
        if wrap_article:
            return f'<div class="wiki-article">\n<p>{body}</p>\n</div>'
        return f'<p>{body}</p>'

    resolved = _resolve_overlaps(confirmed_links, strategy=overlap_strategy)

    # Process spans in reverse order (last → first) to preserve earlier offsets
    resolved_sorted = sorted(resolved, key=lambda x: x["char_start"], reverse=True)

    chars = list(text)
    for link in resolved_sorted:
        start  = link["char_start"]
        end    = link["char_end"]
        url    = link.get("fandom_url", "#")
        anchor = html.escape(text[start:end])
        tag    = f'<a href="{html.escape(url)}" class="{link_class}">{anchor}</a>'
        # Replace the span in the character list with the tag
        chars[start:end] = list(tag)

    body = html.escape("".join(chars), quote=False)
    # Unescape the <a> tags we intentionally inserted (they're already safe)
    # We need a different approach: build from segments instead
    return _render_from_segments(text, resolved_sorted, wrap_article, link_class)


def _render_from_segments(
    text: str,
    resolved: list[dict],   # sorted by char_start descending
    wrap_article: bool,
    link_class: str,
) -> str:
    """Build HTML by concatenating plain-text segments and <a> tags."""
    # Work forward through the text
    segments: list[str] = []
    cursor = 0
    # Sort ascending for forward pass
    for_links = sorted(resolved, key=lambda x: x["char_start"])

    for link in for_links:
        start = link["char_start"]
        end   = link["char_end"]
        url   = link.get("fandom_url", "#")

        if start > cursor:
            segments.append(html.escape(text[cursor:start]))
        anchor_text = html.escape(text[start:end])
        segments.append(
            f'<a href="{html.escape(url)}" class="{link_class}">{anchor_text}</a>'
        )
        cursor = end

    # Remaining text after last span
    if cursor < len(text):
        segments.append(html.escape(text[cursor:]))

    body = "".join(segments)
    if wrap_article:
        return f'<div class="wiki-article">\n<p>{body}</p>\n</div>'
    return f'<p>{body}</p>'


# ── Save HTML ─────────────────────────────────────────────────────────────────

def save_html(html_str: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    full = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<style>\n"
        "  body { font-family: Georgia, serif; max-width: 860px; margin: 2em auto; }\n"
        "  .wiki-article p { line-height: 1.7; font-size: 1.05em; }\n"
        "  a.wiki-link { color: #0645ad; text-decoration: none; }\n"
        "  a.wiki-link:hover { text-decoration: underline; }\n"
        "</style>\n</head>\n<body>\n"
        + html_str
        + "\n</body>\n</html>"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(full)
