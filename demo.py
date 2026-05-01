"""AutoWiki Demo — Wiki Editor Tool.

A wiki editor pastes a paragraph, clicks "Auto-Link", and gets back
the same text with Fandom wiki links automatically inserted.

Span detection uses a hybrid approach:
  1. BERT token classifier (trained on wiki sentences)
  2. Dictionary matcher (exact article-title lookup, case-insensitive)
Both sets of spans are merged and deduplicated before retrieval.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import gradio as gr
from transformers import AutoModelForTokenClassification, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
INDEX_DIR = ROOT / "data/article_retrieval/money-heist/article_index"
CHECKPOINT = (
    ROOT
    / "data/span_id/money-heist/checkpoints/20260305_120731"
    / "sentence_money-heist_bert-base-uncased_BIO_seed42_frac1.0"
    / "checkpoint-368"
)
FAISS_MODEL       = "sentence-transformers/msmarco-distilbert-base-v4"
FAISS_INDEX_FILE  = INDEX_DIR / "faiss_sentence_transformers_msmarco_distilbert_base_v4_title_full_article_flat.index"
FAISS_IDS_FILE    = INDEX_DIR / "embeddings_sentence_transformers_msmarco_distilbert_base_v4_title_full_article_ids.json"
ARTICLES_JSONL    = INDEX_DIR / "articles_title_full_article.jsonl"
FANDOM_BASE_URL   = "https://money-heist.fandom.com/wiki"
QUERY_TEMPLATE    = "Retrieve the topic discussing '{word}'."

EXAMPLES = [
    "The Professor planned the heist at the Royal Mint of Spain with Berlin and Tokyo.",
    "Nairobi was in charge of printing money and became one of the most beloved characters.",
    "Alicia Sierra interrogated Rio during his captivity and used extreme methods.",
    "Denver and Moscow were assigned to guard the hostages on the ground floor.",
    "Lisbon joined the gang in the second part and took command inside the Mint.",
]

# ── Load models (once at startup) ─────────────────────────────────────────────

print("Loading span identification model...", flush=True)
span_tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT))
span_model = AutoModelForTokenClassification.from_pretrained(str(CHECKPOINT))
span_model.eval()

print("Loading article index...", flush=True)
import faiss  # noqa: E402
faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
with open(FAISS_IDS_FILE) as f:
    faiss_article_ids: list[int] = json.load(f)
article_lookup: dict[int, dict] = {}
with open(ARTICLES_JSONL) as f:
    for line in f:
        line = line.strip()
        if line:
            rec = json.loads(line)
            article_lookup[rec["article_id"]] = rec

print("Loading retrieval encoder...", flush=True)
from sentence_transformers import SentenceTransformer  # noqa: E402
retrieval_encoder = SentenceTransformer(FAISS_MODEL)

print("Loading cross-encoder reranker...", flush=True)
from sentence_transformers.cross_encoder import CrossEncoder  # noqa: E402
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Build title index: (title_lower, article_id) sorted longest-first
# so multi-word titles match before their substrings (e.g. "Royal Mint of Spain" before "Spain")
_title_index: list[tuple[str, int]] = sorted(
    [(rec["title"].lower(), rec["article_id"]) for rec in article_lookup.values()
     if len(rec["title"]) > 2],
    key=lambda x: len(x[0]),
    reverse=True,
)
# Fast lookup: title_lower → article_id
_title_to_id: dict[str, int] = {t: aid for t, aid in _title_index}

print("Ready.", flush=True)

# ── Pipeline ──────────────────────────────────────────────────────────────────

def _detect_spans_model(text: str) -> list[tuple[int, int, str]]:
    """BERT token classifier: good on wiki-style sentences."""
    enc = span_tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    offsets: list[tuple[int, int]] = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        logits = span_model(**enc).logits[0]
    pred_ids = logits.argmax(-1).tolist()
    id2label: dict[int, str] = span_model.config.id2label

    spans = []
    i = 0
    while i < len(pred_ids):
        if id2label[pred_ids[i]] == "B-SPAN":
            start_tok = i
            i += 1
            while i < len(pred_ids) and id2label[pred_ids[i]] == "I-SPAN":
                i += 1
            end_tok = i - 1
            c_start, c_end = offsets[start_tok][0], offsets[end_tok][1]
            if c_start < c_end and c_start != 0:
                spans.append((c_start, c_end, text[c_start:c_end]))
        else:
            i += 1
    return spans


def _detect_spans_dictionary(text: str) -> list[tuple[int, int, str, int]]:
    """Exact case-insensitive match of known article titles.
    Returns (char_start, char_end, anchor_text, article_id) — article already known.
    """
    spans = []
    text_lower = text.lower()
    for title, art_id in _title_index:
        start = 0
        while True:
            idx = text_lower.find(title, start)
            if idx == -1:
                break
            end = idx + len(title)
            before_ok = idx == 0 or not text[idx - 1].isalpha()
            after_ok  = end == len(text) or not text[end].isalpha()
            if before_ok and after_ok:
                spans.append((idx, end, text[idx:end], art_id))
            start = idx + 1
    return spans


def _merge_spans_3(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Merge (start, end, anchor) spans; on overlap keep the longer one."""
    if not spans:
        return []
    sorted_spans = sorted(set(spans), key=lambda x: (x[0], -(x[1] - x[0])))
    merged: list[tuple[int, int, str]] = []
    last_end = -1
    for c_start, c_end, anchor in sorted_spans:
        if c_start >= last_end:
            merged.append((c_start, c_end, anchor))
            last_end = c_end
        elif (c_end - c_start) > (merged[-1][1] - merged[-1][0]):
            merged[-1] = (c_start, c_end, anchor)
            last_end = c_end
    return merged


def _detect_spans(text: str):
    """Return (dict_spans, model_spans) separately so caller can handle them differently.

    dict_spans  : list of (start, end, anchor, article_id)  — article already resolved
    model_spans : list of (start, end, anchor)               — need retrieval
    """
    raw_dict   = _detect_spans_dictionary(text)   # (start, end, anchor, article_id)
    raw_model  = _detect_spans_model(text)         # (start, end, anchor)

    # Positions already covered by dictionary spans
    dict_positions = set()
    for c_start, c_end, anchor, art_id in raw_dict:
        dict_positions.update(range(c_start, c_end))

    # Keep model spans only where dict has no match
    model_only = [
        s for s in raw_model
        if not dict_positions.intersection(range(s[0], s[1]))
    ]

    dict_merged  = sorted(raw_dict, key=lambda x: x[0])
    model_merged = _merge_spans_3(model_only)
    return dict_merged, model_merged


def _retrieve_and_rerank(anchor: str, top_k: int) -> list[dict]:
    query = QUERY_TEMPLATE.format(word=anchor)
    emb = retrieval_encoder.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    scores, indices = faiss_index.search(emb, top_k + 1)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(faiss_article_ids):
            continue
        art_id = faiss_article_ids[idx]
        if art_id in article_lookup:
            candidates.append({"article_id": art_id, "retrieval_score": float(score)})
        if len(candidates) == top_k:
            break

    if not candidates:
        return []

    pairs = [[query, article_lookup[c["article_id"]]["text"][:512]] for c in candidates]
    rerank_scores = reranker.predict(pairs, show_progress_bar=False)
    for c, s in zip(candidates, rerank_scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates


def _render_linked_html(text: str, confirmed_links: list[dict]) -> str:
    """Render text as HTML with injected <a> tags, styled like a wiki editor preview."""
    if not confirmed_links:
        import html
        return f'<div class="article-preview"><p>{html.escape(text)}</p></div>'

    sys.path.insert(0, str(ROOT / "src"))
    from linking_pipeline.html_renderer import render_html
    inner = render_html(text, confirmed_links, wrap_article=False)
    return f'<div class="article-preview">{inner}</div>'


def _links_summary_html(confirmed_links: list[dict], skipped_spans: list[str]) -> str:
    """Small status block shown below the preview."""
    lines = []
    if confirmed_links:
        lines.append(f"<b>{len(confirmed_links)} link(s) added:</b>")
        for lnk in confirmed_links:
            url = lnk["fandom_url"]
            art = article_lookup[lnk["article_id"]]
            lines.append(
                f'&nbsp;&nbsp;• <code>{lnk["anchor_text"]}</code> → '
                f'<a href="{url}" target="_blank">{art["title"]}</a>'
            )
    if skipped_spans:
        lines.append(f'<b style="color:#888">{len(skipped_spans)} span(s) below NIL threshold (not linked):</b>')
        for s in skipped_spans:
            lines.append(f'&nbsp;&nbsp;• <code>{s}</code>')
    if not lines:
        lines.append('<span style="color:#888">No entity spans detected in this text.</span>')
    return "<br>".join(lines)


# ── Main handler ──────────────────────────────────────────────────────────────

def auto_link(text: str, nil_threshold: float):
    text = text.strip()
    if not text:
        return (
            '<div class="article-preview"><p style="color:#aaa">Your linked text will appear here.</p></div>',
            "",
        )

    dict_spans, model_spans = _detect_spans(text)
    confirmed_links = []
    skipped_spans   = []

    # Dictionary spans: article already known — link directly, no retrieval needed
    for c_start, c_end, anchor, art_id in dict_spans:
        art = article_lookup.get(art_id)
        if art is None:
            continue
        confirmed_links.append({
            "char_start":  c_start,
            "char_end":    c_end,
            "anchor_text": anchor,
            "fandom_url":  f"{FANDOM_BASE_URL}/{art['page_name']}",
            "article_id":  art_id,
            "page_name":   art["page_name"],
        })

    # Model spans: novel entities not in the title dictionary — use retrieval + rerank
    for c_start, c_end, anchor in model_spans:
        candidates = _retrieve_and_rerank(anchor, top_k=10)
        if not candidates:
            skipped_spans.append(anchor)
            continue
        best = candidates[0]
        if best["rerank_score"] < nil_threshold:
            skipped_spans.append(anchor)
            continue
        art = article_lookup[best["article_id"]]
        confirmed_links.append({
            "char_start":  c_start,
            "char_end":    c_end,
            "anchor_text": anchor,
            "fandom_url":  f"{FANDOM_BASE_URL}/{art['page_name']}",
            "article_id":  best["article_id"],
            "page_name":   art["page_name"],
        })

    preview_html = _render_linked_html(text, confirmed_links)
    summary_html = _links_summary_html(confirmed_links, skipped_spans)
    return preview_html, summary_html


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
.article-preview {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 18px 22px;
    background: #fafafa;
    min-height: 80px;
}
.article-preview p {
    font-family: Georgia, serif;
    font-size: 1.08em;
    line-height: 1.8;
    margin: 0;
    color: #222;
}
a.wiki-link {
    color: #0645ad;
    text-decoration: none;
    border-bottom: 1px solid #a2a9b1;
}
a.wiki-link:hover { background: #eaf3fb; }
#run-btn { font-size: 1.1em; }
"""

with gr.Blocks(title="AutoWiki — Wiki Editor", css=CSS) as demo:

    gr.Markdown(
        "## AutoWiki &nbsp;·&nbsp; Automatic Wiki Linking\n"
        "Write or paste a paragraph below. AutoWiki will detect entity mentions "
        "and automatically link them to the correct Fandom wiki article."
    )

    with gr.Row():

        # Left: input
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Your text",
                placeholder="Write about Money Heist characters, places, or events...",
                lines=6,
                max_lines=12,
            )
            nil_slider = gr.Slider(
                minimum=-5.0,
                maximum=5.0,
                value=-3.0,
                step=0.25,
                label="Confidence threshold  (lower = more links, higher = fewer but more precise)",
            )
            run_btn = gr.Button("Auto-Link →", variant="primary", elem_id="run-btn")
            gr.Examples(
                examples=EXAMPLES,
                inputs=text_input,
                label="Try an example",
            )

        # Right: output
        with gr.Column(scale=1):
            preview = gr.HTML(
                value='<div class="article-preview"><p style="color:#aaa">Your linked text will appear here.</p></div>',
                label="Linked preview",
            )
            summary = gr.HTML(label="Links found")

    run_btn.click(
        fn=auto_link,
        inputs=[text_input, nil_slider],
        outputs=[preview, summary],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # listen on all interfaces, not just localhost
        server_port=8093,         # as requested by professor
        share=True,
    )
