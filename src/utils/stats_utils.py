# src/utils/stats_utils.py
"""Utilities for writing scraping and pipeline statistics to disk."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATS_DIR = PROJECT_ROOT / "data" / "stats"

log = logging.getLogger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_stats(domain: str) -> Dict[str, Any]:
    path = STATS_DIR / f"{domain}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"domain": domain}


def _write_stats(domain: str, data: Dict[str, Any]) -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATS_DIR / f"{domain}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("[stats] updated %s", path)


# ── Scraping ──────────────────────────────────────────────────────────────────

def update_scraping_stats(domain: str, stats: Dict[str, Any]) -> None:
    """
    Write scraping statistics to data/stats/<domain>.json.
    Merges with existing stats (if present) and adds a timestamp.
    """
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = STATS_DIR / f"{domain}.json"

    payload: Dict[str, Any] = {
        "domain": domain,
        "updated_at": datetime.now().isoformat(),
        "scraping": stats,
    }

    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            payload["previous_runs"] = existing.get("previous_runs", [])
            if "updated_at" in existing:
                payload["previous_runs"].append(
                    {"updated_at": existing["updated_at"], "scraping": existing.get("scraping")}
                )
        except (json.JSONDecodeError, OSError):
            pass

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ── Ground Truth / Dataset Stats ─────────────────────────────────────────────

def update_dataset_stats(domain: str, stats: dict[str, Any]) -> None:
    """
    Write ground-truth dataset statistics to data/stats/<domain>.json
    under the key "dataset_stats".

    Automatically enriches the provided stats dict with:
      - avg_article_length_chars
      - avg_links_per_article
      - num_articles_with_no_links
      - split_sizes  (train/val/test per granularity, if splits exist)

    Minimum expected keys in stats dict:
        num_articles, num_paragraphs, num_sentences,
        num_links, link_type_counts
    """
    # ── Enrich from the articles_page_granularity file ────────────────────────
    art_pg_path = (
        PROJECT_ROOT / "data" / "processed" / domain
        / f"articles_page_granularity_{domain}.jsonl"
    )
    if art_pg_path.exists():
        text_lengths: list[int] = []
        links_per_article: list[int] = []
        no_links = 0
        try:
            with open(art_pg_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    text = rec.get("article_plain_text", "")
                    text_lengths.append(len(text))
                    n_internal = sum(
                        1 for lk in rec.get("links", [])
                        if lk.get("link_type") == "internal"
                    )
                    links_per_article.append(n_internal)
                    if n_internal == 0:
                        no_links += 1
            if text_lengths:
                stats["avg_article_length_chars"] = round(
                    sum(text_lengths) / len(text_lengths)
                )
                stats["max_article_length_chars"] = max(text_lengths)
            if links_per_article:
                stats["avg_internal_links_per_article"] = round(
                    sum(links_per_article) / len(links_per_article), 2
                )
            stats["num_articles_with_no_internal_links"] = no_links
        except Exception as e:
            log.warning("[stats] could not compute article-level stats: %s", e)

    # ── Read split sizes if they exist ────────────────────────────────────────
    splits_dir = PROJECT_ROOT / "data" / "span_id" / domain / "splits"
    split_sizes: dict[str, Any] = {}
    for granularity in ("sentence", "paragraph", "article"):
        gran_splits: dict[str, int] = {}
        for split in ("train", "val", "test"):
            p = splits_dir / f"{split}_{granularity}.jsonl"
            if p.exists():
                try:
                    gran_splits[split] = sum(
                        1 for ln in open(p, encoding="utf-8") if ln.strip()
                    )
                except OSError:
                    pass
        if gran_splits:
            split_sizes[granularity] = gran_splits
    if split_sizes:
        stats["split_sizes"] = split_sizes

    existing = _read_stats(domain)
    existing["dataset_stats"] = {
        **stats,
        "updated_at": datetime.now().isoformat(),
    }
    _write_stats(domain, existing)
    log.info(
        "[stats] dataset_stats updated for domain=%s | "
        "articles=%d paragraphs=%d sentences=%d internal_links=%d",
        domain,
        stats.get("num_articles", 0),
        stats.get("num_paragraphs", 0),
        stats.get("num_sentences", 0),
        stats.get("link_type_counts", {}).get("internal", 0),
    )


# ── Task 1: Span Identification ───────────────────────────────────────────────

def update_span_id_stats(domain: str, csv_path: Path) -> None:
    """
    Read the span_id research CSV and write the best result per model
    (by span_f1, model-type rows only) into data/stats/<domain>.json
    under the key "span_id".

    Also records the single overall best model across all configs.
    """
    try:
        import pandas as pd
    except ImportError:
        log.warning("[stats] pandas not available — skipping span_id stats update")
        return

    if not csv_path.exists():
        log.warning("[stats] span_id CSV not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    df = df[(df["domain"] == domain) & (df["experiment_type"] == "model")]
    if df.empty:
        log.info("[stats] no span_id model rows for domain=%s", domain)
        return

    for col in ["span_f1", "char_f1", "exact_match_pct",
                "span_precision", "span_recall"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Best per model (highest span_f1 across granularity / label_scheme / seed)
    best_by_model: Dict[str, Any] = {}
    for model, grp in df.groupby("model"):
        idx = grp["span_f1"].idxmax()
        row = grp.loc[idx]
        best_by_model[str(model)] = {
            "label_scheme":    str(row.get("label_scheme", "")),
            "granularity":     str(row.get("granularity", "")),
            "seed":            int(row.get("seed", -1)),
            "span_f1":         round(float(row.get("span_f1", 0)), 4),
            "span_precision":  round(float(row.get("span_precision", 0)), 4),
            "span_recall":     round(float(row.get("span_recall", 0)), 4),
            "char_f1":         round(float(row.get("char_f1", 0)), 4),
            "exact_match_pct": round(float(row.get("exact_match_pct") or 0), 4),
            "updated_at":      datetime.now().isoformat(),
        }

    # Overall best
    best_idx = df["span_f1"].idxmax()
    best_row = df.loc[best_idx]
    overall_best = {
        "model":           str(best_row.get("model", "")),
        "label_scheme":    str(best_row.get("label_scheme", "")),
        "granularity":     str(best_row.get("granularity", "")),
        "span_f1":         round(float(best_row.get("span_f1", 0)), 4),
        "char_f1":         round(float(best_row.get("char_f1", 0)), 4),
        "exact_match_pct": round(float(best_row.get("exact_match_pct") or 0), 4),
        "updated_at":      datetime.now().isoformat(),
    }

    existing = _read_stats(domain)
    existing["span_id"] = {
        "best_by_model": best_by_model,
        "overall_best":  overall_best,
    }
    _write_stats(domain, existing)
    log.info(
        "[stats] span_id updated for domain=%s | best model=%s span_f1=%.4f",
        domain, overall_best["model"], overall_best["span_f1"],
    )


# ── Task 2: Article Retrieval ─────────────────────────────────────────────────

def update_article_retrieval_stats(domain: str, csv_path: Path) -> None:
    """
    Read the article_retrieval research CSV and write the best retrieval
    and best reranking result (by MRR) into data/stats/<domain>.json
    under the key "article_retrieval".
    """
    try:
        import pandas as pd
    except ImportError:
        log.warning("[stats] pandas not available — skipping article_retrieval stats update")
        return

    if not csv_path.exists():
        log.warning("[stats] article_retrieval CSV not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    df = df[df["domain"] == domain]
    if df.empty:
        log.info("[stats] no article_retrieval rows for domain=%s", domain)
        return

    for col in ["mrr", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_100"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def _best_row(stage: str) -> Dict[str, Any] | None:
        sub = df[df["stage"] == stage]
        if sub.empty:
            return None
        idx = sub["mrr"].idxmax()
        r = sub.loc[idx]
        return {
            "retriever":        str(r.get("retriever", "")),
            "reranker":         str(r.get("reranker", "")) if stage == "reranking" else None,
            "version":          int(r.get("version", 0)),
            "recall_at_1":      round(float(r.get("recall_at_1", 0)), 4),
            "recall_at_5":      round(float(r.get("recall_at_5", 0)), 4),
            "recall_at_10":     round(float(r.get("recall_at_10", 0)), 4),
            "recall_at_100":    round(float(r.get("recall_at_100", 0)), 4),
            "mrr":              round(float(r.get("mrr", 0)), 4),
            "n_queries":        int(r.get("n_queries", 0)),
            "updated_at":       datetime.now().isoformat(),
        }

    best_retrieval = _best_row("retrieval")
    best_reranking = _best_row("reranking")

    section: Dict[str, Any] = {}
    if best_retrieval:
        section["best_retrieval"] = best_retrieval
    if best_reranking:
        section["best_reranking"] = best_reranking

    existing = _read_stats(domain)
    existing["article_retrieval"] = section
    _write_stats(domain, existing)
    log.info(
        "[stats] article_retrieval updated for domain=%s | best retrieval MRR=%.4f | "
        "best reranking MRR=%.4f",
        domain,
        best_retrieval["mrr"] if best_retrieval else 0,
        best_reranking["mrr"] if best_reranking else 0,
    )


# ── Task 3: Linking Pipeline ──────────────────────────────────────────────────

def update_linking_stats(domain: str, csv_path: Path) -> None:
    """
    Read the linking_experiments research CSV and write the best run
    (by linking_f1) into data/stats/<domain>.json under the key
    "linking_pipeline".
    """
    try:
        import pandas as pd
    except ImportError:
        log.warning("[stats] pandas not available — skipping linking stats update")
        return

    if not csv_path.exists():
        log.warning("[stats] linking CSV not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    df = df[df["domain"] == domain]
    if df.empty:
        log.info("[stats] no linking rows for domain=%s", domain)
        return

    for col in ["linking_f1", "linking_precision", "linking_recall",
                "span_f1", "entity_accuracy", "nil_rate", "coverage"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    idx = df["linking_f1"].idxmax()
    r = df.loc[idx]
    best = {
        "retriever":          str(r.get("retriever", "")),
        "reranker":           str(r.get("reranker", "")),
        "stage":              str(r.get("stage", "")),
        "query_version":      str(r.get("query_version", "")),
        "nil_threshold":      round(float(r.get("nil_threshold", 0)), 4),
        "linking_f1":         round(float(r.get("linking_f1", 0)), 4),
        "linking_precision":  round(float(r.get("linking_precision", 0)), 4),
        "linking_recall":     round(float(r.get("linking_recall", 0)), 4),
        "span_f1":            round(float(r.get("span_f1", 0)), 4),
        "entity_accuracy":    round(float(r.get("entity_accuracy", 0)), 4),
        "nil_rate":           round(float(r.get("nil_rate", 0)), 4),
        "coverage":           round(float(r.get("coverage", 0)), 4),
        "n_articles":         int(r.get("n_articles", 0)),
        "updated_at":         datetime.now().isoformat(),
    }

    existing = _read_stats(domain)
    existing["linking_pipeline"] = {"best": best}
    _write_stats(domain, existing)
    log.info(
        "[stats] linking_pipeline updated for domain=%s | best linking_f1=%.4f",
        domain, best["linking_f1"],
    )
