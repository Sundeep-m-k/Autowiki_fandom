"""Config loading and path resolution for the linking pipeline (Task 3).

All artifact paths are derived from the config so nothing is hardcoded in scripts.
Task 2 artifact paths are reconstructed using the same naming logic as
article_retrieval/config_utils.py so the correct pre-computed files are loaded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config with base: inheritance (same pattern as Tasks 1 & 2)."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("base", None)
    if base_ref:
        base_path = path.parent / base_ref
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


# ── Linking artifact paths ─────────────────────────────────────────────────────

def get_linking_root(config: dict, domain: str) -> Path:
    return Path(config.get("linking_dir", "data/linking")) / domain


def get_linking_results_path(config: dict, domain: str) -> Path:
    """linking_results.jsonl — one record per article with predicted links."""
    t2 = config.get("task2", {})
    retriever_slug = _model_slug(t2.get("retriever", "unknown"))
    stage   = t2.get("stage", "reranking")
    version = t2.get("query_version", 6)
    nil_thr = config.get("nil_detection", {}).get("threshold", 0.0)
    name = f"linking_{retriever_slug}_{stage}_v{version}_nil{nil_thr}.jsonl"
    return get_linking_root(config, domain) / name


def get_html_dir(config: dict, domain: str) -> Path:
    return get_linking_root(config, domain) / "html"


def get_metrics_path(config: dict, domain: str) -> Path:
    t2 = config.get("task2", {})
    retriever_slug = _model_slug(t2.get("retriever", "unknown"))
    stage   = t2.get("stage", "reranking")
    version = t2.get("query_version", 6)
    nil_thr = config.get("nil_detection", {}).get("threshold", 0.0)
    name = f"metrics_{retriever_slug}_{stage}_v{version}_nil{nil_thr}.json"
    return get_linking_root(config, domain) / "metrics" / name


def get_log_dir(config: dict, domain: str) -> Path:
    return Path(config.get("log_dir", "data/logs")) / domain / "linking"


def get_research_csv_path(config: dict) -> Path:
    return Path(config["research_dir"]) / config.get("research_csv", "linking_experiments.csv")


# ── Task 1 path helpers ────────────────────────────────────────────────────────

def get_task1_split_path(config: dict, domain: str) -> Path:
    """Path to Task 1 test split JSONL (gold spans source)."""
    span_id_dir  = Path(config.get("span_id_dir", "data/span_id"))
    t1           = config.get("task1", {})
    split        = t1.get("split", "test")
    granularity  = t1.get("granularity", "article")
    return span_id_dir / domain / "splits" / f"{split}_{granularity}.jsonl"


# ── Task 2 path helpers (reconstruct filenames from config) ────────────────────

def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def _index_suffix(t2_cfg: dict) -> str:
    repr_ = t2_cfg.get("corpus_representation", "title_full")
    gran  = t2_cfg.get("corpus_granularity", "article")
    return f"{repr_}_{gran}"


def _query_suffix(t2_cfg: dict) -> str:
    ctx     = t2_cfg.get("query_context_mode", "anchor_sentence")
    preproc = t2_cfg.get("anchor_preprocessing", "raw")
    n       = t2_cfg.get("n_sample") or "all"
    return f"{ctx}_{preproc}_n{n}"


def get_task2_query_dataset_path(config: dict, domain: str) -> Path:
    """Path to the Task 2 query dataset JSONL for this domain."""
    ar_dir  = Path(config.get("article_retrieval_dir", "data/article_retrieval"))
    t2      = config.get("task2", {})
    suffix  = _query_suffix(t2)
    return ar_dir / domain / "queries" / f"query_dataset_{suffix}.jsonl"


def get_task2_retrieval_path(config: dict, domain: str) -> Path:
    """Path to pre-computed retrieval results JSONL."""
    ar_dir   = Path(config.get("article_retrieval_dir", "data/article_retrieval"))
    t2       = config.get("task2", {})
    retriever = t2.get("retriever", "")
    version   = t2.get("query_version", 6)
    top_k     = t2.get("top_k", 100)
    isuffix   = _index_suffix(t2)
    qsuffix   = _query_suffix(t2)
    slug      = _model_slug(retriever)
    fname     = f"{slug}_{isuffix}_{qsuffix}_v{version}_top{top_k}.jsonl"
    return ar_dir / domain / "retrieval" / fname


def get_task2_reranking_path(config: dict, domain: str) -> Path:
    """Path to pre-computed reranking results JSONL."""
    ar_dir    = Path(config.get("article_retrieval_dir", "data/article_retrieval"))
    t2        = config.get("task2", {})
    retriever = t2.get("retriever", "")
    reranker  = t2.get("reranker", "")
    version   = t2.get("query_version", 6)
    top_k_in  = t2.get("top_k_rerank", 20)
    isuffix   = _index_suffix(t2)
    qsuffix   = _query_suffix(t2)
    ret_slug  = _model_slug(retriever)
    rer_slug  = _model_slug(reranker)
    fname     = f"{ret_slug}_{rer_slug}_topk{top_k_in}_{isuffix}_{qsuffix}_v{version}.jsonl"
    return ar_dir / domain / "reranking" / fname


def get_task2_articles_jsonl_path(config: dict, domain: str) -> Path:
    """Path to persisted article records (for page_name lookup)."""
    ar_dir  = Path(config.get("article_retrieval_dir", "data/article_retrieval"))
    t2      = config.get("task2", {})
    repr_   = t2.get("corpus_representation", "title_full")
    gran    = t2.get("corpus_granularity", "article")
    return ar_dir / domain / "article_index" / f"articles_{repr_}_{gran}.jsonl"


def get_fandom_base_url(config: dict, domain: str) -> str:
    """Return the Fandom base wiki URL for this domain."""
    urls = config.get("fandom_base_urls", {})
    return urls.get(domain, f"https://{domain}.fandom.com/wiki")
