"""Config loading and path resolution for article retrieval pipeline.

All artifact paths are derived from the config — nothing is hardcoded in scripts.
Naming convention encodes all experiment dimensions so that artifacts from
different configurations coexist on disk without overwriting each other.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config with base-merging support (same pattern as span_id)."""
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


# ── Root directories ───────────────────────────────────────────────────────────

def get_retrieval_root(config: dict, domain: str) -> Path:
    """Root dir for all article retrieval artifacts: data/article_retrieval/<domain>/"""
    base = Path(config.get("article_retrieval_dir", "data/article_retrieval"))
    return base / domain


def get_log_dir(config: dict, domain: str) -> Path:
    """Log directory: data/logs/<domain>/article_retrieval/"""
    data_dir = Path(config.get("data_dir", "data"))
    return data_dir / "logs" / domain / "article_retrieval"


# ── Article index paths ────────────────────────────────────────────────────────

def get_article_index_dir(config: dict, domain: str) -> Path:
    """Directory for all article index artifacts."""
    return get_retrieval_root(config, domain) / "article_index"


def _index_suffix(config: dict) -> str:
    """Encode corpus_representation and corpus_granularity into artifact names."""
    ai = config.get("article_index", {})
    repr_ = ai.get("corpus_representation", "title_full")
    gran  = ai.get("corpus_granularity", "article")
    return f"{repr_}_{gran}"


def get_articles_jsonl_path(config: dict, domain: str) -> Path:
    """Clean article records: article_index/articles_<repr>_<gran>.jsonl"""
    suffix = _index_suffix(config)
    return get_article_index_dir(config, domain) / f"articles_{suffix}.jsonl"


def get_bm25_index_path(config: dict, domain: str) -> Path:
    """Serialised BM25 index."""
    q = config.get("queries", {})
    preproc = q.get("anchor_preprocessing", "raw")
    suffix  = _index_suffix(config)
    return get_article_index_dir(config, domain) / f"bm25_{suffix}_{preproc}.pkl"


def get_tfidf_index_path(config: dict, domain: str) -> Path:
    """Serialised TF-IDF vectorizer."""
    q = config.get("queries", {})
    preproc = q.get("anchor_preprocessing", "raw")
    suffix  = _index_suffix(config)
    return get_article_index_dir(config, domain) / f"tfidf_{suffix}_{preproc}.pkl"


def get_tfidf_matrix_path(config: dict, domain: str) -> Path:
    """Sparse TF-IDF document matrix (.npz)."""
    q = config.get("queries", {})
    preproc = q.get("anchor_preprocessing", "raw")
    suffix  = _index_suffix(config)
    return get_article_index_dir(config, domain) / f"tfidf_matrix_{suffix}_{preproc}.npz"


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def get_embeddings_path(config: dict, domain: str, model_name: str) -> Path:
    """Article embeddings numpy array."""
    suffix = _index_suffix(config)
    slug   = _model_slug(model_name)
    return get_article_index_dir(config, domain) / f"embeddings_{slug}_{suffix}.npy"


def get_embeddings_ids_path(config: dict, domain: str, model_name: str) -> Path:
    """article_id → embedding row index mapping."""
    suffix = _index_suffix(config)
    slug   = _model_slug(model_name)
    return get_article_index_dir(config, domain) / f"embeddings_{slug}_{suffix}_ids.json"


def get_faiss_index_path(config: dict, domain: str, model_name: str) -> Path:
    """FAISS index file."""
    suffix     = _index_suffix(config)
    slug       = _model_slug(model_name)
    index_type = config.get("faiss_index_type", "flat")
    return get_article_index_dir(config, domain) / f"faiss_{slug}_{suffix}_{index_type}.index"


def get_faiss_meta_path(config: dict, domain: str, model_name: str) -> Path:
    """FAISS index metadata JSON."""
    suffix     = _index_suffix(config)
    slug       = _model_slug(model_name)
    index_type = config.get("faiss_index_type", "flat")
    return get_article_index_dir(config, domain) / f"faiss_{slug}_{suffix}_{index_type}_meta.json"


def get_index_meta_path(config: dict, domain: str) -> Path:
    """Overall index metadata (which indexes have been built)."""
    return get_article_index_dir(config, domain) / "index_meta.json"


# ── Query dataset paths ────────────────────────────────────────────────────────

def get_query_dir(config: dict, domain: str) -> Path:
    return get_retrieval_root(config, domain) / "queries"


def _query_suffix(config: dict) -> str:
    """Encode query context mode, preprocessing, sample size into artifact names."""
    q       = config.get("queries", {})
    ctx     = q.get("query_context_mode", "anchor_sentence")
    preproc = q.get("anchor_preprocessing", "raw")
    n       = q.get("n_sample") or "all"
    return f"{ctx}_{preproc}_n{n}"


def get_query_dataset_path(config: dict, domain: str) -> Path:
    """Query dataset JSONL with all 24 variations per query."""
    suffix = _query_suffix(config)
    return get_query_dir(config, domain) / f"query_dataset_{suffix}.jsonl"


def get_query_embeddings_path(
    config: dict, domain: str, model_name: str, version: int
) -> Path:
    """Query embeddings for one (model, query version) pair."""
    suffix = _query_suffix(config)
    slug   = _model_slug(model_name)
    return get_query_dir(config, domain) / f"query_embeddings_{slug}_{suffix}_v{version}.npy"


def get_query_embeddings_ids_path(
    config: dict, domain: str, model_name: str, version: int
) -> Path:
    suffix = _query_suffix(config)
    slug   = _model_slug(model_name)
    return get_query_dir(config, domain) / f"query_embeddings_{slug}_{suffix}_v{version}_ids.json"


# ── Retrieval result paths ─────────────────────────────────────────────────────

def get_retrieval_dir(config: dict, domain: str) -> Path:
    return get_retrieval_root(config, domain) / "retrieval"


def get_retrieval_path(
    config: dict,
    domain: str,
    retriever: str,
    version: int,
    top_k: int | None = None,
) -> Path:
    """Retrieval results JSONL for one (retriever, query version)."""
    top_k   = top_k or config.get("retrieval", {}).get("top_k", 100)
    isuffix = _index_suffix(config)
    qsuffix = _query_suffix(config)
    slug    = _model_slug(retriever)
    return (
        get_retrieval_dir(config, domain)
        / f"{slug}_{isuffix}_{qsuffix}_v{version}_top{top_k}.jsonl"
    )


# ── Reranking result paths ─────────────────────────────────────────────────────

def get_reranking_dir(config: dict, domain: str) -> Path:
    return get_retrieval_root(config, domain) / "reranking"


def get_reranking_path(
    config: dict,
    domain: str,
    retriever: str,
    reranker: str,
    version: int,
) -> Path:
    """Reranking results JSONL for one (retriever, reranker, query version)."""
    top_k_input = config.get("reranking", {}).get("top_k_input", 20)
    isuffix     = _index_suffix(config)
    qsuffix     = _query_suffix(config)
    ret_slug    = _model_slug(retriever)
    rer_slug    = _model_slug(reranker)
    return (
        get_reranking_dir(config, domain)
        / f"{ret_slug}_{rer_slug}_topk{top_k_input}_{isuffix}_{qsuffix}_v{version}.jsonl"
    )


# ── Metrics paths ──────────────────────────────────────────────────────────────

def get_metrics_dir(config: dict, domain: str) -> Path:
    return get_retrieval_root(config, domain) / "metrics"


def get_metrics_path(
    config: dict,
    domain: str,
    retriever: str,
    version: int,
    stage: str = "retrieval",
    reranker: str = "",
) -> Path:
    """Per-version metrics JSON."""
    isuffix  = _index_suffix(config)
    qsuffix  = _query_suffix(config)
    ret_slug = _model_slug(retriever)
    if stage == "reranking" and reranker:
        rer_slug = _model_slug(reranker)
        name = f"reranking_{ret_slug}_{rer_slug}_{isuffix}_{qsuffix}_v{version}.json"
    else:
        name = f"retrieval_{ret_slug}_{isuffix}_{qsuffix}_v{version}.json"
    return get_metrics_dir(config, domain) / name


def get_summary_path(config: dict, domain: str) -> Path:
    """Aggregated summary CSV across all versions and retrievers."""
    return get_metrics_dir(config, domain) / f"summary_{domain}.csv"


# ── Error analysis paths ───────────────────────────────────────────────────────

def get_error_analysis_dir(config: dict, domain: str) -> Path:
    """Root dir for error analysis outputs: data/article_retrieval/<domain>/error_analysis/"""
    return get_retrieval_root(config, domain) / "error_analysis"


def get_error_analysis_path(
    config: dict,
    domain: str,
    retriever: str,
    version: int,
    stage: str = "retrieval",
    reranker: str = "",
) -> Path:
    """Directory for one (stage, retriever, [reranker], version) error analysis run."""
    ret_slug = _model_slug(retriever)
    if stage == "reranking" and reranker:
        rer_slug = _model_slug(reranker)
        name = f"reranking_{ret_slug}_{rer_slug}_v{version}"
    else:
        name = f"retrieval_{ret_slug}_v{version}"
    return get_error_analysis_dir(config, domain) / name


# ── Ablation config expansion ──────────────────────────────────────────────────

def get_ablation_configs(config: dict) -> list[dict]:
    """
    Expand a config that uses plural ablation keys into a list of single-valued
    configs — one per Cartesian-product combination of all ablation dimensions.

    Plural keys (new style)        → Singular keys injected into each config
    ─────────────────────────────────────────────────────────────────────────
    article_index.corpus_representations  → article_index.corpus_representation
    article_index.corpus_granularities    → article_index.corpus_granularity
    queries.query_context_modes           → queries.query_context_mode
    queries.anchor_preprocessings         → queries.anchor_preprocessing
    queries.n_samples                     → queries.n_sample
    reranking.models                      → reranking.model
    reranking.top_k_inputs                → reranking.top_k_input

    If only the singular key is present (standalone script usage), that value is
    wrapped in a one-element list so the loop still works.
    """
    import copy
    import itertools

    ai  = config.get("article_index", {})
    q   = config.get("queries", {})
    rer = config.get("reranking", {})

    def _as_list(d: dict, plural_key: str, singular_key: str) -> list:
        if plural_key in d:
            return list(d[plural_key])
        return [d.get(singular_key, None)]

    corpus_reprs  = _as_list(ai,  "corpus_representations",  "corpus_representation")
    corpus_grans  = _as_list(ai,  "corpus_granularities",    "corpus_granularity")
    ctx_modes     = _as_list(q,   "query_context_modes",     "query_context_mode")
    preprocs      = _as_list(q,   "anchor_preprocessings",   "anchor_preprocessing")
    n_samples     = _as_list(q,   "n_samples",               "n_sample")
    rer_models    = _as_list(rer, "models",                  "model")
    rer_topks     = _as_list(rer, "top_k_inputs",            "top_k_input")

    combos = list(itertools.product(
        corpus_reprs, corpus_grans, ctx_modes, preprocs, n_samples, rer_models, rer_topks,
    ))

    configs = []
    for corpus_repr, corpus_gran, ctx_mode, preproc, n_sample, rer_model, rer_topk in combos:
        c = copy.deepcopy(config)
        c.setdefault("article_index", {})
        c["article_index"]["corpus_representation"] = corpus_repr
        c["article_index"]["corpus_granularity"]    = corpus_gran
        c.setdefault("queries", {})
        c["queries"]["query_context_mode"]   = ctx_mode
        c["queries"]["anchor_preprocessing"] = preproc
        c["queries"]["n_sample"]             = n_sample
        c.setdefault("reranking", {})
        c["reranking"]["model"]       = rer_model
        c["reranking"]["top_k_input"] = rer_topk
        configs.append(c)

    return configs


def resolve_config(config: dict) -> dict:
    """
    For standalone script usage: if the config still has plural ablation keys,
    resolve to the first combination so that path helpers work correctly.
    Has no effect if all singular keys are already present.
    """
    combos = get_ablation_configs(config)
    return combos[0] if combos else config


def ablation_label(config: dict) -> str:
    """Short human-readable label for one ablation config (for logging)."""
    ai  = config.get("article_index", {})
    q   = config.get("queries", {})
    rer = config.get("reranking", {})
    return (
        f"repr={ai.get('corpus_representation','?')} "
        f"gran={ai.get('corpus_granularity','?')} "
        f"ctx={q.get('query_context_mode','?')} "
        f"preproc={q.get('anchor_preprocessing','?')} "
        f"n={q.get('n_sample','all')} "
        f"rer={rer.get('model','?').split('/')[-1]} "
        f"topk={rer.get('top_k_input','?')}"
    )


# ── Research CSV ───────────────────────────────────────────────────────────────

def get_research_csv_path(config: dict, domain: str = "") -> Path:
    """Research experiments CSV, scoped to a domain subdirectory if provided."""
    base = Path(config["research_dir"])
    csv_name = config.get("research_csv", "article_retrieval_experiments.csv")
    if domain:
        return base / domain / csv_name
    return base / csv_name


# ── Re-ranker training paths ───────────────────────────────────────────────────

def get_reranker_training_data_path(config: dict, domain: str) -> Path:
    """JSONL of (query, positive, negatives) training examples for one domain."""
    rt = config.get("reranker_training", {})
    retriever_slug = _model_slug(rt.get("source_retriever", "unknown"))
    version        = rt.get("source_version", 6)
    n_neg          = rt.get("n_hard_negatives", 5)
    isuffix        = _index_suffix(config)
    qsuffix        = _query_suffix(config)
    name = f"reranker_train_{retriever_slug}_{isuffix}_{qsuffix}_v{version}_neg{n_neg}.jsonl"
    return get_retrieval_root(config, domain) / "reranker_training" / name


def get_reranker_checkpoint_dir(config: dict) -> Path:
    """Directory where the fine-tuned cross-encoder is saved."""
    rt = config.get("reranker_training", {})
    return Path(rt.get("output_dir", "data/article_retrieval/checkpoints/reranker_finetuned"))


# ── Task 1 split helpers (reuse existing splits) ───────────────────────────────

def get_task1_split_article_ids(config: dict, domain: str, split: str) -> set[int]:
    """
    Load article IDs from a Task 1 split file (train/val/test).
    Used to filter queries to only source articles in the requested split.
    Returns an empty set if the split file does not exist (no filtering applied).
    """
    import json
    span_id_dir  = Path(config.get("span_id_dir", "data/span_id"))
    splits_sub   = config.get("splits_subdir", "splits")
    # Task 1 splits are at article granularity
    split_path   = span_id_dir / domain / splits_sub / f"{split}_article.jsonl"
    if not split_path.exists():
        return set()
    ids: set[int] = set()
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            aid = rec.get("article_id")
            if aid is not None:
                ids.add(int(aid))
    return ids


# ── Processed data helpers ─────────────────────────────────────────────────────

def get_articles_page_granularity_path(config: dict, domain: str) -> Path:
    """Path to the full-article processed JSONL (source of corpus + links)."""
    processed_dir = Path(config.get("processed_dir", "data/processed"))
    return processed_dir / domain / f"articles_page_granularity_{domain}.jsonl"
