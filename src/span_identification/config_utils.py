"""Config loading and path resolution for span identification."""
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
    """Load YAML config file.

    If the config contains a ``base:`` key pointing to another YAML file,
    that file is loaded first and the current config is deep-merged on top,
    so only the keys that differ need to be specified in the override config.
    The ``base:`` key is resolved relative to the config file's directory.
    """
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


def get_processed_path(config: dict, domain: str, granularity: str) -> Path:
    """Get path to processed JSONL for domain and granularity."""
    processed_dir = Path(config["processed_dir"])
    proc = config.get("processed", {})
    if granularity == "paragraph":
        fname = proc.get("paragraphs_file", "paragraphs_{domain}.jsonl").format(domain=domain)
    elif granularity == "sentence":
        fname = proc.get("sentences_file", "sentences_{domain}.jsonl").format(domain=domain)
    elif granularity == "article":
        fname = proc.get("articles_file", "articles_page_granularity_{domain}.jsonl").format(domain=domain)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")
    return processed_dir / domain / fname


def get_splits_root(config: dict, domain: str) -> Path:
    """Get path to span_id splits for domain: data/span_id/<domain>/splits/"""
    span_id_base = Path(config.get("span_id_dir", "data/span_id"))
    return span_id_base / domain / config.get("splits_subdir", "splits")


def get_split_path(config: dict, domain: str, granularity: str, split: str) -> Path:
    """Get path for a specific split file: train_paragraph.jsonl, val_sentence.jsonl, etc."""
    root = get_splits_root(config, domain)
    return root / f"{split}_{granularity}.jsonl"


def get_token_data_dir(
    config: dict,
    domain: str,
    granularity: str,
    model_name: str,
    label_scheme: str = "BILOU",
) -> Path:
    """Token-level dataset dir: span_id/<domain>/token_data/<granularity>_<model>_<scheme>/

    The label scheme is included in the directory name so that BIO and BILOU
    datasets can coexist on disk without overwriting each other.
    """
    span_id_base = Path(config.get("span_id_dir", "data/span_id"))
    model_suffix = model_name.replace("/", "_")
    return span_id_base / domain / "token_data" / f"{granularity}_{model_suffix}_{label_scheme}"


def get_token_data_path(
    config: dict,
    domain: str,
    granularity: str,
    model_name: str,
    split: str = "train",
    label_scheme: str = "BILOU",
) -> Path:
    """Path to token-level JSONL: train.jsonl, dev.jsonl, test.jsonl."""
    base = get_token_data_dir(config, domain, granularity, model_name, label_scheme)
    return base / f"{split}.jsonl"


def get_split_meta_path(config: dict, domain: str) -> Path:
    """Get path to split_meta.json for domain: data/span_id/<domain>/split_meta.json"""
    span_id_base = Path(config.get("span_id_dir", "data/span_id"))
    return span_id_base / domain / "split_meta.json"


def get_checkpoint_dir(config: dict, run_id: str, domain: str) -> Path:
    """Get checkpoint base dir for a run: span_id/<domain>/checkpoints/<run_id>/"""
    span_id_base = Path(config.get("span_id_dir", "data/span_id"))
    return span_id_base / domain / "checkpoints" / run_id


def get_research_csv_path(config: dict) -> Path:
    """Get path to research experiments CSV."""
    return Path(config["research_dir"]) / config.get("research_csv", "span_id_experiments.csv")


def get_span_id_log_dir(config: dict, domain: str) -> Path:
    """Get log directory for span_id: data/logs/<domain>/span_id/"""
    data_dir = Path(config.get("data_dir", "data"))
    return data_dir / "logs" / domain / "span_id"
