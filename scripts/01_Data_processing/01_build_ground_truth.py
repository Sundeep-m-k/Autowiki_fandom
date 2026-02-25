# scripts/01_Data_processing/01_build_ground_truth.py
"""
01_build_ground_truth.py

Build ground truth from scraped HTML:
- Reads configs/data_processing/ground_truth.yaml
- Parses HTML in data/raw/<domain>/
- Outputs paragraphs, sentences, articles (JSONL/CSV) to data/processed/<domain>/
"""

import sys
import time
from pathlib import Path

# Add project root and src so "src" and "utils" packages are findable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
for p in (_PROJECT_ROOT, _SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from src.data_processing.ground_truth import GroundTruthConfig, run_ground_truth_build
from src.utils.logging_utils import create_logger, get_log_dir
from src.utils.stats_utils import update_dataset_stats


def main() -> None:
    config_path = _PROJECT_ROOT / "configs" / "data_processing" / "ground_truth.yaml"
    domain = None
    if len(sys.argv) >= 2:
        domain = sys.argv[1].strip()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    config = GroundTruthConfig.load(config_path)
    if domain:
        config.domain = domain
    domain = config.domain
    if not domain:
        print("Domain required (in config or as first arg)")
        sys.exit(1)

    log_dir = get_log_dir("ground_truth", domain=domain)
    logger, log_path = create_logger(log_dir=log_dir, script_name="01_build_ground_truth")

    start = time.time()
    logger.info("=== Starting 01_build_ground_truth for domain=%s ===", domain)
    logger.info("Building ground truth for domain=%s", domain)
    html_dir = config.raw_dir / domain
    out_dir = config.processed_dir / domain
    logger.info("HTML dir: %s", html_dir)
    logger.info(
        "Paragraph master: %s",
        out_dir / f"paragraphs_{domain}.jsonl",
    )
    logger.info(
        "Sentence master:  %s",
        out_dir / f"sentences_{domain}.jsonl",
    )
    logger.info(
        "Articles page granularity: %s",
        out_dir / f"articles_page_granularity_{domain}.jsonl",
    )
    logger.info("Articles index: %s", out_dir / f"articles_{domain}.jsonl")

    try:
        result_paths = run_ground_truth_build(config)
    except Exception as e:
        logger.error("Unhandled error in 01_build_ground_truth: %s", e, exc_info=True)
        raise

    elapsed = time.time() - start
    logger.info("Finished 01_build_ground_truth in %.2f seconds", elapsed)
    for k, p in result_paths.items():
        if p.exists():
            logger.info("%s: %s", k, p)
    logger.info("Log file: %s", log_path)

    # ── Compute and persist dataset stats ─────────────────────────────────────
    import json as _json

    def _count_jsonl(path) -> int:
        if not path.exists():
            return 0
        return sum(1 for line in open(path, encoding="utf-8") if line.strip())

    art_pg_path = result_paths.get(
        "articles_page_granularity_jsonl",
        config.processed_dir / domain / f"articles_page_granularity_{domain}.jsonl",
    )
    para_path = result_paths.get(
        "paragraphs_jsonl",
        config.processed_dir / domain / f"paragraphs_{domain}.jsonl",
    )
    sent_path = result_paths.get(
        "sentences_jsonl",
        config.processed_dir / domain / f"sentences_{domain}.jsonl",
    )

    link_type_counts: dict[str, int] = {}
    num_links = 0
    if art_pg_path.exists():
        with open(art_pg_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = _json.loads(line)
                for lk in rec.get("links", []):
                    lt = lk.get("link_type", "other")
                    link_type_counts[lt] = link_type_counts.get(lt, 0) + 1
                    num_links += 1

    dataset_stats = {
        "num_articles":    _count_jsonl(art_pg_path),
        "num_paragraphs":  _count_jsonl(para_path),
        "num_sentences":   _count_jsonl(sent_path),
        "num_links":       num_links,
        "link_type_counts": link_type_counts,
    }
    update_dataset_stats(domain, dataset_stats)
    logger.info(
        "dataset_stats: articles=%d paragraphs=%d sentences=%d links=%d (internal=%d)",
        dataset_stats["num_articles"],
        dataset_stats["num_paragraphs"],
        dataset_stats["num_sentences"],
        dataset_stats["num_links"],
        dataset_stats["link_type_counts"].get("internal", 0),
    )

    print(f"Paragraph master file: {result_paths.get('paragraphs_jsonl', 'N/A')}")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
