# scripts/01_Data_processing/00_parse_wikipedia_dump.py
"""
00_parse_wikipedia_dump.py

Build ground truth from a Wikipedia XML dump (bz2):
- Reads configs/data_processing/wikipedia_ground_truth.yaml
- Streams through the XML dump with mwparserfromhell
- Outputs paragraphs, sentences, articles (JSONL/CSV) to data/processed/wikipedia/

Usage:
    python scripts/01_Data_processing/00_parse_wikipedia_dump.py
    python scripts/01_Data_processing/00_parse_wikipedia_dump.py --max-articles 500
    python scripts/01_Data_processing/00_parse_wikipedia_dump.py --dump path/to/dump.bz2
"""

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
for p in (_PROJECT_ROOT, _SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from src.data_processing.wikipedia_ground_truth import (
    WikipediaGroundTruthConfig,
    run_wikipedia_ground_truth_build,
)
from src.utils.logging_utils import create_logger, get_log_dir
from src.utils.stats_utils import update_dataset_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Wikipedia ground truth from XML dump")
    p.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "data_processing" / "wikipedia_ground_truth.yaml",
        help="Path to wikipedia_ground_truth.yaml",
    )
    p.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="Override dump_path from config",
    )
    p.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Override max_articles from config (0 = no limit)",
    )
    p.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Override domain name (default: wikipedia)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    config = WikipediaGroundTruthConfig.load(args.config)

    if args.dump is not None:
        config.dump_path = args.dump
    if args.max_articles is not None:
        config.max_articles = args.max_articles
    if args.domain is not None:
        config.domain = args.domain

    domain = config.domain

    log_dir = get_log_dir("ground_truth", domain=domain)
    logger, log_path = create_logger(log_dir=log_dir, script_name="00_parse_wikipedia_dump")

    logger.info("=== Starting 00_parse_wikipedia_dump ===")
    logger.info("Domain:       %s", domain)
    logger.info("Dump path:    %s", config.dump_path)
    logger.info("Max articles: %s", config.max_articles or "unlimited")
    logger.info("Output dir:   %s", config.processed_dir / domain)

    if not config.dump_path.exists():
        logger.error("Dump file not found: %s", config.dump_path)
        sys.exit(1)

    start = time.time()

    try:
        result_paths = run_wikipedia_ground_truth_build(config)
    except Exception as e:
        logger.error("Unhandled error: %s", e, exc_info=True)
        raise

    elapsed = time.time() - start
    logger.info("Finished in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    for k, p in result_paths.items():
        if p.exists():
            logger.info("  %s → %s", k, p)
    logger.info("Log file: %s", log_path)

    # Compute and persist dataset stats (same as Fandom pipeline)
    import json as _json

    def _count_jsonl(path: Path) -> int:
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
        "num_articles":     _count_jsonl(art_pg_path),
        "num_paragraphs":   _count_jsonl(para_path),
        "num_sentences":    _count_jsonl(sent_path),
        "num_links":        num_links,
        "link_type_counts": link_type_counts,
    }
    update_dataset_stats(domain, dataset_stats)

    logger.info(
        "Stats: articles=%d paragraphs=%d sentences=%d links=%d (internal=%d)",
        dataset_stats["num_articles"],
        dataset_stats["num_paragraphs"],
        dataset_stats["num_sentences"],
        dataset_stats["num_links"],
        dataset_stats["link_type_counts"].get("internal", 0),
    )

    print(f"\nDone in {elapsed:.0f}s")
    print(f"Articles processed: {dataset_stats['num_articles']}")
    print(f"Paragraphs:         {dataset_stats['num_paragraphs']}")
    print(f"Sentences:          {dataset_stats['num_sentences']}")
    print(f"Internal links:     {dataset_stats['link_type_counts'].get('internal', 0)}")
    print(f"Output dir:         {config.processed_dir / domain}")
    print(f"Log:                {log_path}")


if __name__ == "__main__":
    main()
