# scripts/01_Data_processing/00_scrape_fandom.py
"""
00_scrape_fandom.py

High-level wrapper around scraping pipeline:
- Reads configs/data_processing/scraping.yaml
- Generates URL list
- Downloads HTML + plain text
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

from src.data_scraping.scrape_pipeline import load_scraping_config, run_full_scrape
from src.utils.logging_utils import create_logger, get_log_dir

def main() -> None:
    config = load_scraping_config(_PROJECT_ROOT / "configs" / "data_processing" / "scraping.yaml")
    log_dir = get_log_dir("scraping", domain=config.domain)
    logger, log_path = create_logger(log_dir=log_dir, script_name="00_scrape_fandom")

    start = time.time()
    logger.info("=== Starting 00_scrape_fandom ===")

    try:
        url_list_path = run_full_scrape()
    except Exception as e:
        logger.error(f"Unhandled error in 00_scrape_fandom: {e}", exc_info=True)
        raise

    elapsed = time.time() - start
    logger.info(f"Finished 00_scrape_fandom in {elapsed:.2f} seconds")
    logger.info(f"URL list: {url_list_path}")
    logger.info(f"Log file: {log_path}")

    print(f"Saved URL list to: {url_list_path}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()