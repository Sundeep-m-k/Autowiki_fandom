"""Step 1: Render linked articles as HTML.

Reads linking_results.jsonl produced by step 00 and writes one HTML file
per article to data/linking/<domain>/html/<article_id>.html

Run:
  python scripts/04_Linking_pipeline/01_render_html.py
  python scripts/04_Linking_pipeline/01_render_html.py --domain beverlyhillscop
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import linking_pipeline.config_utils as cu
from linking_pipeline.logging_utils import setup_logger
from linking_pipeline.html_renderer import render_html, save_html

log = logging.getLogger("linking")


def run_for_domain(config: dict, domain: str, force: bool) -> None:
    log_dir = cu.get_log_dir(config, domain)
    setup_logger(log_dir, script_name="01_render_html")

    results_path = cu.get_linking_results_path(config, domain)
    if not results_path.exists():
        log.error("[01] linking results not found: %s — run step 00 first", results_path)
        return

    html_dir = cu.get_html_dir(config, domain)
    html_dir.mkdir(parents=True, exist_ok=True)

    html_cfg          = config.get("html", {})
    wrap_article      = html_cfg.get("wrap_article", True)
    link_class        = html_cfg.get("link_class", "wiki-link")
    overlap_strategy  = html_cfg.get("overlap_strategy", "longest")

    n_rendered = 0
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            article_id = rec["article_id"]
            html_path  = html_dir / f"{article_id}.html"

            if html_path.exists() and not force:
                continue

            # Only pass confirmed (linked=True) links to the renderer
            confirmed = [
                p for p in rec.get("predicted_links", [])
                if p.get("linked") and p.get("fandom_url")
            ]

            html_str = render_html(
                text=rec["text"],
                confirmed_links=confirmed,
                wrap_article=wrap_article,
                link_class=link_class,
                overlap_strategy=overlap_strategy,
            )
            save_html(html_str, html_path)
            n_rendered += 1

    log.info("[01] rendered %d HTML files → %s", n_rendered, html_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render linked articles as HTML.")
    parser.add_argument("--config", default="configs/linking.yaml")
    parser.add_argument("--domain", help="Single domain override.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config  = cu.load_config(ROOT / args.config)
    domains = [args.domain] if args.domain else config.get("domains", [])

    for domain in domains:
        run_for_domain(config, domain, force=args.force)


if __name__ == "__main__":
    main()
