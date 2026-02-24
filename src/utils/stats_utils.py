# src/utils/stats_utils.py
"""Utilities for writing scraping and pipeline statistics to disk."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATS_DIR = PROJECT_ROOT / "data" / "stats"


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
