# tests/test_stats_utils.py
"""Tests for utils.stats_utils."""

import json
from pathlib import Path

import pytest

from src.utils.stats_utils import STATS_DIR, PROJECT_ROOT, update_scraping_stats


def test_project_root_is_directory():
    assert PROJECT_ROOT.is_dir()
    assert (PROJECT_ROOT / "src").is_dir()


def test_update_scraping_stats_creates_file(tmp_path, monkeypatch):
    """update_scraping_stats writes stats to data/stats/<domain>.json."""
    monkeypatch.setattr("src.utils.stats_utils.STATS_DIR", tmp_path)

    stats = {
        "total_urls": 10,
        "downloaded": 8,
        "skipped": 1,
        "failed": 1,
        "html_bytes": {"total": 1000, "avg": 125, "max": 200},
        "text_bytes": {"total": 500, "avg": 62.5, "max": 100},
    }
    update_scraping_stats("test_domain", stats)

    out = tmp_path / "test_domain.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["domain"] == "test_domain"
    assert data["scraping"] == stats
    assert "updated_at" in data


def test_update_scraping_stats_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.stats_utils.STATS_DIR", tmp_path)

    update_scraping_stats("domain_a", {"total_urls": 5})
    update_scraping_stats("domain_a", {"total_urls": 10})

    data = json.loads((tmp_path / "domain_a.json").read_text())
    assert data["scraping"]["total_urls"] == 10
    assert "previous_runs" in data
    assert len(data["previous_runs"]) == 1
    assert data["previous_runs"][0]["scraping"]["total_urls"] == 5
