# tests/test_scrape_pipeline.py
"""Tests for data_scraping.scrape_pipeline."""

import tempfile
from pathlib import Path

import pytest
import yaml

# Import after conftest adds path
from src.data_scraping.scrape_pipeline import (
    ScrapingConfig,
    extract_article_id,
    extract_plain_text,
    filter_article,
    load_scraping_config,
    normalize_url,
    read_url_list,
    safe_filename_from_url,
)


class TestNormalizeUrl:
    def test_basic(self):
        assert normalize_url("https://example.com/wiki/Page") == "https://example.com/wiki/Page"

    def test_decode_spaces_to_underscores(self):
        # unquote replaces %20 with space, then replace(" ", "_") for consistent filenames
        assert normalize_url("https://a.com/wiki/Page%20Name") == "https://a.com/wiki/Page_Name"

    def test_strip_query(self):
        assert normalize_url("https://a.com/wiki/X?query=1") == "https://a.com/wiki/X"

    def test_trailing_slash_handling(self):
        assert normalize_url("https://a.com/wiki/") == "https://a.com/wiki/"


class TestFilterArticle:
    def test_article_url_passes(self):
        assert filter_article("https://x.fandom.com/wiki/Some_Article") is True
        assert filter_article("https://x.fandom.com/wiki/Main_Page") is True

    def test_non_wiki_fails(self):
        assert filter_article("https://x.fandom.com/other") is False

    def test_file_prefix_fails(self):
        assert filter_article("https://x.fandom.com/wiki/File:Image.png") is False

    def test_template_prefix_fails(self):
        assert filter_article("https://x.fandom.com/wiki/Template:Thing") is False

    def test_category_prefix_fails(self):
        assert filter_article("https://x.fandom.com/wiki/Category:Cat") is False

    def test_user_prefix_fails(self):
        assert filter_article("https://x.fandom.com/wiki/User:Admin") is False

    def test_special_prefix_fails(self):
        assert filter_article("https://x.fandom.com/wiki/Special:AllPages") is False


class TestSafeFilenameFromUrl:
    def test_basic(self):
        name = safe_filename_from_url("https://example.com/wiki/Article_Name")
        assert name.endswith(".html")
        assert "example_com" in name
        assert "Article_Name" in name or "Article" in name

    def test_max_length(self):
        long_path = "/wiki/" + "A" * 300
        name = safe_filename_from_url(f"https://x.com{long_path}")
        assert len(name) <= 185  # 180 + ".html"

    def test_special_chars_sanitized(self):
        name = safe_filename_from_url("https://x.com/wiki/Page@#$%")
        assert all(c.isalnum() or c in "._-" or c == "/" for c in name.split("/")[-1] or "")


class TestReadUrlList:
    def test_read_valid_urls(self, tmp_path):
        f = tmp_path / "urls.txt"
        f.write_text("https://a.com/wiki/1\nhttps://a.com/wiki/2\n")
        assert read_url_list(f) == ["https://a.com/wiki/1", "https://a.com/wiki/2"]

    def test_skips_empty_and_comments(self, tmp_path):
        f = tmp_path / "urls.txt"
        f.write_text("# comment\nhttps://a.com/wiki/1\n\nhttps://a.com/wiki/2\n")
        assert read_url_list(f) == ["https://a.com/wiki/1", "https://a.com/wiki/2"]

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "urls.txt"
        f.write_text("  https://a.com/wiki/1  \n")
        assert read_url_list(f) == ["https://a.com/wiki/1"]


class TestExtractPlainText:
    def test_extracts_mw_parser_output(self):
        html = '<div class="mw-parser-output"><p>Hello world</p></div>'
        assert "Hello world" in extract_plain_text(html)

    def test_strips_script_and_style(self):
        html = '<html><script>alert(1)</script><body>Text</body></html>'
        assert "alert" not in extract_plain_text(html)
        assert "Text" in extract_plain_text(html)

    def test_uses_body_when_no_mw_parser(self):
        html = "<html><body><p>Fallback text</p></body></html>"
        assert "Fallback text" in extract_plain_text(html)


class TestExtractArticleId:
    def test_extracts_from_json_pattern(self):
        html = '{"wgArticleId": 12345}'
        assert extract_article_id(html) == "12345"

    def test_extracts_from_js_pattern(self):
        html = "var wgArticleId = 99999;"
        assert extract_article_id(html) == "99999"

    def test_extracts_from_meta_mw_pageid(self):
        html = '<meta property="mw:pageId" content="42" />'
        assert extract_article_id(html) == "42"

    def test_returns_none_when_missing(self):
        assert extract_article_id("<html><body>no id</body></html>") is None


class TestLoadScrapingConfig:
    def test_load_valid_config(self, tmp_path):
        cfg_path = tmp_path / "scraping.yaml"
        cfg_path.write_text(yaml.dump({"base_url": "https://test.fandom.com/", "start_url": "https://test.fandom.com/wiki/Special:AllPages"}))
        cfg = load_scraping_config(cfg_path)
        assert cfg.base_url == "https://test.fandom.com"
        assert cfg.start_url == "https://test.fandom.com/wiki/Special:AllPages"
        assert cfg.domain == "test"

    def test_missing_base_url_raises(self, tmp_path):
        cfg_path = tmp_path / "scraping.yaml"
        cfg_path.write_text("start_url: https://x.com\n")
        with pytest.raises(ValueError, match="base_url"):
            load_scraping_config(cfg_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_scraping_config(tmp_path / "nonexistent.yaml")

    def test_invalid_base_url_scheme_raises(self, tmp_path):
        cfg_path = tmp_path / "scraping.yaml"
        cfg_path.write_text(yaml.dump({"base_url": "ftp://invalid.com/"}))
        with pytest.raises(ValueError, match="HTTP"):
            load_scraping_config(cfg_path)

    def test_invalid_base_url_host_raises(self, tmp_path):
        cfg_path = tmp_path / "scraping.yaml"
        cfg_path.write_text(yaml.dump({"base_url": "https://x"}))
        with pytest.raises(ValueError, match="host"):
            load_scraping_config(cfg_path)

    def test_invalid_start_url_raises(self, tmp_path):
        cfg_path = tmp_path / "scraping.yaml"
        cfg_path.write_text(yaml.dump({"base_url": "https://x.fandom.com/", "start_url": "not-a-url"}))
        with pytest.raises(ValueError, match="start_url"):
            load_scraping_config(cfg_path)

    def test_invalid_category_urls_type_raises(self, tmp_path):
        cfg_path = tmp_path / "scraping.yaml"
        cfg_path.write_text(yaml.dump({"base_url": "https://x.fandom.com/", "category_urls": "string"}))
        with pytest.raises(ValueError, match="list"):
            load_scraping_config(cfg_path)
