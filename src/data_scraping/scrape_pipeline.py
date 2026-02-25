# src/data_scraping/scrape_pipeline.py
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit, urlunsplit, unquote, urlparse

import requests
from bs4 import BeautifulSoup
import yaml

from utils.stats_utils import update_scraping_stats

logger = logging.getLogger()
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------- CONFIG ----------------------

# NOTE:
# - Removed "br" from Accept-Encoding to avoid Brotli decode issues if brotli isn't installed.
# - Kept browser-ish headers to reduce 403s.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

DEFAULT_TIMEOUT = 30


@dataclass
class ScrapingConfig:
    base_url: str
    start_url: Optional[str]
    category_urls: List[str]
    delay_seconds: float = 2.0
    use_api_fallback: bool = True

    @property
    def domain(self) -> str:
        return urlparse(self.base_url).netloc.split(".")[0]


def load_scraping_config(config_path: Optional[Path] = None) -> ScrapingConfig:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "data_processing" / "scraping.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

    cfg = cfg or {}

    if "base_url" not in cfg:
        raise ValueError("'base_url' must be defined in configs/data_processing/scraping.yaml")

    base_url = str(cfg["base_url"]).strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError(f"base_url must be a valid HTTP(S) URL, got: {base_url!r}")
    parsed = urlparse(base_url)
    if not parsed.netloc or "." not in parsed.netloc:
        raise ValueError(
            f"base_url must have a valid host (e.g. example.fandom.com), got: {base_url!r}"
        )

    start_url = cfg.get("start_url")
    if start_url is not None:
        start_url = str(start_url).strip()
        if start_url and not start_url.startswith(("http://", "https://")):
            raise ValueError(
                f"start_url must be a valid HTTP(S) URL or empty, got: {start_url!r}"
            )

    category_urls = cfg.get("category_urls") or []
    if not isinstance(category_urls, list):
        raise ValueError(f"category_urls must be a list, got {type(category_urls).__name__}")
    category_urls = [str(u).strip() for u in category_urls if u]

    delay_seconds = float(cfg.get("delay_seconds", 2.0))
    if delay_seconds < 0.5:
        delay_seconds = 0.5

    use_api_fallback = bool(cfg.get("use_api_fallback", True))

    return ScrapingConfig(
        base_url=base_url,
        start_url=start_url or None,
        category_urls=category_urls,
        delay_seconds=delay_seconds,
        use_api_fallback=use_api_fallback,
    )


# ---------------------- URL GENERATION ----------------------

def normalize_url(url: str) -> str:
    parts = urlsplit(url)
    path = unquote(parts.path).replace(" ", "_")
    # strip query/fragment (MediaWiki pages often add noise params)
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def filter_article(url: str) -> bool:
    if "/wiki/" not in url:
        return False
    page = url.split("/wiki/", 1)[1]
    skip_prefixes = (
        "File:",
        "User:",
        "User_blog:",
        "Template:",
        "Help:",
        "Special:",
        "Forum:",
        "Message_Wall:",
        "Category:",
        "Talk:",
        "MediaWiki:",
        "Module:",
    )
    return not any(page.startswith(p) for p in skip_prefixes)


def _page_title_from_url(url: str) -> str:
    """Extract MediaWiki page title from wiki URL (part after /wiki/)."""
    if "/wiki/" not in url:
        return ""
    return unquote(url.split("/wiki/", 1)[1])


def _api_url(base_url: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", "api.php")


def get_page_id_via_api(
    base_url: str,
    title: str,
    session: requests.Session,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[str]:
    """
    Reliable way to get a page/article id (pageid) on MediaWiki/Fandom.

    Uses:
      action=query&titles=...&prop=info&redirects=1
    """
    if not title:
        return None

    api = _api_url(base_url)
    params = {
        "action": "query",
        "format": "json",
        "prop": "info",
        "redirects": "1",
        "titles": title,
    }
    r = session.get(api, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    pages = (data.get("query", {}) or {}).get("pages", {}) or {}
    for _, page in pages.items():
        # missing pages have "missing": "" and no pageid
        pageid = page.get("pageid")
        if pageid is not None:
            return str(pageid)
    return None


def scrape_allpages_api(base_url: str, session: requests.Session, delay: float = 0.1) -> List[str]:
    api_url = _api_url(base_url)
    results: List[str] = []
    seen = set()
    params: Dict[str, str] = {
        "action": "query",
        "list": "allpages",
        "aplimit": "500",
        "format": "json",
    }

    while True:
        logger.info(f"[AllPages API] Fetching {api_url} with params={params}")
        r = session.get(api_url, params=params, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        for page in (data.get("query", {}).get("allpages", []) or []):
            title = page.get("title")
            if not title:
                continue
            href = f"/wiki/{title.replace(' ', '_')}"
            full = normalize_url(urljoin(base_url, href))
            if full not in seen:
                seen.add(full)
                results.append(full)

        cont = data.get("continue", {}).get("apcontinue")
        if not cont:
            break
        params["apcontinue"] = cont
        time.sleep(delay)

    logger.info(f"[AllPages API] Collected {len(results)} raw URLs")
    return results


def scrape_allpages(base_url: str, start_url: str, session: requests.Session, delay: float = 0.5) -> List[str]:
    url = start_url
    results: List[str] = []
    seen = set()
    first_page = True

    while url:
        logger.info(f"[AllPages] Fetching {url}")
        try:
            r = session.get(url, timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 403:
                logger.warning("[AllPages] 403 Forbidden, switching to API fallback.")
                return scrape_allpages_api(base_url, session)
            raise

        soup = BeautifulSoup(r.text, "html.parser")

        for a in soup.select(".mw-allpages-chunk li > a, .mw-allpages-group li > a"):
            href = a.get("href")
            if not href or not href.startswith("/wiki/"):
                continue
            full = normalize_url(urljoin(base_url, href))
            if full not in seen:
                seen.add(full)
                results.append(full)

        if first_page and not results:
            logger.warning("[AllPages] No links found on first page, switching to API fallback.")
            return scrape_allpages_api(base_url, session)

        next_url = None

        head_next = soup.find("link", rel=lambda v: v and "next" in v.lower() if v else False)
        if head_next and head_next.get("href"):
            next_url = head_next.get("href")

        if not next_url:
            a_next = soup.select_one("a.mw-nextlink")
            if a_next and a_next.get("href"):
                next_url = a_next.get("href")

        if not next_url:
            for a in soup.select(".mw-allpages-nav a[href]"):
                if a.get_text(strip=True).lower().startswith("next page"):
                    next_url = a.get("href")
                    break

        url = urljoin(base_url, next_url) if next_url else None
        time.sleep(delay)
        first_page = False

    logger.info(f"[AllPages] Collected {len(results)} raw URLs")
    return results


def scrape_category_paginated(base_url: str, category_url: str, session: requests.Session) -> List[str]:
    """
    Fandom categories are often paginated. This follows "next" until done.
    """
    collected: List[str] = []
    seen_pages = set()
    url = category_url

    while url and url not in seen_pages:
        seen_pages.add(url)
        logger.info(f"[Category] Fetching {url}")
        r = session.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        selectors = [
            "a.category-page__member-link",
            ".category-page__members a",
            ".category-page__content a",
        ]

        for sel in selectors:
            for a in soup.select(sel):
                href = a.get("href")
                if not href:
                    continue
                if href.startswith("/"):
                    full = normalize_url(base_url.rstrip("/") + href)
                else:
                    full = normalize_url(href)
                collected.append(full)

        # find next page
        next_url = None
        # common fandom next button
        a_next = soup.select_one("a.category-page__pagination-next, a.category-page__pagination-next-button")
        if a_next and a_next.get("href"):
            next_url = a_next.get("href")

        if not next_url:
            # sometimes generic "next" rel link
            head_next = soup.find("link", rel=lambda v: v and "next" in v.lower() if v else False)
            if head_next and head_next.get("href"):
                next_url = head_next.get("href")

        url = urljoin(base_url, next_url) if next_url else None

    logger.info(f"[Category] Found {len(collected)} URLs (including pagination) from {category_url}")
    return collected


def scrape_all_categories(base_url: str, category_urls: List[str], session: requests.Session) -> List[str]:
    if not category_urls:
        logger.info("No category_urls in scraping.yaml — skipping category scrape.")
        return []

    collected: List[str] = []
    for cu in category_urls:
        try:
            collected.extend(scrape_category_paginated(base_url, cu, session))
        except Exception as e:
            logger.warning(f"Error in category {cu}: {e}", exc_info=True)

    logger.info(f"[Category] Total raw category URLs: {len(collected)}")
    return collected


def build_url_list(cfg: ScrapingConfig, project_root: Path) -> Path:
    base_url = cfg.base_url
    start_url = cfg.start_url
    category_urls = cfg.category_urls

    logger.info(f"Base URL: {base_url}")
    logger.info(f"Start URL (AllPages): {start_url}")
    logger.info(f"Category URLs: {category_urls}")

    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)
    session.headers["Referer"] = base_url + "/"

    combined: List[str] = []

    if start_url:
        combined.extend(scrape_allpages(base_url, start_url, session))
    else:
        logger.info("start_url not set in scraping.yaml — skipping AllPages.")

    combined.extend(scrape_all_categories(base_url, category_urls, session))

    # normalize + dedupe early
    combined = [normalize_url(u) for u in combined]
    combined = list(dict.fromkeys(combined))  # stable unique
    logger.info(f"Combined (before filtering): {len(combined)}")

    filtered = [u for u in combined if filter_article(u)]
    logger.info(f"After filtering to real article pages: {len(filtered)}")

    for u in filtered[:5]:
        logger.info(f"Sample URL: {u}")

    url_list_dir = project_root / "data" / "raw" / "url_lists"
    url_list_dir.mkdir(parents=True, exist_ok=True)
    output_path = url_list_dir / f"{cfg.domain}_urls.txt"

    output_path.write_text("\n".join(sorted(filtered)), encoding="utf-8")
    logger.info(f"Saved URL list to: {output_path}")

    return output_path


# ---------------------- PAGE SCRAPING ----------------------

def read_url_list(path: Path) -> List[str]:
    urls: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def extract_plain_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    content = soup.find("div", class_="mw-parser-output")
    if content:
        text = content.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join([line for line in lines if line])
    return cleaned


def fetch_page_via_api(base_url: str, title: str, session: requests.Session, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Fetch page HTML via MediaWiki parse API. Robust fallback when direct HTML fetch is blocked.
    """
    api_url = _api_url(base_url)
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
        "redirects": "1",
    }
    resp = session.get(api_url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise ValueError(f"API error: {data['error'].get('info', data['error'])}")

    text = data.get("parse", {}).get("text", {}).get("*", "")
    if not text:
        raise ValueError(f"Empty parse result for page: {title}")

    return f'<div class="mw-parser-output">{text}</div>'


def fetch_html(session: requests.Session, url: str, timeout: int = DEFAULT_TIMEOUT) -> Tuple[str, str]:
    resp = session.get(url, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    return resp.text, resp.url


def fetch_html_or_api(
    session: requests.Session,
    url: str,
    base_url: str,
    use_api_fallback: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[str, str]:
    """
    Try HTML fetch first. If blocked (403) and allowed, fall back to parse API.
    Returns: (html, final_url)
    """
    try:
        html, final_url = fetch_html(session, url, timeout=timeout)
        return html, final_url
    except requests.HTTPError as e:
        if use_api_fallback and e.response is not None and e.response.status_code == 403:
            logger.info(f"    403 on HTML fetch, trying MediaWiki API parse for {url}")
            title = _page_title_from_url(url)
            if not title:
                raise
            html = fetch_page_via_api(base_url, title, session, timeout=timeout)
            return html, url
        raise


def scrape_pages(cfg: ScrapingConfig, project_root: Path) -> None:
    base_url = cfg.base_url
    domain = cfg.domain

    url_list_path = project_root / "data" / "raw" / "url_lists" / f"{domain}_urls.txt"
    output_dir = project_root / "data" / "raw" / domain
    output_dir.mkdir(parents=True, exist_ok=True)

    if not url_list_path.exists():
        logger.error(f"URL list file not found: {url_list_path}")
        raise FileNotFoundError(url_list_path)

    urls = read_url_list(url_list_path)
    logger.info(f"Loaded {len(urls)} URLs from {url_list_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Delay: {cfg.delay_seconds}s, API fallback: {cfg.use_api_fallback}")

    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)
    session.headers["Referer"] = base_url + "/"

    total = len(urls)
    success = skipped = failed = 0

    total_html_bytes = 0
    total_text_bytes = 0
    max_html_bytes = 0
    max_text_bytes = 0

    url_mapping: List[Tuple[str, str]] = []

    for i, url in enumerate(urls, start=1):
        logger.info(f"[{i}/{total}] Fetching → {url}")
        ok = False

        for attempt in range(1, 4):
            try:
                html, final_url = fetch_html_or_api(
                    session,
                    url,
                    base_url,
                    use_api_fallback=cfg.use_api_fallback,
                    timeout=DEFAULT_TIMEOUT,
                )

                if url != final_url:
                    logger.info(f"    Redirected to final URL: {final_url}")
                url_mapping.append((url, final_url))

                # ✅ Reliable ID: pageid via API (using final title when possible)
                title = _page_title_from_url(final_url) or _page_title_from_url(url)
                page_id = get_page_id_via_api(base_url, title, session)
                if not page_id:
                    logger.warning(f"    Could not resolve pageid via API; skipping (title={title!r})")
                    skipped += 1
                    ok = True
                    break

                out_path = output_dir / f"{page_id}.html"
                if out_path.exists():
                    logger.info(f"    SKIP (exists) → {out_path.name}")
                    skipped += 1
                    ok = True
                    break

                # Write HTML
                html_bytes = html.encode("utf-8")
                out_path.write_text(html, encoding="utf-8")
                logger.info(f"    Saved HTML → {out_path}")

                total_html_bytes += len(html_bytes)
                max_html_bytes = max(max_html_bytes, len(html_bytes))

                # Write TEXT
                plain_text = extract_plain_text(html)
                txt_path = out_path.with_suffix(".txt")
                text_bytes = plain_text.encode("utf-8")
                txt_path.write_text(plain_text, encoding="utf-8")
                logger.info(f"    Saved TEXT → {txt_path}")

                total_text_bytes += len(text_bytes)
                max_text_bytes = max(max_text_bytes, len(text_bytes))

                success += 1
                ok = True
                break

            except Exception as e:
                logger.warning(f"    Attempt {attempt}/3 failed for {url}: {e}", exc_info=True)
                time.sleep(1.0 * attempt)

        if not ok:
            logger.error(f"    FAILED: {url}")
            failed += 1

        if i % 50 == 0:
            logger.info(
                f"[Progress] Processed {i}/{total} pages "
                f"(success={success}, skipped={skipped}, failed={failed})"
            )

        time.sleep(cfg.delay_seconds)

    mapping_path = output_dir / "url_redirect_mapping.tsv"
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("original_url\tfinal_url\n")
        for orig, final in url_mapping:
            f.write(f"{orig}\t{final}\n")
    logger.info(f"Saved URL redirect mapping to {mapping_path}")

    logger.info("=== Scrape summary ===")
    logger.info(f"Total URLs:    {total}")
    logger.info(f"Downloaded:    {success}")
    logger.info(f"Skipped (had): {skipped}")
    logger.info(f"Failed:        {failed}")

    avg_html_bytes = total_html_bytes / success if success > 0 else 0
    avg_text_bytes = total_text_bytes / success if success > 0 else 0

    scraping_stats = {
        "total_urls": total,
        "downloaded": success,
        "skipped": skipped,
        "failed": failed,
        "html_bytes": {"total": total_html_bytes, "avg": avg_html_bytes, "max": max_html_bytes},
        "text_bytes": {"total": total_text_bytes, "avg": avg_text_bytes, "max": max_text_bytes},
    }

    logger.info(f"Scraping stats: {scraping_stats}")
    update_scraping_stats(domain, scraping_stats)


# ---------------------- PUBLIC ENTRYPOINT ----------------------

def run_full_scrape(config_path: Optional[Path] = None) -> Path:
    cfg = load_scraping_config(config_path)
    logger.info(f"Starting full scrape for domain={cfg.domain}")
    url_list_path = build_url_list(cfg, PROJECT_ROOT)
    scrape_pages(cfg, PROJECT_ROOT)
    logger.info("Finished full scrape.")
    return url_list_path