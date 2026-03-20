#!/usr/bin/env python3
"""Scrape Magicbricks.com property listings (raw HTML only, gzip compressed).

This script collects raw pages WITHOUT parsing. Parsing is done by 02_parse_magicbricks.py.

Workflow:
1. Check robots.txt before each fetch.
2. Paginate through search results, save HTML as .html.gz.
3. Extract detail page links from search pages.
4. Visit each detail page, save HTML as .html.gz.
5. Output: scrape_manifest.jsonl tracking what was scraped.

Example
-------
python scripts/01_collect_magicbricks.py --city delhi-ncr --max-pages 5 --max-detail-pages 200
python scripts/01_collect_magicbricks.py --all-cities --max-pages 3 --max-detail-pages 100
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlsplit
from urllib.robotparser import RobotFileParser

try:
    from playwright.sync_api import sync_playwright
except Exception as exc:
    raise SystemExit(
        "Playwright is required. Install dependencies and run 'playwright install chromium'."
    ) from exc


logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DOMAIN = "https://www.magicbricks.com"

MAX_RETRIES = 3
BACKOFF_BASE = 2.0


def setup_logging(verbose: bool = False, quiet: bool = False, log_file: Optional[str] = None) -> None:
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handlers: List[logging.Handler] = []

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handlers.append(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers)


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root_from_here()
    parser = argparse.ArgumentParser(description=__doc__)
    city_group = parser.add_mutually_exclusive_group(required=True)
    city_group.add_argument("--city", help="City key from data/config/magicbricks_cities.json")
    city_group.add_argument(
        "--all-cities",
        action="store_true",
        help="Collect data for all cities in the config file",
    )
    parser.add_argument(
        "--config",
        default=str(root / "data" / "config" / "magicbricks_cities.json"),
        help="Path to city-URL config JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for raw pages. Defaults to data/raw/magicbricks/<city>",
    )
    parser.add_argument("--max-pages", type=int, default=5, help="Number of search-result pages to visit")
    parser.add_argument(
        "--max-detail-pages",
        type=int,
        default=250,
        help="Hard cap on listing detail pages fetched after deduplication",
    )
    parser.add_argument("--min-sleep", type=float, default=2.0, help="Minimum sleep between requests")
    parser.add_argument("--max-sleep", type=float, default=4.0, help="Maximum sleep between requests")
    parser.add_argument("--timeout-ms", type=int, default=60000, help="Playwright page timeout in ms")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium headless (default is headful for Magicbricks due to bot detection)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip URLs already in scrape_manifest.jsonl",
    )
    parser.add_argument(
        "--proxy",
        help="Proxy server URL (e.g., http://user:pass@host:port)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress INFO messages, show only warnings and errors",
    )
    parser.add_argument(
        "--log-file",
        help="Write logs to this file in addition to console",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "na"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_city_config(path: Path) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a JSON object of city -> url.")
    return data


def parse_proxy_url(proxy_url: Optional[str]) -> Optional[Dict[str, Any]]:
    if not proxy_url:
        return None
    parsed = urlsplit(proxy_url)
    scheme = parsed.scheme or "http"
    config = {"server": f"{scheme}://{parsed.hostname}:{parsed.port}"}
    if parsed.username:
        config["username"] = parsed.username
    if parsed.password:
        config["password"] = parsed.password
    return config


class RobotsGuard:
    def __init__(self, root_url: str, user_agent: str = USER_AGENT, proxy: Optional[str] = None) -> None:
        self.root_url = root_url.rstrip("/")
        self.user_agent = user_agent
        self.proxy = proxy
        self.rp = RobotFileParser()
        self.rp.set_url(f"{self.root_url}/robots.txt")
        self.robots_available = False
        self._fetch_robots_with_playwright()

    def _fetch_robots_with_playwright(self) -> None:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=False,
                    args=["--disable-blink-features=AutomationControlled"]
                )
                proxy_config = parse_proxy_url(self.proxy)
                context = browser.new_context(
                    user_agent=self.user_agent,
                    locale="en-IN",
                    timezone_id="Asia/Kolkata",
                    proxy=proxy_config,  # type: ignore[arg-type]
                    ignore_https_errors=bool(proxy_config),
                )
                page = context.new_page()
                page.goto(f"{self.root_url}/robots.txt", wait_until="domcontentloaded")
                page.wait_for_timeout(2000)
                text = page.locator("body").inner_text()
                browser.close()

                if "User-agent" in text or "Disallow" in text or "Allow" in text:
                    self.rp.parse(text.splitlines())
                    self.robots_available = True
        except Exception:
            pass

    def is_allowed(self, url: str) -> bool:
        if not self.robots_available:
            return True
        return self.rp.can_fetch(self.user_agent, url)

    def assert_allowed(self, url: str) -> None:
        if not self.is_allowed(url):
            raise PermissionError(f"robots.txt disallows fetch: {url}")


RE_PROPERTY_ID = re.compile(r"pdpid-([a-zA-Z0-9]+)|(?:id[=:]|propertyid[=:]|/id/)(\d+)", re.I)


def jitter_sleep(lo: float, hi: float) -> None:
    if hi <= 0:
        return
    time.sleep(random.uniform(max(0.0, lo), max(lo, hi)))


def is_blocked_response(text: str) -> bool:
    blocked_patterns = [
        "access denied",
        "too many requests",
        "rate limit",
        "please try again later",
        "captcha",
        "verify you are human",
        "403 forbidden",
        "429 too many",
    ]
    text_lower = text.lower()
    if any(pat in text_lower for pat in blocked_patterns):
        return True
    if len(text.strip()) < 500 and ("error" in text_lower or "denied" in text_lower):
        return True
    return False


def fetch_with_retry(
    page,
    url: str,
    wait_ms: int = 3000,
    max_retries: int = MAX_RETRIES,
) -> Tuple[str, str, bool]:
    for attempt in range(max_retries):
        try:
            response = page.goto(url, wait_until="domcontentloaded")

            if response and response.status >= 400:
                backoff = BACKOFF_BASE ** (attempt + 1) + random.uniform(0, 2)
                logger.warning(
                    "HTTP %d on attempt %d/%d, backing off %.1fs",
                    response.status, attempt + 1, max_retries, backoff
                )
                time.sleep(backoff)
                continue

            page.wait_for_timeout(wait_ms)
            html = page.content()
            text = page.locator("body").inner_text()

            if is_blocked_response(text):
                backoff = BACKOFF_BASE ** (attempt + 1) + random.uniform(0, 2)
                logger.warning(
                    "Blocked on attempt %d/%d, backing off %.1fs",
                    attempt + 1, max_retries, backoff
                )
                time.sleep(backoff)
                continue

            return html, text, True
        except Exception as e:
            backoff = BACKOFF_BASE ** (attempt + 1) + random.uniform(0, 2)
            logger.warning(
                "Attempt %d/%d failed: %s, backing off %.1fs",
                attempt + 1, max_retries, e, backoff
            )
            time.sleep(backoff)

    return "", "", False


def property_id_from_url(url: str) -> str:
    m = RE_PROPERTY_ID.search(url)
    if m:
        return m.group(1) or m.group(2)
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "id" in qs:
        return qs["id"][0]
    stem = Path(parsed.path).name
    return slugify(stem)


def extract_detail_links(hrefs: Iterable[str], root_url: str) -> List[str]:
    out: List[str] = []
    for href in hrefs:
        if not href:
            continue
        url = urljoin(root_url, href)
        parsed = urlparse(url)
        if parsed.netloc and "magicbricks.com" not in parsed.netloc:
            continue
        path = parsed.path.lower()
        bad_paths = [
            "/price-trends", "/locality-insights", "/new-projects", "/search/",
            "/advice/", "/homeloan/", "/accounts", "/post.", "/help", "/blog"
        ]
        if any(bad in path for bad in bad_paths):
            continue
        if "pdpid-" in path:
            if url not in out:
                out.append(url)
    return out


def add_page_param(base_url: str, page_num: int) -> str:
    parsed = urlparse(base_url)
    qs = parse_qs(parsed.query)
    qs["page"] = [str(page_num)]
    new_query = urlencode(qs, doseq=True)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"


def save_html_gz(path: Path, html: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(html)


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_scraped_urls(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    urls = set()
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "url" in entry:
                    urls.add(entry["url"])
            except json.JSONDecodeError:
                continue
    return urls


def scrape_city(
    city: str,
    base_url: str,
    outdir: Path,
    args: argparse.Namespace,
    guard: RobotsGuard,
) -> dict:
    pages_search = ensure_dir(outdir / "pages" / "search")
    pages_detail = ensure_dir(outdir / "pages" / "detail")
    ensure_dir(outdir)

    manifest_path = outdir / "scrape_manifest.jsonl"
    scraped_urls = load_scraped_urls(manifest_path) if args.resume else set()

    proxy = getattr(args, 'proxy', None)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                headless=args.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ]
            )
        except Exception as exc:
            raise SystemExit(
                "Chromium did not launch. Install it with: playwright install chromium"
            ) from exc

        proxy_config = parse_proxy_url(proxy)
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 2200},
            locale="en-IN",
            timezone_id="Asia/Kolkata",
            proxy=proxy_config,  # type: ignore[arg-type]
            ignore_https_errors=bool(proxy_config),
        )
        page = context.new_page()
        page.set_default_timeout(args.timeout_ms)

        detail_links: List[Tuple[str, int]] = []
        search_pages_scraped = 0

        for pageno in range(1, args.max_pages + 1):
            url = add_page_param(base_url, pageno)

            if args.resume and url in scraped_urls:
                logger.debug("[%s] Skipping search page %d (already scraped)", city, pageno)
                continue

            if not guard.is_allowed(url):
                logger.warning("[%s] Skipping search page %d (robots.txt)", city, pageno)
                continue

            logger.info("[%s] Fetching search page %d: %s", city, pageno, url)
            html, _, success = fetch_with_retry(page, url, wait_ms=3000)

            if not success:
                logger.error("[%s] Failed to load search page %d after retries", city, pageno)
                continue

            hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")

            html_path = pages_search / f"{slugify(city)}_p{pageno}.html.gz"
            save_html_gz(html_path, html)

            links = extract_detail_links(hrefs, base_url)
            detail_links.extend((u, pageno) for u in links)
            search_pages_scraped += 1

            append_jsonl(manifest_path, [{
                "type": "search",
                "city": city,
                "page": pageno,
                "url": url,
                "html_path": str(html_path.relative_to(outdir)),
                "n_detail_links": len(links),
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }])

            jitter_sleep(args.min_sleep, args.max_sleep)

        seen: set[str] = set()
        unique_links: List[Tuple[str, int]] = []
        for url, pageno in detail_links:
            pid = property_id_from_url(url)
            if pid in seen:
                continue
            seen.add(pid)
            unique_links.append((url, pageno))

        if args.max_detail_pages > 0:
            unique_links = unique_links[: args.max_detail_pages]

        logger.info("[%s] Found %d unique detail URLs to fetch", city, len(unique_links))

        detail_pages_scraped = 0
        for idx, (url, search_page) in enumerate(unique_links, start=1):
            if args.resume and url in scraped_urls:
                continue

            if not guard.is_allowed(url):
                logger.warning("[%s] Skipping detail page (robots.txt): %s", city, url)
                continue

            pid = property_id_from_url(url)
            logger.info("[%s] Fetching detail %d/%d: %s", city, idx, len(unique_links), pid)

            html, _, success = fetch_with_retry(page, url, wait_ms=2000)

            if not success:
                logger.error("[%s] Failed to load detail page %s after retries", city, pid)
                continue

            html_path = pages_detail / f"{slugify(city)}_{pid}.html.gz"
            save_html_gz(html_path, html)
            detail_pages_scraped += 1

            append_jsonl(manifest_path, [{
                "type": "detail",
                "city": city,
                "property_id": pid,
                "url": url,
                "html_path": str(html_path.relative_to(outdir)),
                "search_page": search_page,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }])

            if idx % 25 == 0:
                logger.info("[%s] Scraped %d detail pages...", city, idx)
            jitter_sleep(args.min_sleep, args.max_sleep)

        browser.close()

    return {
        "city": city,
        "base_url": base_url,
        "search_pages_scraped": search_pages_scraped,
        "detail_pages_scraped": detail_pages_scraped,
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)
    random.seed(42)

    root = project_root_from_here()
    config = load_city_config(Path(args.config))

    proxy = args.proxy or os.environ.get("BRIGHT_DATA_PROXY_URL")
    if proxy:
        proxy_display = proxy.split('@')[-1] if '@' in proxy else proxy
        logger.info("Using proxy: %s", proxy_display)
    else:
        logger.warning("No proxy configured. Set BRIGHT_DATA_PROXY_URL or use --proxy")

    args.proxy = proxy
    guard = RobotsGuard(DOMAIN, proxy=proxy)

    if args.all_cities:
        unique_urls = {}
        for city, url in config.items():
            if url not in unique_urls.values():
                unique_urls[city] = url
        cities_to_collect = list(unique_urls.keys())
        logger.info("Collecting %d cities: %s", len(cities_to_collect), cities_to_collect)

        all_summaries = []
        for city in cities_to_collect:
            base_url = config[city]
            outdir = root / "data" / "raw" / "magicbricks" / slugify(city)
            logger.info("=== Collecting %s ===", city)
            summary = scrape_city(city, base_url, outdir, args, guard)
            all_summaries.append(summary)
            print(json.dumps(summary, indent=2))

        total_detail = sum(s["detail_pages_scraped"] for s in all_summaries)
        logger.info("=== Collection complete ===")
        logger.info("Cities: %d, Total detail pages: %d", len(all_summaries), total_detail)
    else:
        if args.city not in config:
            raise SystemExit(f"City '{args.city}' not found in {args.config}. Known keys: {sorted(config)}")

        base_url = config[args.city]
        if args.output_dir:
            outdir = Path(args.output_dir)
        else:
            outdir = root / "data" / "raw" / "magicbricks" / slugify(args.city)
        summary = scrape_city(args.city, base_url, outdir, args, guard)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
