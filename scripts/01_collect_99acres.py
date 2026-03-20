#!/usr/bin/env python3
"""Scrape 99acres.com property listings (raw HTML only, gzip compressed).

This script collects raw pages WITHOUT parsing. Parsing is done by 02_parse_99acres.py.

Workflow:
1. Check robots.txt before each fetch.
2. Paginate through search results, save HTML as .html.gz.
3. Extract detail page links from search pages.
4. Visit each detail page, save HTML as .html.gz.
5. Output: scrape_manifest.jsonl tracking what was scraped.

Notes
-----
- 99acres uses dynamic rendering, so Playwright is required.
- The default city URLs live in data/config/99acres_cities.json and can be edited without
  touching the code.
- Supports both houses (independent-house-villa) and flats.

Example
-------
python scripts/01_collect_99acres.py --city gurgaon --property-type both --max-pages 5
python scripts/01_collect_99acres.py --all-cities --property-type house --max-pages 3
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlsplit
from urllib.robotparser import RobotFileParser

try:
    from playwright.sync_api import sync_playwright
except Exception as exc:
    raise SystemExit(
        "Playwright is required. Install dependencies and run 'playwright install chromium'."
    ) from exc


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DOMAIN = "https://www.99acres.com"

MAX_RETRIES = 3
BACKOFF_BASE = 2.0


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root_from_here()
    parser = argparse.ArgumentParser(description=__doc__)
    city_group = parser.add_mutually_exclusive_group(required=True)
    city_group.add_argument("--city", help="City key from data/config/99acres_cities.json")
    city_group.add_argument(
        "--all-cities",
        action="store_true",
        help="Collect data for all cities in the config file",
    )
    parser.add_argument(
        "--config",
        default=str(root / "data" / "config" / "99acres_cities.json"),
        help="Path to city-URL config JSON",
    )
    parser.add_argument(
        "--property-type",
        choices=["house", "flat", "both"],
        default="both",
        help="Property type to collect",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for raw pages. Defaults to data/raw/99acres/<city>",
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
        help="Run Chromium in headless mode (99acres blocks headless, so headful is default)",
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
    return parser.parse_args()


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "na"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_city_config(path: Path) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a JSON object of city -> {{house, flat}} URLs.")
    return data


def parse_proxy_url(proxy_url: Optional[str]) -> Optional[Dict[str, Any]]:
    if not proxy_url:
        return None
    parsed = urlsplit(proxy_url)
    scheme = parsed.scheme or "http"
    config: Dict[str, Any] = {"server": f"{scheme}://{parsed.hostname}:{parsed.port}"}
    if parsed.username:
        config["username"] = parsed.username
    if parsed.password:
        config["password"] = parsed.password
    return config


class RobotsGuard:
    def __init__(
        self, root_url: str, user_agent: str = USER_AGENT, proxy: Optional[str] = None
    ) -> None:
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


RE_PROPERTY_ID = re.compile(r"spid-([A-Z]?\d{7,})")


def jitter_sleep(lo: float, hi: float) -> None:
    if hi <= 0:
        return
    time.sleep(random.uniform(max(0.0, lo), max(lo, hi)))


def is_blocked_response(html: str, text: str) -> bool:
    blocked_patterns = [
        "access denied",
        "too many requests",
        "rate limit",
        "please try again later",
        "captcha",
        "verify you are human",
        "blocked",
        "403 forbidden",
        "429 too many",
    ]
    combined = (html + text).lower()
    return any(pat in combined for pat in blocked_patterns)


def fetch_with_retry(
    page,
    url: str,
    wait_ms: int = 3000,
    max_retries: int = MAX_RETRIES,
) -> Tuple[str, str, bool]:
    for attempt in range(max_retries):
        try:
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(wait_ms)
            html = page.content()
            text = page.locator("body").inner_text()

            if is_blocked_response(html, text):
                backoff = BACKOFF_BASE ** (attempt + 1) + random.uniform(0, 2)
                print(f"    [BLOCKED] Attempt {attempt + 1}/{max_retries}, backing off {backoff:.1f}s", file=sys.stderr)
                time.sleep(backoff)
                continue

            return html, text, True
        except Exception as e:
            backoff = BACKOFF_BASE ** (attempt + 1) + random.uniform(0, 2)
            print(f"    [RETRY] Attempt {attempt + 1}/{max_retries} failed: {e}, backing off {backoff:.1f}s", file=sys.stderr)
            time.sleep(backoff)

    return "", "", False


def property_id_from_url(url: str) -> str:
    m = RE_PROPERTY_ID.search(url)
    if m:
        return m.group(1)
    stem = Path(urlparse(url).path).name
    return slugify(stem)


def extract_detail_links(hrefs: Iterable[str], root_url: str) -> List[str]:
    out: List[str] = []
    for href in hrefs:
        if not href:
            continue
        url = urljoin(root_url, href)
        parsed = urlparse(url)
        if parsed.netloc and "99acres.com" not in parsed.netloc:
            continue
        path = parsed.path.lower()
        if "spid-" not in path:
            continue
        if any(bad in path for bad in ["/projects/", "/builder/", "/price-trends", "/search/"]):
            continue
        if url not in out:
            out.append(url)
    return out


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
    city_urls: Dict[str, str],
    outdir: Path,
    args: argparse.Namespace,
    guard: RobotsGuard,
) -> dict:
    pages_search = ensure_dir(outdir / "pages" / "search")
    pages_detail = ensure_dir(outdir / "pages" / "detail")
    ensure_dir(outdir)

    manifest_path = outdir / "scrape_manifest.jsonl"
    scraped_urls = load_scraped_urls(manifest_path) if args.resume else set()

    property_types = []
    if args.property_type in ("house", "both"):
        property_types.append("house")
    if args.property_type in ("flat", "both"):
        property_types.append("flat")

    search_pages_scraped = 0
    detail_pages_scraped = 0

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

        proxy_config = parse_proxy_url(args.proxy)
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 900},
            locale="en-IN",
            timezone_id="Asia/Kolkata",
            proxy=proxy_config,  # type: ignore[arg-type]
            ignore_https_errors=bool(proxy_config),
        )
        page = context.new_page()
        page.set_default_timeout(args.timeout_ms)

        for prop_type in property_types:
            if prop_type not in city_urls:
                print(f"  [WARN] No URL for {prop_type} in {city}, skipping", file=sys.stderr)
                continue

            base_url = city_urls[prop_type]
            print(f"  Collecting {prop_type} listings from {city}...", file=sys.stderr)

            detail_links: List[Tuple[str, int]] = []

            for pageno in range(1, args.max_pages + 1):
                url = f"{base_url}&page={pageno}" if "?" in base_url else f"{base_url}?page={pageno}"

                if args.resume and url in scraped_urls:
                    continue

                if not guard.is_allowed(url):
                    print(f"  [SKIP] robots.txt disallows: {url}", file=sys.stderr)
                    continue

                print(f"  [{prop_type}] Fetching search page {pageno}...", file=sys.stderr)
                html, _, success = fetch_with_retry(page, url, wait_ms=3000)
                if not success:
                    print(f"  [ERROR] Failed to load search page {pageno} after retries", file=sys.stderr)
                    continue

                hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")

                html_path = pages_search / f"{slugify(city)}_{prop_type}_p{pageno}.html.gz"
                save_html_gz(html_path, html)

                links = extract_detail_links(hrefs, base_url)
                detail_links.extend((u, pageno) for u in links)
                search_pages_scraped += 1

                append_jsonl(manifest_path, [{
                    "type": "search",
                    "city": city,
                    "property_type": prop_type,
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

            print(f"  [{prop_type}] Found {len(unique_links)} unique detail URLs to fetch", file=sys.stderr)

            for idx, (url, search_page) in enumerate(unique_links, start=1):
                if args.resume and url in scraped_urls:
                    continue

                if not guard.is_allowed(url):
                    print(f"  [SKIP] robots.txt disallows: {url}", file=sys.stderr)
                    continue

                pid = property_id_from_url(url)
                print(f"  [{prop_type}] Fetching detail {idx}/{len(unique_links)}: {pid}", file=sys.stderr)

                html, _, success = fetch_with_retry(page, url, wait_ms=2000)
                if not success:
                    print(f"  [ERROR] Failed to load detail page {pid} after retries", file=sys.stderr)
                    continue

                html_path = pages_detail / f"{slugify(city)}_{prop_type}_{pid}.html.gz"
                save_html_gz(html_path, html)
                detail_pages_scraped += 1

                append_jsonl(manifest_path, [{
                    "type": "detail",
                    "city": city,
                    "property_type": prop_type,
                    "property_id": pid,
                    "url": url,
                    "html_path": str(html_path.relative_to(outdir)),
                    "search_page": search_page,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }])

                if idx % 25 == 0:
                    print(f"  [{prop_type}] Scraped {idx} detail pages...", file=sys.stderr)

                jitter_sleep(args.min_sleep, args.max_sleep)

        browser.close()

    return {
        "city": city,
        "property_types": property_types,
        "search_pages_scraped": search_pages_scraped,
        "detail_pages_scraped": detail_pages_scraped,
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    random.seed(42)

    root = project_root_from_here()
    config = load_city_config(Path(args.config))
    guard = RobotsGuard(DOMAIN, proxy=args.proxy)

    if args.all_cities:
        cities_to_collect = list(config.keys())
        unique_cities = []
        seen_urls = set()
        for city in cities_to_collect:
            url_tuple = tuple(sorted(config[city].items()))
            if url_tuple not in seen_urls:
                seen_urls.add(url_tuple)
                unique_cities.append(city)

        print(f"Collecting {len(unique_cities)} cities: {unique_cities}", file=sys.stderr)

        all_summaries = []
        for city in unique_cities:
            city_urls = config[city]
            outdir = root / "data" / "raw" / "99acres" / slugify(city)
            print(f"\n=== Collecting {city} ===", file=sys.stderr)
            summary = scrape_city(city, city_urls, outdir, args, guard)
            all_summaries.append(summary)
            print(json.dumps(summary, indent=2))

        total_detail = sum(s["detail_pages_scraped"] for s in all_summaries)
        print("\n=== Collection complete ===", file=sys.stderr)
        print(f"Cities: {len(all_summaries)}, Total detail pages: {total_detail}", file=sys.stderr)
    else:
        if args.city not in config:
            raise SystemExit(f"City '{args.city}' not found in {args.config}. Known keys: {sorted(config)}")

        city_urls = config[args.city]
        outdir = Path(args.output_dir) if args.output_dir else root / "data" / "raw" / "99acres" / slugify(args.city)
        summary = scrape_city(args.city, city_urls, outdir, args, guard)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
