#!/usr/bin/env python3
"""Scrape Housing.com property listings (raw HTML only, gzip compressed).

This script collects raw pages WITHOUT parsing. Parsing is done by 02_parse_housingcom.py.

Workflow:
1. Check robots.txt before each fetch.
2. Paginate through search results, save HTML as .html.gz.
3. Extract detail page links from search pages.
4. Visit each detail page, save HTML as .html.gz.
5. Output: scrape_manifest.jsonl tracking what was scraped.

Notes
-----
- Housing portals change their DOMs often. This collector saves raw files first.
- The default city URLs live in data/config/housingcom_cities.json.
- The script prefers Playwright because some pages are rendered dynamically.

Example
-------
python scripts/01_collect_housingcom.py --city hyderabad --max-pages 5 --max-detail-pages 200
python scripts/01_collect_housingcom.py --all-cities --max-pages 3 --max-detail-pages 100
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
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
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
DOMAIN = "https://housing.com"

MAX_RETRIES = 3
BACKOFF_BASE = 2.0


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root_from_here()
    parser = argparse.ArgumentParser(description=__doc__)
    city_group = parser.add_mutually_exclusive_group(required=True)
    city_group.add_argument("--city", help="City key from data/config/housingcom_cities.json")
    city_group.add_argument(
        "--all-cities",
        action="store_true",
        help="Collect data for all cities in the config file",
    )
    parser.add_argument(
        "--config",
        default=str(root / "data" / "config" / "housingcom_cities.json"),
        help="Path to city-URL config JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for raw pages. Defaults to data/raw/housingcom/<city>",
    )
    parser.add_argument("--max-pages", type=int, default=5, help="Number of search-result pages to visit")
    parser.add_argument(
        "--max-detail-pages",
        type=int,
        default=250,
        help="Hard cap on listing detail pages fetched after deduplication",
    )
    parser.add_argument("--min-sleep", type=float, default=1.0, help="Minimum sleep between requests")
    parser.add_argument("--max-sleep", type=float, default=2.5, help="Maximum sleep between requests")
    parser.add_argument("--timeout-ms", type=int, default=60000, help="Playwright page timeout in ms")
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run Chromium with a visible window (helpful when debugging anti-bot issues)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip URLs already in scrape_manifest.jsonl",
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


class RobotsGuard:
    def __init__(self, root_url: str, user_agent: str = USER_AGENT) -> None:
        self.root_url = root_url.rstrip("/")
        self.user_agent = user_agent
        self.rp = RobotFileParser()
        self.rp.set_url(f"{self.root_url}/robots.txt")
        self.rp.read()

    def is_allowed(self, url: str) -> bool:
        return self.rp.can_fetch(self.user_agent, url)


RE_SHOWING = re.compile(r"Showing\s+\d+\s*-\s*\d+\s+of\s+([\d,]+)", re.I)
RE_PROPERTY_ID = re.compile(r"(?:/page/|/)(\d{5,})[^/]*$")


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
    return any(pat in text_lower for pat in blocked_patterns)


def fetch_with_retry(
    page,
    url: str,
    wait_ms: int = 2000,
    max_retries: int = MAX_RETRIES,
) -> Tuple[str, str, bool]:
    for attempt in range(max_retries):
        try:
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(wait_ms)
            html = page.content()
            text = page.locator("body").inner_text()

            if is_blocked_response(text):
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


def inventory_total_from_text(text: str) -> Optional[int]:
    m = RE_SHOWING.search(text)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def extract_detail_links(hrefs: Iterable[str], root_url: str) -> List[str]:
    out: List[str] = []
    for href in hrefs:
        if not href:
            continue
        url = urljoin(root_url, href)
        parsed = urlparse(url)
        if parsed.netloc and "housing.com" not in parsed.netloc:
            continue
        path = parsed.path.lower()
        if "/in/buy/" not in path:
            continue
        if any(bad in path for bad in ["/price-trends", "/locality-insights", "/builder/", "/projects/"]):
            continue
        if not ("/page/" in path or "-for-rs-" in path):
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

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=not args.headful)
        except Exception as exc:
            raise SystemExit(
                "Chromium did not launch. Install it with: playwright install chromium"
            ) from exc

        context = browser.new_context(user_agent=USER_AGENT, viewport={"width": 1440, "height": 2200})
        page = context.new_page()
        page.set_default_timeout(args.timeout_ms)

        detail_links: List[Tuple[str, int]] = []
        inventory_totals: List[Optional[int]] = []
        search_pages_scraped = 0

        for pageno in range(1, args.max_pages + 1):
            url = f"{base_url}?page={pageno}"

            if args.resume and url in scraped_urls:
                continue

            if not guard.is_allowed(url):
                print(f"  [SKIP] robots.txt disallows: {url}", file=sys.stderr)
                continue

            print(f"[{city}] Fetching search page {pageno}: {url}", file=sys.stderr)
            html, text, success = fetch_with_retry(page, url, wait_ms=2000)

            if not success:
                print(f"  [ERROR] Failed to load search page {pageno} after retries", file=sys.stderr)
                continue

            hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")

            html_path = pages_search / f"{slugify(city)}_p{pageno}.html.gz"
            save_html_gz(html_path, html)

            total = inventory_total_from_text(text)
            inventory_totals.append(total)
            links = extract_detail_links(hrefs, base_url)
            detail_links.extend((u, pageno) for u in links)
            search_pages_scraped += 1

            append_jsonl(manifest_path, [{
                "type": "search",
                "city": city,
                "page": pageno,
                "url": url,
                "html_path": str(html_path.relative_to(outdir)),
                "inventory_total": total,
                "n_detail_links": len(links),
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }])

            jitter_sleep(args.min_sleep, args.max_sleep)

        seen: set[str] = set()
        unique_links: List[Tuple[str, int]] = []
        for url, pageno in detail_links:
            if url in seen:
                continue
            seen.add(url)
            unique_links.append((url, pageno))

        if args.max_detail_pages > 0:
            unique_links = unique_links[: args.max_detail_pages]

        print(f"[{city}] Found {len(unique_links)} unique detail URLs to fetch", file=sys.stderr)

        detail_pages_scraped = 0
        for idx, (url, search_page) in enumerate(unique_links, start=1):
            pid = property_id_from_url(url)

            if args.resume and url in scraped_urls:
                continue

            if not guard.is_allowed(url):
                print(f"  [SKIP] robots.txt disallows: {url}", file=sys.stderr)
                continue

            print(f"[{city}] Fetching detail {idx}/{len(unique_links)}: {pid}", file=sys.stderr)
            html, text, success = fetch_with_retry(page, url, wait_ms=1500)

            if not success:
                print(f"  [ERROR] Failed to load detail page {pid} after retries", file=sys.stderr)
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
                print(f"[{city}] Scraped {idx} detail pages...", file=sys.stderr)

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
    random.seed(42)

    root = project_root_from_here()
    config = load_city_config(Path(args.config))
    guard = RobotsGuard(DOMAIN)

    if args.all_cities:
        unique_urls = {}
        for city, url in config.items():
            if url not in unique_urls.values():
                unique_urls[city] = url
        cities_to_collect = list(unique_urls.keys())
        print(f"Collecting {len(cities_to_collect)} cities: {cities_to_collect}", file=sys.stderr)

        all_summaries = []
        for city in cities_to_collect:
            base_url = config[city]
            outdir = root / "data" / "raw" / "housingcom" / slugify(city)
            print(f"\n=== Collecting {city} ===", file=sys.stderr)
            summary = scrape_city(city, base_url, outdir, args, guard)
            all_summaries.append(summary)
            print(json.dumps(summary, indent=2))

        total_detail = sum(s["detail_pages_scraped"] for s in all_summaries)
        print("\n=== Collection complete ===", file=sys.stderr)
        print(f"Cities: {len(all_summaries)}, Total detail pages: {total_detail}", file=sys.stderr)
    else:
        if args.city not in config:
            raise SystemExit(f"City '{args.city}' not found in {args.config}. Known keys: {sorted(config)}")

        base_url = config[args.city]
        outdir = Path(args.output_dir) if args.output_dir else root / "data" / "raw" / "housingcom" / slugify(args.city)
        summary = scrape_city(args.city, base_url, outdir, args, guard)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
