#!/usr/bin/env python3
"""Collect MagicBricks search result pages.

Paginate through search results and save HTML as .html.gz files.
Outputs search_manifest.jsonl tracking what was collected.

Example
-------
python scripts/magicbricks/01_collect_search.py --city delhi-ncr_apartment --max-pages 10
python scripts/magicbricks/01_collect_search.py --all-cities --max-pages 5
python scripts/magicbricks/01_collect_search.py --city delhi-ncr_apartment --resume
python scripts/magicbricks/01_collect_search.py --city delhi-ncr_apartment --retry-errors
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from playwright.sync_api import sync_playwright

from scripts.utils import (
    append_jsonl,
    ensure_dir,
    load_manifest,
    now_iso,
    project_root,
    slugify,
    write_html_gz,
)
from scripts.utils.scraping import (
    DOMAIN,
    RobotsGuard,
    city_outdir,
    create_browser_context,
    fetch_with_retry,
    get_proxy,
    jitter_sleep,
    load_city_config,
    logger,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    city_group = parser.add_mutually_exclusive_group(required=True)
    city_group.add_argument(
        "--city", help="City key from data/config/magicbricks_cities.json"
    )
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
        help="Base directory for raw data. Defaults to data/raw/magicbricks/",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Number of search-result pages to visit",
    )
    parser.add_argument(
        "--min-sleep", type=float, default=2.0, help="Minimum sleep between requests"
    )
    parser.add_argument(
        "--max-sleep", type=float, default=4.0, help="Maximum sleep between requests"
    )
    parser.add_argument(
        "--timeout-ms", type=int, default=60000, help="Playwright page timeout in ms"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium headless (default is headful due to bot detection)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip pages already in search_manifest.jsonl",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-attempt pages with status=error or status=blocked",
    )
    parser.add_argument(
        "--proxy",
        help="Proxy server URL (e.g., http://user:pass@host:port)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress INFO messages, show only warnings and errors",
    )
    parser.add_argument(
        "--log-file",
        help="Write logs to this file in addition to console",
    )
    return parser.parse_args()


def add_page_param(base_url: str, page_num: int) -> str:
    parsed = urlparse(base_url)
    qs = parse_qs(parsed.query)
    qs["page"] = [str(page_num)]
    new_query = urlencode(qs, doseq=True)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"


def extract_detail_links(hrefs: list[str], root_url: str) -> list[str]:
    out: list[str] = []
    for href in hrefs:
        if not href:
            continue
        url = urljoin(root_url, href)
        parsed = urlparse(url)
        if parsed.netloc and "magicbricks.com" not in parsed.netloc:
            continue
        path = parsed.path.lower()
        bad_paths = [
            "/price-trends",
            "/locality-insights",
            "/new-projects",
            "/search/",
            "/advice/",
            "/homeloan/",
            "/accounts",
            "/post.",
            "/help",
            "/blog",
        ]
        if any(bad in path for bad in bad_paths):
            continue
        if "pdpid-" in path:
            if url not in out:
                out.append(url)
    return out


def get_scraped_pages(manifest_path: Path) -> dict[tuple[str, int], dict]:
    entries = load_manifest(manifest_path)
    result = {}
    for e in entries:
        key = (e.get("city", ""), e.get("page", 0))
        result[key] = e
    return result


def collect_search_city(
    city: str,
    base_url: str,
    outdir: Path,
    args: argparse.Namespace,
    guard: RobotsGuard,
) -> dict:
    pages_search = ensure_dir(outdir / "pages" / "search")
    manifest_path = outdir / "search_manifest.jsonl"

    scraped = get_scraped_pages(manifest_path)

    pages_to_collect = []
    for pageno in range(1, args.max_pages + 1):
        key = (city, pageno)
        entry = scraped.get(key)

        if args.retry_errors:
            if entry and entry.get("status") in ("error", "blocked"):
                pages_to_collect.append(pageno)
        elif args.resume:
            if not entry or entry.get("status") not in ("success",):
                pages_to_collect.append(pageno)
        else:
            pages_to_collect.append(pageno)

    if not pages_to_collect:
        logger.info("[%s] No pages to collect", city)
        return {
            "city": city,
            "base_url": base_url,
            "pages_collected": 0,
            "manifest_path": str(manifest_path),
        }

    logger.info("[%s] Collecting %d search pages", city, len(pages_to_collect))

    proxy = get_proxy(args.proxy)
    pages_collected = 0

    with sync_playwright() as p:
        browser, _, page = create_browser_context(
            p, headless=args.headless, proxy=proxy, timeout_ms=args.timeout_ms
        )

        for pageno in pages_to_collect:
            url = add_page_param(base_url, pageno)

            if not guard.is_allowed(url):
                logger.warning("[%s] Skipping page %d (robots.txt)", city, pageno)
                append_jsonl(
                    manifest_path,
                    [
                        {
                            "city": city,
                            "page": pageno,
                            "url": url,
                            "html_path": None,
                            "status": "blocked",
                            "error_msg": "robots.txt disallows",
                            "n_links_found": 0,
                            "collected_at": now_iso(),
                        }
                    ],
                )
                continue

            logger.info("[%s] Fetching search page %d: %s", city, pageno, url)
            html, _, success = fetch_with_retry(page, url, wait_ms=3000)

            if not success:
                logger.error(
                    "[%s] Failed to load search page %d after retries", city, pageno
                )
                append_jsonl(
                    manifest_path,
                    [
                        {
                            "city": city,
                            "page": pageno,
                            "url": url,
                            "html_path": None,
                            "status": "error",
                            "error_msg": "Failed after retries",
                            "n_links_found": 0,
                            "collected_at": now_iso(),
                        }
                    ],
                )
                continue

            hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
            links = extract_detail_links(hrefs, base_url)

            html_path = pages_search / f"{slugify(city)}_p{pageno}.html.gz"
            write_html_gz(html_path, html)
            pages_collected += 1

            append_jsonl(
                manifest_path,
                [
                    {
                        "city": city,
                        "page": pageno,
                        "url": url,
                        "html_path": str(html_path.relative_to(outdir)),
                        "status": "success",
                        "error_msg": None,
                        "n_links_found": len(links),
                        "collected_at": now_iso(),
                    }
                ],
            )

            jitter_sleep(args.min_sleep, args.max_sleep)

        browser.close()

    return {
        "city": city,
        "base_url": base_url,
        "pages_collected": pages_collected,
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)
    random.seed(42)

    config = load_city_config(Path(args.config))

    proxy = get_proxy(args.proxy)
    if proxy:
        proxy_display = proxy.split("@")[-1] if "@" in proxy else proxy
        logger.info("Using proxy: %s", proxy_display)
    else:
        logger.warning("No proxy configured. Set BRIGHT_DATA_PROXY_URL or use --proxy")

    guard = RobotsGuard(DOMAIN, proxy=proxy)

    base_outdir = Path(args.output_dir) if args.output_dir else None

    if args.all_cities:
        cities_to_collect = list(config.keys())
        logger.info(
            "Collecting %d cities: %s",
            len(cities_to_collect),
            cities_to_collect[:5],
        )

        all_summaries = []
        for city_key in cities_to_collect:
            base_url = config[city_key]
            outdir = city_outdir(city_key, base_outdir)
            ensure_dir(outdir)
            logger.info("=== Collecting %s ===", city_key)
            summary = collect_search_city(city_key, base_url, outdir, args, guard)
            all_summaries.append(summary)
            print(json.dumps(summary, indent=2))

        total_pages = sum(s["pages_collected"] for s in all_summaries)
        logger.info("=== Collection complete ===")
        logger.info(
            "Cities: %d, Total search pages: %d", len(all_summaries), total_pages
        )
    else:
        if args.city not in config:
            raise SystemExit(
                f"City '{args.city}' not found in {args.config}. Known keys: {sorted(config)}"
            )

        base_url = config[args.city]
        outdir = city_outdir(args.city, base_outdir)
        ensure_dir(outdir)
        summary = collect_search_city(args.city, base_url, outdir, args, guard)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
