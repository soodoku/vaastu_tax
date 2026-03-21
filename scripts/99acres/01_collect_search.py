#!/usr/bin/env python3
"""Collect 99acres search result pages.

Paginate through search results and save HTML as .html.gz files.
Outputs search_manifest.jsonl tracking what was collected.

Example
-------
python scripts/99acres/01_collect_search.py --city gurgaon --max-pages 10
python scripts/99acres/01_collect_search.py --all-cities --max-pages 5
python scripts/99acres/01_collect_search.py --city gurgaon --resume
python scripts/99acres/01_collect_search.py --city gurgaon --retry-errors
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

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
    RobotsGuard,
    create_browser_context,
    fetch_with_retry,
    get_proxy,
    jitter_sleep,
    logger,
    setup_logging,
)

DOMAIN_99ACRES = "https://www.99acres.com"
USER_AGENT_99ACRES = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    city_group = parser.add_mutually_exclusive_group(required=True)
    city_group.add_argument(
        "--city", help="City key from data/config/99acres_cities.json"
    )
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
        help="Base directory for raw data. Defaults to data/raw/99acres/",
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


def load_city_config_99acres(path: Path) -> dict[str, dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a JSON object of city -> {{house, flat}} URLs.")
    return data


def city_outdir_99acres(city: str, base_dir: Path | None = None) -> Path:
    if base_dir:
        return base_dir / slugify(city)
    return project_root() / "data" / "raw" / "99acres" / slugify(city)


def add_page_param(base_url: str, page_num: int) -> str:
    if "?" in base_url:
        return f"{base_url}&page={page_num}"
    return f"{base_url}?page={page_num}"


RE_PROPERTY_ID = re.compile(r"spid-([A-Z]?\d{7,})")


def extract_detail_links(hrefs: list[str], root_url: str) -> list[str]:
    from urllib.parse import urljoin, urlparse

    out: list[str] = []
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


def get_scraped_pages(manifest_path: Path) -> dict[tuple[str, str, int], dict]:
    entries = load_manifest(manifest_path)
    result = {}
    for e in entries:
        key = (e.get("city", ""), e.get("property_type", ""), e.get("page", 0))
        result[key] = e
    return result


def collect_search_city(
    city: str,
    city_urls: dict[str, str],
    outdir: Path,
    args: argparse.Namespace,
    guard: RobotsGuard,
) -> dict:
    pages_search = ensure_dir(outdir / "pages" / "search")
    manifest_path = outdir / "search_manifest.jsonl"

    scraped = get_scraped_pages(manifest_path)

    property_types: list[str] = []
    match args.property_type:
        case "house":
            property_types = ["house"]
        case "flat":
            property_types = ["flat"]
        case "both":
            property_types = ["house", "flat"]

    proxy = get_proxy(args.proxy)
    pages_collected = 0

    with sync_playwright() as p:
        browser, _, page = create_browser_context(
            p, headless=args.headless, proxy=proxy, timeout_ms=args.timeout_ms
        )

        for prop_type in property_types:
            if prop_type not in city_urls:
                logger.warning("[%s] No URL for %s, skipping", city, prop_type)
                continue

            base_url = city_urls[prop_type]
            logger.info("[%s] Collecting %s listings", city, prop_type)

            pages_to_collect = []
            for pageno in range(1, args.max_pages + 1):
                key = (city, prop_type, pageno)
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
                logger.info("[%s/%s] No pages to collect", city, prop_type)
                continue

            logger.info("[%s/%s] Collecting %d search pages", city, prop_type, len(pages_to_collect))

            for pageno in pages_to_collect:
                url = add_page_param(base_url, pageno)

                if not guard.is_allowed(url):
                    logger.warning("[%s/%s] Skipping page %d (robots.txt)", city, prop_type, pageno)
                    append_jsonl(
                        manifest_path,
                        [
                            {
                                "city": city,
                                "property_type": prop_type,
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

                logger.info("[%s/%s] Fetching search page %d: %s", city, prop_type, pageno, url)
                html, _, success = fetch_with_retry(page, url, wait_ms=3000)

                if not success:
                    logger.error("[%s/%s] Failed to load search page %d after retries", city, prop_type, pageno)
                    append_jsonl(
                        manifest_path,
                        [
                            {
                                "city": city,
                                "property_type": prop_type,
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

                html_path = pages_search / f"{slugify(city)}_{prop_type}_p{pageno}.html.gz"
                write_html_gz(html_path, html)
                pages_collected += 1

                append_jsonl(
                    manifest_path,
                    [
                        {
                            "city": city,
                            "property_type": prop_type,
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
        "property_types": property_types,
        "pages_collected": pages_collected,
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)
    random.seed(42)

    config = load_city_config_99acres(Path(args.config))

    proxy = get_proxy(args.proxy)
    if proxy:
        proxy_display = proxy.split("@")[-1] if "@" in proxy else proxy
        logger.info("Using proxy: %s", proxy_display)
    else:
        logger.warning("No proxy configured. Set BRIGHT_DATA_PROXY_URL or use --proxy")

    guard = RobotsGuard(DOMAIN_99ACRES, proxy=proxy)

    base_outdir = Path(args.output_dir) if args.output_dir else None

    if args.all_cities:
        cities_to_collect = list(config.keys())
        logger.info("Collecting %d cities: %s", len(cities_to_collect), cities_to_collect[:5])

        all_summaries = []
        for city_key in cities_to_collect:
            city_urls = config[city_key]
            outdir = city_outdir_99acres(city_key, base_outdir)
            ensure_dir(outdir)
            logger.info("=== Collecting %s ===", city_key)
            summary = collect_search_city(city_key, city_urls, outdir, args, guard)
            all_summaries.append(summary)
            print(json.dumps(summary, indent=2))

        total_pages = sum(s["pages_collected"] for s in all_summaries)
        logger.info("=== Collection complete ===")
        logger.info("Cities: %d, Total search pages: %d", len(all_summaries), total_pages)
    else:
        if args.city not in config:
            raise SystemExit(
                f"City '{args.city}' not found in {args.config}. Known keys: {sorted(config)}"
            )

        city_urls = config[args.city]
        outdir = city_outdir_99acres(args.city, base_outdir)
        ensure_dir(outdir)
        summary = collect_search_city(args.city, city_urls, outdir, args, guard)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
