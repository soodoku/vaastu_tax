#!/usr/bin/env python3
"""Collect MagicBricks property detail pages.

Fetch detail pages from detail_urls.jsonl and save HTML as .html.gz files.
Outputs detail_manifest.jsonl tracking what was collected.

Example
-------
python scripts/magicbricks/03_collect_detail.py --city delhi-ncr_apartment --max-pages 500
python scripts/magicbricks/03_collect_detail.py --city delhi-ncr_apartment --resume
python scripts/magicbricks/03_collect_detail.py --city delhi-ncr_apartment --retry-errors
python scripts/magicbricks/03_collect_detail.py --all-cities --max-pages 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from playwright.sync_api import sync_playwright  # noqa: E402

from scripts.utils import (  # noqa: E402
    append_jsonl,
    ensure_dir,
    load_manifest,
    now_iso,
    project_root,
    slugify,
    write_html_gz,
)
from scripts.utils.scraping import (  # noqa: E402
    DOMAIN,
    RobotsGuard,
    create_browser_context,
    fetch_with_retry,
    get_proxy,
    jitter_sleep,
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
        help="Collect detail pages for all city directories",
    )
    parser.add_argument(
        "--data-dir",
        default=str(root / "data" / "raw" / "magicbricks"),
        help="Base directory containing city subdirectories",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=250,
        help="Maximum number of detail pages to fetch",
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
        help="Skip URLs already in detail_manifest.jsonl",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-attempt URLs with status=error or status=blocked",
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


def get_collected_urls(manifest_path: Path) -> dict[str, dict]:
    entries = load_manifest(manifest_path)
    result = {}
    for e in entries:
        prop_id = e.get("property_id", "")
        if prop_id:
            result[prop_id] = e
    return result


def collect_detail_city(
    city: str,
    city_dir: Path,
    args: argparse.Namespace,
    guard: RobotsGuard,
) -> dict:
    detail_urls_path = city_dir / "detail_urls.jsonl"
    detail_manifest_path = city_dir / "detail_manifest.jsonl"
    pages_detail = ensure_dir(city_dir / "pages" / "detail")

    urls_to_fetch = load_manifest(detail_urls_path)
    if not urls_to_fetch:
        logger.warning("[%s] No URLs found in detail_urls.jsonl", city)
        return {"city": city, "pages_collected": 0}

    collected = get_collected_urls(detail_manifest_path)

    to_collect: list[dict] = []
    for url_entry in urls_to_fetch:
        prop_id = url_entry.get("property_id", "")
        existing = collected.get(prop_id)

        if args.retry_errors:
            if existing and existing.get("status") in ("error", "blocked"):
                to_collect.append(url_entry)
        elif args.resume:
            if not existing or existing.get("status") not in ("success",):
                to_collect.append(url_entry)
        else:
            to_collect.append(url_entry)

    if args.max_pages > 0:
        to_collect = to_collect[: args.max_pages]

    if not to_collect:
        logger.info("[%s] No detail pages to collect", city)
        return {
            "city": city,
            "pages_collected": 0,
            "manifest_path": str(detail_manifest_path),
        }

    logger.info("[%s] Collecting %d detail pages", city, len(to_collect))

    proxy = get_proxy(args.proxy)
    pages_collected = 0

    with sync_playwright() as p:
        browser, _, page = create_browser_context(
            p, headless=args.headless, proxy=proxy, timeout_ms=args.timeout_ms
        )

        for idx, url_entry in enumerate(to_collect, start=1):
            prop_id = url_entry.get("property_id", "")
            url = url_entry.get("url", "")

            if not guard.is_allowed(url):
                logger.warning("[%s] Skipping %s (robots.txt)", city, prop_id)
                append_jsonl(
                    detail_manifest_path,
                    [
                        {
                            "city": city,
                            "property_id": prop_id,
                            "url": url,
                            "html_path": None,
                            "status": "blocked",
                            "error_msg": "robots.txt disallows",
                            "retry_count": 0,
                            "collected_at": now_iso(),
                        }
                    ],
                )
                continue

            logger.info(
                "[%s] Fetching detail %d/%d: %s", city, idx, len(to_collect), prop_id
            )
            html, _, success = fetch_with_retry(page, url, wait_ms=2000)

            if not success:
                logger.error("[%s] Failed to load detail page %s", city, prop_id)
                existing = collected.get(prop_id, {})
                retry_count = existing.get("retry_count", 0) + 1
                append_jsonl(
                    detail_manifest_path,
                    [
                        {
                            "city": city,
                            "property_id": prop_id,
                            "url": url,
                            "html_path": None,
                            "status": "error",
                            "error_msg": "Failed after retries",
                            "retry_count": retry_count,
                            "collected_at": now_iso(),
                        }
                    ],
                )
                continue

            html_path = pages_detail / f"{slugify(city)}_{prop_id}.html.gz"
            write_html_gz(html_path, html)
            pages_collected += 1

            append_jsonl(
                detail_manifest_path,
                [
                    {
                        "city": city,
                        "property_id": prop_id,
                        "url": url,
                        "html_path": str(html_path.relative_to(city_dir)),
                        "status": "success",
                        "error_msg": None,
                        "retry_count": 0,
                        "collected_at": now_iso(),
                    }
                ],
            )

            if idx % 25 == 0:
                logger.info("[%s] Collected %d detail pages...", city, idx)

            jitter_sleep(args.min_sleep, args.max_sleep)

        browser.close()

    return {
        "city": city,
        "pages_collected": pages_collected,
        "manifest_path": str(detail_manifest_path),
    }


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)
    random.seed(42)

    data_dir = Path(args.data_dir)

    proxy = get_proxy(args.proxy)
    if proxy:
        proxy_display = proxy.split("@")[-1] if "@" in proxy else proxy
        logger.info("Using proxy: %s", proxy_display)
    else:
        logger.warning("No proxy configured. Set BRIGHT_DATA_PROXY_URL or use --proxy")

    guard = RobotsGuard(DOMAIN, proxy=proxy)

    if args.all_cities:
        if not data_dir.exists():
            raise SystemExit(f"Data directory not found: {data_dir}")

        city_dirs = [
            d
            for d in data_dir.iterdir()
            if d.is_dir() and (d / "detail_urls.jsonl").exists()
        ]
        logger.info("Collecting detail pages for %d cities", len(city_dirs))

        all_summaries = []
        for city_dir in sorted(city_dirs):
            city = city_dir.name
            logger.info("=== Collecting %s ===", city)
            summary = collect_detail_city(city, city_dir, args, guard)
            all_summaries.append(summary)
            print(json.dumps(summary, indent=2))

        total_pages = sum(s["pages_collected"] for s in all_summaries)
        logger.info("=== Collection complete ===")
        logger.info(
            "Cities: %d, Total detail pages: %d", len(all_summaries), total_pages
        )
    else:
        city_dir = data_dir / slugify(args.city)
        if not city_dir.exists():
            raise SystemExit(f"City directory not found: {city_dir}")

        summary = collect_detail_city(args.city, city_dir, args, guard)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
