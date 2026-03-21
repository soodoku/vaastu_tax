#!/usr/bin/env python3
"""Extract URLs from collected search HTML pages.

Parses search HTML files to extract two types of URLs:
1. Project URLs (pdpid-) -> project_urls.jsonl
2. Listing URLs (/propertyDetails/) -> listing_urls.jsonl

Example
-------
python scripts/magicbricks/02_extract_urls.py --city delhi-ncr_apartment
python scripts/magicbricks/02_extract_urls.py --all-cities
"""

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bs4 import BeautifulSoup  # noqa: E402

from scripts.utils import (load_manifest, now_iso, project_root,  # noqa: E402
                           read_html_gz, slugify)
from scripts.utils.scraping import (city_outdir, logger,  # noqa: E402
                                    setup_logging)

RE_PDPID = re.compile(r"pdpid-([a-zA-Z0-9]+)", re.I)
RE_PROPERTY_DETAILS = re.compile(r"/propertyDetails/([^/?]+)", re.I)


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
        help="Extract URLs for all city directories",
    )
    parser.add_argument(
        "--data-dir",
        default=str(root / "data" / "raw" / "magicbricks"),
        help="Base directory containing city subdirectories",
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
    return parser.parse_args()


def extract_project_id_from_url(url: str) -> str | None:
    m = RE_PDPID.search(url)
    if m:
        return m.group(1)
    return None


def extract_listing_id_from_url(url: str) -> str | None:
    m = RE_PROPERTY_DETAILS.search(url)
    if m:
        slug = m.group(1)
        id_match = re.search(r"-(\d+)$", slug)
        if id_match:
            return id_match.group(1)
        return slugify(slug)
    return None


def extract_urls_from_html(html: str, base_url: str) -> tuple[list[str], list[str]]:
    """Extract project URLs (pdpid-) and listing URLs (/propertyDetails/) from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    project_urls: list[str] = []
    listing_urls: list[str] = []

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

    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if not href:
            continue
        url = urljoin(base_url, href)
        parsed = urlparse(url)
        if parsed.netloc and "magicbricks.com" not in parsed.netloc:
            continue
        path = parsed.path.lower()
        if any(bad in path for bad in bad_paths):
            continue

        if "pdpid-" in path:
            if url not in project_urls:
                project_urls.append(url)
        elif "/propertydetails/" in path:
            if url not in listing_urls:
                listing_urls.append(url)

    return project_urls, listing_urls


def extract_urls_city(city: str, city_dir: Path) -> dict:
    search_manifest_path = city_dir / "search_manifest.jsonl"
    project_urls_path = city_dir / "project_urls.jsonl"
    listing_urls_path = city_dir / "listing_urls.jsonl"

    entries = load_manifest(search_manifest_path)
    successful_entries = [e for e in entries if e.get("status") == "success"]

    if not successful_entries:
        logger.warning("[%s] No successful search pages found", city)
        return {
            "city": city,
            "project_urls_extracted": 0,
            "listing_urls_extracted": 0,
        }

    all_project_urls: list[dict] = []
    all_listing_urls: list[dict] = []
    seen_project_ids: set[str] = set()
    seen_listing_ids: set[str] = set()

    for entry in successful_entries:
        html_path_rel = entry.get("html_path")
        if not html_path_rel:
            continue

        html_path = city_dir / html_path_rel
        if not html_path.exists():
            logger.warning("[%s] HTML file not found: %s", city, html_path)
            continue

        try:
            html = read_html_gz(html_path)
        except Exception as e:
            logger.warning("[%s] Error reading %s: %s", city, html_path, e)
            continue

        base_url = entry.get("url", "https://www.magicbricks.com")
        project_urls, listing_urls = extract_urls_from_html(html, base_url)

        page_num = entry.get("page", 0)

        for url in project_urls:
            prop_id = extract_project_id_from_url(url)
            if not prop_id or prop_id in seen_project_ids:
                continue
            seen_project_ids.add(prop_id)

            all_project_urls.append(
                {
                    "city": city,
                    "property_id": prop_id,
                    "url": url,
                    "source_page": page_num,
                    "extracted_at": now_iso(),
                }
            )

        for url in listing_urls:
            listing_id = extract_listing_id_from_url(url)
            if not listing_id or listing_id in seen_listing_ids:
                continue
            seen_listing_ids.add(listing_id)

            all_listing_urls.append(
                {
                    "city": city,
                    "property_id": listing_id,
                    "url": url,
                    "source": "search",
                    "source_page": page_num,
                    "extracted_at": now_iso(),
                }
            )

    with open(project_urls_path, "w", encoding="utf-8") as fh:
        for row in all_project_urls:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(listing_urls_path, "w", encoding="utf-8") as fh:
        for row in all_listing_urls:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(
        "[%s] Extracted %d project URLs and %d listing URLs from %d search pages",
        city,
        len(all_project_urls),
        len(all_listing_urls),
        len(successful_entries),
    )

    return {
        "city": city,
        "search_pages_processed": len(successful_entries),
        "project_urls_extracted": len(all_project_urls),
        "listing_urls_extracted": len(all_listing_urls),
        "project_urls_path": str(project_urls_path),
        "listing_urls_path": str(listing_urls_path),
    }


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    data_dir = Path(args.data_dir)

    if args.all_cities:
        if not data_dir.exists():
            raise SystemExit(f"Data directory not found: {data_dir}")

        city_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        logger.info("Extracting URLs for %d cities", len(city_dirs))

        all_summaries = []
        for city_dir in sorted(city_dirs):
            city = city_dir.name
            logger.info("=== Extracting %s ===", city)
            summary = extract_urls_city(city, city_dir)
            all_summaries.append(summary)
            print(json.dumps(summary, indent=2))

        total_project = sum(s["project_urls_extracted"] for s in all_summaries)
        total_listing = sum(s["listing_urls_extracted"] for s in all_summaries)
        logger.info("=== Extraction complete ===")
        logger.info(
            "Cities: %d, Project URLs: %d, Listing URLs: %d",
            len(all_summaries),
            total_project,
            total_listing,
        )
    else:
        city_dir = city_outdir(args.city, Path(args.data_dir).parent)
        if args.data_dir:
            city_dir = Path(args.data_dir) / slugify(args.city)

        if not city_dir.exists():
            raise SystemExit(f"City directory not found: {city_dir}")

        summary = extract_urls_city(args.city, city_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
