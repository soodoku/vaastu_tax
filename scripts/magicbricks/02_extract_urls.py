#!/usr/bin/env python3
"""Extract detail page URLs from collected search HTML pages.

Parses search HTML files to extract property detail URLs.
Outputs detail_urls.jsonl with URL metadata.

Example
-------
python scripts/magicbricks/02_extract_urls.py --city delhi-ncr_apartment
python scripts/magicbricks/02_extract_urls.py --all-cities
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bs4 import BeautifulSoup

from scripts.utils import (
    load_manifest,
    now_iso,
    project_root,
    read_html_gz,
    slugify,
)
from scripts.utils.scraping import (
    city_outdir,
    logger,
    setup_logging,
)

RE_PROPERTY_ID = re.compile(
    r"pdpid-([a-zA-Z0-9]+)|(?:id[=:]|propertyid[=:]|/id/)(\d+)", re.I
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


def property_id_from_url(url: str) -> str:
    m = RE_PROPERTY_ID.search(url)
    if m:
        return m.group(1) or m.group(2)
    parsed = urlparse(url)
    from urllib.parse import parse_qs

    qs = parse_qs(parsed.query)
    if "id" in qs:
        return qs["id"][0]
    stem = Path(parsed.path).name
    return slugify(stem)


def extract_detail_links_from_html(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []

    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if not href:
            continue
        url = urljoin(base_url, href)
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


def extract_urls_city(city: str, city_dir: Path) -> dict:
    search_manifest_path = city_dir / "search_manifest.jsonl"
    detail_urls_path = city_dir / "detail_urls.jsonl"

    entries = load_manifest(search_manifest_path)
    successful_entries = [e for e in entries if e.get("status") == "success"]

    if not successful_entries:
        logger.warning("[%s] No successful search pages found", city)
        return {"city": city, "urls_extracted": 0, "unique_properties": 0}

    all_urls: list[dict] = []
    seen_property_ids: set[str] = set()

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
        links = extract_detail_links_from_html(html, base_url)

        page_num = entry.get("page", 0)
        for url in links:
            prop_id = property_id_from_url(url)
            if prop_id in seen_property_ids:
                continue
            seen_property_ids.add(prop_id)

            all_urls.append(
                {
                    "city": city,
                    "property_id": prop_id,
                    "url": url,
                    "source_page": page_num,
                    "extracted_at": now_iso(),
                }
            )

    with open(detail_urls_path, "w", encoding="utf-8") as fh:
        for row in all_urls:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(
        "[%s] Extracted %d unique URLs from %d search pages",
        city,
        len(all_urls),
        len(successful_entries),
    )

    return {
        "city": city,
        "search_pages_processed": len(successful_entries),
        "urls_extracted": len(all_urls),
        "output_path": str(detail_urls_path),
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

        total_urls = sum(s["urls_extracted"] for s in all_summaries)
        logger.info("=== Extraction complete ===")
        logger.info("Cities: %d, Total URLs: %d", len(all_summaries), total_urls)
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
