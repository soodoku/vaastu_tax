#!/usr/bin/env python3
"""Extract detail page URLs from collected 99acres search HTML pages.

Parses search HTML files to extract property detail URLs.
Outputs detail_urls.jsonl with URL metadata.

Example
-------
python scripts/99acres/02_extract_urls.py --city gurgaon
python scripts/99acres/02_extract_urls.py --all-cities
"""

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
    logger,
    setup_logging,
)

RE_PROPERTY_ID = re.compile(r"spid-([A-Z]?\d{7,})")


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
        help="Extract URLs for all city directories",
    )
    parser.add_argument(
        "--data-dir",
        default=str(root / "data" / "raw" / "99acres"),
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
        return m.group(1)
    stem = Path(urlparse(url).path).name
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

        base_url = entry.get("url", "https://www.99acres.com")
        links = extract_detail_links_from_html(html, base_url)

        page_num = entry.get("page", 0)
        property_type = entry.get("property_type", "")
        for url in links:
            prop_id = property_id_from_url(url)
            if prop_id in seen_property_ids:
                continue
            seen_property_ids.add(prop_id)

            all_urls.append(
                {
                    "city": city,
                    "property_id": prop_id,
                    "property_type": property_type,
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
        city_dir = data_dir / slugify(args.city)
        if not city_dir.exists():
            raise SystemExit(f"City directory not found: {city_dir}")

        summary = extract_urls_city(args.city, city_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
