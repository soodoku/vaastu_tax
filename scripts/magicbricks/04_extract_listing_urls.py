#!/usr/bin/env python3
"""Extract individual listing URLs from collected project HTML pages.

Reads project HTML files, extracts SERVER_PRELOADED_STATE_ JSON,
and constructs /propertyDetails/ URLs for each listing.
Appends to listing_urls.jsonl (deduplicating by property_id).

Example
-------
python scripts/magicbricks/04_extract_listing_urls.py --city delhi-ncr_apartment
python scripts/magicbricks/04_extract_listing_urls.py --all-cities
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bs4 import BeautifulSoup  # noqa: E402

from scripts.utils import (load_manifest, now_iso, project_root,  # noqa: E402
                           read_html_gz, slugify)
from scripts.utils.scraping import (city_outdir, logger,  # noqa: E402
                                    setup_logging)

BASE_LISTING_URL = "https://www.magicbricks.com/propertyDetails/"


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
        help="Extract listing URLs for all city directories",
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


def extract_preloaded_state(html: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")

    for script in soup.find_all("script"):
        text = script.string or ""
        if "window.SERVER_PRELOADED_STATE_" in text:
            start = text.find("window.SERVER_PRELOADED_STATE_ = ")
            if start >= 0:
                json_start = start + len("window.SERVER_PRELOADED_STATE_ = ")
                json_text = text[json_start:].strip()
                if json_text.endswith(";"):
                    json_text = json_text[:-1]
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    return None
    return None


def extract_listings_from_project(data: dict, project_id: str, city: str) -> list[dict]:
    """Extract listing URLs from project JSON data."""
    listings: list[dict] = []

    bhk_details = (
        data.get("projectPageSeoStaticData", {})
        .get("bhkDetailsDTO", {})
        .get("bhkProjectDetailsMap", {})
    )

    all_data = bhk_details.get("ALL", {})
    sale_listings = all_data.get("groupedResult") or []
    rent_listings = all_data.get("groupedRentResult") or []

    for transaction_type, listing_list in [
        ("sale", sale_listings),
        ("rent", rent_listings),
    ]:
        for listing in listing_list:
            prop_id = listing.get("id")
            url_slug = listing.get("url")
            if not prop_id:
                continue

            if url_slug:
                full_url = BASE_LISTING_URL + url_slug
            else:
                continue

            listings.append(
                {
                    "property_id": str(prop_id),
                    "url": full_url,
                    "transaction_type": transaction_type,
                    "project_id": project_id,
                    "city": city,
                }
            )

    return listings


def load_existing_listing_ids(listing_urls_path: Path) -> set[str]:
    """Load existing property IDs from listing_urls.jsonl."""
    existing_ids: set[str] = set()
    if listing_urls_path.exists():
        entries = load_manifest(listing_urls_path)
        for e in entries:
            prop_id = e.get("property_id", "")
            if prop_id:
                existing_ids.add(prop_id)
    return existing_ids


def extract_listing_urls_city(city: str, city_dir: Path) -> dict:
    project_manifest_path = city_dir / "project_manifest.jsonl"
    listing_urls_path = city_dir / "listing_urls.jsonl"

    if not project_manifest_path.exists():
        logger.warning("[%s] No project_manifest.jsonl found", city)
        return {"city": city, "listings_extracted": 0}

    entries = load_manifest(project_manifest_path)
    successful_entries = [e for e in entries if e.get("status") == "success"]

    if not successful_entries:
        logger.warning("[%s] No successful project pages found", city)
        return {"city": city, "listings_extracted": 0}

    existing_ids = load_existing_listing_ids(listing_urls_path)
    logger.info("[%s] Found %d existing listing IDs", city, len(existing_ids))

    new_listings: list[dict] = []
    projects_processed = 0
    errors = 0

    for entry in successful_entries:
        project_id = entry.get("property_id", "")
        raw_path = entry.get("html_path", "")
        html_path = city_dir / raw_path

        if not html_path.exists():
            logger.warning("[%s] HTML file not found: %s", city, html_path)
            errors += 1
            continue

        try:
            html = read_html_gz(html_path)
        except Exception as e:
            logger.warning("[%s] Error reading %s: %s", city, html_path, e)
            errors += 1
            continue

        try:
            data = extract_preloaded_state(html)
            if not data:
                logger.debug("[%s] No SERVER_PRELOADED_STATE_ in %s", city, project_id)
                errors += 1
                continue

            listings = extract_listings_from_project(data, project_id, city)
            projects_processed += 1

            for listing in listings:
                if listing["property_id"] not in existing_ids:
                    listing["source"] = "project"
                    listing["source_project"] = project_id
                    listing["extracted_at"] = now_iso()
                    new_listings.append(listing)
                    existing_ids.add(listing["property_id"])

        except Exception as e:
            logger.warning("[%s] Error parsing %s: %s", city, project_id, e)
            errors += 1
            continue

    if new_listings:
        with open(listing_urls_path, "a", encoding="utf-8") as fh:
            for row in new_listings:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(
        "[%s] Extracted %d new listing URLs from %d projects (errors: %d)",
        city,
        len(new_listings),
        projects_processed,
        errors,
    )

    return {
        "city": city,
        "projects_processed": projects_processed,
        "listings_extracted": len(new_listings),
        "errors": errors,
        "listing_urls_path": str(listing_urls_path),
    }


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    data_dir = Path(args.data_dir)

    if args.all_cities:
        if not data_dir.exists():
            raise SystemExit(f"Data directory not found: {data_dir}")

        city_dirs = [
            d
            for d in data_dir.iterdir()
            if d.is_dir() and (d / "project_manifest.jsonl").exists()
        ]
        logger.info("Extracting listing URLs for %d cities", len(city_dirs))

        all_summaries = []
        for city_dir in sorted(city_dirs):
            city = city_dir.name
            logger.info("=== Extracting %s ===", city)
            summary = extract_listing_urls_city(city, city_dir)
            all_summaries.append(summary)
            print(json.dumps(summary, indent=2))

        total_listings = sum(s["listings_extracted"] for s in all_summaries)
        logger.info("=== Extraction complete ===")
        logger.info(
            "Cities: %d, Total listing URLs: %d", len(all_summaries), total_listings
        )
    else:
        city_dir = city_outdir(args.city, Path(args.data_dir).parent)
        if args.data_dir:
            city_dir = Path(args.data_dir) / slugify(args.city)

        if not city_dir.exists():
            raise SystemExit(f"City directory not found: {city_dir}")

        summary = extract_listing_urls_city(args.city, city_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
