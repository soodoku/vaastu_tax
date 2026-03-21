#!/usr/bin/env python3
"""Parse Housing.com detail HTML files into structured Parquet.

Reads detail HTML files and extracts property listings.
Outputs parsed_listings.parquet with zstd compression.

Example
-------
python scripts/housingcom/04_parse.py --city hyderabad
python scripts/housingcom/04_parse.py --all-cities
python scripts/housingcom/04_parse.py --city hyderabad --force
"""

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from bs4 import BeautifulSoup

from scripts.utils import (
    RE_BALCONY,
    RE_BATH,
    RE_BHK,
    RE_FACING,
    RE_FURNISHING,
    RE_LAST_UPDATED,
    RE_POSSESSION,
    RE_PPSF,
    RE_SHOWING,
    RE_SQFT,
    extract_lines,
    extract_vaastu_mentions,
    find_first_matching_line,
    find_price_display,
    load_manifest,
    normalize_direction,
    normalize_ws,
    number_from_match,
    price_to_crore,
    project_root,
    read_html_gz,
    slugify,
)
from scripts.utils.scraping import logger, setup_logging

SEARCH_STOP_HEADS = {
    "property location",
    "amenities",
    "about this property",
    "overview",
    "project details",
    "nearby places",
    "specifications",
    "furnishing",
    "configuration",
    "ratings",
    "locality",
    "calculator",
    "sale and rent trends",
    "home loan offers",
    "similar properties",
    "ratings by features",
    "what is good here",
    "what can be better",
    "reviews by residents",
    "faq",
    "read more",
}


@dataclass
class ListingRecord:
    property_id: str
    url: str
    city: str
    source_page: int
    title: str | None
    locality_line: str | None
    price_display: str | None
    price_crore: float | None
    builtup_area_sqft: float | None
    avg_price_per_sqft: float | None
    bhk: float | None
    bathrooms: float | None
    balconies: float | None
    furnishing: str | None
    facing: str | None
    possession_status: str | None
    special_highlights: str | None
    amenities: str | None
    about_this_property: str | None
    vaastu_mentioned: int
    vaastu_mentions_text: str | None
    last_updated: str | None
    collected_at: str | None
    raw_html_path: str


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    city_group = parser.add_mutually_exclusive_group(required=True)
    city_group.add_argument("--city", help="City key to parse")
    city_group.add_argument(
        "--all-cities",
        action="store_true",
        help="Parse data for all cities found in data/raw/housingcom/",
    )
    parser.add_argument(
        "--data-dir",
        default=str(root / "data" / "raw" / "housingcom"),
        help="Base directory containing city subdirectories",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-parse all files even if already in parsed_listings.parquet",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip property_ids already in parquet (default behavior)",
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


RE_TITLE = re.compile(r"^(\d+(?:\.\d+)?)\s*BHK\s+.+?(Independent House|Villa|Row House)\b", re.I)


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.body:
        return soup.body.get_text(separator="\n", strip=True)
    return soup.get_text(separator="\n", strip=True)


def find_title(lines: Sequence[str]) -> str | None:
    for line in lines:
        if RE_TITLE.search(line):
            return line
    return None


def find_locality_line(lines: Sequence[str], title: str | None) -> str | None:
    if not title:
        return None
    try:
        idx = list(lines).index(title)
    except ValueError:
        return None
    for cand in lines[idx + 1 : idx + 6]:
        lower = cand.lower()
        if not cand:
            continue
        if "₹" in cand:
            continue
        if RE_SHOWING.search(cand):
            continue
        if any(tok in lower for tok in [
            "emi", "sq.ft", "special highlights", "amenities", "about this property"
        ]):
            continue
        return cand
    return None


def extract_section_housing(lines: Sequence[str], heading: str) -> str | None:
    heading_l = heading.lower().strip()
    start = None
    for idx, line in enumerate(lines):
        if line.lower() == heading_l:
            start = idx + 1
            break
    if start is None:
        return None

    buffer: list[str] = []
    for line in lines[start:]:
        line_l = line.lower().strip()
        if line_l in SEARCH_STOP_HEADS:
            break
        buffer.append(line)
    out = normalize_ws(" | ".join(buffer))
    return out or None


def parse_detail_text(
    text: str,
    url: str,
    city: str,
    property_id: str,
    source_page: int,
    raw_html_path: str,
    collected_at: str | None = None,
) -> ListingRecord:
    lines = extract_lines(text)
    title = find_title(lines)
    title_idx = lines.index(title) if title in lines else None
    locality_line = find_locality_line(lines, title)
    price_display = find_price_display(lines, title_idx)
    price_crore = price_to_crore(price_display)

    title_and_body = "\n".join(lines)
    bhk = number_from_match(RE_BHK, title or title_and_body)
    bathrooms = number_from_match(RE_BATH, title_and_body)
    balconies = number_from_match(RE_BALCONY, title_and_body)
    builtup_area_sqft = number_from_match(RE_SQFT, title_and_body)
    avg_price_per_sqft = number_from_match(RE_PPSF, title_and_body)

    facing_line = find_first_matching_line(lines, RE_FACING)
    furnishing_line = find_first_matching_line(lines, RE_FURNISHING)
    possession_line = find_first_matching_line(lines, RE_POSSESSION)

    facing = None
    if facing_line:
        m = RE_FACING.search(facing_line)
        if m:
            facing = normalize_direction(m.group(1) or m.group(2))

    furnishing = None
    if furnishing_line:
        m = RE_FURNISHING.search(furnishing_line)
        furnishing = normalize_ws(m.group(1)) if m else None

    possession_status = None
    if possession_line:
        m = RE_POSSESSION.search(possession_line)
        possession_status = normalize_ws(m.group(1)) if m else None

    special_highlights = extract_section_housing(lines, "Special Highlights")
    amenities = extract_section_housing(lines, "Amenities")
    about_this_property = extract_section_housing(lines, "About this property")

    combined_text = " ".join(filter(None, [title, special_highlights, amenities, about_this_property]))
    vaastu_mentioned, vaastu_mentions_text = extract_vaastu_mentions(combined_text)

    last_updated = None
    for line in lines:
        m = RE_LAST_UPDATED.search(line)
        if m:
            last_updated = normalize_ws(m.group(1))
            break

    return ListingRecord(
        property_id=property_id,
        url=url,
        city=city,
        source_page=source_page,
        title=title,
        locality_line=locality_line,
        price_display=price_display,
        price_crore=price_crore,
        builtup_area_sqft=builtup_area_sqft,
        avg_price_per_sqft=avg_price_per_sqft,
        bhk=bhk,
        bathrooms=bathrooms,
        balconies=balconies,
        furnishing=furnishing,
        facing=facing,
        possession_status=possession_status,
        special_highlights=special_highlights,
        amenities=amenities,
        about_this_property=about_this_property,
        vaastu_mentioned=int(vaastu_mentioned),
        vaastu_mentions_text=vaastu_mentions_text,
        last_updated=last_updated,
        collected_at=collected_at,
        raw_html_path=raw_html_path,
    )


def read_existing_property_ids(parquet_path: Path) -> set[str]:
    if not parquet_path.exists():
        return set()
    try:
        df = pd.read_parquet(parquet_path, columns=["property_id"])
        return set(df["property_id"].dropna().astype(str))
    except Exception:
        return set()


def parse_city(city: str, city_dir: Path, force: bool) -> dict:
    detail_manifest_path = city_dir / "detail_manifest.jsonl"
    parsed_parquet = city_dir / "parsed_listings.parquet"

    if not detail_manifest_path.exists():
        logger.warning("[%s] No detail_manifest.jsonl found", city)
        return {"city": city, "parsed": 0, "skipped": 0, "errors": 0}

    entries = load_manifest(detail_manifest_path)
    successful_entries = [e for e in entries if e.get("status") == "success"]

    if not successful_entries:
        logger.warning("[%s] No successful detail pages found", city)
        return {"city": city, "parsed": 0, "skipped": 0, "errors": 0}

    existing_ids = set() if force else read_existing_property_ids(parsed_parquet)

    parsed_rows: list[dict] = []
    skipped = 0
    errors = 0

    for entry in successful_entries:
        property_id = entry.get("property_id", "")
        if property_id in existing_ids:
            skipped += 1
            continue

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

        text = extract_text_from_html(html)

        try:
            record = parse_detail_text(
                text=text,
                url=entry.get("url", ""),
                city=city,
                property_id=property_id,
                source_page=entry.get("source_page", 0),
                raw_html_path=entry.get("html_path", ""),
                collected_at=entry.get("collected_at"),
            )
            parsed_rows.append(asdict(record))
            existing_ids.add(property_id)
        except Exception as e:
            logger.warning("[%s] Error parsing %s: %s", city, property_id, e)
            errors += 1
            continue

    if parsed_rows:
        if force or not parsed_parquet.exists():
            df = pd.DataFrame(parsed_rows)
            df.to_parquet(parsed_parquet, index=False, compression="zstd")
        else:
            existing_df = (
                pd.read_parquet(parsed_parquet)
                if parsed_parquet.exists()
                else pd.DataFrame()
            )
            new_df = pd.DataFrame(parsed_rows)
            all_df = pd.concat([existing_df, new_df], ignore_index=True)
            all_df.to_parquet(parsed_parquet, index=False, compression="zstd")

    logger.info(
        "[%s] Parsed: %d listings, Skipped: %d, Errors: %d",
        city,
        len(parsed_rows),
        skipped,
        errors,
    )

    return {
        "city": city,
        "parsed": len(parsed_rows),
        "skipped": skipped,
        "errors": errors,
        "output_parquet": str(parsed_parquet),
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
            if d.is_dir() and (d / "detail_manifest.jsonl").exists()
        ]
        logger.info("Parsing %d cities", len(city_dirs))

        all_summaries = []
        for city_dir in sorted(city_dirs):
            city = city_dir.name
            logger.info("=== Parsing %s ===", city)
            summary = parse_city(city, city_dir, args.force)
            all_summaries.append(summary)
            print(json.dumps(summary, indent=2))

        total_parsed = sum(s["parsed"] for s in all_summaries)
        total_errors = sum(s["errors"] for s in all_summaries)
        logger.info("=== Parsing complete ===")
        logger.info(
            "Cities: %d, Parsed: %d, Errors: %d",
            len(all_summaries),
            total_parsed,
            total_errors,
        )
    else:
        city_dir = data_dir / slugify(args.city)
        if not city_dir.exists():
            raise SystemExit(f"City directory not found: {city_dir}")

        summary = parse_city(args.city, city_dir, args.force)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
