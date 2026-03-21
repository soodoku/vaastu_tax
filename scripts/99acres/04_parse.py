#!/usr/bin/env python3
"""Parse 99acres detail HTML files into structured Parquet.

Reads detail HTML files and extracts property listings.
Outputs parsed_listings.parquet with zstd compression.

Example
-------
python scripts/99acres/04_parse.py --city gurgaon
python scripts/99acres/04_parse.py --all-cities
python scripts/99acres/04_parse.py --city gurgaon --force
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from bs4 import BeautifulSoup

from scripts.utils import (
    RE_AMENITIES_SECTION,
    RE_BALCONY,
    RE_BATH,
    RE_BHK,
    RE_FACING,
    RE_FLOOR,
    RE_FURNISHING,
    RE_LAST_UPDATED,
    RE_POSSESSION,
    RE_POSTED_DATE,
    RE_PROPERTY_AGE,
    RE_SELLER_TYPE,
    RE_SQFT,
    extract_lines,
    extract_next_data,
    extract_vaastu_mentions,
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

RE_PRICE_UNIT = re.compile(r"₹?\s*([0-9][\d,]*(?:\.\d+)?)\s*(Cr|Crore|L|Lac|Lakh|Lakhs|K)?\b", re.I)


@dataclass
class ListingRecord:
    property_id: str
    url: str
    city: str
    property_type: str
    source_page: int
    title: str | None
    locality: str | None
    price_display: str | None
    price_crore: float | None
    builtup_area_sqft: float | None
    carpet_area_sqft: float | None
    bhk: float | None
    bathrooms: float | None
    balconies: float | None
    furnishing: str | None
    facing: str | None
    possession_status: str | None
    floor_number: int | None
    total_floors: int | None
    property_age: str | None
    amenities: str | None
    description: str | None
    vaastu_mentioned: int
    vaastu_mentions_text: str | None
    seller_type: str | None
    posted_date: str | None
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
        help="Parse data for all cities found in data/raw/99acres/",
    )
    parser.add_argument(
        "--data-dir",
        default=str(root / "data" / "raw" / "99acres"),
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


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.body:
        return soup.body.get_text(separator="\n", strip=True)
    return soup.get_text(separator="\n", strip=True)


def parse_detail_from_next_data(
    next_data: dict,
    url: str,
    city: str,
    property_type: str,
    property_id: str,
    source_page: int,
    raw_html_path: str,
    collected_at: str | None = None,
) -> ListingRecord | None:
    try:
        props = next_data.get("props", {})
        page_props = props.get("pageProps", {})
        listing_data = page_props.get("listingData", {}) or page_props.get("propertyData", {}) or {}
        if not listing_data:
            return None

        title = listing_data.get("title") or listing_data.get("heading")
        locality = listing_data.get("locality") or listing_data.get("localityName")
        price_display = listing_data.get("price") or listing_data.get("priceDisplay")
        price_crore_val = None
        if isinstance(price_display, (int, float)):
            price_crore_val = price_display / 10000000.0
        elif isinstance(price_display, str):
            price_crore_val = price_to_crore(price_display)

        builtup = listing_data.get("builtUpArea") or listing_data.get("superBuiltupArea")
        carpet = listing_data.get("carpetArea")
        bhk = listing_data.get("bhk") or listing_data.get("bedroom")
        bathrooms = listing_data.get("bathroom")
        balconies = listing_data.get("balcony")
        furnishing = listing_data.get("furnishing")
        facing = listing_data.get("facing")
        floor_num = listing_data.get("floor") or listing_data.get("floorNumber")
        total_floors = listing_data.get("totalFloor")
        possession = listing_data.get("possession") or listing_data.get("possessionStatus")
        age = listing_data.get("propertyAge") or listing_data.get("age")
        amenities = listing_data.get("amenities")
        if isinstance(amenities, list):
            amenities = ", ".join(str(a) for a in amenities)
        description = listing_data.get("description") or listing_data.get("about")
        seller = listing_data.get("sellerType") or listing_data.get("postedBy")
        posted = listing_data.get("postedDate") or listing_data.get("createdAt")
        updated = listing_data.get("updatedAt") or listing_data.get("lastUpdated")

        all_text = json.dumps(listing_data, ensure_ascii=False)
        vaastu_mentioned, vaastu_text = extract_vaastu_mentions(all_text)

        return ListingRecord(
            property_id=property_id,
            url=url,
            city=city,
            property_type=property_type,
            source_page=source_page,
            title=title,
            locality=locality,
            price_display=str(price_display) if price_display else None,
            price_crore=price_crore_val,
            builtup_area_sqft=float(builtup) if builtup else None,
            carpet_area_sqft=float(carpet) if carpet else None,
            bhk=float(bhk) if bhk else None,
            bathrooms=float(bathrooms) if bathrooms else None,
            balconies=float(balconies) if balconies else None,
            furnishing=furnishing,
            facing=normalize_direction(facing) if facing else None,
            possession_status=possession,
            floor_number=int(floor_num) if floor_num else None,
            total_floors=int(total_floors) if total_floors else None,
            property_age=str(age) if age else None,
            amenities=amenities,
            description=description,
            vaastu_mentioned=int(vaastu_mentioned),
            vaastu_mentions_text=vaastu_text,
            seller_type=seller,
            posted_date=str(posted) if posted else None,
            last_updated=str(updated) if updated else None,
            collected_at=collected_at,
            raw_html_path=raw_html_path,
        )
    except Exception:
        return None


def parse_detail_from_text(
    text: str,
    url: str,
    city: str,
    property_type: str,
    property_id: str,
    source_page: int,
    raw_html_path: str,
    collected_at: str | None = None,
) -> ListingRecord:
    lines = extract_lines(text)
    all_text = "\n".join(lines)

    title = None
    for line in lines:
        line_lower = line.lower()
        if RE_BHK.search(line) and (
            "house" in line_lower or "flat" in line_lower
            or "villa" in line_lower or "apartment" in line_lower
        ):
            title = line
            break

    locality = None
    for line in lines:
        line_lower = line.lower()
        if "address" in line_lower:
            continue
        if ("sector" in line_lower or ", " in line) and "₹" not in line:
            if not RE_PRICE_UNIT.search(line) and len(line) < 100:
                locality = line
                break

    price_display = None
    price_crore_val = None
    for line in lines:
        if "₹" in line and ("cr" in line.lower() or "lac" in line.lower() or "lakh" in line.lower()):
            price_display = line
            price_crore_val = price_to_crore(line)
            break
        if "price on request" in line.lower():
            price_display = "Price on Request"
            break

    bhk = number_from_match(RE_BHK, all_text)
    bathrooms = number_from_match(RE_BATH, all_text)
    balconies = number_from_match(RE_BALCONY, all_text)

    builtup = None
    carpet = None
    sqft_matches = RE_SQFT.findall(all_text)
    if sqft_matches:
        builtup = float(sqft_matches[0].replace(",", ""))
        if len(sqft_matches) > 1:
            carpet = float(sqft_matches[1].replace(",", ""))

    facing = None
    m = RE_FACING.search(all_text)
    if m:
        facing = normalize_direction(m.group(1))

    furnishing = None
    m = RE_FURNISHING.search(all_text)
    if m:
        furnishing = normalize_ws(m.group(1))

    possession = None
    m = RE_POSSESSION.search(all_text)
    if m:
        possession = normalize_ws(m.group(1))

    floor_num = None
    total_floors = None
    m = RE_FLOOR.search(all_text)
    if m:
        floor_num = int(m.group(1))
        if m.group(2):
            total_floors = int(m.group(2))

    age = None
    m = RE_PROPERTY_AGE.search(all_text)
    if m:
        age = normalize_ws(m.group(1))

    seller = None
    m = RE_SELLER_TYPE.search(all_text)
    if m:
        seller = normalize_ws(m.group(1))

    posted = None
    m = RE_POSTED_DATE.search(all_text)
    if m:
        posted = normalize_ws(m.group(1))

    updated = None
    m = RE_LAST_UPDATED.search(all_text)
    if m:
        updated = normalize_ws(m.group(1))

    description = None
    for i, line in enumerate(lines):
        if "about" in line.lower() and "property" in line.lower():
            desc_lines = []
            for j in range(i + 1, min(i + 20, len(lines))):
                if any(stop in lines[j].lower() for stop in [
                    "amenities", "features", "specifications", "overview"
                ]):
                    break
                desc_lines.append(lines[j])
            if desc_lines:
                description = " ".join(desc_lines)
            break

    amenities = None
    for i, line in enumerate(lines):
        if RE_AMENITIES_SECTION.search(line):
            amen_lines = []
            for j in range(i + 1, min(i + 30, len(lines))):
                if any(stop in lines[j].lower() for stop in [
                    "about", "description", "overview", "contact"
                ]):
                    break
                amen_lines.append(lines[j])
            if amen_lines:
                amenities = " | ".join(amen_lines)
            break

    combined_text = " ".join(filter(None, [title, description, amenities]))
    vaastu_mentioned, vaastu_text = extract_vaastu_mentions(combined_text)

    return ListingRecord(
        property_id=property_id,
        url=url,
        city=city,
        property_type=property_type,
        source_page=source_page,
        title=title,
        locality=locality,
        price_display=price_display,
        price_crore=price_crore_val,
        builtup_area_sqft=builtup,
        carpet_area_sqft=carpet,
        bhk=bhk,
        bathrooms=bathrooms,
        balconies=balconies,
        furnishing=furnishing,
        facing=facing,
        possession_status=possession,
        floor_number=floor_num,
        total_floors=total_floors,
        property_age=age,
        amenities=amenities,
        description=description,
        vaastu_mentioned=int(vaastu_mentioned),
        vaastu_mentions_text=vaastu_text,
        seller_type=seller,
        posted_date=posted,
        last_updated=updated,
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

        record = None
        next_data = extract_next_data(html)
        if next_data:
            record = parse_detail_from_next_data(
                next_data=next_data,
                url=entry.get("url", ""),
                city=city,
                property_type=entry.get("property_type", ""),
                property_id=property_id,
                source_page=entry.get("source_page", 0),
                raw_html_path=entry.get("html_path", ""),
                collected_at=entry.get("collected_at"),
            )

        if not record:
            try:
                record = parse_detail_from_text(
                    text=text,
                    url=entry.get("url", ""),
                    city=city,
                    property_type=entry.get("property_type", ""),
                    property_id=property_id,
                    source_page=entry.get("source_page", 0),
                    raw_html_path=entry.get("html_path", ""),
                    collected_at=entry.get("collected_at"),
                )
            except Exception as e:
                logger.warning("[%s] Error parsing %s: %s", city, property_id, e)
                errors += 1
                continue

        parsed_rows.append(asdict(record))
        existing_ids.add(property_id)

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
