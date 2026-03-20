#!/usr/bin/env python3
"""Parse raw 99acres HTML files into structured Parquet.

This script parses raw files collected by 01_collect_99acres.py.

Workflow:
1. Read scrape_manifest.jsonl to find all scraped detail pages.
2. Parse each detail page's .html.gz file to extract structured fields.
3. Output: parsed_listings.parquet

Example
-------
python scripts/02_parse_99acres.py --city gurgaon
python scripts/02_parse_99acres.py --all-cities
"""

import argparse
import gzip
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.utils import (
    RE_BHK,
    RE_SQFT,
    RE_BATH,
    RE_BALCONY,
    RE_FLOOR,
    RE_LAST_UPDATED,
    RE_POSTED_DATE,
    RE_POSSESSION,
    RE_FACING,
    RE_FURNISHING,
    RE_PROPERTY_AGE,
    RE_SELLER_TYPE,
    RE_AMENITIES_SECTION,
    normalize_ws,
    slugify,
    extract_lines,
    price_to_crore,
    number_from_match,
    normalize_direction,
    write_parquet,
    load_manifest,
    read_existing_property_ids_parquet,
    extract_next_data,
    extract_vaastu_mentions,
)


@dataclass
class ListingRecord:
    property_id: str
    url: str
    city: str
    property_type: str
    search_page: int
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


RE_PRICE_UNIT = re.compile(r"₹?\s*([0-9][\d,]*(?:\.\d+)?)\s*(Cr|Crore|L|Lac|Lakh|Lakhs|K)?\b", re.I)


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root_from_here()
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
        help="Re-parse all files even if already in parsed_listings.csv",
    )
    return parser.parse_args()


def parse_detail_from_next_data(
    next_data: dict[str, Any],
    url: str,
    city: str,
    property_type: str,
    property_id: str,
    search_page: int,
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
            search_page=search_page,
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
    search_page: int,
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
        search_page=search_page,
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


def get_fieldnames() -> list[str]:
    return list(asdict(ListingRecord(
        property_id="", url="", city="", property_type="", search_page=0,
        title=None, locality=None, price_display=None, price_crore=None,
        builtup_area_sqft=None, carpet_area_sqft=None, bhk=None, bathrooms=None,
        balconies=None, furnishing=None, facing=None, possession_status=None,
        floor_number=None, total_floors=None, property_age=None, amenities=None,
        description=None, vaastu_mentioned=0, vaastu_mentions_text=None,
        seller_type=None, posted_date=None, last_updated=None, collected_at=None,
        raw_html_path="",
    )).keys())


def read_html_gz(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return fh.read()


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.body:
        return soup.body.get_text(separator="\n", strip=True)
    return soup.get_text(separator="\n", strip=True)


def parse_city(city: str, city_dir: Path, force: bool) -> dict:
    manifest_path = city_dir / "scrape_manifest.jsonl"
    parsed_parquet = city_dir / "parsed_listings.parquet"

    entries = load_manifest(manifest_path)
    detail_entries = [e for e in entries if e.get("type") == "detail"]

    if not detail_entries:
        print(f"[{city}] No detail entries found in scrape_manifest.jsonl", file=sys.stderr)
        return {"city": city, "parsed": 0, "skipped": 0, "errors": 0}

    existing_ids = set() if force else read_existing_property_ids_parquet(parsed_parquet)

    parsed_rows = []
    skipped = 0
    errors = 0

    for entry in detail_entries:
        property_id = entry.get("property_id", "")
        if property_id in existing_ids:
            skipped += 1
            continue

        raw_path = entry.get("html_path", "")
        html_path = city_dir / raw_path
        if not html_path.exists():
            alt_path = raw_path.replace("raw/", "pages/")
            if not alt_path.endswith(".gz"):
                alt_path += ".gz"
            html_path = city_dir / alt_path
        if not html_path.exists():
            print(f"[{city}] HTML file not found: {html_path}", file=sys.stderr)
            errors += 1
            continue

        try:
            html = read_html_gz(html_path)
        except Exception as e:
            print(f"[{city}] Error reading {html_path}: {e}", file=sys.stderr)
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
                search_page=entry.get("search_page", 0),
                raw_html_path=entry.get("html_path", ""),
                collected_at=entry.get("collected_at") or entry.get("timestamp"),
            )

        if not record:
            try:
                record = parse_detail_from_text(
                    text=text,
                    url=entry.get("url", ""),
                    city=city,
                    property_type=entry.get("property_type", ""),
                    property_id=property_id,
                    search_page=entry.get("search_page", 0),
                    raw_html_path=entry.get("html_path", ""),
                    collected_at=entry.get("collected_at") or entry.get("timestamp"),
                )
            except Exception as e:
                print(f"[{city}] Error parsing {property_id}: {e}", file=sys.stderr)
                errors += 1
                continue

        parsed_rows.append(asdict(record))

    if parsed_rows:
        if force or not parsed_parquet.exists():
            write_parquet(parsed_parquet, parsed_rows)
        else:
            import pandas as pd
            existing_df = pd.read_parquet(parsed_parquet) if parsed_parquet.exists() else pd.DataFrame()
            new_df = pd.DataFrame(parsed_rows)
            all_df = pd.concat([existing_df, new_df], ignore_index=True)
            all_df.to_parquet(parsed_parquet, index=False, compression="zstd")

    print(f"[{city}] Parsed: {len(parsed_rows)}, Skipped: {skipped}, Errors: {errors}", file=sys.stderr)

    return {
        "city": city, "parsed": len(parsed_rows), "skipped": skipped, "errors": errors,
        "output_parquet": str(parsed_parquet),
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    if args.all_cities:
        if not data_dir.exists():
            raise SystemExit(f"Data directory not found: {data_dir}")

        city_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        print(f"Parsing {len(city_dirs)} cities: {[d.name for d in city_dirs]}", file=sys.stderr)

        all_summaries = []
        for city_dir in sorted(city_dirs):
            city = city_dir.name
            print(f"\n=== Parsing {city} ===", file=sys.stderr)
            summary = parse_city(city, city_dir, args.force)
            all_summaries.append(summary)
            print(json.dumps(summary, indent=2))

        total_parsed = sum(s["parsed"] for s in all_summaries)
        total_errors = sum(s["errors"] for s in all_summaries)
        print("\n=== Parsing complete ===", file=sys.stderr)
        print(f"Cities: {len(all_summaries)}, Parsed: {total_parsed}, Errors: {total_errors}", file=sys.stderr)
    else:
        city_dir = data_dir / slugify(args.city)
        if not city_dir.exists():
            raise SystemExit(f"City directory not found: {city_dir}")

        summary = parse_city(args.city, city_dir, args.force)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
