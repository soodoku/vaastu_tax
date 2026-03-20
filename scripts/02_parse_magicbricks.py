#!/usr/bin/env python3
"""Parse raw Magicbricks HTML files into structured Parquet.

This script parses raw files collected by 01_collect_magicbricks.py.

Workflow:
1. Read scrape_manifest.jsonl to find all scraped detail pages.
2. Parse each detail page's .html.gz file to extract structured fields.
3. Output: parsed_listings.parquet

Example
-------
python scripts/02_parse_magicbricks.py --city delhi-ncr
python scripts/02_parse_magicbricks.py --all-cities
"""

import argparse
import gzip
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.utils import (
    RE_BHK,
    RE_SQFT,
    RE_CARPET,
    RE_BUILTUP,
    RE_BATH,
    RE_BALCONY,
    RE_POSTED_DATE,
    RE_POSSESSION,
    RE_FACING,
    RE_FURNISHING,
    RE_PROPERTY_AGE,
    RE_SELLER_TYPE,
    RE_RESULTS_COUNT,
    normalize_ws,
    slugify,
    extract_lines,
    price_to_crore,
    number_from_match,
    normalize_direction,
    find_price_display,
    find_first_matching_line,
    extract_section,
    write_parquet,
    load_manifest,
    read_existing_property_ids_parquet,
    extract_vaastu_mentions,
)


RE_TITLE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*BHK\s+.+?"
    r"(Independent House|Villa|Row House|Builder Floor|Penthouse|Apartment|Flat)\b",
    re.I
)


@dataclass
class ListingRecord:
    property_id: str
    url: str
    city: str
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
    property_age: str | None
    amenities: str | None
    description: str | None
    vaastu_mentioned: int
    vaastu_mentions_text: str | None
    seller_type: str | None
    posted_date: str | None
    collected_at: str | None
    raw_html_path: str


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
        help="Parse data for all cities found in data/raw/magicbricks/",
    )
    parser.add_argument(
        "--data-dir",
        default=str(root / "data" / "raw" / "magicbricks"),
        help="Base directory containing city subdirectories",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-parse all files even if already in parsed_listings.csv",
    )
    return parser.parse_args()


def find_title(lines: Sequence[str]) -> str | None:
    for line in lines:
        if RE_TITLE.search(line):
            return line
    return None


def find_locality(lines: Sequence[str], title: str | None) -> str | None:
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
        if RE_RESULTS_COUNT.search(cand):
            continue
        if any(tok in lower for tok in ["emi", "sq.ft", "amenities", "overview", "call for price"]):
            continue
        if len(cand) > 10:
            return cand
    return None


def find_price_magicbricks(lines: Sequence[str], full_text: str) -> tuple[str | None, float | None]:
    """Extract price from Magicbricks listing.

    Magicbricks often has price split across lines:
    - Line N: "₹"
    - Line N+1: "19 Lac" or "1.5 Cr"

    Also extracts from description text as fallback.
    """
    # Pattern 1: Look for "₹" followed by price on next line
    for i, line in enumerate(lines):
        if line.strip() == "₹" and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Check if next line is a price value (e.g., "19 Lac", "1.5 Cr")
            if re.match(r"^[\d,.]+\s*(Lac|Lakh|L|Cr|Crore|K)?\s*$", next_line, re.I):
                price_str = f"₹ {next_line}"
                price_val = price_to_crore(price_str)
                if price_val and price_val > 0.01:  # Reasonable price
                    return price_str, price_val

    # Pattern 2: Extract from description (e.g., "at a cost of ₹19 Lac")
    price_pattern = re.compile(
        r"(?:price|cost|available at|offered at|priced at)[:\s]*₹?\s*([\d,.]+)\s*(Lac|Lakh|L|Cr|Crore)?",
        re.I
    )
    m = price_pattern.search(full_text)
    if m:
        value = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        if unit in ("cr", "crore"):
            price_crore = value
        elif unit in ("lac", "lakh", "l"):
            price_crore = value / 100
        else:
            price_crore = None
        if price_crore and price_crore > 0.01:
            return f"₹{m.group(1)} {m.group(2) or ''}", price_crore

    # Pattern 3: Look for INR ₹XX Lac pattern
    inr_pattern = re.compile(r"INR\s*₹?\s*([\d,.]+)\s*(Lac|Lakh|L|Cr|Crore)?", re.I)
    m = inr_pattern.search(full_text)
    if m:
        value = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        if unit in ("cr", "crore"):
            price_crore = value
        elif unit in ("lac", "lakh", "l"):
            price_crore = value / 100
        else:
            price_crore = None
        if price_crore and price_crore > 0.01:
            return f"₹{m.group(1)} {m.group(2) or ''}", price_crore

    return None, None


def parse_detail_text(
    text: str,
    url: str,
    city: str,
    property_id: str,
    search_page: int,
    raw_html_path: str,
    collected_at: str | None = None,
) -> ListingRecord:
    lines = extract_lines(text)
    title = find_title(lines)
    title_idx = lines.index(title) if title in lines else None
    locality = find_locality(lines, title)

    full_text = "\n".join(lines)
    price_display, price_crore_val = find_price_magicbricks(lines, full_text)

    full_text = "\n".join(lines)
    bhk = number_from_match(RE_BHK, title or full_text)
    bathrooms = number_from_match(RE_BATH, full_text)
    balconies = number_from_match(RE_BALCONY, full_text)

    carpet_area_sqft = number_from_match(RE_CARPET, full_text)
    builtup_area_sqft = number_from_match(RE_BUILTUP, full_text)
    if not builtup_area_sqft:
        builtup_area_sqft = number_from_match(RE_SQFT, full_text)

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

    property_age = None
    age_match = RE_PROPERTY_AGE.search(full_text)
    if age_match:
        property_age = normalize_ws(age_match.group(1))

    amenities = extract_section(lines, "Amenities")
    if not amenities:
        amenities = extract_section(lines, "Features")
    description = extract_section(lines, "Description")
    if not description:
        description = extract_section(lines, "About this property")
    if not description:
        description = extract_section(lines, "Property Description")

    combined_text = " ".join(filter(None, [title, amenities, description]))
    vaastu_mentioned, vaastu_mentions_text = extract_vaastu_mentions(combined_text)

    seller_type = None
    seller_match = RE_SELLER_TYPE.search(full_text)
    if seller_match:
        seller_type = normalize_ws(seller_match.group(1))

    posted_date = None
    posted_match = RE_POSTED_DATE.search(full_text)
    if posted_match:
        posted_date = normalize_ws(posted_match.group(1))

    return ListingRecord(
        property_id=property_id,
        url=url,
        city=city,
        search_page=search_page,
        title=title,
        locality=locality,
        price_display=price_display,
        price_crore=price_crore_val,
        builtup_area_sqft=builtup_area_sqft,
        carpet_area_sqft=carpet_area_sqft,
        bhk=bhk,
        bathrooms=bathrooms,
        balconies=balconies,
        furnishing=furnishing,
        facing=facing,
        possession_status=possession_status,
        property_age=property_age,
        amenities=amenities,
        description=description,
        vaastu_mentioned=int(vaastu_mentioned),
        vaastu_mentions_text=vaastu_mentions_text,
        seller_type=seller_type,
        posted_date=posted_date,
        collected_at=collected_at,
        raw_html_path=raw_html_path,
    )


def get_fieldnames() -> list[str]:
    return list(asdict(ListingRecord(
        property_id="",
        url="",
        city="",
        search_page=0,
        title=None,
        locality=None,
        price_display=None,
        price_crore=None,
        builtup_area_sqft=None,
        carpet_area_sqft=None,
        bhk=None,
        bathrooms=None,
        balconies=None,
        furnishing=None,
        facing=None,
        possession_status=None,
        property_age=None,
        amenities=None,
        description=None,
        vaastu_mentioned=0,
        vaastu_mentions_text=None,
        seller_type=None,
        posted_date=None,
        collected_at=None,
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
            # Try alternate path: raw/ -> pages/, and add .gz
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

        try:
            record = parse_detail_text(
                text=text,
                url=entry.get("url", ""),
                city=city,
                property_id=property_id,
                search_page=entry.get("search_page", 0),
                raw_html_path=entry.get("html_path", ""),
                collected_at=entry.get("collected_at") or entry.get("timestamp"),
            )
            parsed_rows.append(asdict(record))
        except Exception as e:
            print(f"[{city}] Error parsing {property_id}: {e}", file=sys.stderr)
            errors += 1
            continue

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
        "city": city,
        "parsed": len(parsed_rows),
        "skipped": skipped,
        "errors": errors,
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
        print(f"\n=== Parsing complete ===", file=sys.stderr)
        print(f"Cities: {len(all_summaries)}, Parsed: {total_parsed}, Errors: {total_errors}", file=sys.stderr)
    else:
        city_dir = data_dir / slugify(args.city)
        if not city_dir.exists():
            raise SystemExit(f"City directory not found: {city_dir}")

        summary = parse_city(args.city, city_dir, args.force)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
