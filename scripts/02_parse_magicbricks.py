#!/usr/bin/env python3
"""Parse raw Magicbricks HTML files into structured Parquet.

This script parses raw files collected by 01_collect_magicbricks.py.

Magicbricks project detail pages contain multiple individual property listings
embedded in the SERVER_PRELOADED_STATE_ JSON. This script extracts each
individual listing as a separate record.

Workflow:
1. Read scrape_manifest.jsonl to find all scraped detail pages.
2. Parse each detail page's .html.gz file to extract SERVER_PRELOADED_STATE_ JSON.
3. Extract individual listings from bhkProjectDetailsMap.
4. Output: parsed_listings.parquet

Example
-------
python scripts/02_parse_magicbricks.py --city delhi-ncr
python scripts/02_parse_magicbricks.py --all-cities
"""

import argparse
import gzip
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.utils import (extract_vaastu_mentions, load_manifest,
                           read_existing_property_ids_parquet, slugify,
                           write_parquet)


@dataclass
class ListingRecord:
    property_id: str
    project_id: str
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
    floor_no: int | None
    total_floors: int | None
    amenities: str | None
    description: str | None
    vaastu_mentioned: int
    vaastu_mentions_text: str | None
    seller_type: str | None
    posted_date: str | None
    rating_overall: float | None
    rating_connectivity: float | None
    rating_neighbourhood: float | None
    rating_safety: float | None
    project_name: str | None
    developer_name: str | None
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
        help="Re-parse all files even if already in parsed_listings.parquet",
    )
    return parser.parse_args()


def read_html_gz(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return fh.read()


def extract_preloaded_state(html: str) -> dict | None:
    """Extract SERVER_PRELOADED_STATE_ JSON from HTML."""
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


def extract_project_ratings(data: dict) -> dict[str, float | None]:
    """Extract project-level ratings from the data.

    Magicbricks uses various rating keys:
    - psmAvgRt: overall project rating
    - securityRt: safety
    - prjInfraRt: infrastructure/connectivity proxy
    - prjMaintainanceRt: maintenance/neighbourhood proxy
    """
    ratings: dict[str, float | None] = {
        "overall": None,
        "connectivity": None,
        "neighbourhood": None,
        "safety": None,
    }

    try:
        prj_mobile = data.get("projectDetailData", {}).get("prjMobileBean", {})
        rating_bean = prj_mobile.get("ratingBean", {})
        if rating_bean:
            overall = rating_bean.get("psmAvgRt")
            if overall:
                ratings["overall"] = float(overall)

            security = rating_bean.get("securityRt")
            if security:
                ratings["safety"] = float(security)

            infra = rating_bean.get("prjInfraRt")
            if infra:
                ratings["connectivity"] = float(infra)

            maint = rating_bean.get("prjMaintainanceRt")
            if maint:
                ratings["neighbourhood"] = float(maint)
    except (ValueError, TypeError, AttributeError):
        pass

    return ratings


def extract_listings_from_json(
    data: dict,
    project_id: str,
    project_url: str,
    city: str,
    search_page: int,
    raw_html_path: str,
    collected_at: str | None,
) -> list[dict]:
    """Extract individual listing records from the parsed JSON."""
    records = []

    bhk_details = (
        data.get("projectPageSeoStaticData", {})
        .get("bhkDetailsDTO", {})
        .get("bhkProjectDetailsMap", {})
    )

    all_data = bhk_details.get("ALL", {})
    sale_listings = all_data.get("groupedResult") or []
    rent_listings = all_data.get("groupedRentResult") or []

    project_ratings = extract_project_ratings(data)

    prj_mobile = data.get("projectDetailData", {}).get("prjMobileBean", {})
    project_name = prj_mobile.get("opsmname")
    dev_list = prj_mobile.get("projectDeveloperList")
    developer_name = None
    if isinstance(dev_list, list) and dev_list:
        first_dev = dev_list[0]
        if isinstance(first_dev, dict):
            developer_name = first_dev.get("developerName")

    for listing in sale_listings + rent_listings:
        prop_id = listing.get("id")
        if not prop_id:
            continue

        price = listing.get("price")
        price_crore = None
        if price and isinstance(price, (int, float)) and price > 0:
            if price > 10000:
                price_crore = price / 10000000
            else:
                price_crore = price / 100

        price_display = listing.get("priceD")

        bhk = None
        bedroom_d = listing.get("bedroomD")
        if bedroom_d:
            try:
                bhk = float(bedroom_d)
            except (ValueError, TypeError):
                pass

        bathrooms = None
        bath_d = listing.get("bathD")
        if bath_d:
            try:
                bathrooms = float(bath_d)
            except (ValueError, TypeError):
                pass

        balconies = None
        balc_d = listing.get("balconiesD")
        if balc_d:
            try:
                balconies = float(balc_d)
            except (ValueError, TypeError):
                pass

        builtup_area = None
        ca_sqft = (
            listing.get("caSqFt") or listing.get("coveredArea") or listing.get("ca")
        )
        if ca_sqft:
            try:
                builtup_area = float(ca_sqft)
            except (ValueError, TypeError):
                pass

        floor_no = None
        floor_str = listing.get("floorNo")
        if floor_str:
            try:
                floor_no = int(floor_str)
            except (ValueError, TypeError):
                pass

        total_floors = None
        floors_str = listing.get("floors")
        if floors_str:
            try:
                total_floors = int(floors_str)
            except (ValueError, TypeError):
                pass

        furnishing = listing.get("furnishedD")
        possession_status = listing.get("possStatusD")
        property_age = listing.get("acD")

        locality = listing.get("locSeoName")
        city_name = listing.get("ctName") or city

        seo_desc = listing.get("seoDesc") or ""
        auto_desc = listing.get("auto_desc") or ""
        amenities_str = listing.get("amenities") or ""
        ad_text = listing.get("ad_text") or ""
        dtldesc = listing.get("dtldesc") or ""
        plgdtldesc = listing.get("plgdtldesc") or ""

        combined_text = " ".join(
            filter(None, [seo_desc, auto_desc, amenities_str, ad_text, dtldesc, plgdtldesc])
        )
        vaastu_mentioned, vaastu_mentions_text = extract_vaastu_mentions(combined_text)

        user_type = listing.get("userType")

        title = None
        if bhk and listing.get("propTypeD"):
            title = f"{int(bhk)} BHK {listing.get('propTypeD')}"
        elif auto_desc:
            title = auto_desc[:100]

        listing_url = listing.get("url") or project_url

        record = ListingRecord(
            property_id=str(prop_id),
            project_id=project_id,
            url=listing_url,
            city=city_name,
            search_page=search_page,
            title=title,
            locality=locality,
            price_display=price_display,
            price_crore=price_crore,
            builtup_area_sqft=builtup_area,
            carpet_area_sqft=None,
            bhk=bhk,
            bathrooms=bathrooms,
            balconies=balconies,
            furnishing=furnishing,
            facing=None,
            possession_status=possession_status,
            property_age=property_age,
            floor_no=floor_no,
            total_floors=total_floors,
            amenities=amenities_str if amenities_str else None,
            description=seo_desc if seo_desc else None,
            vaastu_mentioned=int(vaastu_mentioned),
            vaastu_mentions_text=vaastu_mentions_text,
            seller_type=user_type,
            posted_date=None,
            rating_overall=project_ratings["overall"],
            rating_connectivity=project_ratings["connectivity"],
            rating_neighbourhood=project_ratings["neighbourhood"],
            rating_safety=project_ratings["safety"],
            project_name=listing.get("prjname") or project_name,
            developer_name=listing.get("devName") or developer_name,
            collected_at=collected_at,
            raw_html_path=raw_html_path,
        )
        records.append(asdict(record))

    return records


def parse_city(city: str, city_dir: Path, force: bool) -> dict:
    manifest_path = city_dir / "scrape_manifest.jsonl"
    parsed_parquet = city_dir / "parsed_listings.parquet"

    entries = load_manifest(manifest_path)
    detail_entries = [e for e in entries if e.get("type") == "detail"]

    if not detail_entries:
        print(
            f"[{city}] No detail entries found in scrape_manifest.jsonl",
            file=sys.stderr,
        )
        return {"city": city, "parsed": 0, "skipped": 0, "errors": 0}

    existing_ids = (
        set() if force else read_existing_property_ids_parquet(parsed_parquet)
    )

    seen_project_ids = set()
    parsed_rows = []
    skipped = 0
    errors = 0

    for entry in detail_entries:
        project_id = entry.get("property_id", "")

        if project_id in seen_project_ids:
            skipped += 1
            continue
        seen_project_ids.add(project_id)

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

        try:
            data = extract_preloaded_state(html)
            if not data:
                print(
                    f"[{city}] No SERVER_PRELOADED_STATE_ found in {project_id}",
                    file=sys.stderr,
                )
                errors += 1
                continue

            records = extract_listings_from_json(
                data=data,
                project_id=project_id,
                project_url=entry.get("url", ""),
                city=city,
                search_page=entry.get("search_page", 0),
                raw_html_path=entry.get("html_path", ""),
                collected_at=entry.get("collected_at") or entry.get("timestamp"),
            )

            if not records:
                errors += 1
                continue

            for rec in records:
                if rec["property_id"] not in existing_ids:
                    parsed_rows.append(rec)
                else:
                    skipped += 1

        except Exception as e:
            import traceback

            print(f"[{city}] Error parsing {project_id}: {e}", file=sys.stderr)
            traceback.print_exc()
            errors += 1
            continue

    if parsed_rows:
        if force or not parsed_parquet.exists():
            write_parquet(parsed_parquet, parsed_rows)
        else:
            import pandas as pd

            existing_df = (
                pd.read_parquet(parsed_parquet)
                if parsed_parquet.exists()
                else pd.DataFrame()
            )
            new_df = pd.DataFrame(parsed_rows)
            all_df = pd.concat([existing_df, new_df], ignore_index=True)
            all_df.to_parquet(parsed_parquet, index=False, compression="zstd")

    print(
        f"[{city}] Parsed: {len(parsed_rows)} listings, "
        f"Skipped: {skipped}, Errors: {errors}",
        file=sys.stderr,
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
    data_dir = Path(args.data_dir)

    if args.all_cities:
        if not data_dir.exists():
            raise SystemExit(f"Data directory not found: {data_dir}")

        city_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        print(
            f"Parsing {len(city_dirs)} cities: {[d.name for d in city_dirs]}",
            file=sys.stderr,
        )

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
        print(
            f"Cities: {len(all_summaries)}, Parsed: {total_parsed}, Errors: {total_errors}",
            file=sys.stderr,
        )
    else:
        city_dir = data_dir / slugify(args.city)
        if not city_dir.exists():
            raise SystemExit(f"City directory not found: {city_dir}")

        summary = parse_city(args.city, city_dir, args.force)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
