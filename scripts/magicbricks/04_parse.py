#!/usr/bin/env python3
"""Parse MagicBricks detail HTML files into structured Parquet.

Reads detail HTML files and extracts property listings.
Outputs parsed_listings.parquet with zstd compression.

Example
-------
python scripts/magicbricks/04_parse.py --city delhi-ncr_apartment
python scripts/magicbricks/04_parse.py --all-cities
python scripts/magicbricks/04_parse.py --city delhi-ncr_apartment --force
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

from scripts.utils import (  # noqa: E402
    extract_vaastu_mentions,
    load_manifest,
    project_root,
    read_html_gz,
    slugify,
)
from scripts.utils.scraping import logger, setup_logging  # noqa: E402


@dataclass
class ListingRecord:
    property_id: str
    project_id: str
    url: str
    city: str
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


def parse_args() -> argparse.Namespace:
    root = project_root()
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


def extract_project_ratings(data: dict) -> dict[str, float | None]:
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
    source_page: int,
    raw_html_path: str,
    collected_at: str | None,
) -> list[dict]:
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
            filter(
                None, [seo_desc, auto_desc, amenities_str, ad_text, dtldesc, plgdtldesc]
            )
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
            source_page=source_page,
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

    entries = load_manifest(detail_manifest_path)
    successful_entries = [e for e in entries if e.get("status") == "success"]

    if not successful_entries:
        logger.warning("[%s] No successful detail pages found", city)
        return {"city": city, "parsed": 0, "skipped": 0, "errors": 0}

    existing_ids = set() if force else read_existing_property_ids(parsed_parquet)

    seen_project_ids: set[str] = set()
    parsed_rows: list[dict] = []
    skipped = 0
    errors = 0

    for entry in successful_entries:
        project_id = entry.get("property_id", "")

        if project_id in seen_project_ids:
            skipped += 1
            continue
        seen_project_ids.add(project_id)

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

            records = extract_listings_from_json(
                data=data,
                project_id=project_id,
                project_url=entry.get("url", ""),
                city=city,
                source_page=0,
                raw_html_path=entry.get("html_path", ""),
                collected_at=entry.get("collected_at"),
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
            logger.warning("[%s] Error parsing %s: %s", city, project_id, e)
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
