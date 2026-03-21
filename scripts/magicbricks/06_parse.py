#!/usr/bin/env python3
"""Parse MagicBricks individual listing HTML files into structured Parquet.

Reads individual listing HTML files and extracts property data.
Outputs parsed_listings.parquet with zstd compression.

Example
-------
python scripts/magicbricks/06_parse.py --city delhi-ncr_apartment
python scripts/magicbricks/06_parse.py --all-cities
python scripts/magicbricks/06_parse.py --city delhi-ncr_apartment --force
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

from scripts.utils import extract_vaastu_mentions  # noqa: E402
from scripts.utils import load_manifest, project_root, read_html_gz, slugify
from scripts.utils.scraping import logger, setup_logging  # noqa: E402


@dataclass
class ListingRecord:
    property_id: str
    project_id: str | None
    url: str
    city: str
    title: str | None
    property_type: str | None
    locality: str | None
    address: str | None
    latitude: float | None
    longitude: float | None
    price_display: str | None
    price_crore: float | None
    price_per_sqft: float | None
    builtup_area_sqft: float | None
    carpet_area_sqft: float | None
    bhk: float | None
    bathrooms: float | None
    balconies: float | None
    furnishing: str | None
    flooring_type: str | None
    facing: str | None
    possession_status: str | None
    property_age: str | None
    floor_no: int | None
    total_floors: int | None
    ownership_type: str | None
    transaction_type: str | None
    amenities: str | None
    description: str | None
    vaastu_mentioned: int
    vaastu_mentions_text: str | None
    seller_type: str | None
    seller_name: str | None
    posted_date: str | None
    image_count: int | None
    is_luxury: bool | None
    connectivity_score: float | None
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
    """Extract SERVER_PRELOADED_STATE data from HTML.

    Individual listing pages use SERVER_PRELOADED_STATE_DETAILS
    while project pages use SERVER_PRELOADED_STATE_
    """
    soup = BeautifulSoup(html, "html.parser")

    patterns = [
        "window.SERVER_PRELOADED_STATE_DETAILS = ",
        "window.SERVER_PRELOADED_STATE_ = ",
    ]

    for script in soup.find_all("script"):
        text = script.string or ""
        for pattern in patterns:
            if pattern in text:
                start = text.find(pattern)
                if start >= 0:
                    json_start = start + len(pattern)
                    json_text = text[json_start:].strip()
                    if json_text.endswith(";"):
                        json_text = json_text[:-1]
                    try:
                        return json.loads(json_text)
                    except json.JSONDecodeError:
                        continue
    return None


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def extract_listing_from_individual_page(
    data: dict,
    manifest_entry: dict,
    city: str,
    raw_html_path: str,
) -> dict | None:
    """Extract listing data from individual /propertyDetails/ page JSON."""

    property_bean = None

    if "propertyDetailInfoBeanData" in data:
        info_bean = data["propertyDetailInfoBeanData"]
        property_detail = info_bean.get("propertyDetail", {})
        property_bean = property_detail.get("detailBean", {})

    if not property_bean:
        property_bean = data.get("propertyPageBean", {})

    if not property_bean:
        property_bean = data.get("propertyData", {})

    if not property_bean:
        for key in ["pdpData", "listingData", "propertyDetail"]:
            if key in data and isinstance(data[key], dict):
                property_bean = data[key]
                break

    if not property_bean:
        return None

    prop_id = property_bean.get("id") or property_bean.get("propertyId")
    if not prop_id:
        prop_id = manifest_entry.get("property_id", "")

    prop_id = str(prop_id)

    price = property_bean.get("price")
    price_crore = None
    if price and isinstance(price, (int, float)) and price > 0:
        if price > 10000:
            price_crore = price / 10000000
        else:
            price_crore = price / 100

    price_display = (
        property_bean.get("priceD")
        or property_bean.get("priceDisplay")
        or property_bean.get("saleRentPrice")
    )

    bhk = safe_float(
        property_bean.get("bedroomD")
        or property_bean.get("bedroom")
        or property_bean.get("bedrooms")
        or property_bean.get("bhk")
    )
    bathrooms = safe_float(
        property_bean.get("bathD")
        or property_bean.get("bathroom")
        or property_bean.get("bathrooms")
    )
    balconies = safe_float(
        property_bean.get("balconiesD")
        or property_bean.get("balconies")
        or property_bean.get("numberOfBalconied")
    )

    builtup_area = safe_float(
        property_bean.get("caSqFt")
        or property_bean.get("coveredArea")
        or property_bean.get("ca")
        or property_bean.get("builtupArea")
    )

    carpet_area = safe_float(property_bean.get("carpetArea"))

    facing = property_bean.get("facingD") or property_bean.get("facing")

    posted_date = (
        property_bean.get("postDateT")
        or property_bean.get("postedLabelD")
        or property_bean.get("postedDate")
        or property_bean.get("postedOn")
    )

    floor_no = safe_int(
        property_bean.get("floorNo") or property_bean.get("floorNumber")
    )
    total_floors = safe_int(
        property_bean.get("floors")
        or property_bean.get("totalFloors")
        or property_bean.get("totalFloorNumber")
    )

    furnishing = (
        property_bean.get("furnishedD")
        or property_bean.get("furnishing")
        or property_bean.get("furnished")
    )
    flooring_type = property_bean.get("flooringTyD") or property_bean.get(
        "flooringType"
    )
    possession_status = property_bean.get("possStatusD") or property_bean.get(
        "possessionStatus"
    )
    property_age = (
        property_bean.get("acD")
        or property_bean.get("propertyAge")
        or property_bean.get("ageofcons")
    )
    ownership_type = property_bean.get("OwnershipTypeD") or property_bean.get(
        "ownershipType"
    )
    transaction_type_from_data = property_bean.get(
        "transactionTypeD"
    ) or property_bean.get("transactionType")

    transaction_type = (
        manifest_entry.get("transaction_type") or transaction_type_from_data
    )

    locality = (
        property_bean.get("locSeoName")
        or property_bean.get("locality")
        or property_bean.get("localityName")
    )
    address = (
        property_bean.get("psmAdd")
        or property_bean.get("address")
        or property_bean.get("propertyAddress")
    )
    city_name = (
        property_bean.get("ctName")
        or property_bean.get("cityName")
        or property_bean.get("city")
        or city
    )

    latitude = None
    longitude = None
    coords = property_bean.get("ltcoordGeo") or property_bean.get("coordinates")
    if coords and "," in str(coords):
        try:
            lat_str, lon_str = str(coords).split(",")
            latitude = float(lat_str)
            longitude = float(lon_str)
        except (ValueError, TypeError):
            pass

    if latitude is None:
        latitude = safe_float(property_bean.get("latitude") or property_bean.get("lat"))
    if longitude is None:
        longitude = safe_float(
            property_bean.get("longitude") or property_bean.get("lng")
        )

    price_per_sqft = safe_float(
        property_bean.get("sqFtPrice")
        or property_bean.get("pricePerSqft")
        or property_bean.get("sqftPrice")
    )

    image_count = safe_int(
        property_bean.get("imgCt")
        or property_bean.get("imageCount")
        or property_bean.get("propImageCount")
    )

    is_luxury = (
        property_bean.get("isLuxury") == "T" or property_bean.get("luxury") is True
    )

    connectivity_score = safe_float(
        property_bean.get("cScore") or property_bean.get("connectivityScore")
    )

    property_type = property_bean.get("propTypeD") or property_bean.get("propertyType")
    seller_name = (
        property_bean.get("oname")
        or property_bean.get("ownerName")
        or property_bean.get("contactPersonName")
    )
    seller_type = (
        property_bean.get("userType")
        or property_bean.get("sellerType")
        or property_bean.get("postedBy")
    )

    description = (
        property_bean.get("dtldesc")
        or property_bean.get("propertyDescription")
        or property_bean.get("desc")
        or ""
    )

    seo_desc = property_bean.get("seoDesc") or ""
    auto_desc = property_bean.get("auto_desc") or ""
    amenities_str = property_bean.get("amenities") or ""
    if not amenities_str:
        facilities = property_bean.get("facilitiesDesc") or ""
        prop_amenities = property_bean.get("propertyAmenities", {})
        if isinstance(prop_amenities, dict):
            amenities_str = ", ".join(prop_amenities.values())
        elif facilities:
            amenities_str = facilities
    ad_text = property_bean.get("ad_text") or ""
    plgdtldesc = property_bean.get("plgdtldesc") or ""

    combined_text = " ".join(
        filter(
            None, [description, seo_desc, auto_desc, amenities_str, ad_text, plgdtldesc]
        )
    )
    vaastu_mentioned, vaastu_mentions_text = extract_vaastu_mentions(combined_text)

    title = None
    if bhk and property_type:
        title = f"{int(bhk)} BHK {property_type}"
    elif auto_desc:
        title = auto_desc[:100]
    elif property_bean.get("title"):
        title = property_bean.get("title")

    project_id = (
        property_bean.get("prjId")
        or property_bean.get("projectId")
        or property_bean.get("propPsm")
    )
    project_name = (
        property_bean.get("prjname")
        or property_bean.get("projectName")
        or property_bean.get("nameOfSociety")
    )
    developer_name = property_bean.get("devName") or property_bean.get("developerName")

    rating_overall = None
    rating_connectivity = None
    rating_neighbourhood = None
    rating_safety = None

    rating_bean = property_bean.get("ratingBean", {})
    if rating_bean:
        rating_overall = safe_float(rating_bean.get("psmAvgRt"))
        rating_safety = safe_float(rating_bean.get("securityRt"))
        rating_connectivity = safe_float(rating_bean.get("prjInfraRt"))
        rating_neighbourhood = safe_float(rating_bean.get("prjMaintainanceRt"))

    record = ListingRecord(
        property_id=prop_id,
        project_id=str(project_id) if project_id else None,
        url=manifest_entry.get("url", ""),
        city=city_name,
        title=title,
        property_type=property_type,
        locality=locality,
        address=address,
        latitude=latitude,
        longitude=longitude,
        price_display=price_display,
        price_crore=price_crore,
        price_per_sqft=price_per_sqft,
        builtup_area_sqft=builtup_area,
        carpet_area_sqft=carpet_area,
        bhk=bhk,
        bathrooms=bathrooms,
        balconies=balconies,
        furnishing=furnishing,
        flooring_type=flooring_type,
        facing=facing,
        possession_status=possession_status,
        property_age=property_age,
        floor_no=floor_no,
        total_floors=total_floors,
        ownership_type=ownership_type,
        transaction_type=transaction_type,
        amenities=amenities_str if amenities_str else None,
        description=description if description else None,
        vaastu_mentioned=int(vaastu_mentioned),
        vaastu_mentions_text=vaastu_mentions_text,
        seller_type=seller_type,
        seller_name=seller_name,
        posted_date=posted_date,
        image_count=image_count,
        is_luxury=is_luxury,
        connectivity_score=connectivity_score,
        rating_overall=rating_overall,
        rating_connectivity=rating_connectivity,
        rating_neighbourhood=rating_neighbourhood,
        rating_safety=rating_safety,
        project_name=project_name,
        developer_name=developer_name,
        collected_at=manifest_entry.get("collected_at"),
        raw_html_path=raw_html_path,
    )

    return asdict(record)


def read_existing_property_ids(parquet_path: Path) -> set[str]:
    if not parquet_path.exists():
        return set()
    try:
        df = pd.read_parquet(parquet_path, columns=["property_id"])
        return set(df["property_id"].dropna().astype(str))
    except Exception:
        return set()


def parse_city(city: str, city_dir: Path, force: bool) -> dict:
    listing_manifest_path = city_dir / "listing_manifest.jsonl"
    parsed_parquet = city_dir / "parsed_listings.parquet"

    if not listing_manifest_path.exists():
        logger.warning("[%s] No listing_manifest.jsonl found", city)
        return {"city": city, "parsed": 0, "skipped": 0, "errors": 0}

    entries = load_manifest(listing_manifest_path)
    successful_entries = [e for e in entries if e.get("status") == "success"]

    if not successful_entries:
        logger.warning("[%s] No successful listing pages found", city)
        return {"city": city, "parsed": 0, "skipped": 0, "errors": 0}

    existing_ids = set() if force else read_existing_property_ids(parsed_parquet)

    seen_property_ids: set[str] = set()
    parsed_rows: list[dict] = []
    skipped = 0
    errors = 0

    for entry in successful_entries:
        prop_id = entry.get("property_id", "")

        if prop_id in seen_property_ids:
            skipped += 1
            continue

        if prop_id in existing_ids:
            skipped += 1
            continue

        seen_property_ids.add(prop_id)

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
                logger.debug("[%s] No SERVER_PRELOADED_STATE_ in %s", city, prop_id)
                errors += 1
                continue

            record = extract_listing_from_individual_page(
                data=data,
                manifest_entry=entry,
                city=city,
                raw_html_path=raw_path,
            )

            if not record:
                logger.debug("[%s] Could not extract listing from %s", city, prop_id)
                errors += 1
                continue

            parsed_rows.append(record)

        except Exception as e:
            logger.warning("[%s] Error parsing %s: %s", city, prop_id, e)
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
            if d.is_dir() and (d / "listing_manifest.jsonl").exists()
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
