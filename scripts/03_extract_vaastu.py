#!/usr/bin/env python3
"""Extract Vaastu mentions from all 99acres datasets and create unified output.

Processes:
1. CampusX data (data/raw/99acres_campusx/)
2. Kaggle datasets (data/raw/99acres_kaggle/)
3. Our scraper output (data/raw/99acres/) if present

Outputs:
- data/derived/all_99acres_vaastu.csv - unified dataset with vaastu_mentioned flag
- data/derived/data_manifest.json - summary statistics

Usage
-----
python scripts/03_extract_vaastu.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("pandas required: pip install pandas")
    sys.exit(1)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


RE_VAASTU = re.compile(r"\bvaa?stu\b", re.IGNORECASE)


def detect_vaastu(text: str | None) -> bool:
    if not text or pd.isna(text):
        return False
    return bool(RE_VAASTU.search(str(text)))


def extract_vaastu_mentions(text: str | None) -> str | None:
    if not text or pd.isna(text):
        return None
    text = str(text)
    matches = RE_VAASTU.findall(text)
    if not matches:
        return None
    sentences = []
    for sentence in re.split(r"[.!?\n]", text):
        if RE_VAASTU.search(sentence):
            sentences.append(sentence.strip())
    return " || ".join(sentences[:3]) if sentences else "vaastu mentioned"


def normalize_price_to_crore(price: float | None) -> float | None:
    if price is None or pd.isna(price):
        return None
    try:
        price = float(price)
    except (ValueError, TypeError):
        return None
    if price <= 0:
        return None
    if price > 100:
        return price / 10000000.0
    return price


RE_SECTOR = re.compile(r"sector\s*(\d+[a-z]?)", re.IGNORECASE)


def extract_sector_from_text(text: str | None) -> str | None:
    """Extract sector number from address or location text."""
    if not text or pd.isna(text):
        return None
    text = str(text)
    match = RE_SECTOR.search(text)
    if match:
        return f"sector_{match.group(1).lower()}"
    return None


def extract_city_from_address(address: str | None) -> str | None:
    if not address or pd.isna(address):
        return None
    address = str(address).lower()
    city_patterns = {
        "gurgaon": ["gurgaon", "gurugram"],
        "mumbai": ["mumbai"],
        "hyderabad": ["hyderabad"],
        "kolkata": ["kolkata", "calcutta"],
        "delhi": ["delhi", "new delhi"],
        "bangalore": ["bangalore", "bengaluru"],
        "chennai": ["chennai"],
        "pune": ["pune"],
        "noida": ["noida"],
        "greater noida": ["greater noida"],
        "faridabad": ["faridabad"],
        "ghaziabad": ["ghaziabad"],
        "thane": ["thane"],
        "navi mumbai": ["navi mumbai"],
    }
    for city, patterns in city_patterns.items():
        for pattern in patterns:
            if pattern in address:
                return city
    return None


def load_campusx_data(root: Path) -> pd.DataFrame:
    """Load CampusX 99acres data."""
    campusx_dir = root / "data" / "raw" / "99acres_campusx"
    if not campusx_dir.exists():
        return pd.DataFrame()

    dfs = []
    for csv_file in campusx_dir.glob("*.csv"):
        if "gurgaon_properties_cleaned" in csv_file.name:
            continue
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            prop_type = "flat" if "flat" in csv_file.name.lower() else "house"
            if "property_type" not in df.columns:
                df["property_type"] = prop_type
            df["source"] = "campusx"
            df["source_file"] = csv_file.name
            dfs.append(df)
        except Exception as e:
            print(f"  Error loading {csv_file.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    text_cols = ["description", "features", "property_name", "furnishDetails"]
    combined["_vaastu_text"] = ""
    for col in text_cols:
        if col in combined.columns:
            combined["_vaastu_text"] += " " + combined[col].fillna("").astype(str)

    combined["vaastu_mentioned"] = combined["_vaastu_text"].apply(detect_vaastu).astype(int)
    combined["vaastu_mentions_text"] = combined["_vaastu_text"].apply(extract_vaastu_mentions)
    combined.drop(columns=["_vaastu_text"], inplace=True)

    combined["city"] = "gurgaon"
    if "address" in combined.columns:
        combined["city"] = combined["address"].apply(extract_city_from_address).fillna("gurgaon")

    if "sector" not in combined.columns:
        combined["sector"] = None
    for col in ["address", "society", "nearbyLocations"]:
        if col in combined.columns:
            mask = combined["sector"].isna()
            combined.loc[mask, "sector"] = combined.loc[mask, col].apply(extract_sector_from_text)

    rename_map = {
        "price": "price_crore",
        "area": "builtup_area_sqft",
        "bedRoom": "bhk",
        "bathroom": "bathrooms",
        "balcony": "balconies",
    }
    combined.rename(columns={k: v for k, v in rename_map.items() if k in combined.columns}, inplace=True)

    return combined


def load_arvanshul_data(root: Path) -> pd.DataFrame:
    """Load arvanshul multi-city Kaggle dataset."""
    arvanshul_dir = root / "data" / "raw" / "99acres_kaggle" / "arvanshul"
    if not arvanshul_dir.exists():
        return pd.DataFrame()

    city_files = {
        "gurgaon_10k.csv": "gurgaon",
        "mumbai.csv": "mumbai",
        "hyderabad.csv": "hyderabad",
        "kolkata.csv": "kolkata",
    }

    dfs = []
    for filename, city in city_files.items():
        csv_path = arvanshul_dir / filename
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            df["city"] = city
            df["source"] = "kaggle_arvanshul"
            df["source_file"] = filename
            dfs.append(df)
        except Exception as e:
            print(f"  Error loading {filename}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    text_cols = ["DESCRIPTION", "FEATURES", "AMENITIES", "PROP_HEADING"]
    combined["_vaastu_text"] = ""
    for col in text_cols:
        if col in combined.columns:
            combined["_vaastu_text"] += " " + combined[col].fillna("").astype(str)

    combined["vaastu_mentioned"] = combined["_vaastu_text"].apply(detect_vaastu).astype(int)
    combined["vaastu_mentions_text"] = combined["_vaastu_text"].apply(extract_vaastu_mentions)
    combined.drop(columns=["_vaastu_text"], inplace=True)

    rename_map = {
        "PROP_ID": "property_id",
        "PROPERTY_TYPE": "property_type",
        "BEDROOM_NUM": "bhk",
        "BATHROOM_NUM": "bathrooms",
        "BALCONY_NUM": "balconies",
        "CARPET_SQFT": "carpet_area_sqft",
        "SUPERBUILTUP_SQFT": "builtup_area_sqft",
        "MIN_PRICE": "price_crore",
        "PRICE_SQFT": "price_per_sqft",
        "FURNISH": "furnishing",
        "FACING": "facing",
        "AGE": "property_age",
        "FLOOR_NUM": "floor_number",
        "TOTAL_FLOOR": "total_floors",
        "LOCALITY": "locality",
        "DESCRIPTION": "description",
        "FEATURES": "features",
        "AMENITIES": "amenities",
    }
    combined.rename(columns={k: v for k, v in rename_map.items() if k in combined.columns}, inplace=True)

    if "price_crore" in combined.columns:
        combined["price_crore"] = pd.to_numeric(combined["price_crore"], errors="coerce")
        mask = combined["price_crore"] > 100
        combined.loc[mask, "price_crore"] = combined.loc[mask, "price_crore"] / 10000000.0

    return combined




def load_our_scraper_data(root: Path) -> pd.DataFrame:
    """Load data from our 99acres scraper if present."""
    scraper_dir = root / "data" / "raw" / "99acres"
    if not scraper_dir.exists():
        return pd.DataFrame()

    dfs = []
    for city_dir in scraper_dir.iterdir():
        if not city_dir.is_dir():
            continue
        csv_path = city_dir / "parsed_listings.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                df["source"] = "our_scraper"
                df["source_file"] = f"{city_dir.name}/parsed_listings.csv"
                dfs.append(df)
            except Exception as e:
                print(f"  Error loading {csv_path}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column names and types."""
    standard_cols = [
        "property_id",
        "source",
        "source_file",
        "city",
        "property_type",
        "locality",
        "sector",
        "price_crore",
        "price_per_sqft",
        "builtup_area_sqft",
        "carpet_area_sqft",
        "bhk",
        "bathrooms",
        "balconies",
        "furnishing",
        "facing",
        "floor_number",
        "total_floors",
        "property_age",
        "description",
        "features",
        "amenities",
        "vaastu_mentioned",
        "vaastu_mentions_text",
    ]

    for col in standard_cols:
        if col not in df.columns:
            df[col] = None

    if "property_id" not in df.columns or df["property_id"].isna().all():
        df["property_id"] = range(1, len(df) + 1)

    return df


def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute summary statistics for the manifest."""
    stats = {
        "total_listings": len(df),
        "vaastu_mentioned_count": int(df["vaastu_mentioned"].sum()),
        "vaastu_mention_rate": float(df["vaastu_mentioned"].mean() * 100),
    }

    by_source = df.groupby("source").agg(
        count=("source", "size"),
        vaastu_count=("vaastu_mentioned", "sum"),
        vaastu_rate=("vaastu_mentioned", "mean"),
    ).round(4)
    by_source["vaastu_rate"] = (by_source["vaastu_rate"] * 100).round(2)
    stats["by_source"] = by_source.to_dict("index")

    by_city = df.groupby("city").agg(
        count=("city", "size"),
        vaastu_count=("vaastu_mentioned", "sum"),
        vaastu_rate=("vaastu_mentioned", "mean"),
    ).round(4)
    by_city["vaastu_rate"] = (by_city["vaastu_rate"] * 100).round(2)
    stats["by_city"] = by_city.to_dict("index")

    if "property_type" in df.columns:
        df = df.copy()
        df["property_type_clean"] = df["property_type"].fillna("unknown").astype(str).str.lower()
        by_type = df.groupby("property_type_clean").agg(
            count=("property_type_clean", "size"),
            vaastu_count=("vaastu_mentioned", "sum"),
            vaastu_rate=("vaastu_mentioned", "mean"),
        ).round(4)
        by_type["vaastu_rate"] = (by_type["vaastu_rate"] * 100).round(2)
        stats["by_property_type"] = by_type.to_dict("index")

    numeric_cols = ["price_crore", "builtup_area_sqft", "bhk", "bathrooms"]
    field_coverage = {}
    for col in numeric_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            field_coverage[col] = {
                "non_null": int(non_null),
                "coverage_pct": round(non_null / len(df) * 100, 2),
            }
    stats["field_coverage"] = field_coverage

    return stats


def main() -> None:
    root = project_root()
    derived_dir = root / "data" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")

    print("  Loading CampusX data...")
    campusx_df = load_campusx_data(root)
    print(f"    {len(campusx_df)} rows")

    print("  Loading arvanshul Kaggle data...")
    arvanshul_df = load_arvanshul_data(root)
    print(f"    {len(arvanshul_df)} rows")

    print("  Loading our scraper data...")
    scraper_df = load_our_scraper_data(root)
    print(f"    {len(scraper_df)} rows")

    all_dfs = [df for df in [campusx_df, arvanshul_df, scraper_df] if len(df) > 0]

    if not all_dfs:
        print("ERROR: No data loaded!")
        sys.exit(1)

    print("\nCombining datasets...")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = normalize_columns(combined)
    print(f"  Total: {len(combined)} rows")

    print("\nComputing statistics...")
    stats = compute_statistics(combined)

    print(f"\n=== Summary ===")
    print(f"Total listings: {stats['total_listings']}")
    print(f"Vaastu mentions: {stats['vaastu_mentioned_count']} ({stats['vaastu_mention_rate']:.1f}%)")

    print("\nBy source:")
    for source, data in stats["by_source"].items():
        print(f"  {source}: {int(data['count'])} listings, {data['vaastu_rate']:.1f}% vaastu")

    print("\nBy city:")
    for city, data in stats["by_city"].items():
        print(f"  {city}: {int(data['count'])} listings, {data['vaastu_rate']:.1f}% vaastu")

    output_cols = [
        "property_id",
        "source",
        "city",
        "property_type",
        "locality",
        "sector",
        "price_crore",
        "price_per_sqft",
        "builtup_area_sqft",
        "carpet_area_sqft",
        "bhk",
        "bathrooms",
        "balconies",
        "furnishing",
        "facing",
        "floor_number",
        "total_floors",
        "property_age",
        "vaastu_mentioned",
        "vaastu_mentions_text",
        "description",
        "features",
        "amenities",
    ]

    output_cols = [c for c in output_cols if c in combined.columns]
    output_df = combined[output_cols]

    output_csv = derived_dir / "all_99acres_vaastu.csv"
    output_df.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")

    manifest = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "statistics": stats,
        "output_file": str(output_csv.relative_to(root)),
        "columns": output_cols,
    }

    manifest_path = derived_dir / "data_manifest.json"

    def convert_numpy(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=convert_numpy)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
