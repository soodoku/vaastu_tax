#!/usr/bin/env python3
"""Download Kaggle datasets for 99acres real estate data.

This script downloads public Kaggle datasets and validates that they contain
text fields (description, amenities, features) necessary for Vaastu extraction.

Datasets without text fields suitable for Vaastu detection are removed.

Usage
-----
python scripts/02_download_kaggle.py

Requires kaggle CLI to be installed and configured:
    pip install kaggle
    # Place kaggle.json in ~/.kaggle/
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("pandas required: pip install pandas")
    sys.exit(1)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


KAGGLE_DATASETS = [
    {
        "name": "arvanshul/gurgaon-real-estate-99acres-com",
        "author": "arvanshul",
        "expected_text_fields": ["DESCRIPTION", "FEATURES", "AMENITIES"],
        "cities": ["gurgaon", "hyderabad", "mumbai", "kolkata"],
        "notes": "Multi-city dataset with 60+ columns including description, features, amenities",
    },
    {
        "name": "shubhammkumaar/real-estate-listings-and-prices-in-india-2025",
        "author": "shubhammkumaar",
        "expected_text_fields": [],  # Need to verify
        "cities": ["multi-city"],
        "notes": "Feb 2025 dataset - need to verify text fields exist",
    },
    {
        "name": "aniket7089/99acres-housing-dataset",
        "author": "aniket7089",
        "expected_text_fields": [],  # Likely minimal - BHK, area, price only
        "cities": ["mumbai"],
        "notes": "Mumbai flats - appears to be numerical only, may lack text fields",
    },
    {
        "name": "arnabk123/real-estate-data-sourced-from-99acres-com",
        "author": "arnabk123",
        "expected_text_fields": [],  # Need to verify
        "cities": ["gurgaon"],
        "notes": "Gurgaon data - need to verify amenities field exists",
    },
]

VAASTU_FIELDS = [
    "description",
    "features",
    "amenities",
    "prop_heading",
    "title",
    "property_details",
    "remarks",
    "about",
    "overview",
]


def run_kaggle_download(dataset_name: str, output_dir: Path) -> bool:
    """Download a Kaggle dataset using the CLI."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle",
        "datasets",
        "download",
        dataset_name,
        "-p",
        str(output_dir),
        "--unzip",
    ]

    print(f"Downloading {dataset_name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False

    print(f"  Downloaded to {output_dir}")
    return True


def find_csv_files(directory: Path) -> list[Path]:
    """Find all CSV files in a directory (recursively)."""
    return list(directory.rglob("*.csv"))


def check_vaastu_text_fields(csv_path: Path) -> tuple[list[str], int, int]:
    """Check if a CSV has text fields suitable for Vaastu extraction.

    Returns: (matching_fields, total_rows, rows_with_vaastu)
    """
    try:
        df = pd.read_csv(csv_path, nrows=5000, low_memory=False)
    except Exception as e:
        print(f"  Error reading {csv_path.name}: {e}")
        return [], 0, 0

    columns_lower = {c.lower(): c for c in df.columns}
    matching = []

    for field in VAASTU_FIELDS:
        if field in columns_lower:
            matching.append(columns_lower[field])

    total_rows = len(df)
    vaastu_count = 0

    if matching:
        for col in matching:
            if col in df.columns:
                text_data = df[col].astype(str).str.lower()
                vaastu_matches = text_data.str.contains(r"\bvaa?stu\b", regex=True, na=False)
                vaastu_count += vaastu_matches.sum()

    return matching, total_rows, vaastu_count


def validate_dataset(dataset_dir: Path, author: str) -> dict:
    """Validate a downloaded dataset has usable text fields."""
    csv_files = find_csv_files(dataset_dir)

    if not csv_files:
        return {
            "author": author,
            "valid": False,
            "reason": "No CSV files found",
            "files": [],
            "text_fields": [],
            "total_rows": 0,
            "vaastu_mentions": 0,
        }

    all_text_fields = set()
    total_rows = 0
    total_vaastu = 0
    file_info = []

    for csv_path in csv_files:
        text_fields, rows, vaastu = check_vaastu_text_fields(csv_path)
        all_text_fields.update(text_fields)
        total_rows += rows
        total_vaastu += vaastu
        file_info.append({
            "file": csv_path.name,
            "rows": int(rows),
            "text_fields": text_fields,
            "vaastu_mentions": int(vaastu),
        })
        print(f"  {csv_path.name}: {rows} rows, text fields: {text_fields}, vaastu: {vaastu}")

    has_text_fields = len(all_text_fields) > 0
    has_vaastu_mentions = total_vaastu > 0

    return {
        "author": author,
        "valid": has_text_fields and has_vaastu_mentions,
        "reason": (
            "Has text fields with Vaastu mentions"
            if (has_text_fields and has_vaastu_mentions)
            else "No Vaastu mentions found" if has_text_fields else "No text fields for Vaastu extraction"
        ),
        "files": file_info,
        "text_fields": list(all_text_fields),
        "total_rows": int(total_rows),
        "vaastu_mentions": int(total_vaastu),
    }


def main() -> None:
    root = project_root()
    kaggle_dir = root / "data" / "raw" / "99acres_kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for dataset in KAGGLE_DATASETS:
        author = dataset["author"]
        name = dataset["name"]
        output_dir = kaggle_dir / author

        print(f"\n=== {name} ===")

        if not run_kaggle_download(name, output_dir):
            results.append({
                "author": author,
                "valid": False,
                "reason": "Download failed",
            })
            continue

        validation = validate_dataset(output_dir, author)
        results.append(validation)

        if not validation["valid"]:
            print(f"  REMOVING: {validation['reason']}")
            shutil.rmtree(output_dir, ignore_errors=True)

    print("\n=== Summary ===")
    valid_datasets = [r for r in results if r.get("valid")]
    invalid_datasets = [r for r in results if not r.get("valid")]

    print(f"Valid datasets: {len(valid_datasets)}")
    for r in valid_datasets:
        print(f"  - {r['author']}: {r['total_rows']} rows, {r['vaastu_mentions']} vaastu mentions")
        print(f"    Text fields: {r['text_fields']}")

    print(f"\nRemoved datasets: {len(invalid_datasets)}")
    for r in invalid_datasets:
        print(f"  - {r['author']}: {r['reason']}")

    manifest_path = kaggle_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"downloads": results}, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
