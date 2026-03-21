#!/usr/bin/env python3
"""Export script to package raw HTML archives and parsed data for Harvard Dataverse.

Usage:
    python scripts/export_dataverse.py --output-dir exports/dataverse_v1
    python scripts/export_dataverse.py --output-dir exports/dataverse_v1 --dry-run
"""

import argparse
import gzip
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def scan_source_files(source_dir: Path, pattern: str = "**/*.html.gz") -> list[Path]:
    """Scan directory for files matching pattern."""
    return sorted(source_dir.glob(pattern))


def create_tar_archive(
    files: list[Path],
    archive_path: Path,
    base_dir: Path,
    compression_level: int = 1,
    dry_run: bool = False,
) -> dict:
    """Create tar.gz archive with low compression (files already gzipped)."""
    if dry_run:
        total_size = sum(f.stat().st_size for f in files)
        return {
            "archive_path": str(archive_path),
            "file_count": len(files),
            "source_size_bytes": total_size,
            "dry_run": True,
        }

    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(archive_path, "wb", compresslevel=compression_level) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for filepath in files:
                arcname = filepath.relative_to(base_dir)
                tar.add(filepath, arcname=arcname)

    archive_size = archive_path.stat().st_size
    return {
        "archive_path": str(archive_path),
        "file_count": len(files),
        "archive_size_bytes": archive_size,
        "archive_size_mb": round(archive_size / (1024 * 1024), 2),
    }


def merge_parquet_files(
    parquet_files: list[Path], output_path: Path, dry_run: bool = False
) -> dict:
    """Merge multiple parquet files into one."""
    import pandas as pd

    if dry_run:
        return {
            "output_path": str(output_path),
            "source_files": len(parquet_files),
            "dry_run": True,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        df["source_file"] = pf.parent.name
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_parquet(output_path, index=False)

    return {
        "output_path": str(output_path),
        "source_files": len(parquet_files),
        "total_rows": len(merged),
        "size_bytes": output_path.stat().st_size,
    }


def merge_csv_to_parquet(
    csv_files: list[Path], output_path: Path, dry_run: bool = False
) -> dict:
    """Merge CSV files and save as parquet."""
    import pandas as pd

    if dry_run:
        return {
            "output_path": str(output_path),
            "source_files": len(csv_files),
            "dry_run": True,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    for cf in csv_files:
        df = pd.read_csv(cf)
        df["source_city"] = cf.parent.name
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_parquet(output_path, index=False)

    return {
        "output_path": str(output_path),
        "source_files": len(csv_files),
        "total_rows": len(merged),
        "size_bytes": output_path.stat().st_size,
    }


def generate_file_manifest(
    files: list[Path], base_dir: Path, dry_run: bool = False
) -> dict:
    """Generate manifest with SHA256 checksums for each file."""
    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file_count": len(files),
        "files": {},
    }

    if dry_run:
        manifest["dry_run"] = True
        return manifest

    for filepath in files:
        rel_path = str(filepath.relative_to(base_dir))
        manifest["files"][rel_path] = {
            "sha256": compute_sha256(filepath),
            "size_bytes": filepath.stat().st_size,
        }

    return manifest


def generate_readme(output_dir: Path, stats: dict, dry_run: bool = False) -> None:
    """Generate README.md for the dataset."""
    readme_content = f"""# Vaastu Premium Dataset - Raw Scraped Data

## Overview

This dataset contains raw HTML archives and parsed listing data from Indian real estate
websites, collected for research on Vaastu compliance pricing premiums.

## Data Sources

| Source | Description | Files | Size |
|--------|-------------|-------|------|
| Magicbricks | Property listings from magicbricks.com | {stats.get('magicbricks_html_count', 'N/A')} HTML files | ~{stats.get('magicbricks_html_size_mb', 'N/A')} MB |
| Housing.com | Property listings from housing.com | {stats.get('housingcom_html_count', 'N/A')} HTML files | ~{stats.get('housingcom_html_size_mb', 'N/A')} MB |
| 99acres | Property listings from 99acres.com | {stats.get('99acres_file_count', 'N/A')} files | ~{stats.get('99acres_size_mb', 'N/A')} MB |

## Directory Structure

```
raw/
├── magicbricks_raw_html.tar.gz   # Compressed HTML pages
├── housingcom_raw_html.tar.gz    # Compressed HTML pages
└── 99acres_scraped_raw.tar.gz    # Compressed raw scraped data

parsed/
├── magicbricks_listings.parquet  # Parsed property listings
├── housingcom_listings.parquet   # Parsed property listings
└── 99acres_listings.parquet      # Parsed property listings (if available)

manifests/
├── magicbricks_manifest.json     # File checksums
├── housingcom_manifest.json      # File checksums
└── collection_metadata.json      # Collection metadata
```

## File Formats

- **HTML Archives**: tar.gz files containing gzipped HTML pages
- **Parsed Data**: Apache Parquet format for efficient storage and querying
- **Manifests**: JSON files with SHA256 checksums for integrity verification

## Verification

Verify archive integrity using the provided checksums:
```bash
sha256sum -c checksums.sha256
```

## Collection Period

Data collected: March 2025

## License

This dataset is released for academic research purposes.

## Citation

If using this data, please cite:
[Citation information to be added]

---
Generated: {datetime.now(timezone.utc).isoformat()}Z
"""

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "README.md").write_text(readme_content)


def generate_codebook(output_dir: Path, dry_run: bool = False) -> None:
    """Generate CODEBOOK.md with variable definitions."""
    codebook_content = """# Codebook - Variable Definitions

## Magicbricks Listings (magicbricks_listings.parquet)

| Variable | Type | Description |
|----------|------|-------------|
| property_id | string | Unique identifier for the property listing |
| project_id | string | Project/development identifier |
| url | string | Original listing URL |
| city | string | City name |
| search_page | int | Search results page number |
| title | string | Listing title |
| locality | string | Locality/neighborhood |
| price_display | string | Price as displayed on website |
| price_crore | float | Price in crores (10 million INR) |
| builtup_area_sqft | float | Built-up area in square feet |
| carpet_area_sqft | float | Carpet area in square feet |
| bhk | float | Number of bedrooms |
| bathrooms | float | Number of bathrooms |
| balconies | float | Number of balconies |
| furnishing | string | Furnishing status (Unfurnished/Semi-Furnished/Furnished) |
| facing | string | Property facing direction |
| possession_status | string | Ready to move / Under construction |
| property_age | string | Age of property |
| floor_no | float | Floor number |
| total_floors | float | Total floors in building |
| amenities | string | Comma-separated amenities list |
| description | string | Full property description |
| vaastu_mentioned | int | 1 if Vaastu mentioned in listing, 0 otherwise |
| vaastu_mentions_text | string | Text containing Vaastu mentions |
| seller_type | string | Owner/Agent/Builder |
| rating_overall | float | Overall locality rating |
| rating_connectivity | float | Connectivity rating |
| rating_neighbourhood | float | Neighbourhood rating |
| rating_safety | float | Safety rating |
| project_name | string | Name of the project |
| developer_name | string | Developer/builder name |
| collected_at | string | Data collection timestamp |
| raw_html_path | string | Path to raw HTML file |
| source_file | string | Source city/property type |

## Housing.com Listings (housingcom_listings.parquet)

| Variable | Type | Description |
|----------|------|-------------|
| property_id | string | Unique identifier |
| url | string | Original listing URL |
| city | string | City name |
| search_page | int | Search results page number |
| title | string | Listing title |
| locality_line | string | Locality information |
| price_display | string | Price as displayed |
| price_crore | float | Price in crores |
| builtup_area_sqft | float | Built-up area in square feet |
| avg_price_per_sqft | float | Average price per square foot |
| bhk | float | Number of bedrooms |
| bathrooms | float | Number of bathrooms |
| balconies | float | Number of balconies |
| furnishing | string | Furnishing status |
| facing | string | Property facing direction |
| possession_status | string | Possession/availability status |
| special_highlights | string | Special features highlighted |
| amenities | string | Available amenities |
| about_this_property | string | Property description |
| vaastu_mentioned | int | 1 if Vaastu mentioned, 0 otherwise |
| vaastu_mentions_text | string | Text containing Vaastu mentions |
| last_updated | string | Last update timestamp |
| raw_text_path | string | Path to extracted text |
| raw_html_path | string | Path to raw HTML |
| source_city | string | Source city directory |

## Notes

- Prices in crores: 1 crore = 10 million INR
- Missing values represented as null/NaN
- Vaastu compliance indicators extracted via keyword matching
"""

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "CODEBOOK.md").write_text(codebook_content)


def generate_checksums(output_dir: Path, dry_run: bool = False) -> None:
    """Generate checksums.sha256 for all archive files."""
    if dry_run:
        return

    archives = list(output_dir.rglob("*.tar.gz")) + list(output_dir.rglob("*.parquet"))

    lines = []
    for archive in sorted(archives):
        checksum = compute_sha256(archive)
        rel_path = archive.relative_to(output_dir)
        lines.append(f"{checksum}  {rel_path}")

    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n")


def verify_archive(archive_path: Path) -> bool:
    """Verify archive by test extraction."""
    try:
        with gzip.open(archive_path, "rb") as gz:
            with tarfile.open(fileobj=gz, mode="r") as tar:
                members = tar.getmembers()
                return len(members) > 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export data for Harvard Dataverse upload"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports/dataverse_v1"),
        help="Output directory for exports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without creating files",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    dry_run = args.dry_run

    print(f"{'DRY RUN - ' if dry_run else ''}Dataverse Export Pipeline")
    print("=" * 60)

    data_raw = Path("data/raw")
    stats = {}

    # 1. Scan source directories
    print("\n1. Scanning source directories...")

    mb_html_files = scan_source_files(data_raw / "magicbricks", "**/*.html.gz")
    hc_html_files = scan_source_files(data_raw / "housingcom", "**/*.html.gz")
    acres_files = list((data_raw / "99acres").rglob("*"))
    acres_files = [f for f in acres_files if f.is_file()]

    print(f"   Magicbricks HTML: {len(mb_html_files)} files")
    print(f"   Housing.com HTML: {len(hc_html_files)} files")
    print(f"   99acres files: {len(acres_files)} files")

    stats["magicbricks_html_count"] = len(mb_html_files)
    stats["housingcom_html_count"] = len(hc_html_files)
    stats["99acres_file_count"] = len(acres_files)

    # Calculate sizes
    mb_size = sum(f.stat().st_size for f in mb_html_files)
    hc_size = sum(f.stat().st_size for f in hc_html_files)
    acres_size = sum(f.stat().st_size for f in acres_files)

    stats["magicbricks_html_size_mb"] = round(mb_size / (1024 * 1024), 1)
    stats["housingcom_html_size_mb"] = round(hc_size / (1024 * 1024), 1)
    stats["99acres_size_mb"] = round(acres_size / (1024 * 1024), 1)

    print(f"   Magicbricks size: {stats['magicbricks_html_size_mb']} MB")
    print(f"   Housing.com size: {stats['housingcom_html_size_mb']} MB")
    print(f"   99acres size: {stats['99acres_size_mb']} MB")

    # 2. Create raw HTML archives
    print("\n2. Creating raw HTML archives...")

    raw_dir = output_dir / "raw"

    # Magicbricks archive
    print("   Creating magicbricks_raw_html.tar.gz...")
    mb_archive_result = create_tar_archive(
        mb_html_files,
        raw_dir / "magicbricks_raw_html.tar.gz",
        data_raw / "magicbricks",
        dry_run=dry_run,
    )
    print(
        f"   -> {mb_archive_result.get('archive_size_mb', 'N/A')} MB, {mb_archive_result['file_count']} files"
    )

    # Housing.com archive
    print("   Creating housingcom_raw_html.tar.gz...")
    hc_archive_result = create_tar_archive(
        hc_html_files,
        raw_dir / "housingcom_raw_html.tar.gz",
        data_raw / "housingcom",
        dry_run=dry_run,
    )
    print(
        f"   -> {hc_archive_result.get('archive_size_mb', 'N/A')} MB, {hc_archive_result['file_count']} files"
    )

    # 99acres archive
    print("   Creating 99acres_scraped_raw.tar.gz...")
    acres_archive_result = create_tar_archive(
        acres_files,
        raw_dir / "99acres_scraped_raw.tar.gz",
        data_raw / "99acres",
        dry_run=dry_run,
    )
    print(
        f"   -> {acres_archive_result.get('archive_size_mb', 'N/A')} MB, {acres_archive_result['file_count']} files"
    )

    # 3. Merge parquet/CSV files
    print("\n3. Merging parsed data files...")

    parsed_dir = output_dir / "parsed"

    # Magicbricks parquet files
    mb_parquet_files = sorted((data_raw / "magicbricks").glob("*/parsed_listings.parquet"))
    print(f"   Found {len(mb_parquet_files)} magicbricks parquet files")
    if mb_parquet_files:
        mb_merged = merge_parquet_files(
            mb_parquet_files,
            parsed_dir / "magicbricks_listings.parquet",
            dry_run=dry_run,
        )
        print(f"   -> magicbricks_listings.parquet: {mb_merged.get('total_rows', 'N/A')} rows")

    # Housing.com CSV files -> parquet
    hc_csv_files = sorted((data_raw / "housingcom").glob("*/parsed_listings.csv"))
    print(f"   Found {len(hc_csv_files)} housingcom CSV files")
    if hc_csv_files:
        hc_merged = merge_csv_to_parquet(
            hc_csv_files,
            parsed_dir / "housingcom_listings.parquet",
            dry_run=dry_run,
        )
        print(f"   -> housingcom_listings.parquet: {hc_merged.get('total_rows', 'N/A')} rows")

    # 99acres - check for parsed data
    acres_parquet = list((data_raw / "99acres").glob("*/parsed_listings.parquet"))
    acres_csv = list((data_raw / "99acres").glob("*/parsed_listings.csv"))
    if acres_parquet:
        print(f"   Found {len(acres_parquet)} 99acres parquet files")
        acres_merged = merge_parquet_files(
            acres_parquet,
            parsed_dir / "99acres_listings.parquet",
            dry_run=dry_run,
        )
        print(f"   -> 99acres_listings.parquet: {acres_merged.get('total_rows', 'N/A')} rows")
    elif acres_csv:
        print(f"   Found {len(acres_csv)} 99acres CSV files")
        acres_merged = merge_csv_to_parquet(
            acres_csv,
            parsed_dir / "99acres_listings.parquet",
            dry_run=dry_run,
        )
        print(f"   -> 99acres_listings.parquet: {acres_merged.get('total_rows', 'N/A')} rows")
    else:
        print("   No parsed 99acres data found (raw files only)")

    # 4. Generate manifests
    print("\n4. Generating manifests...")

    manifests_dir = output_dir / "manifests"
    if not dry_run:
        manifests_dir.mkdir(parents=True, exist_ok=True)

    # Magicbricks manifest
    print("   Generating magicbricks_manifest.json...")
    mb_manifest = generate_file_manifest(
        mb_html_files, data_raw / "magicbricks", dry_run=dry_run
    )
    if not dry_run:
        (manifests_dir / "magicbricks_manifest.json").write_text(
            json.dumps(mb_manifest, indent=2)
        )

    # Housing.com manifest
    print("   Generating housingcom_manifest.json...")
    hc_manifest = generate_file_manifest(
        hc_html_files, data_raw / "housingcom", dry_run=dry_run
    )
    if not dry_run:
        (manifests_dir / "housingcom_manifest.json").write_text(
            json.dumps(hc_manifest, indent=2)
        )

    # Collection metadata
    print("   Generating collection_metadata.json...")
    collection_metadata = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "magicbricks": {
                "html_files": len(mb_html_files),
                "size_bytes": mb_size,
            },
            "housingcom": {
                "html_files": len(hc_html_files),
                "size_bytes": hc_size,
            },
            "99acres": {
                "files": len(acres_files),
                "size_bytes": acres_size,
            },
        },
    }
    if not dry_run:
        (manifests_dir / "collection_metadata.json").write_text(
            json.dumps(collection_metadata, indent=2)
        )

    # 5. Generate documentation
    print("\n5. Generating documentation...")
    print("   README.md")
    generate_readme(output_dir, stats, dry_run=dry_run)
    print("   CODEBOOK.md")
    generate_codebook(output_dir, dry_run=dry_run)

    # 6. Generate top-level checksums
    print("\n6. Generating checksums.sha256...")
    generate_checksums(output_dir, dry_run=dry_run)

    # 7. Verify archives
    if not dry_run:
        print("\n7. Verifying archive integrity...")
        for archive in sorted((output_dir / "raw").glob("*.tar.gz")):
            print(f"   Verifying {archive.name}...", end=" ")
            if verify_archive(archive):
                print("OK")
            else:
                print("FAILED")

    # 8. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not dry_run:
        total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        print(f"Total export size: {total_size / (1024 * 1024):.1f} MB")
        print(f"Output directory: {output_dir}")

        print("\nArchive sizes:")
        for archive in sorted(output_dir.rglob("*.tar.gz")):
            size_mb = archive.stat().st_size / (1024 * 1024)
            print(f"  {archive.name}: {size_mb:.1f} MB")

        print("\nParsed data sizes:")
        for pq_file in sorted(output_dir.rglob("*.parquet")):
            size_mb = pq_file.stat().st_size / (1024 * 1024)
            print(f"  {pq_file.name}: {size_mb:.1f} MB")
    else:
        print("Dry run complete. No files created.")
        print(f"Would create exports in: {output_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
