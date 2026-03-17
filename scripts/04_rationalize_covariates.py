#!/usr/bin/env python3
"""Verify covariate availability across data sources.

Outputs:
- Console report showing field coverage by source
- data/derived/covariate_coverage.json - structured coverage data

Usage
-----
python scripts/04_rationalize_covariates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("pandas required: pip install pandas")
    sys.exit(1)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def compute_coverage(df: pd.DataFrame, source_name: str) -> dict:
    """Compute field coverage statistics for a source."""
    n = len(df)
    if n == 0:
        return {}

    key_fields = [
        "price_crore",
        "builtup_area_sqft",
        "carpet_area_sqft",
        "bhk",
        "bathrooms",
        "balconies",
        "locality",
        "sector",
        "facing",
        "furnishing",
        "floor_number",
        "total_floors",
        "property_age",
        "vaastu_mentioned",
    ]

    coverage = {"n": n, "source": source_name, "fields": {}}

    for field in key_fields:
        if field in df.columns:
            non_null = df[field].notna().sum()
            if df[field].dtype == "object":
                non_null = (df[field].notna() & (df[field].astype(str).str.strip() != "")).sum()
            pct = round(non_null / n * 100, 1)
            coverage["fields"][field] = {"non_null": int(non_null), "coverage_pct": pct}
        else:
            coverage["fields"][field] = {"non_null": 0, "coverage_pct": 0.0}

    return coverage


def print_coverage_table(coverage_by_source: dict) -> None:
    """Print coverage table to console."""
    fields = [
        "price_crore",
        "builtup_area_sqft",
        "bhk",
        "bathrooms",
        "locality",
        "sector",
        "facing",
        "vaastu_mentioned",
    ]

    sources = list(coverage_by_source.keys())
    header = f"{'Field':<20}" + "".join(f"{s:<18}" for s in sources)
    print(header)
    print("-" * len(header))

    for field in fields:
        row = f"{field:<20}"
        for source in sources:
            cov = coverage_by_source[source]["fields"].get(field, {})
            pct = cov.get("coverage_pct", 0)
            row += f"{pct:>6.1f}%{' ':>10}"
        print(row)

    print()
    print("Sample sizes:")
    for source in sources:
        n = coverage_by_source[source]["n"]
        print(f"  {source}: n={n:,}")


def check_warnings(coverage_by_source: dict) -> list[str]:
    """Check for fields with low coverage."""
    warnings = []
    critical_fields = ["price_crore", "bhk", "vaastu_mentioned"]

    for source, cov in coverage_by_source.items():
        n = cov["n"]
        if n < 100:
            warnings.append(f"WARNING: {source} has only {n} observations (too small for analysis)")

        for field in critical_fields:
            field_cov = cov["fields"].get(field, {}).get("coverage_pct", 0)
            if field_cov < 50:
                warnings.append(
                    f"WARNING: {source} has only {field_cov:.1f}% coverage for critical field '{field}'"
                )

    return warnings


def main() -> None:
    root = project_root()
    derived_dir = root / "data" / "derived"

    csv_path = derived_dir / "all_99acres_vaastu.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run 03_extract_vaastu.py first.")
        sys.exit(1)

    print("Loading unified data...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Total: {len(df)} rows")

    print("\nComputing coverage by source...")
    coverage_by_source = {}
    for source in df["source"].unique():
        source_df = df[df["source"] == source]
        coverage_by_source[source] = compute_coverage(source_df, source)
        print(f"  {source}: {len(source_df)} rows")

    print("\n=== Covariate Coverage by Source ===\n")
    print_coverage_table(coverage_by_source)

    warnings = check_warnings(coverage_by_source)
    if warnings:
        print("\n=== Warnings ===")
        for w in warnings:
            print(f"  {w}")

    output = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "coverage_by_source": coverage_by_source,
        "warnings": warnings,
    }

    output_path = derived_dir / "covariate_coverage.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
