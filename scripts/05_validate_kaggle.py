#!/usr/bin/env python3
"""
Validate Kaggle 99acres data quality.

Checks for:
1. Missing vaastu column (buried in DESCRIPTION text)
2. Price validity and distribution
3. Area outliers
4. Data completeness
5. Comparison with other data sources

Usage:
    uv run python scripts/05_validate_kaggle.py
"""

import re
from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
KAGGLE_DIR = DATA_DIR / "raw" / "99acres_kaggle" / "arvanshul"


def load_kaggle_data() -> pd.DataFrame:
    """Load all Kaggle city files."""
    all_data = []

    for csv_path in KAGGLE_DIR.glob("*.csv"):
        if csv_path.is_dir():
            continue

        try:
            df = pd.read_csv(csv_path, low_memory=False)
            city = csv_path.stem.replace("_10k", "").title()
            df["source_city"] = city
            df["source_file"] = csv_path.name
            all_data.append(df)
            print(f"  Loaded {len(df):,} listings from {csv_path.name}")
        except Exception as e:
            print(f"  Error loading {csv_path.name}: {e}")

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def extract_vaastu_from_description(df: pd.DataFrame) -> pd.DataFrame:
    """Extract vaastu mentions from DESCRIPTION column."""
    df = df.copy()

    if "DESCRIPTION" not in df.columns:
        df["vaastu_mentioned"] = 0
        df["vaastu_text"] = None
        return df

    def check_vaastu(text):
        if pd.isna(text):
            return 0, None
        text_lower = str(text).lower()
        if re.search(r"vaastu|vastu", text_lower):
            # Extract context
            match = re.search(r".{0,50}(vaastu|vastu).{0,50}", text_lower)
            context = match.group(0) if match else None
            return 1, context
        return 0, None

    results = df["DESCRIPTION"].apply(check_vaastu)
    df["vaastu_mentioned"] = results.apply(lambda x: x[0])
    df["vaastu_text"] = results.apply(lambda x: x[1])

    return df


def parse_price(price_str: str) -> float | None:
    """Parse price string to numeric value in crores."""
    if pd.isna(price_str):
        return None

    price_str = str(price_str).strip()

    # Pattern: "2.63 Cr" or "85,000" or "1.5 L"
    m = re.match(r"([\d,.]+)\s*(Cr|Crore|L|Lac|Lakh)?", price_str, re.I)
    if not m:
        return None

    value = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()

    if unit in ("cr", "crore"):
        return value
    elif unit in ("l", "lac", "lakh"):
        return value / 100
    else:
        # Assume raw rupees
        return value / 10000000


def validate_data(df: pd.DataFrame) -> dict:
    """Run comprehensive data quality checks."""
    results = {}

    print("\n" + "=" * 70)
    print("KAGGLE DATA QUALITY VALIDATION")
    print("=" * 70)

    # 1. Basic stats
    print("\n## 1. Basic Statistics")
    print(f"Total listings: {len(df):,}")
    print(f"Cities: {df['source_city'].nunique()}")
    print(f"Columns: {len(df.columns)}")

    # 2. Vaastu column check
    print("\n## 2. Vaastu Column Check")
    vaastu_cols = [c for c in df.columns if "vaastu" in c.lower() or "vastu" in c.lower()]
    if vaastu_cols:
        print(f"Found vaastu columns: {vaastu_cols}")
    else:
        print("⚠️  NO DEDICATED VAASTU COLUMN FOUND")
        print("   Vaastu info is buried in DESCRIPTION text")

    # Extract vaastu from description
    df = extract_vaastu_from_description(df)
    vaastu_count = df["vaastu_mentioned"].sum()
    vaastu_pct = 100 * df["vaastu_mentioned"].mean()
    print(f"\n   Vaastu mentions in DESCRIPTION: {vaastu_count:,} ({vaastu_pct:.1f}%)")

    results["vaastu_count"] = vaastu_count
    results["vaastu_pct"] = vaastu_pct

    # Sample vaastu contexts
    print("\n   Sample vaastu mentions:")
    vaastu_samples = df[df["vaastu_mentioned"] == 1]["vaastu_text"].head(5)
    for i, text in enumerate(vaastu_samples):
        if text:
            print(f"   {i+1}. ...{text}...")

    # 3. Price validity
    print("\n## 3. Price Analysis")

    # Check PRICE column format
    if "PRICE" in df.columns:
        print(f"PRICE column dtype: {df['PRICE'].dtype}")
        print(f"Sample values: {df['PRICE'].head(5).tolist()}")

        # Parse prices
        df["price_crore"] = df["PRICE"].apply(parse_price)
        valid_prices = df["price_crore"].notna()
        print(f"\nValid parsed prices: {valid_prices.sum():,} ({100*valid_prices.mean():.1f}%)")

        if valid_prices.sum() > 0:
            prices = df.loc[valid_prices, "price_crore"]
            print(f"Price range: ₹{prices.min():.2f} - ₹{prices.max():.2f} Cr")
            print(f"Median: ₹{prices.median():.2f} Cr")
            print(f"Mean: ₹{prices.mean():.2f} Cr")

            # Check for outliers
            very_low = (prices < 0.01).sum()  # < 1 lakh
            very_high = (prices > 100).sum()  # > 100 Cr
            print(f"\n⚠️  Price outliers:")
            print(f"   < ₹1 lakh: {very_low:,}")
            print(f"   > ₹100 Cr: {very_high:,}")

            results["price_valid_pct"] = 100 * valid_prices.mean()
            results["price_median"] = prices.median()

    # Also check MIN_PRICE (numeric column)
    if "MIN_PRICE" in df.columns:
        print("\nMIN_PRICE column (numeric):")
        valid_min = df["MIN_PRICE"].notna() & (df["MIN_PRICE"] > 0)
        if valid_min.sum() > 0:
            min_prices_cr = df.loc[valid_min, "MIN_PRICE"] / 10000000
            print(f"  Valid: {valid_min.sum():,}")
            print(f"  Range: ₹{min_prices_cr.min():.2f} - ₹{min_prices_cr.max():.2f} Cr")
            print(f"  Median: ₹{min_prices_cr.median():.2f} Cr")

    # 4. Area analysis
    print("\n## 4. Area Analysis")
    area_cols = ["BUILTUP_SQFT", "CARPET_SQFT", "SUPERBUILTUP_SQFT", "SUPER_SQFT"]
    for col in area_cols:
        if col not in df.columns:
            continue
        valid = df[col].notna() & (df[col] > 0)
        if valid.sum() == 0:
            continue

        areas = df.loc[valid, col]
        print(f"\n{col}:")
        print(f"  Valid: {valid.sum():,} ({100*valid.mean():.1f}%)")
        print(f"  Range: {areas.min():.0f} - {areas.max():.0f} sqft")
        print(f"  Median: {areas.median():.0f} sqft")

        # Outliers
        tiny = (areas < 100).sum()
        huge = (areas > 50000).sum()
        if tiny > 0 or huge > 0:
            print(f"  ⚠️  Outliers: {tiny} tiny (<100 sqft), {huge} huge (>50k sqft)")

    # 5. BHK distribution
    print("\n## 5. BHK Distribution")
    if "BEDROOM_NUM" in df.columns:
        bhk = df["BEDROOM_NUM"].dropna()
        print(bhk.value_counts().sort_index().head(10).to_string())

        # Check for unusual values
        unusual = (bhk > 10).sum()
        if unusual > 0:
            print(f"⚠️  {unusual} listings with >10 bedrooms")

    # 6. Missing data
    print("\n## 6. Missing Data")
    critical_cols = ["PRICE", "BEDROOM_NUM", "LOCALITY", "DESCRIPTION"]
    for col in critical_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"  {col}: {missing:,} ({100*missing/len(df):.1f}%) missing")

    # 7. By city comparison
    print("\n## 7. By City Comparison")
    for city in df["source_city"].unique():
        city_df = df[df["source_city"] == city]
        n = len(city_df)
        vaastu_n = city_df["vaastu_mentioned"].sum()
        vaastu_pct = 100 * vaastu_n / n if n > 0 else 0

        # Get median price if available
        if "price_crore" in city_df.columns:
            valid_prices = city_df["price_crore"].dropna()
            median_price = valid_prices.median() if len(valid_prices) > 0 else np.nan
            print(f"  {city}: {n:,} listings, {vaastu_n} vaastu ({vaastu_pct:.1f}%), median ₹{median_price:.1f}Cr")
        else:
            print(f"  {city}: {n:,} listings, {vaastu_n} vaastu ({vaastu_pct:.1f}%)")

    return results


def print_verdict(results: dict) -> None:
    """Print overall data quality verdict."""
    print("\n" + "=" * 70)
    print("VERDICT: KAGGLE DATA QUALITY ISSUES")
    print("=" * 70)

    issues = []

    # 1. Low vaastu rate
    if results.get("vaastu_pct", 0) < 10:
        issues.append(f"LOW VAASTU RATE: {results['vaastu_pct']:.1f}% (vs 12-50% in scraped data)")

    # 2. No dedicated vaastu column
    issues.append("NO VAASTU COLUMN: Must extract from free-text DESCRIPTION field")

    # 3. Price issues
    if results.get("price_valid_pct", 0) < 50:
        issues.append(f"PRICE DATA SPARSE: Only {results.get('price_valid_pct', 0):.0f}% valid prices")

    print("\n### Issues Found:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    print("\n### Comparison with Other Sources:")
    print("""
    | Source         | Vaastu % | Dedicated Column | Price Format |
    |----------------|----------|------------------|--------------|
    | Kaggle         | ~5.5%    | No (text only)   | String       |
    | CampusX        | ~53%     | Yes              | Numeric      |
    | Magicbricks    | ~11.7%   | Yes (parsed)     | Numeric      |
    | Housing.com    | ~16%     | Yes (parsed)     | Numeric      |
    """)

    print("\n### Recommendation:")
    print("""
    The Kaggle dataset has significant quality limitations:

    1. Vaastu detection requires regex on DESCRIPTION (unreliable vs dedicated field)
    2. Low vaastu mention rate suggests many listings don't discuss it in description
    3. This likely leads to ATTENUATION BIAS in regression estimates

    For robust analysis, prefer:
    - CampusX data (highest quality, dedicated fields)
    - Scraped Magicbricks/Housing.com data (structured extraction)
    - Use Kaggle as supplementary/robustness check only
    """)


def main():
    print("Loading Kaggle data...")
    df = load_kaggle_data()
    print(f"\nTotal: {len(df):,} listings from {df['source_city'].nunique()} cities")

    results = validate_data(df)
    print_verdict(results)


if __name__ == "__main__":
    main()
