#!/usr/bin/env python3
"""
Analysis of Magicbricks data for Vaastu premium estimation.

Runs hedonic regressions on Magicbricks listings.

Usage:
    uv run python scripts/04_analyze_magicbricks.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.utils import get_significance_stars

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "magicbricks"
TABS_DIR = Path(__file__).parent.parent / "tabs"


def load_magicbricks_data() -> pd.DataFrame:
    """Load and combine all Magicbricks city data."""
    all_data = []

    for city_dir in DATA_DIR.iterdir():
        if not city_dir.is_dir():
            continue

        parquet_path = city_dir / "parsed_listings.parquet"
        if not parquet_path.exists():
            continue

        df = pd.read_parquet(parquet_path)
        if len(df) < 10:
            continue

        df["city"] = city_dir.name
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for regression."""
    df = df.copy()

    df = df.dropna(subset=["price_crore", "builtup_area_sqft", "bhk"])

    df = df[df["price_crore"] > 0]
    df = df[df["builtup_area_sqft"] > 0]
    df = df[df["bhk"] > 0]
    df = df[df["bhk"] <= 10]

    df["ln_price"] = np.log(df["price_crore"])
    df["ln_area"] = np.log(df["builtup_area_sqft"])
    df["vaastu"] = df["vaastu_mentioned"].fillna(0).astype(int)
    df["bhk_cat"] = pd.Categorical(df["bhk"].astype(int).clip(upper=6))

    df["facing_clean"] = df["facing"].fillna("Unknown").astype(str).str.strip()
    df["facing_clean"] = df["facing_clean"].replace({"": "Unknown", "nan": "Unknown"})

    df["furnishing_clean"] = df["furnishing"].fillna("Unknown").astype(str).str.strip()
    df["furnishing_clean"] = df["furnishing_clean"].replace({"": "Unknown", "nan": "Unknown"})

    # Extract base city (remove property type suffix)
    df["base_city"] = df["city"].apply(lambda x: x.split("-")[0] if "-" in x else x)

    return df


def print_data_summary(df: pd.DataFrame) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("MAGICBRICKS DATA SUMMARY")
    print("=" * 70)

    print(f"\nTotal listings: {len(df):,}")
    print(f"Cities: {df['city'].nunique()}")
    print(f"Base cities: {df['base_city'].nunique()}")

    print("\n## By Base City")
    summary = (
        df.groupby("base_city")
        .agg(
            n=("vaastu", "count"),
            vaastu_n=("vaastu", "sum"),
            vaastu_pct=("vaastu", lambda x: 100 * x.mean()),
            mean_price=("price_crore", "mean"),
            mean_area=("builtup_area_sqft", "mean"),
        )
        .round(2)
    )
    summary["vaastu_pct"] = summary["vaastu_pct"].round(1)
    summary = summary.sort_values("n", ascending=False)
    print(summary.to_string())

    vaastu_n = df["vaastu"].sum()
    vaastu_pct = 100 * df["vaastu"].mean()
    print(f"\nTotal: {len(df):,} listings, {vaastu_n:,} vaastu ({vaastu_pct:.1f}%)")

    print("\n## Price Summary")
    print(f"Valid prices: {len(df):,}")
    print(f"Min: ₹{df['price_crore'].min():.2f} Cr")
    print(f"Max: ₹{df['price_crore'].max():.2f} Cr")
    print(f"Median: ₹{df['price_crore'].median():.2f} Cr")

    print("\n## Facing Distribution")
    print(df["facing_clean"].value_counts().head(10).to_string())


def run_regressions(df: pd.DataFrame) -> dict:
    """Run hedonic regressions and return results."""
    results = {}

    print("\n" + "=" * 70)
    print("MAGICBRICKS VAASTU PREMIUM ANALYSIS")
    print("=" * 70)

    # Model 1: Basic
    print("\n" + "-" * 70)
    print("Model 1: ln_price ~ vaastu")
    print("-" * 70)
    m1 = smf.ols("ln_price ~ vaastu", data=df).fit(cov_type="HC3")
    coef = m1.params["vaastu"]
    se = m1.bse["vaastu"]
    pval = m1.pvalues["vaastu"]
    pct = (np.exp(coef) - 1) * 100
    print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
    print(f"Premium: {pct:+.1f}%")
    results["m1"] = {"coef": coef, "se": se, "pval": pval, "pct": pct}

    # Model 2: + BHK
    print("\n" + "-" * 70)
    print("Model 2: ln_price ~ vaastu + bhk_cat")
    print("-" * 70)
    m2 = smf.ols("ln_price ~ vaastu + bhk_cat", data=df).fit(cov_type="HC3")
    coef = m2.params["vaastu"]
    se = m2.bse["vaastu"]
    pval = m2.pvalues["vaastu"]
    pct = (np.exp(coef) - 1) * 100
    print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
    print(f"Premium: {pct:+.1f}%")
    results["m2"] = {"coef": coef, "se": se, "pval": pval, "pct": pct}

    # Model 3: + Area
    print("\n" + "-" * 70)
    print("Model 3: ln_price ~ vaastu + bhk_cat + ln_area")
    print("-" * 70)
    m3 = smf.ols("ln_price ~ vaastu + bhk_cat + ln_area", data=df).fit(cov_type="HC3")
    coef = m3.params["vaastu"]
    se = m3.bse["vaastu"]
    pval = m3.pvalues["vaastu"]
    pct = (np.exp(coef) - 1) * 100
    print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
    print(f"Premium: {pct:+.1f}%")
    results["m3"] = {"coef": coef, "se": se, "pval": pval, "pct": pct}

    # Model 4: + City FE
    print("\n" + "-" * 70)
    print("Model 4: ln_price ~ vaastu + bhk_cat + ln_area + C(base_city)")
    print("-" * 70)
    m4 = smf.ols("ln_price ~ vaastu + bhk_cat + ln_area + C(base_city)", data=df).fit(
        cov_type="HC3"
    )
    coef = m4.params["vaastu"]
    se = m4.bse["vaastu"]
    pval = m4.pvalues["vaastu"]
    pct = (np.exp(coef) - 1) * 100
    print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
    print(f"Premium: {pct:+.1f}%")
    print(f"R²: {m4.rsquared:.3f}, N: {m4.nobs:.0f}")
    results["m4"] = {
        "coef": coef,
        "se": se,
        "pval": pval,
        "pct": pct,
        "r2": m4.rsquared,
        "n": m4.nobs,
    }

    # Model 5: + Facing
    print("\n" + "-" * 70)
    print("Model 5: + C(facing_clean)")
    print("-" * 70)
    m5 = smf.ols(
        "ln_price ~ vaastu + bhk_cat + ln_area + C(base_city) + C(facing_clean)", data=df
    ).fit(cov_type="HC3")
    coef = m5.params["vaastu"]
    se = m5.bse["vaastu"]
    pval = m5.pvalues["vaastu"]
    pct = (np.exp(coef) - 1) * 100
    print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
    print(f"Premium: {pct:+.1f}%")
    print(f"R²: {m5.rsquared:.3f}, N: {m5.nobs:.0f}")
    results["m5"] = {
        "coef": coef,
        "se": se,
        "pval": pval,
        "pct": pct,
        "r2": m5.rsquared,
        "n": m5.nobs,
    }

    return results


def run_city_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """Run separate regressions by base city."""
    print("\n" + "=" * 70)
    print("CITY-SPECIFIC VAASTU EFFECTS")
    print("=" * 70)
    print("\nModel: ln_price ~ vaastu + bhk_cat + ln_area\n")

    city_results = []

    for city in sorted(df["base_city"].unique()):
        city_df = df[df["base_city"] == city]
        n = len(city_df)
        vaastu_n = city_df["vaastu"].sum()
        vaastu_pct = 100 * city_df["vaastu"].mean()

        if vaastu_n < 10:
            print(
                f"{city:15s}: n={n:5d}, vaastu={vaastu_n:3d} ({vaastu_pct:5.1f}%) - SKIPPED (too few vaastu)"
            )
            continue

        try:
            m = smf.ols("ln_price ~ vaastu + bhk_cat + ln_area", data=city_df).fit(
                cov_type="HC3"
            )
            coef = m.params["vaastu"]
            se = m.bse["vaastu"]
            pval = m.pvalues["vaastu"]
            pct = (np.exp(coef) - 1) * 100
            sig = get_significance_stars(pval)

            print(
                f"{city:15s}: n={n:5d}, vaastu={vaastu_n:3d} ({vaastu_pct:5.1f}%) | "
                f"Premium: {pct:+6.1f}% (SE: {se:.3f}) {sig}"
            )

            city_results.append({
                "city": city,
                "n": n,
                "vaastu_n": vaastu_n,
                "vaastu_pct": vaastu_pct,
                "coef": coef,
                "se": se,
                "pval": pval,
                "pct": pct,
            })
        except Exception as e:
            print(
                f"{city:15s}: n={n:5d}, vaastu={vaastu_n:3d} ({vaastu_pct:5.1f}%) - ERROR: {e}"
            )

    return pd.DataFrame(city_results)


def main():
    print("Loading Magicbricks data...")
    df = load_magicbricks_data()
    print(f"Loaded {len(df):,} raw listings from {df['city'].nunique()} cities")

    df = prepare_data(df)
    print(f"After cleaning: {len(df):,} listings")

    if len(df) == 0:
        print("ERROR: No valid data after cleaning")
        print("\nNote: Price data appears to be broken - all prices show as 0.50 Cr")
        print("This is due to the parser capturing filter text instead of actual prices.")
        return

    print_data_summary(df)

    # Check for valid price variation
    if df["price_crore"].nunique() <= 1:
        print("\n" + "=" * 70)
        print("WARNING: PRICE DATA ISSUE DETECTED")
        print("=" * 70)
        print(f"All prices are: ₹{df['price_crore'].iloc[0]:.2f} Cr")
        print("This indicates a price parsing bug - regression analysis not meaningful.")
        print("\nVaastu detection is working correctly:")
        print(f"  Vaastu mentioned: {df['vaastu'].sum():,} ({100*df['vaastu'].mean():.1f}%)")
        print("\nTo fix: Update 02_parse_magicbricks.py price extraction logic")
        return

    results = run_regressions(df)
    city_results = run_city_regressions(df)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    m5 = results.get("m5", results.get("m4", results["m3"]))
    sig = get_significance_stars(m5["pval"])
    print(f"""
Magicbricks Overall Effect (with controls):
  Premium: {m5['pct']:+.1f}% {sig}
  p-value: {m5['pval']:.4f}
  N: {m5.get('n', len(df)):.0f}
""")


if __name__ == "__main__":
    main()
