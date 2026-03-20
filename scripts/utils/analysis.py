"""Shared utilities for hedonic regression analysis."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


def prepare_regression_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for hedonic regression.

    Requires columns: price_crore, builtup_area_sqft (or similar), bhk, vaastu_mentioned
    """
    df = df.copy()

    # Handle price - try various column names
    price_col = None
    for col in ["price_crore", "price_cr", "PRICE"]:
        if col in df.columns:
            price_col = col
            break

    if price_col is None:
        raise ValueError("No price column found")

    # Handle area - try various column names
    area_col = None
    for col in ["builtup_area_sqft", "super_area", "BUILDUP_AREA", "carpet_area_sqft"]:
        if col in df.columns:
            area_col = col
            break

    if area_col is None:
        raise ValueError("No area column found")

    # Handle BHK
    bhk_col = None
    for col in ["bhk", "BHK", "BEDROOM_NUM", "bedrooms"]:
        if col in df.columns:
            bhk_col = col
            break

    if bhk_col is None:
        raise ValueError("No BHK column found")

    # Normalize column names
    df["price_crore"] = pd.to_numeric(df[price_col], errors="coerce")
    df["area_sqft"] = pd.to_numeric(df[area_col], errors="coerce")
    df["bhk"] = pd.to_numeric(df[bhk_col], errors="coerce")

    # Handle vaastu
    if "vaastu_mentioned" in df.columns:
        df["vaastu"] = df["vaastu_mentioned"].fillna(0).astype(int)
    elif "vaastu" in df.columns:
        df["vaastu"] = df["vaastu"].fillna(0).astype(int)
    else:
        raise ValueError("No vaastu column found")

    # Filter valid data
    df = df.dropna(subset=["price_crore", "area_sqft", "bhk"])
    df = df[df["price_crore"] > 0]
    df = df[df["area_sqft"] > 0]
    df = df[df["bhk"] > 0]
    df = df[df["bhk"] <= 10]

    # Create regression variables
    df["ln_price"] = np.log(df["price_crore"])
    df["ln_area"] = np.log(df["area_sqft"])
    df["bhk_cat"] = pd.Categorical(df["bhk"].astype(int).clip(upper=6))

    # Clean facing
    if "facing" in df.columns:
        df["facing_clean"] = df["facing"].fillna("Unknown").astype(str).str.strip()
        df["facing_clean"] = df["facing_clean"].replace({"": "Unknown", "nan": "Unknown"})

    # Clean furnishing
    if "furnishing" in df.columns:
        df["furnishing_clean"] = df["furnishing"].fillna("Unknown").astype(str).str.strip()
        df["furnishing_clean"] = df["furnishing_clean"].replace({"": "Unknown", "nan": "Unknown"})

    return df


def run_hedonic_models(
    df: pd.DataFrame,
    city_col: str | None = "city",
    print_output: bool = True,
) -> dict:
    """Run progressive hedonic regression models.

    Returns dict with model results for each specification.
    """
    results = {}

    if print_output:
        print("\n" + "-" * 70)
        print("Model 1: ln_price ~ vaastu")
        print("-" * 70)
    m1 = smf.ols("ln_price ~ vaastu", data=df).fit(cov_type="HC3")
    coef = m1.params["vaastu"]
    se = m1.bse["vaastu"]
    pval = m1.pvalues["vaastu"]
    pct = (np.exp(coef) - 1) * 100
    if print_output:
        print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
        print(f"Premium: {pct:+.1f}%")
    results["m1"] = {"coef": coef, "se": se, "pval": pval, "pct": pct, "n": int(m1.nobs)}

    if print_output:
        print("\n" + "-" * 70)
        print("Model 2: ln_price ~ vaastu + bhk_cat")
        print("-" * 70)
    m2 = smf.ols("ln_price ~ vaastu + bhk_cat", data=df).fit(cov_type="HC3")
    coef = m2.params["vaastu"]
    se = m2.bse["vaastu"]
    pval = m2.pvalues["vaastu"]
    pct = (np.exp(coef) - 1) * 100
    if print_output:
        print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
        print(f"Premium: {pct:+.1f}%")
    results["m2"] = {"coef": coef, "se": se, "pval": pval, "pct": pct, "n": int(m2.nobs)}

    if print_output:
        print("\n" + "-" * 70)
        print("Model 3: ln_price ~ vaastu + bhk_cat + ln_area")
        print("-" * 70)
    m3 = smf.ols("ln_price ~ vaastu + bhk_cat + ln_area", data=df).fit(cov_type="HC3")
    coef = m3.params["vaastu"]
    se = m3.bse["vaastu"]
    pval = m3.pvalues["vaastu"]
    pct = (np.exp(coef) - 1) * 100
    if print_output:
        print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
        print(f"Premium: {pct:+.1f}%")
    results["m3"] = {"coef": coef, "se": se, "pval": pval, "pct": pct, "n": int(m3.nobs), "r2": m3.rsquared}

    # Model 4: + City FE (if city column exists)
    if city_col and city_col in df.columns and df[city_col].nunique() > 1:
        if print_output:
            print("\n" + "-" * 70)
            print(f"Model 4: ln_price ~ vaastu + bhk_cat + ln_area + C({city_col})")
            print("-" * 70)
        m4 = smf.ols(f"ln_price ~ vaastu + bhk_cat + ln_area + C({city_col})", data=df).fit(cov_type="HC3")
        coef = m4.params["vaastu"]
        se = m4.bse["vaastu"]
        pval = m4.pvalues["vaastu"]
        pct = (np.exp(coef) - 1) * 100
        if print_output:
            print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
            print(f"Premium: {pct:+.1f}%")
            print(f"R²: {m4.rsquared:.3f}, N: {m4.nobs:.0f}")
        results["m4"] = {"coef": coef, "se": se, "pval": pval, "pct": pct, "n": int(m4.nobs), "r2": m4.rsquared}

    # Model 5: + Facing (if available)
    if "facing_clean" in df.columns and df["facing_clean"].nunique() > 1:
        formula = "ln_price ~ vaastu + bhk_cat + ln_area + C(facing_clean)"
        if city_col and city_col in df.columns and df[city_col].nunique() > 1:
            formula = f"ln_price ~ vaastu + bhk_cat + ln_area + C({city_col}) + C(facing_clean)"

        if print_output:
            print("\n" + "-" * 70)
            print(f"Model 5: + facing")
            print("-" * 70)
        m5 = smf.ols(formula, data=df).fit(cov_type="HC3")
        coef = m5.params["vaastu"]
        se = m5.bse["vaastu"]
        pval = m5.pvalues["vaastu"]
        pct = (np.exp(coef) - 1) * 100
        if print_output:
            print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
            print(f"Premium: {pct:+.1f}%")
            print(f"R²: {m5.rsquared:.3f}, N: {m5.nobs:.0f}")
        results["m5"] = {"coef": coef, "se": se, "pval": pval, "pct": pct, "n": int(m5.nobs), "r2": m5.rsquared}

    return results


def print_data_summary(df: pd.DataFrame, source_name: str, city_col: str | None = "city") -> None:
    """Print summary statistics for a dataset."""
    print("\n" + "=" * 70)
    print(f"{source_name.upper()} DATA SUMMARY")
    print("=" * 70)

    print(f"\nTotal listings: {len(df):,}")

    if city_col and city_col in df.columns:
        print(f"Cities: {df[city_col].nunique()}")
        print("\n## By City")
        summary = df.groupby(city_col).agg(
            n=("vaastu", "count"),
            vaastu_n=("vaastu", "sum"),
            vaastu_pct=("vaastu", lambda x: 100 * x.mean()),
            mean_price=("price_crore", "mean"),
            mean_area=("area_sqft", "mean"),
        ).round(2)
        summary["vaastu_pct"] = summary["vaastu_pct"].round(1)
        print(summary.to_string())

    vaastu_n = df["vaastu"].sum()
    vaastu_pct = 100 * df["vaastu"].mean()
    print(f"\nTotal: {len(df):,} listings, {vaastu_n:,} vaastu ({vaastu_pct:.1f}%)")

    print("\n## Price Summary")
    print(f"Min: ₹{df['price_crore'].min():.2f} Cr")
    print(f"Max: ₹{df['price_crore'].max():.2f} Cr")
    print(f"Median: ₹{df['price_crore'].median():.2f} Cr")
    print(f"Mean: ₹{df['price_crore'].mean():.2f} Cr")

    print("\n## Area Summary")
    print(f"Min: {df['area_sqft'].min():.0f} sqft")
    print(f"Max: {df['area_sqft'].max():.0f} sqft")
    print(f"Median: {df['area_sqft'].median():.0f} sqft")


def get_significance_stars(pval: float) -> str:
    """Return significance stars for p-value."""
    if pval < 0.01:
        return "***"
    elif pval < 0.05:
        return "**"
    elif pval < 0.1:
        return "*"
    return ""
