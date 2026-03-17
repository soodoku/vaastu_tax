#!/usr/bin/env python3
"""
Standalone analysis of Housing.com data for Vaastu premium estimation.

Runs hedonic regressions on housing.com listings:
1. Overall vaastu premium across all cities
2. City-specific effects
3. Comparison summary
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "housingcom"


def load_housingcom_data() -> pd.DataFrame:
    """Load and combine all housing.com city data."""
    all_data = []

    for city_dir in DATA_DIR.iterdir():
        if not city_dir.is_dir():
            continue

        csv_path = city_dir / "parsed_listings.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
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

    df["facing_clean"] = df["facing"].fillna("Unknown").str.strip()
    df["facing_clean"] = df["facing_clean"].replace("", "Unknown")

    df["furnishing_clean"] = df["furnishing"].fillna("Unknown").str.strip()
    df["furnishing_clean"] = df["furnishing_clean"].replace("", "Unknown")

    return df


def run_regressions(df: pd.DataFrame) -> dict:
    """Run hedonic regressions and return results."""
    results = {}

    print("\n" + "=" * 70)
    print("HOUSING.COM VAASTU PREMIUM ANALYSIS")
    print("=" * 70)

    # Data summary
    print("\n## Data Summary by City\n")
    summary = (
        df.groupby("city")
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
    print(summary.to_string())
    print(
        f"\nTotal: {len(df):,} listings, {df['vaastu'].sum():,} vaastu ({100*df['vaastu'].mean():.1f}%)"
    )

    # Model 1: Basic (no controls)
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
    print("Model 4: ln_price ~ vaastu + bhk_cat + ln_area + C(city)")
    print("-" * 70)
    m4 = smf.ols("ln_price ~ vaastu + bhk_cat + ln_area + C(city)", data=df).fit(
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

    # Model 5: + Facing direction
    print("\n" + "-" * 70)
    print("Model 5: ln_price ~ vaastu + bhk_cat + ln_area + C(city) + C(facing_clean)")
    print("-" * 70)
    m5 = smf.ols(
        "ln_price ~ vaastu + bhk_cat + ln_area + C(city) + C(facing_clean)", data=df
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

    # Model 6: + Furnishing
    print("\n" + "-" * 70)
    print("Model 6: + C(furnishing_clean) [Full Controls]")
    print("-" * 70)
    m6 = smf.ols(
        "ln_price ~ vaastu + bhk_cat + ln_area + C(city) + C(facing_clean) + C(furnishing_clean)",
        data=df,
    ).fit(cov_type="HC3")
    coef = m6.params["vaastu"]
    se = m6.bse["vaastu"]
    pval = m6.pvalues["vaastu"]
    pct = (np.exp(coef) - 1) * 100
    print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
    print(f"Premium: {pct:+.1f}%")
    print(f"R²: {m6.rsquared:.3f}, N: {m6.nobs:.0f}")
    results["m6"] = {
        "coef": coef,
        "se": se,
        "pval": pval,
        "pct": pct,
        "r2": m6.rsquared,
        "n": m6.nobs,
    }

    return results


def run_city_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """Run separate regressions by city."""
    print("\n" + "=" * 70)
    print("CITY-SPECIFIC VAASTU EFFECTS")
    print("=" * 70)
    print("\nModel: ln_price ~ vaastu + bhk_cat + ln_area\n")

    city_results = []

    for city in sorted(df["city"].unique()):
        city_df = df[df["city"] == city]
        n = len(city_df)
        vaastu_n = city_df["vaastu"].sum()
        vaastu_pct = 100 * city_df["vaastu"].mean()

        if vaastu_n < 20:
            print(
                f"{city:12s}: n={n:5d}, vaastu={vaastu_n:3d} ({vaastu_pct:5.1f}%) - SKIPPED (too few vaastu)"
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

            sig = ""
            if pval < 0.01:
                sig = "***"
            elif pval < 0.05:
                sig = "**"
            elif pval < 0.1:
                sig = "*"

            print(
                f"{city:12s}: n={n:5d}, vaastu={vaastu_n:3d} ({vaastu_pct:5.1f}%) | "
                f"Premium: {pct:+6.1f}% (SE: {se:.3f}) {sig}"
            )

            city_results.append(
                {
                    "city": city,
                    "n": n,
                    "vaastu_n": vaastu_n,
                    "vaastu_pct": vaastu_pct,
                    "coef": coef,
                    "se": se,
                    "pval": pval,
                    "pct": pct,
                }
            )
        except Exception as e:
            print(
                f"{city:12s}: n={n:5d}, vaastu={vaastu_n:3d} ({vaastu_pct:5.1f}%) - ERROR: {e}"
            )

    return pd.DataFrame(city_results)


def print_comparison():
    """Print comparison with other data sources."""
    print("\n" + "=" * 70)
    print("COMPARISON WITH OTHER DATA SOURCES")
    print("=" * 70)
    print(
        """
| Source         | N       | Vaastu % | Premium (w/ controls) | Significance |
|----------------|---------|----------|----------------------|--------------|
| CampusX        | 7,621   | 53%      | +6.7%                | ***          |
| Kaggle         | 21,512  | 5.5%     | -3.2%                | n.s.         |
| Housing.com    | (above) | (above)  | (see above)          | (see above)  |

Notes:
- CampusX: 99acres Gurgaon data, high vaastu mention rate
- Kaggle: 99acres multi-city data, low vaastu mention rate
- Housing.com: Our scraped data, independent source
"""
    )


def main():
    print("Loading housing.com data...")
    df = load_housingcom_data()
    print(f"Loaded {len(df):,} raw listings from {df['city'].nunique()} cities")

    df = prepare_data(df)
    print(f"After cleaning: {len(df):,} listings")

    results = run_regressions(df)

    city_results = run_city_regressions(df)

    print_comparison()

    if len(city_results) > 0:
        print("\n## City Results Summary Table")
        print(
            city_results[
                ["city", "n", "vaastu_n", "vaastu_pct", "pct", "pval"]
            ].to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    m6 = results["m6"]
    print(
        f"""
Housing.com Overall Effect (Full Model with all controls):
  Premium: {m6['pct']:+.1f}%
  p-value: {m6['pval']:.4f}
  N: {m6['n']:.0f}

Interpretation:
  - Compare to CampusX (+6.7%***) and Kaggle (-3.2% n.s.)
  - Housing.com provides independent validation
"""
    )


if __name__ == "__main__":
    main()
