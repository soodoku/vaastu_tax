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

    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["property_id"])
    after_dedup = len(combined)
    if before_dedup > after_dedup:
        print(f"Deduplicated: {before_dedup} -> {after_dedup} listings")

    return combined


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for regression (shared preprocessing)."""
    df = df.copy()

    # Add listing_type column based on URL pattern
    df["listing_type"] = "sale"
    if "url" in df.columns:
        df.loc[df["url"].str.contains("Rent", case=False, na=False), "listing_type"] = (
            "rent"
        )

    df = df.dropna(subset=["price_crore", "builtup_area_sqft", "bhk"])

    df = df[df["price_crore"] > 0]
    df = df[df["builtup_area_sqft"] > 0]
    df = df[df["bhk"] > 0]
    df = df[df["bhk"] <= 10]

    df["ln_price"] = np.log(df["price_crore"])
    df["ln_area"] = np.log(df["builtup_area_sqft"])
    df["vaastu"] = df["vaastu_mentioned"].fillna(0).astype(int)
    df["bhk_cat"] = pd.Categorical(df["bhk"].astype(int).clip(upper=6))

    df["furnishing_clean"] = df["furnishing"].fillna("Unknown").astype(str).str.strip()
    df["furnishing_clean"] = df["furnishing_clean"].replace(
        {"": "Unknown", "nan": "Unknown"}
    )

    df["possession_clean"] = (
        df["possession_status"].fillna("Unknown").astype(str).str.strip()
    )
    df["possession_clean"] = df["possession_clean"].replace(
        {"": "Unknown", "nan": "Unknown"}
    )

    df["seller_clean"] = df["seller_type"].fillna("Unknown").astype(str).str.strip()
    df["seller_clean"] = df["seller_clean"].replace({"": "Unknown", "nan": "Unknown"})

    df["base_city"] = df["city"].apply(lambda x: x.split("-")[0] if "-" in x else x)

    return df


def filter_sale_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for sale listings with appropriate price threshold."""
    df_sale = df[df["listing_type"] == "sale"].copy()
    df_sale = df_sale[df_sale["price_crore"] >= 0.05]  # >= 5 lakh
    return df_sale


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

    vaastu_n = int(df["vaastu"].sum())
    vaastu_pct = 100 * df["vaastu"].mean()
    print(f"\nTotal: {len(df):,} listings, {vaastu_n:,} vaastu ({vaastu_pct:.1f}%)")

    print("\n## Covariate Coverage")
    covariates = [
        "bathrooms",
        "furnishing",
        "possession_status",
        "seller_type",
        "floor_no",
        "total_floors",
        "rating_overall",
    ]
    for cov in covariates:
        if cov in df.columns:
            n_valid = df[cov].notna().sum()
            pct = 100 * n_valid / len(df)
            print(f"  {cov:20s}: {n_valid:,} ({pct:.1f}%)")

    print("\n## Price Summary")
    print(f"Min: ₹{df['price_crore'].min():.2f} Cr")
    print(f"Max: ₹{df['price_crore'].max():.2f} Cr")
    print(f"Median: ₹{df['price_crore'].median():.2f} Cr")


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

    # Model 2: + BHK + Area
    print("\n" + "-" * 70)
    print("Model 2: ln_price ~ vaastu + bhk_cat + ln_area")
    print("-" * 70)
    m2 = smf.ols("ln_price ~ vaastu + bhk_cat + ln_area", data=df).fit(cov_type="HC3")
    coef = m2.params["vaastu"]
    se = m2.bse["vaastu"]
    pval = m2.pvalues["vaastu"]
    pct = (np.exp(coef) - 1) * 100
    print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
    print(f"Premium: {pct:+.1f}%")
    print(f"R²: {m2.rsquared:.3f}, N: {int(m2.nobs)}")
    results["m2"] = {
        "coef": coef,
        "se": se,
        "pval": pval,
        "pct": pct,
        "r2": m2.rsquared,
        "n": m2.nobs,
    }

    # Model 3: + City FE
    print("\n" + "-" * 70)
    print("Model 3: + C(base_city)")
    print("-" * 70)
    m3 = smf.ols("ln_price ~ vaastu + bhk_cat + ln_area + C(base_city)", data=df).fit(
        cov_type="HC3"
    )
    coef = m3.params["vaastu"]
    se = m3.bse["vaastu"]
    pval = m3.pvalues["vaastu"]
    pct = (np.exp(coef) - 1) * 100
    print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
    print(f"Premium: {pct:+.1f}%")
    print(f"R²: {m3.rsquared:.3f}, N: {int(m3.nobs)}")
    results["m3"] = {
        "coef": coef,
        "se": se,
        "pval": pval,
        "pct": pct,
        "r2": m3.rsquared,
        "n": m3.nobs,
    }

    # Model 4: + Bathrooms
    df_bath = df.dropna(subset=["bathrooms"])
    if len(df_bath) >= 100 and df_bath["vaastu"].sum() >= 5:
        print("\n" + "-" * 70)
        print("Model 4: + bathrooms")
        print("-" * 70)
        m4 = smf.ols(
            "ln_price ~ vaastu + bhk_cat + ln_area + C(base_city) + bathrooms",
            data=df_bath,
        ).fit(cov_type="HC3")
        coef = m4.params["vaastu"]
        se = m4.bse["vaastu"]
        pval = m4.pvalues["vaastu"]
        pct = (np.exp(coef) - 1) * 100
        print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
        print(f"Premium: {pct:+.1f}%")
        print(f"R²: {m4.rsquared:.3f}, N: {int(m4.nobs)}")
        results["m4"] = {
            "coef": coef,
            "se": se,
            "pval": pval,
            "pct": pct,
            "r2": m4.rsquared,
            "n": m4.nobs,
        }

    # Model 5: + Furnishing
    furn_levels = df_bath["furnishing_clean"].nunique()
    if furn_levels > 1:
        print("\n" + "-" * 70)
        print("Model 5: + C(furnishing_clean)")
        print("-" * 70)
        m5 = smf.ols(
            "ln_price ~ vaastu + bhk_cat + ln_area + C(base_city) + bathrooms + C(furnishing_clean)",
            data=df_bath,
        ).fit(cov_type="HC3")
        coef = m5.params["vaastu"]
        se = m5.bse["vaastu"]
        pval = m5.pvalues["vaastu"]
        pct = (np.exp(coef) - 1) * 100
        print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
        print(f"Premium: {pct:+.1f}%")
        print(f"R²: {m5.rsquared:.3f}, N: {int(m5.nobs)}")
        results["m5"] = {
            "coef": coef,
            "se": se,
            "pval": pval,
            "pct": pct,
            "r2": m5.rsquared,
            "n": m5.nobs,
        }

    # Model 6: + Possession Status
    poss_levels = df_bath["possession_clean"].nunique()
    if poss_levels > 1:
        print("\n" + "-" * 70)
        print("Model 6: + C(possession_clean)")
        print("-" * 70)
        formula = "ln_price ~ vaastu + bhk_cat + ln_area + C(base_city) + bathrooms"
        if furn_levels > 1:
            formula += " + C(furnishing_clean)"
        formula += " + C(possession_clean)"
        m6 = smf.ols(formula, data=df_bath).fit(cov_type="HC3")
        coef = m6.params["vaastu"]
        se = m6.bse["vaastu"]
        pval = m6.pvalues["vaastu"]
        pct = (np.exp(coef) - 1) * 100
        print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
        print(f"Premium: {pct:+.1f}%")
        print(f"R²: {m6.rsquared:.3f}, N: {int(m6.nobs)}")
        results["m6"] = {
            "coef": coef,
            "se": se,
            "pval": pval,
            "pct": pct,
            "r2": m6.rsquared,
            "n": m6.nobs,
        }

    # Model 7: + Seller Type
    seller_levels = df_bath["seller_clean"].nunique()
    if seller_levels > 1:
        print("\n" + "-" * 70)
        print("Model 7: + C(seller_clean)")
        print("-" * 70)
        formula = "ln_price ~ vaastu + bhk_cat + ln_area + C(base_city) + bathrooms"
        if furn_levels > 1:
            formula += " + C(furnishing_clean)"
        if poss_levels > 1:
            formula += " + C(possession_clean)"
        formula += " + C(seller_clean)"
        m7 = smf.ols(formula, data=df_bath).fit(cov_type="HC3")
        coef = m7.params["vaastu"]
        se = m7.bse["vaastu"]
        pval = m7.pvalues["vaastu"]
        pct = (np.exp(coef) - 1) * 100
        print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
        print(f"Premium: {pct:+.1f}%")
        print(f"R²: {m7.rsquared:.3f}, N: {int(m7.nobs)}")
        results["m7"] = {
            "coef": coef,
            "se": se,
            "pval": pval,
            "pct": pct,
            "r2": m7.rsquared,
            "n": m7.nobs,
        }

    # Model 8: + Ratings (if available)
    df_ratings = df_bath.dropna(subset=["rating_overall"])
    if len(df_ratings) >= 100 and df_ratings["vaastu"].sum() >= 5:
        print("\n" + "-" * 70)
        print("Model 8: + rating_overall")
        print("-" * 70)
        formula = "ln_price ~ vaastu + bhk_cat + ln_area + C(base_city) + bathrooms"
        if furn_levels > 1:
            formula += " + C(furnishing_clean)"
        if poss_levels > 1:
            formula += " + C(possession_clean)"
        if seller_levels > 1:
            formula += " + C(seller_clean)"
        formula += " + rating_overall"
        m8 = smf.ols(formula, data=df_ratings).fit(cov_type="HC3")
        coef = m8.params["vaastu"]
        se = m8.bse["vaastu"]
        pval = m8.pvalues["vaastu"]
        pct = (np.exp(coef) - 1) * 100
        print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
        print(f"Premium: {pct:+.1f}%")
        print(f"R²: {m8.rsquared:.3f}, N: {int(m8.nobs)}")
        results["m8"] = {
            "coef": coef,
            "se": se,
            "pval": pval,
            "pct": pct,
            "r2": m8.rsquared,
            "n": m8.nobs,
        }

    # Model 9: Project FE (within-project variation)
    if "project_name" in df_bath.columns:
        df_proj = df_bath.dropna(subset=["project_name"])
        proj_counts = df_proj["project_name"].value_counts()
        valid_projs = proj_counts[proj_counts >= 2].index
        df_proj = df_proj[df_proj["project_name"].isin(valid_projs)]

        if len(df_proj) >= 100 and df_proj["vaastu"].sum() >= 5:
            proj_vaastu_var = df_proj.groupby("project_name")["vaastu"].nunique()
            projs_with_variation = (proj_vaastu_var > 1).sum()
            print("\n" + "-" * 70)
            print(f"Model 9: Project FE ({len(valid_projs)} projects, {projs_with_variation} with vaastu variation)")
            print("-" * 70)
            formula = "ln_price ~ vaastu + bhk_cat + ln_area + bathrooms + C(project_name)"
            try:
                m9 = smf.ols(formula, data=df_proj).fit(cov_type="HC3")
                coef = m9.params["vaastu"]
                se = m9.bse["vaastu"]
                pval = m9.pvalues["vaastu"]
                pct = (np.exp(coef) - 1) * 100
                print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
                print(f"Premium: {pct:+.1f}%")
                print(f"R²: {m9.rsquared:.3f}, N: {int(m9.nobs)}")
                results["m9_project"] = {
                    "coef": coef,
                    "se": se,
                    "pval": pval,
                    "pct": pct,
                    "r2": m9.rsquared,
                    "n": m9.nobs,
                }
            except Exception as e:
                print(f"Project FE failed: {e}")

    # Model 10: Developer FE
    if "developer_name" in df_bath.columns:
        df_dev = df_bath.dropna(subset=["developer_name"])
        dev_counts = df_dev["developer_name"].value_counts()
        valid_devs = dev_counts[dev_counts >= 5].index
        df_dev = df_dev[df_dev["developer_name"].isin(valid_devs)]

        if len(df_dev) >= 100 and df_dev["vaastu"].sum() >= 5:
            print("\n" + "-" * 70)
            print(f"Model 10: Developer FE ({len(valid_devs)} developers)")
            print("-" * 70)
            formula = "ln_price ~ vaastu + bhk_cat + ln_area + C(base_city) + bathrooms + C(developer_name)"
            try:
                m10 = smf.ols(formula, data=df_dev).fit(cov_type="HC3")
                coef = m10.params["vaastu"]
                se = m10.bse["vaastu"]
                pval = m10.pvalues["vaastu"]
                pct = (np.exp(coef) - 1) * 100
                print(f"Vaastu coef: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
                print(f"Premium: {pct:+.1f}%")
                print(f"R²: {m10.rsquared:.3f}, N: {int(m10.nobs)}")
                results["m10_developer"] = {
                    "coef": coef,
                    "se": se,
                    "pval": pval,
                    "pct": pct,
                    "r2": m10.rsquared,
                    "n": m10.nobs,
                }
            except Exception as e:
                print(f"Developer FE failed: {e}")

    return results


def run_city_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """Run separate regressions by base city."""
    print("\n" + "=" * 70)
    print("CITY-SPECIFIC VAASTU EFFECTS")
    print("=" * 70)
    print(
        "\nModel: ln_price ~ vaastu + bhk_cat + ln_area + bathrooms + C(furnishing_clean)\n"
    )

    df_bath = df.dropna(subset=["bathrooms"])
    city_results = []

    for city in sorted(df_bath["base_city"].unique()):
        city_df = df_bath[df_bath["base_city"] == city]
        n = len(city_df)
        vaastu_n = int(city_df["vaastu"].sum())
        vaastu_pct = 100 * city_df["vaastu"].mean()

        if vaastu_n < 5:
            print(
                f"{city:15s}: n={n:5d}, vaastu={vaastu_n:3d} ({vaastu_pct:5.1f}%) - SKIPPED (too few vaastu)"
            )
            continue

        try:
            furn_levels = city_df["furnishing_clean"].nunique()
            formula = "ln_price ~ vaastu + bhk_cat + ln_area + bathrooms"
            if furn_levels > 1:
                formula += " + C(furnishing_clean)"

            m = smf.ols(formula, data=city_df).fit(cov_type="HC3")
            coef = m.params["vaastu"]
            se = m.bse["vaastu"]
            pval = m.pvalues["vaastu"]
            pct = (np.exp(coef) - 1) * 100
            sig = get_significance_stars(pval)

            print(
                f"{city:15s}: n={n:5d}, vaastu={vaastu_n:3d} ({vaastu_pct:5.1f}%) | "
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
        return

    # Filter to sales only (rentals have no vaastu mentions)
    n_sale = (df["listing_type"] == "sale").sum()
    n_rent = (df["listing_type"] == "rent").sum()
    print(f"  Sale: {n_sale:,}, Rent: {n_rent:,} (excluded - no vaastu data)")

    df = filter_sale_data(df)
    print(f"Sale listings after price filter: {len(df):,}")

    if len(df) == 0:
        print("ERROR: No valid sale data after filtering")
        return

    print_data_summary(df)

    if df["price_crore"].nunique() <= 1:
        print("\n" + "=" * 70)
        print("WARNING: PRICE DATA ISSUE DETECTED")
        print("=" * 70)
        print(f"All prices are: ₹{df['price_crore'].iloc[0]:.2f} Cr")
        return

    results = run_regressions(df)
    run_city_regressions(df)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    best_model = results.get(
        "m8",
        results.get(
            "m7", results.get("m6", results.get("m5", results.get("m4", results["m3"])))
        ),
    )
    sig = get_significance_stars(best_model["pval"])
    print(
        f"""
Magicbricks Overall Effect (with controls):
  Premium: {best_model['pct']:+.1f}% {sig}
  p-value: {best_model['pval']:.4f}
  N: {best_model.get('n', len(df)):.0f}
"""
    )


if __name__ == "__main__":
    main()
