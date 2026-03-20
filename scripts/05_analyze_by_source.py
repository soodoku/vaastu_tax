#!/usr/bin/env python3
"""Run hedonic regressions by source + aggregate estimate.

Outputs:
- tabs/tab_by_source_coefs.tex - Vaastu coefficient by source
- tabs/tab_aggregate_coef.tex - Pooled estimate
- figs/coef_by_source.png - Forest plot of source-specific estimates
- data/derived/regression_results.json - Full regression results

Usage
-----
python scripts/05_analyze_by_source.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("numpy and pandas required: pip install numpy pandas")
    sys.exit(1)

try:
    import statsmodels.formula.api as smf
except ImportError:
    print("statsmodels required: pip install statsmodels")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_magicbricks_data() -> pd.DataFrame:
    """Load magicbricks data from parquet files."""
    root = project_root()
    mb_dir = root / "data" / "raw" / "magicbricks"
    all_data = []

    for city_dir in mb_dir.iterdir():
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

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["property_id"])
    combined["source"] = "magicbricks"
    return combined


def load_housingcom_data() -> pd.DataFrame:
    """Load housingcom data from CSV files."""
    root = project_root()
    hc_dir = root / "data" / "raw" / "housingcom"
    all_data = []

    for city_dir in hc_dir.iterdir():
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

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined["source"] = "housingcom"
    return combined


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for regression analysis with quality filters."""
    df = df.copy()
    n_start = len(df)

    df["price_crore"] = pd.to_numeric(df["price_crore"], errors="coerce")
    df = df[(df["price_crore"] >= 0.1) & (df["price_crore"] <= 100)]

    df["bhk"] = pd.to_numeric(df["bhk"], errors="coerce")
    df = df[(df["bhk"] >= 1) & (df["bhk"] <= 10)]

    df["ln_price"] = np.log(df["price_crore"])
    df["ln_area"] = np.log(df["builtup_area_sqft"].replace(0, np.nan))
    df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")

    n_end = len(df)
    if n_start - n_end > 0:
        print(f"  (filtered {n_start - n_end} rows with invalid price/bhk)")

    return df


def create_feature_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Create dummy variables from comma-separated feature codes."""
    if "features" not in df.columns:
        return df
    df = df.copy()
    all_codes = set()
    for features in df["features"].dropna():
        for code in str(features).split(","):
            code = code.strip()
            if code.isdigit():
                all_codes.add(int(code))
    for code in sorted(all_codes):
        col_name = f"feat_{code}"
        df[col_name] = df["features"].fillna("").apply(
            lambda x, c=code: 1 if str(c) in [s.strip() for s in str(x).split(",")] else 0
        )
    return df


def run_regression(
    df: pd.DataFrame, formula: str, source_name: str
) -> dict | None:
    """Run OLS regression and return results."""
    work_df = df.dropna(subset=["ln_price", "vaastu_mentioned"])
    formula_vars = formula.replace("ln_price ~ ", "").replace(" + ", " ").replace("C(", " ").replace(")", " ").split()
    for var in formula_vars:
        if var in work_df.columns:
            work_df = work_df.dropna(subset=[var])

    if len(work_df) < 30:
        return None

    try:
        model = smf.ols(formula, data=work_df).fit()
    except Exception as e:
        print(f"  Regression failed for {source_name}: {e}")
        return None

    if "vaastu_mentioned" not in model.params.index:
        return None

    coef = model.params["vaastu_mentioned"]
    se = model.bse["vaastu_mentioned"]
    ci_low, ci_high = model.conf_int().loc["vaastu_mentioned"]
    pval = model.pvalues["vaastu_mentioned"]

    pct_effect = (np.exp(coef) - 1) * 100
    pct_ci_low = (np.exp(ci_low) - 1) * 100
    pct_ci_high = (np.exp(ci_high) - 1) * 100

    return {
        "source": source_name,
        "n": len(work_df),
        "coef": float(coef),
        "se": float(se),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "pval": float(pval),
        "pct_effect": float(pct_effect),
        "pct_ci_low": float(pct_ci_low),
        "pct_ci_high": float(pct_ci_high),
        "r2": float(model.rsquared),
        "r2_adj": float(model.rsquared_adj),
        "formula": formula,
    }


def run_source_analysis(df: pd.DataFrame, source_name: str) -> list[dict]:
    """Run multiple regression specifications for a single source."""
    results = []
    source_df = prepare_data(df)
    source_df = create_feature_dummies(source_df)

    has_price = source_df["ln_price"].notna().sum() > 100
    has_area = source_df["ln_area"].notna().sum() > 100
    has_bhk = source_df["bhk"].notna().sum() > 100
    has_bath = source_df["bathrooms"].notna().sum() > 100
    has_sector = ("sector" in source_df.columns) and (source_df["sector"].notna().sum() > 50)
    has_city = ("city" in source_df.columns) and (source_df["city"].nunique() > 1)

    if not has_price:
        print(f"  {source_name}: insufficient price data, skipping")
        return results

    specs = []
    specs.append(("raw", "ln_price ~ vaastu_mentioned"))

    if has_bhk:
        specs.append(("+ bhk", "ln_price ~ vaastu_mentioned + bhk"))
        if has_area:
            specs.append(("+ bhk + area", "ln_price ~ vaastu_mentioned + bhk + ln_area"))
            if has_bath:
                specs.append(("+ structural", "ln_price ~ vaastu_mentioned + bhk + ln_area + bathrooms"))

    if has_sector:
        base = "ln_price ~ vaastu_mentioned + bhk" if has_bhk else "ln_price ~ vaastu_mentioned"
        if has_area and has_bhk:
            base = "ln_price ~ vaastu_mentioned + bhk + ln_area"
        specs.append(("+ sector FE", f"{base} + C(sector)"))

    if has_city and not has_sector:
        base = "ln_price ~ vaastu_mentioned + bhk" if has_bhk else "ln_price ~ vaastu_mentioned"
        specs.append(("+ city FE", f"{base} + C(city)"))

    feat_cols = [c for c in source_df.columns if c.startswith("feat_")]
    if feat_cols and has_bhk and has_area:
        feat_formula = " + ".join(feat_cols)
        specs.append(("+ features", f"ln_price ~ vaastu_mentioned + bhk + ln_area + {feat_formula}"))

    for spec_name, formula in specs:
        result = run_regression(source_df, formula, source_name)
        if result:
            result["spec_name"] = spec_name
            results.append(result)
            print(f"    {spec_name}: coef={result['coef']:.4f}, n={result['n']}")

    return results


def run_aggregate_analysis(df: pd.DataFrame) -> list[dict]:
    """Run pooled analysis with source fixed effects."""
    results = []
    agg_df = prepare_data(df)

    specs = [
        ("pooled_raw", "ln_price ~ vaastu_mentioned"),
        ("pooled_bhk", "ln_price ~ vaastu_mentioned + bhk"),
        ("pooled_city_fe", "ln_price ~ vaastu_mentioned + bhk + C(city)"),
        ("pooled_source_fe", "ln_price ~ vaastu_mentioned + bhk + C(source)"),
        ("pooled_city_source_fe", "ln_price ~ vaastu_mentioned + bhk + C(city) + C(source)"),
    ]

    for spec_name, formula in specs:
        result = run_regression(agg_df, formula, "aggregate")
        if result:
            result["spec_name"] = spec_name
            results.append(result)
            print(f"  {spec_name}: coef={result['coef']:.4f}, n={result['n']}")

    return results


def generate_latex_table(results: list[dict], output_path: Path, title: str) -> None:
    """Generate LaTeX table from regression results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{title}}}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Source & Spec & N & Coef & SE & 95\% CI & \% Effect \\",
        r"\midrule",
    ]

    for r in results:
        ci = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        pct = f"{r['pct_effect']:.1f}\\%"
        stars = ""
        if r["pval"] < 0.01:
            stars = "***"
        elif r["pval"] < 0.05:
            stars = "**"
        elif r["pval"] < 0.1:
            stars = "*"

        lines.append(
            f"{r['source']} & {r.get('spec_name', '')} & {r['n']:,} & "
            f"{r['coef']:.4f}{stars} & ({r['se']:.4f}) & {ci} & {pct} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Note: * p<0.1, ** p<0.05, *** p<0.01. Dependent variable is ln(price in crore).",
        r"\item \% Effect shows the percentage change in price for Vaastu-mentioned properties.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def generate_forest_plot(results: list[dict], output_path: Path) -> None:
    """Generate forest plot of coefficients by source."""
    if plt is None:
        print("  matplotlib not available, skipping forest plot")
        return

    main_results = []
    for r in results:
        if r.get("spec_name") and "FE" in r["spec_name"]:
            main_results.append(r)
    if not main_results:
        main_results = [r for r in results if r.get("spec_name") == "+ bhk"]
    if not main_results:
        main_results = results[-1:] if results else []

    if not main_results:
        print("  No results for forest plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4 + len(main_results) * 0.3))

    y_positions = range(len(main_results))
    labels = [f"{r['source']} (n={r['n']:,})" for r in main_results]
    coefs = [r["coef"] for r in main_results]
    ci_lows = [r["ci_low"] for r in main_results]
    ci_highs = [r["ci_high"] for r in main_results]

    ax.errorbar(
        coefs,
        y_positions,
        xerr=[[c - ci_low for c, ci_low in zip(coefs, ci_lows)], [h - c for c, h in zip(coefs, ci_highs)]],
        fmt="o",
        capsize=5,
        capthick=2,
        markersize=8,
        color="steelblue",
    )

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Vaastu Coefficient (log points)")
    ax.set_title("Vaastu Premium by Source")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    root = project_root()
    derived_dir = root / "data" / "derived"
    tex_dir = root / "tabs"
    figs_dir = root / "figs"

    csv_path = derived_dir / "all_99acres_vaastu.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run 03_extract_vaastu.py first.")
        sys.exit(1)

    print("Loading 99acres data...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  99acres: {len(df)} rows")

    print("Loading magicbricks data...")
    mb_df = load_magicbricks_data()
    if len(mb_df) > 0:
        print(f"  magicbricks: {len(mb_df)} rows")
        df = pd.concat([df, mb_df], ignore_index=True)
    else:
        print("  magicbricks: no data found")

    print("Loading housingcom data...")
    hc_df = load_housingcom_data()
    if len(hc_df) > 0:
        print(f"  housingcom: {len(hc_df)} rows")
        df = pd.concat([df, hc_df], ignore_index=True)
    else:
        print("  housingcom: no data found")

    print(f"  Total: {len(df)} rows")

    all_results = []

    print("\n=== Source-Specific Analysis ===")
    excluded_sources = {"kaggle_arvanshul"}
    for source in df["source"].unique():
        if source in excluded_sources:
            print(f"\n{source}: excluded from analysis")
            continue
        source_df = df[df["source"] == source]
        if len(source_df) < 100:
            print(f"\n{source}: n={len(source_df)} (too small, skipping)")
            continue

        print(f"\n{source} (n={len(source_df)}):")
        results = run_source_analysis(source_df, source)
        all_results.extend(results)

    print("\n=== Aggregate Analysis ===")
    agg_results = run_aggregate_analysis(df)
    all_results.extend(agg_results)

    source_results = [r for r in all_results if r["source"] != "aggregate"]
    agg_only = [r for r in all_results if r["source"] == "aggregate"]

    print("\nGenerating outputs...")
    generate_latex_table(source_results, tex_dir / "tab_by_source_coefs.tex", "Vaastu Premium by Source")
    generate_latex_table(agg_only, tex_dir / "tab_aggregate_coef.tex", "Aggregate Vaastu Premium")
    generate_forest_plot(source_results, figs_dir / "coef_by_source.png")

    output = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "results": all_results,
    }
    results_path = derived_dir / "regression_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\nOutputs:")
    print(f"  {tex_dir / 'tab_by_source_coefs.tex'}")
    print(f"  {tex_dir / 'tab_aggregate_coef.tex'}")
    print(f"  {figs_dir / 'coef_by_source.png'}")
    print(f"  {results_path}")

    print("\n=== Summary ===")
    for r in all_results:
        if r.get("spec_name") and ("FE" in r["spec_name"] or r["spec_name"] == "+ bhk"):
            print(f"  {r['source']} ({r['spec_name']}): {r['pct_effect']:+.1f}% [{r['pct_ci_low']:.1f}, {r['pct_ci_high']:.1f}]")


if __name__ == "__main__":
    main()
