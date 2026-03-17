#!/usr/bin/env python3
"""Estimate willingness to pay for Vaastu from public listing data.

Modes
-----
1. legacy_gurugram
   Re-runs the Gurugram prototype bundled with this package.
2. housingcom_collected
   Analyzes a normalized CSV created by collect_housingcom_vaastu.py.

Outputs
-------
- Cleaned analysis sample (CSV)
- Result tables (LaTeX)
- Figure (PNG)
- A LaTeX macros file that keeps manuscript numbers synchronized with the results
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Keep linear algebra single-threaded for reproducible, low-memory runs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    root = project_root_from_here()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["legacy_gurugram", "housingcom_collected"],
        default="legacy_gurugram",
        help="Dataset mode to analyze",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Required only for mode=housingcom_collected; CSV produced by the collector",
    )
    parser.add_argument(
        "--root",
        default=str(root),
        help="Project root directory (defaults to the package root)",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Optional override for raw legacy data directory",
    )
    return parser.parse_args()


def norm_text(s: object) -> str:
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def parse_list(s: object) -> List[str]:
    if pd.isna(s):
        return []
    try:
        parsed = ast.literal_eval(str(s))
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed]
    except Exception:
        pass
    text = str(s)
    return [x.strip().strip("'\"").lower() for x in re.split(r",\s*", text.strip("[]")) if x.strip()]


def winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    lo, hi = s.quantile([lower_q, upper_q])
    return s.clip(lower=lo, upper=hi)


def fit_cluster(formula: str, data: pd.DataFrame, cluster_col: str):
    model = smf.ols(formula, data=data).fit()
    used = model.model.data.row_labels
    robust = model.get_robustcov_results(cov_type="cluster", groups=data.loc[used, cluster_col])
    return model, robust


def price_fraction_to_pct(x: float) -> str:
    return f"{100.0 * x:.1f}\\%"


def p_value_str(x: float) -> str:
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def fmt_int(x: float | int) -> str:
    return f"{int(round(float(x))):,}"


def fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{float(x):,.{digits}f}"


def latex_macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def latex_table_from_df(
    df: pd.DataFrame,
    caption: str,
    label: str,
    note: Optional[str] = None,
    size: Optional[str] = None,
    column_format: Optional[str] = None,
) -> str:
    table = df.to_latex(index=False, escape=False, column_format=column_format)
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
    ]
    if size:
        lines.append("{" + size)
        lines.append(table)
        lines.append("}")
    else:
        lines.append(table)
    if note:
        lines += ["\\vspace{0.3em}", f"\\par\\begin{{minipage}}{{0.95\\linewidth}}\\footnotesize {note}\\end{{minipage}}"]
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def build_legacy_gurugram_dataset(raw_dir: Path) -> pd.DataFrame:
    houses = pd.read_csv(raw_dir / "house_cleaned.csv")
    flats = pd.read_csv(raw_dir / "flats_cleaned.csv")
    g2 = pd.read_csv(raw_dir / "gurgaon_properties_cleaned_v2.csv")

    houses["property_type"] = "house"
    flats["property_type"] = "flat"

    comb = pd.concat([houses, flats], ignore_index=True, sort=False).reset_index().rename(columns={"index": "orig_id"})
    comb["vaastu_i"] = (
        comb["features"].fillna("").str.contains("vaastu|vastu", case=False, regex=True)
        | comb["description"].fillna("").str.contains(r"\bvaastu\b|\bvastu\b", case=False, regex=True)
    ).astype(int)

    comb["pooja_room"] = comb["additionalRoom"].fillna("").str.contains("pooja", case=False, regex=False)
    comb["servant_room"] = comb["additionalRoom"].fillna("").str.contains("servant", case=False, regex=False)
    comb["store_room"] = comb["additionalRoom"].fillna("").str.contains("store", case=False, regex=False)
    comb["study_room"] = comb["additionalRoom"].fillna("").str.contains("study", case=False, regex=False)
    comb["others_room"] = comb["additionalRoom"].fillna("").str.contains("others", case=False, regex=False)

    comb["location"] = comb["property_name"].str.extract(r" in (.*)$", flags=re.I, expand=False).str.lower().str.strip()
    comb["location"] = comb["location"].str.replace(r"\s*,?\s*gurgaon$", "", regex=True)

    for d in [comb, g2]:
        d["society_n"] = d["society"].apply(norm_text)

    keys = ["property_type", "society_n", "price", "area", "bedRoom", "bathroom"]
    g2u = g2.sort_values(keys).drop_duplicates(subset=keys, keep="first")

    ana = comb.merge(
        g2u[
            keys
            + [
                "sector",
                "super_built_up_area",
                "built_up_area",
                "carpet_area",
                "study room",
                "servant room",
                "store room",
                "pooja room",
                "others",
                "furnishing_type",
                "luxury_score",
            ]
        ],
        on=keys,
        how="left",
    )

    ana["pooja_room2"] = ana["pooja room"].fillna(ana["pooja_room"].astype(int)).astype(int)
    ana["servant_room2"] = ana["servant room"].fillna(ana["servant_room"].astype(int)).astype(int)
    ana["store_room2"] = ana["store room"].fillna(ana["store_room"].astype(int)).astype(int)
    ana["study_room2"] = ana["study room"].fillna(ana["study_room"].astype(int)).astype(int)
    ana["others_room2"] = ana["others"].fillna(ana["others_room"].astype(int)).astype(int)
    ana["sector"] = ana["sector"].fillna(ana["location"])
    ana["furnishing_type"] = ana["furnishing_type"].fillna("missing")
    ana["luxury_score"] = ana["luxury_score"].fillna(ana["luxury_score"].median())

    ana = ana[(ana["price"].notna()) & (ana["area"].notna()) & (ana["price"] > 0) & (ana["area"] > 100) & (ana["sector"].notna())].copy()

    for col in ["price", "area", "price_per_sqft", "floorNum", "luxury_score"]:
        ana[f"{col}_w"] = winsorize_series(ana[col])

    def parse_balcony(x: object) -> float:
        if pd.isna(x):
            return float("nan")
        s = str(x).strip()
        if s == "3+":
            return 3.0
        m = re.search(r"(\d+)", s)
        return float(m.group(1)) if m else float("nan")

    ana["balcony_n"] = ana["balcony"].apply(parse_balcony)
    ana["balcony_n"] = ana["balcony_n"].fillna(ana["balcony_n"].median())
    ana["facing"] = ana["facing"].fillna("Missing")
    ana["agePossession"] = ana["agePossession"].fillna("Missing")
    ana["property_type"] = pd.Categorical(ana["property_type"], categories=["flat", "house"], ordered=True)
    ana["ln_price"] = np.log(ana["price_w"])
    ana["ln_area"] = np.log(ana["area_w"])
    return ana


def build_collected_housingcom_dataset(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"price_crore", "builtup_area_sqft", "city", "vaastu_mentioned"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    out = df.copy()
    out["property_type"] = "house"
    out["vaastu_i"] = out["vaastu_mentioned"].fillna(0).astype(int)
    out["price"] = out["price_crore"].astype(float)
    out["area"] = out["builtup_area_sqft"].astype(float)
    out["bedRoom"] = out.get("bhk", np.nan)
    out["bathroom"] = out.get("bathrooms", np.nan)
    out["balcony_n"] = out.get("balconies", np.nan)
    out["sector"] = out.get("locality_line", out["city"]).fillna(out["city"])
    out["facing"] = out.get("facing", "Missing").fillna("Missing")
    out["agePossession"] = out.get("possession_status", "Missing").fillna("Missing")
    out["furnishing_type"] = out.get("furnishing", "missing").fillna("missing")
    out["luxury_score_w"] = 0.0
    out["pooja_room2"] = out.get("about_this_property", "").fillna("").str.contains("pooja", case=False).astype(int)
    out["servant_room2"] = 0
    out["store_room2"] = 0
    out["study_room2"] = 0
    out["others_room2"] = 0

    out = out[(out["price"].notna()) & (out["area"].notna()) & (out["price"] > 0) & (out["area"] > 100)].copy()
    out["price_w"] = winsorize_series(out["price"])
    out["area_w"] = winsorize_series(out["area"])
    out["floorNum_w"] = 0.0
    out["ln_price"] = np.log(out["price_w"])
    out["ln_area"] = np.log(out["area_w"])
    return out


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def run_legacy_models(a: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    formula_s1 = "ln_price ~ vaastu_i"
    formula_s2 = "ln_price ~ vaastu_i + ln_area + bedRoom + bathroom + balcony_n + floorNum_w + C(agePossession) + C(facing) + C(property_type)"
    formula_s3 = (
        "ln_price ~ vaastu_i + ln_area + bedRoom + bathroom + balcony_n + floorNum_w + "
        "C(agePossession) + C(facing) + C(property_type) + "
        "pooja_room2 + servant_room2 + store_room2 + study_room2 + others_room2 + "
        "C(furnishing_type) + luxury_score_w"
    )
    formula_s4 = formula_s3 + " + C(sector)"

    average_rows = []
    for label, formula in [
        ("Raw diff", formula_s1),
        ("+ structural", formula_s2),
        ("+ quality/room", formula_s3),
        ("+ sector FE (preferred average)", formula_s4),
    ]:
        model, robust = fit_cluster(formula, a, "sector")
        names = robust.model.exog_names
        ix = names.index("vaastu_i")
        beta = float(robust.params[ix])
        se = float(robust.bse[ix])
        average_rows.append(
            {
                "model": label,
                "n": int(robust.nobs),
                "beta_log_points": beta,
                "se": se,
                "p_value": float(robust.pvalues[ix]),
                "premium_pct": math.exp(beta) - 1.0,
                "premium_pct_ci_low": math.exp(beta - 1.96 * se) - 1.0,
                "premium_pct_ci_high": math.exp(beta + 1.96 * se) - 1.0,
                "r2": float(model.rsquared),
            }
        )

    formula_flex = (
        "ln_price ~ vaastu_i*C(property_type) + ln_area*C(property_type) + bedRoom*C(property_type) + "
        "bathroom*C(property_type) + balcony_n*C(property_type) + floorNum_w*C(property_type) + "
        "C(agePossession)*C(property_type) + C(facing)*C(property_type) + pooja_room2*C(property_type) + "
        "servant_room2*C(property_type) + store_room2*C(property_type) + study_room2*C(property_type) + "
        "others_room2*C(property_type) + C(furnishing_type)*C(property_type) + luxury_score_w*C(property_type) + C(sector)"
    )
    model_f, robust_f = fit_cluster(formula_flex, a, "sector")
    names = robust_f.model.exog_names
    ix_flat = names.index("vaastu_i")
    ix_int = names.index("vaastu_i:C(property_type)[T.house]")

    beta_flat = float(robust_f.params[ix_flat])
    se_flat = float(robust_f.bse[ix_flat])

    cov = robust_f.cov_params()
    beta_house = float(robust_f.params[ix_flat] + robust_f.params[ix_int])
    se_house = float(np.sqrt(cov[ix_flat, ix_flat] + cov[ix_int, ix_int] + 2 * cov[ix_flat, ix_int]))

    R = np.zeros((1, len(names)))
    R[0, ix_flat] = 1
    R[0, ix_int] = 1
    p_house = float(robust_f.t_test(R).pvalue)

    type_rows = []
    for property_type, beta, se, pval in [
        ("flat", beta_flat, se_flat, float(robust_f.pvalues[ix_flat])),
        ("house", beta_house, se_house, p_house),
    ]:
        med_price = float(a.loc[a["property_type"] == property_type, "price"].median())
        prem = math.exp(beta) - 1.0
        type_rows.append(
            {
                "property_type": property_type,
                "n": int((a["property_type"] == property_type).sum()),
                "beta_log_points": beta,
                "se": se,
                "p_value": pval,
                "premium_pct": prem,
                "premium_pct_ci_low": math.exp(beta - 1.96 * se) - 1.0,
                "premium_pct_ci_high": math.exp(beta + 1.96 * se) - 1.0,
                "median_price_cr": med_price,
                "wtp_median_lakh": prem * med_price * 100.0,
            }
        )

    return pd.DataFrame(average_rows), pd.DataFrame(type_rows)


def run_legacy_matching(a: pd.DataFrame) -> pd.DataFrame:
    m = a.copy()
    loc_agg = m.groupby("sector")["vaastu_i"].agg(["sum", "count"])
    sectors_both = loc_agg[(loc_agg["sum"] > 0) & (loc_agg["sum"] < loc_agg["count"])].index
    m = m[m["sector"].isin(sectors_both)].copy()

    matches = []
    cols = ["ln_area", "bathroom", "balcony_n", "floorNum_w", "luxury_score_w"]
    for (sector, property_type), g in m.groupby(["sector", "property_type"], observed=False):
        treated = g[g["vaastu_i"] == 1]
        control = g[g["vaastu_i"] == 0]
        if len(treated) == 0 or len(control) == 0:
            continue
        for _, row in treated.iterrows():
            cand = control[control["bedRoom"].between(row["bedRoom"] - 1, row["bedRoom"] + 1)]
            if cand.empty:
                cand = control.copy()
            same_age = cand[cand["agePossession"] == row["agePossession"]]
            if not same_age.empty:
                cand = same_age
            same_facing = cand[cand["facing"] == row["facing"]]
            if not same_facing.empty:
                cand = same_facing
            X = pd.concat([cand[cols], row[cols].to_frame().T], axis=0).astype(float)
            X = X.fillna(X.median())
            scaler = StandardScaler().fit(X)
            Xc = scaler.transform(X.iloc[:-1])
            xr = scaler.transform(X.iloc[[-1]])
            d = ((Xc - xr) ** 2).sum(axis=1)
            j = cand.index[int(np.argmin(d))]
            matches.append(
                {
                    "sector": sector,
                    "property_type": property_type,
                    "ln_price_diff": row["ln_price"] - m.loc[j, "ln_price"],
                    "price_diff": row["price"] - m.loc[j, "price"],
                }
            )

    match_df = pd.DataFrame(matches)
    point = float(match_df["ln_price_diff"].mean())
    by_sector = match_df.groupby("sector").agg(
        sum_ln_price_diff=("ln_price_diff", "sum"),
        n_matches=("ln_price_diff", "size"),
    )
    sector_sums = np.asarray(by_sector["sum_ln_price_diff"])
    sector_counts = np.asarray(by_sector["n_matches"])
    rng = np.random.default_rng(42)
    boots = np.empty(500)
    n_sectors = len(by_sector)
    for i in range(500):
        idx = rng.integers(0, n_sectors, size=n_sectors)
        boots[i] = sector_sums[idx].sum() / sector_counts[idx].sum()

    out = pd.DataFrame(
        [
            {
                "matching_design": "within sector x property type, nearest neighbor",
                "n_treated_matches": int(match_df.shape[0]),
                "beta_log_points": point,
                "bootstrap_se": float(boots.std(ddof=1)),
                "premium_pct": float(math.exp(point) - 1.0),
                "premium_pct_ci_low": float(math.exp(np.quantile(boots, 0.025)) - 1.0),
                "premium_pct_ci_high": float(math.exp(np.quantile(boots, 0.975)) - 1.0),
                "avg_price_diff_cr": float(match_df["price_diff"].mean()),
            }
        ]
    )
    return out


def run_collected_models(a: pd.DataFrame) -> pd.DataFrame:
    # For portal-collected housing data we only estimate a single house-market specification.
    formula = (
        "ln_price ~ vaastu_i + ln_area + bedRoom + bathroom + balcony_n + "
        "C(agePossession) + C(facing) + C(furnishing_type) + C(sector)"
    )
    model, robust = fit_cluster(formula, a, "sector")
    names = robust.model.exog_names
    ix = names.index("vaastu_i")
    beta = float(robust.params[ix])
    se = float(robust.bse[ix])
    return pd.DataFrame(
        [
            {
                "model": "Portal-collected houses with locality FE",
                "n": int(robust.nobs),
                "beta_log_points": beta,
                "se": se,
                "p_value": float(robust.pvalues[ix]),
                "premium_pct": math.exp(beta) - 1.0,
                "premium_pct_ci_low": math.exp(beta - 1.96 * se) - 1.0,
                "premium_pct_ci_high": math.exp(beta + 1.96 * se) - 1.0,
                "r2": float(model.rsquared),
            }
        ]
    )


# ---------------------------------------------------------------------------
# Diagnostics and exports
# ---------------------------------------------------------------------------


def house_sector_support(a: pd.DataFrame) -> pd.DataFrame:
    house = a[a["property_type"] == "house"].copy()
    g = house.groupby("sector")["vaastu_i"].agg(["sum", "count"])
    positive = g[g["sum"] > 0]
    both = g[(g["sum"] > 0) & (g["sum"] < g["count"])]
    five_plus = g[g["sum"] >= 5]
    rows = [
        {"statistic": "House listings", "value": int(len(house))},
        {"statistic": "Vaastu-tagged house listings", "value": int(house["vaastu_i"].sum())},
        {"statistic": "Non-Vaastu house listings", "value": int((1 - house["vaastu_i"]).sum())},
        {"statistic": "Sectors with any house listings", "value": int(g.shape[0])},
        {"statistic": "Sectors with at least one Vaastu house", "value": int(positive.shape[0])},
        {"statistic": "Sectors with both Vaastu and non-Vaastu houses", "value": int(both.shape[0])},
        {"statistic": "Sectors with at least 5 Vaastu houses", "value": int(five_plus.shape[0])},
        {"statistic": "Median Vaastu-house count among positive sectors", "value": float(positive["sum"].median()) if len(positive) > 0 else float("nan")},
        {"statistic": "Mean Vaastu-house count among positive sectors", "value": float(positive["sum"].mean()) if len(positive) > 0 else float("nan")},
    ]
    return pd.DataFrame(rows)


def sample_counts_by_type(a: pd.DataFrame) -> pd.DataFrame:
    counts = a.groupby("property_type", observed=False)["vaastu_i"].agg(["sum", "count", "mean"]).reset_index()
    counts.rename(columns={"sum": "vaastu_listings", "count": "total_listings", "mean": "vaastu_share"}, inplace=True)
    return counts


def build_macros(a: pd.DataFrame, avg_df: pd.DataFrame, type_df: Optional[pd.DataFrame], match_df: Optional[pd.DataFrame], support_df: pd.DataFrame) -> Dict[str, str]:
    total = int(a.shape[0])
    total_vaastu = int(a["vaastu_i"].sum())
    overall_share = total_vaastu / total if total else float("nan")
    counts = sample_counts_by_type(a).set_index("property_type")
    macros = {
        "TotalListings": fmt_int(total),
        "TotalVaastuListings": fmt_int(total_vaastu),
        "TotalVaastuSharePct": price_fraction_to_pct(overall_share),
        "FlatListings": fmt_int(counts.loc["flat", "total_listings"]) if "flat" in counts.index else "0",
        "FlatVaastuListings": fmt_int(counts.loc["flat", "vaastu_listings"]) if "flat" in counts.index else "0",
        "FlatVaastuSharePct": price_fraction_to_pct(counts.loc["flat", "vaastu_share"]) if "flat" in counts.index else "0.0\\%",
        "HouseListings": fmt_int(counts.loc["house", "total_listings"]) if "house" in counts.index else fmt_int(total),
        "HouseVaastuListings": fmt_int(counts.loc["house", "vaastu_listings"]) if "house" in counts.index else fmt_int(total_vaastu),
        "HouseVaastuSharePct": price_fraction_to_pct(counts.loc["house", "vaastu_share"]) if "house" in counts.index else price_fraction_to_pct(overall_share),
    }

    preferred = avg_df.loc[avg_df["model"].str.contains("preferred", case=False)].iloc[0]
    macros.update(
        {
            "PreferredAveragePremiumPct": price_fraction_to_pct(preferred["premium_pct"]),
            "PreferredAverageCILowPct": price_fraction_to_pct(preferred["premium_pct_ci_low"]),
            "PreferredAverageCIHighPct": price_fraction_to_pct(preferred["premium_pct_ci_high"]),
        }
    )

    if type_df is not None and not type_df.empty:
        flat = type_df.loc[type_df["property_type"] == "flat"].iloc[0]
        house = type_df.loc[type_df["property_type"] == "house"].iloc[0]
        macros.update(
            {
                "FlatPremiumPct": price_fraction_to_pct(flat["premium_pct"]),
                "FlatCILowPct": price_fraction_to_pct(flat["premium_pct_ci_low"]),
                "FlatCIHighPct": price_fraction_to_pct(flat["premium_pct_ci_high"]),
                "HousePremiumPct": price_fraction_to_pct(house["premium_pct"]),
                "HouseCILowPct": price_fraction_to_pct(house["premium_pct_ci_low"]),
                "HouseCIHighPct": price_fraction_to_pct(house["premium_pct_ci_high"]),
                "MedianHousePriceCr": fmt_num(house["median_price_cr"], 2),
                "MedianHouseWTPLakh": fmt_num(house["wtp_median_lakh"], 1),
            }
        )

    if match_df is not None and not match_df.empty:
        match = match_df.iloc[0]
        macros.update(
            {
                "MatchingPremiumPct": price_fraction_to_pct(match["premium_pct"]),
                "MatchingCILowPct": price_fraction_to_pct(match["premium_pct_ci_low"]),
                "MatchingCIHighPct": price_fraction_to_pct(match["premium_pct_ci_high"]),
            }
        )

    support = dict(zip(support_df["statistic"], support_df["value"]))
    macros.update(
        {
            "HouseSectorsTotal": fmt_int(support["Sectors with any house listings"]),
            "HouseSectorsWithVaastu": fmt_int(support["Sectors with at least one Vaastu house"]),
            "HouseSectorsWithBoth": fmt_int(support["Sectors with both Vaastu and non-Vaastu houses"]),
            "HouseSectorsWithFivePlusVaastu": fmt_int(support["Sectors with at least 5 Vaastu houses"]),
            "HouseMedianVaastuCountPositive": fmt_num(support["Median Vaastu-house count among positive sectors"], 1),
            "HouseMeanVaastuCountPositive": fmt_num(support["Mean Vaastu-house count among positive sectors"], 1),
        }
    )
    return macros


def write_macros_tex(path: Path, macros: Dict[str, str]) -> None:
    lines = ["% Auto-generated by scripts/02_analyze.py"]
    for key in sorted(macros):
        lines.append(latex_macro(key, macros[key]))
    write_text(path, "\n".join(lines) + "\n")


def plot_coefficients(avg_df: pd.DataFrame, type_df: Optional[pd.DataFrame], out_path: Path) -> None:
    fig = plt.figure(figsize=(8.2, 4.8))
    ax = fig.add_subplot(111)
    labels = [
        "Raw diff",
        "+ structural",
        "+ quality/room",
        "+ sector FE avg",
    ]
    betas = [
        avg_df.loc[avg_df["model"] == "Raw diff", "beta_log_points"].iloc[0],
        avg_df.loc[avg_df["model"] == "+ structural", "beta_log_points"].iloc[0],
        avg_df.loc[avg_df["model"] == "+ quality/room", "beta_log_points"].iloc[0],
        avg_df.loc[avg_df["model"] == "+ sector FE (preferred average)", "beta_log_points"].iloc[0],
    ]
    ses = [
        avg_df.loc[avg_df["model"] == "Raw diff", "se"].iloc[0],
        avg_df.loc[avg_df["model"] == "+ structural", "se"].iloc[0],
        avg_df.loc[avg_df["model"] == "+ quality/room", "se"].iloc[0],
        avg_df.loc[avg_df["model"] == "+ sector FE (preferred average)", "se"].iloc[0],
    ]
    if type_df is not None and not type_df.empty:
        labels += ["Flat (flex model)", "House (flex model)"]
        betas += [
            type_df.loc[type_df["property_type"] == "flat", "beta_log_points"].iloc[0],
            type_df.loc[type_df["property_type"] == "house", "beta_log_points"].iloc[0],
        ]
        ses += [
            type_df.loc[type_df["property_type"] == "flat", "se"].iloc[0],
            type_df.loc[type_df["property_type"] == "house", "se"].iloc[0],
        ]
    y = np.arange(len(labels))[::-1]
    ax.errorbar(betas, y, xerr=np.asarray(ses) * 1.96, fmt="o")
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Log price effect of Vaastu label (95% CI)")
    ax.set_title("Vaastu capitalization in Gurugram listings")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def export_results_tables(
    *,
    out_tex: Path,
    counts_df: pd.DataFrame,
    avg_df: pd.DataFrame,
    type_df: Optional[pd.DataFrame],
    match_df: Optional[pd.DataFrame],
    support_df: pd.DataFrame,
) -> None:
    ensure_dir(out_tex)

    counts_pub = counts_df.copy()
    counts_pub["property_type"] = counts_pub["property_type"].astype(str).replace({"flat": "Flats", "house": "Houses"})
    counts_pub["vaastu_listings"] = counts_pub["vaastu_listings"].map(fmt_int)
    counts_pub["total_listings"] = counts_pub["total_listings"].map(fmt_int)
    counts_pub["vaastu_share"] = counts_pub["vaastu_share"].map(price_fraction_to_pct)
    counts_pub.columns = pd.Index(["Property type", "Vaastu listings", "Total listings", "Vaastu share"])
    write_text(
        out_tex / "tab_counts_by_type.tex",
        latex_table_from_df(
            counts_pub,
            caption="Sample composition by property type",
            label="tab:counts",
            note="Vaastu is coded from listing text and feature fields. Shares are within property type.",
        ),
    )

    avg_pub = avg_df.copy()
    avg_pub["n"] = avg_pub["n"].map(fmt_int)
    avg_pub["premium_pct"] = avg_pub["premium_pct"].map(price_fraction_to_pct)
    avg_pub["95\\% CI"] = avg_df.apply(
        lambda r: f"[{price_fraction_to_pct(r['premium_pct_ci_low'])}, {price_fraction_to_pct(r['premium_pct_ci_high'])}]",
        axis=1,
    )
    avg_pub["p_value"] = avg_pub["p_value"].map(p_value_str)
    avg_pub["r2"] = avg_pub["r2"].map(lambda x: f"{x:.3f}")
    avg_pub = pd.DataFrame(avg_pub[["model", "n", "premium_pct", "95\\% CI", "p_value", "r2"]])
    avg_pub.columns = pd.Index(["Specification", "N", "Premium", "95\\% CI", "p-value", "$R^2$"])
    write_text(
        out_tex / "tab_average_models.tex",
        latex_table_from_df(
            avg_pub,
            caption="Average Vaastu capitalization estimates",
            label="tab:average-models",
            note="All specifications use log list price as the dependent variable. The preferred specification includes sector fixed effects and the full set of structural and quality controls.",
        ),
    )

    if type_df is not None and not type_df.empty:
        type_pub = type_df.copy()
        type_pub["n"] = type_pub["n"].map(fmt_int)
        type_pub["premium_pct"] = type_pub["premium_pct"].map(price_fraction_to_pct)
        type_pub["95\\% CI"] = type_df.apply(
            lambda r: f"[{price_fraction_to_pct(r['premium_pct_ci_low'])}, {price_fraction_to_pct(r['premium_pct_ci_high'])}]",
            axis=1,
        )
        type_pub["p_value"] = type_pub["p_value"].map(p_value_str)
        type_pub["median_price_cr"] = type_pub["median_price_cr"].map(lambda x: f"{x:.2f}")
        type_pub["wtp_median_lakh"] = type_pub["wtp_median_lakh"].map(lambda x: f"{x:.1f}")
        type_pub = pd.DataFrame(type_pub[["property_type", "n", "premium_pct", "95\\% CI", "p_value", "median_price_cr", "wtp_median_lakh"]])
        type_pub["property_type"] = type_pub["property_type"].replace({"flat": "Flats", "house": "Ind. houses"})
        type_pub.columns = pd.Index(["Segment", "N", "Premium", "95\\% CI", "p", "Median price (Cr)", "WTP (Lakh)"])
        write_text(
            out_tex / "tab_heterogeneity.tex",
            latex_table_from_df(
                type_pub,
                caption="Heterogeneity by property type",
                label="tab:heterogeneity",
                note="The flexible model allows the Vaastu coefficient and all main controls to differ by property type while keeping sector fixed effects common.",
                size=r"\small\setlength{\tabcolsep}{4pt}",
                column_format="lrrrrrr",
            ),
        )

    if match_df is not None and not match_df.empty:
        match_pub = match_df.copy()
        match_pub["matching_design"] = "Within sector x type, NN"
        match_pub["n_treated_matches"] = match_pub["n_treated_matches"].map(fmt_int)
        match_pub["premium_pct"] = match_pub["premium_pct"].map(price_fraction_to_pct)
        match_pub["95\\% CI"] = match_df.apply(
            lambda r: f"[{price_fraction_to_pct(r['premium_pct_ci_low'])}, {price_fraction_to_pct(r['premium_pct_ci_high'])}]",
            axis=1,
        )
        match_pub["avg_price_diff_cr"] = match_pub["avg_price_diff_cr"].map(lambda x: f"{x:.3f}")
        match_pub = pd.DataFrame(match_pub[["matching_design", "n_treated_matches", "premium_pct", "95\\% CI", "avg_price_diff_cr"]])
        match_pub.columns = pd.Index(["Design", "Matches", "Premium", "95\\% CI", "Avg. diff. (Cr)"])
        write_text(
            out_tex / "tab_matching.tex",
            latex_table_from_df(
                match_pub,
                caption="Nearest-neighbor matching robustness check",
                label="tab:matching",
                note="Treated listings are Vaastu-tagged listings matched to non-Vaastu listings within the same sector and property type using nearby structural covariates. Confidence intervals are based on a sector bootstrap.",
                size=r"\small\setlength{\tabcolsep}{4pt}",
                column_format="lrrrr",
            ),
        )

    support_pub = support_df.copy()
    support_pub["value"] = support_pub["value"].map(lambda x: fmt_num(x, 1) if abs(float(x) - round(float(x))) > 1e-9 else fmt_int(x))
    support_pub.columns = pd.Index(["Statistic", "Value"])
    write_text(
        out_tex / "tab_house_sector_support.tex",
        latex_table_from_df(
            support_pub,
            caption="Within-sector support for independent-house comparisons",
            label="tab:house-support",
            note="This table makes the small-cell problem visible: many sectors have some Vaastu houses, but far fewer have enough Vaastu and non-Vaastu houses for tight within-sector comparisons.",
        ),
    )


def save_analysis_sample(a: pd.DataFrame, out_path: Path) -> None:
    cols = [
        c
        for c in [
            "property_type",
            "sector",
            "society",
            "price",
            "price_per_sqft",
            "area",
            "bedRoom",
            "bathroom",
            "balcony_n",
            "floorNum_w",
            "facing",
            "agePossession",
            "pooja_room2",
            "servant_room2",
            "store_room2",
            "study_room2",
            "others_room2",
            "furnishing_type",
            "luxury_score_w",
            "vaastu_i",
            "features",
            "description",
            "property_name",
            "city",
            "url",
        ]
        if c in a.columns
    ]
    a[cols].to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_tex = root / "tex"
    out_fig = root / "figs"
    derived = root / "data" / "derived"
    ensure_dir(out_tex)
    ensure_dir(out_fig)
    ensure_dir(derived)

    if args.mode == "legacy_gurugram":
        raw_dir = Path(args.raw_dir) if args.raw_dir else root / "data" / "raw" / "gurugram_legacy"
        a = build_legacy_gurugram_dataset(raw_dir)
        save_analysis_sample(a, derived / "vaastu_analysis_sample.csv")
        avg_df, type_df = run_legacy_models(a)
        match_df = run_legacy_matching(a)
        counts_df = sample_counts_by_type(a)
        support_df = house_sector_support(a)
        macros = build_macros(a, avg_df, type_df, match_df, support_df)
        export_results_tables(
            out_tex=out_tex,
            counts_df=counts_df,
            avg_df=avg_df,
            type_df=type_df,
            match_df=match_df,
            support_df=support_df,
        )
        write_macros_tex(out_tex / "results_macros.tex", macros)
        plot_coefficients(avg_df, type_df, out_fig / "vaastu_coefficients.png")
    else:
        if not args.input_csv:
            raise SystemExit("--input-csv is required when mode=housingcom_collected")
        a = build_collected_housingcom_dataset(Path(args.input_csv))
        save_analysis_sample(a, derived / "housingcom_analysis_sample.csv")
        avg_df = run_collected_models(a)
        type_df = None
        match_df = None
        counts_df = pd.DataFrame(
            [{"property_type": "house", "vaastu_listings": int(a["vaastu_i"].sum()), "total_listings": int(a.shape[0]), "vaastu_share": float(a["vaastu_i"].mean())}]
        )
        support_df = house_sector_support(a)
        macros = build_macros(a, avg_df, type_df, match_df, support_df)
        export_results_tables(
            out_tex=out_tex,
            counts_df=counts_df,
            avg_df=avg_df,
            type_df=type_df,
            match_df=match_df,
            support_df=support_df,
        )
        write_macros_tex(out_tex / "results_macros.tex", macros)
        plot_coefficients(avg_df, type_df, out_fig / "vaastu_coefficients.png")

    session = {
        "mode": args.mode,
        "root": str(root),
        "rows": int(a.shape[0]),
        "generated_files": [
            str((out_tex / "tab_average_models.tex").relative_to(root)) if (out_tex / "tab_average_models.tex").exists() else None,
            str((out_fig / "vaastu_coefficients.png").relative_to(root)) if (out_fig / "vaastu_coefficients.png").exists() else None,
        ],
    }
    write_text(root / "session_info.txt", "\n".join(f"{k}: {v}" for k, v in session.items()))
    print(f"Analysis complete for mode={args.mode}. Rows={a.shape[0]}")


if __name__ == "__main__":
    main()
