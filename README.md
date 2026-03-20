# Vaastu WTP

Estimating willingness to pay for Vaastu compliance in Indian residential real estate.

## Research Question

Do buyers pay a premium for Vaastu-compliant homes? How much?

## Key Findings

### Summary Across Data Sources

| Source | N | Vaastu % | Premium | Sig. |
|--------|---|----------|---------|------|
| CampusX (Gurgaon) | 3,916 | 53% | +6.7% | *** |
| Magicbricks (5 cities) | 26,621 | 39.4% | +3.8% | *** |
| Housing.com (7 cities) | 1,670 | 16% | +7.1% | n.s. |

### Gurugram Prototype (CampusX Data)

| Specification | N | Vaastu Premium | 95% CI | Significance |
|--------------|---|----------------|--------|--------------|
| + sector FE (preferred) | 3,916 | +2.7% | [-0.9%, 6.4%] | n.s. |
| Matching robustness | 392 | +6.0% | [0.4%, 12.4%] | * |

### Housing.com Robustness (by City)

| City | N | Vaastu % | Premium | Sig. |
|------|---|----------|---------|------|
| Mumbai | 270 | 12% | +36.3% | ** |
| Bangalore | 262 | 16% | +19.3% | n.s. |
| Pune | 274 | 27% | +5.5% | n.s. |
| Chennai | 285 | 12% | +0.6% | n.s. |
| Hyderabad | 277 | 19% | -3.3% | n.s. |

Key finding: The Vaastu premium in Gurugram is modest once sector FE are included (+2.7%, n.s.). Matching robustness check yields +6.0% (* p<0.05). Housing.com data provides directionally consistent results (+7.1%) but with wider confidence intervals due to lower Vaastu mention rates.

### Magicbricks Multi-City Data

| City | N | Vaastu % | Premium | Sig. |
|------|---|----------|---------|------|
| Delhi | 12,368 | 45.4% | +3.8% | *** |
| Pune | 5,953 | 37.1% | +7.7% | *** |
| Navi Mumbai | 3,306 | 34.3% | +2.1% | n.s. |
| Jaipur | 2,089 | 29.8% | +3.9% | * |
| Lucknow | 1,626 | 33.0% | +3.6% | n.s. |
| **Total** | **26,621** | **39.4%** | - | - |

#### Fixed Effects Robustness

| Specification | N | Premium | p-value |
|--------------|---|---------|---------|
| Project FE | 24,604 | +1.4% | 0.005 |
| Developer FE | 20,868 | +3.7% | <0.0001 |

## Data Sources

### Magicbricks Multi-City Data (`data/raw/magicbricks/<city>/`)

- **Source**: magicbricks.com listings (apartments, houses, villas)
- **Collected via**: `scripts/01_collect_magicbricks.py` (Playwright-based scraper)
- **Cities**: Delhi-NCR, Pune, Navi Mumbai, Bangalore, Jaipur, Lucknow, Patna, Chandigarh, Rajkot
- **Sample**: 26,621 listings (sale only, after deduplication and cleaning)
- **Preserves**: Raw HTML for reproducibility

### 99acres CampusX Data (`data/raw/99acres_campusx/`)

- **Source**: 99acres.com listings
- **Obtained via**: Public GitHub repository [campusx-official/dsmp-capstone-project](https://github.com/campusx-official/dsmp-capstone-project)
- **Collection method**: Unknown (no documentation in source repo)
- **Files**: `flats_cleaned.csv`, `house_cleaned.csv`, `gurgaon_properties_cleaned_v2.csv`
- **Sample**: ~6,943 listings (Gurugram only)

### 99acres Kaggle/arvanshul Data (`data/raw/kaggle_arvanshul/`)

- **Source**: 99acres.com listings
- **Obtained via**: Kaggle dataset [arvanshul/real-estate-dataset-99acres](https://www.kaggle.com/datasets/arvanshul/real-estate-dataset-99acres)
- **Cities**: Gurgaon, Mumbai, Hyderabad, Kolkata
- **Sample**: ~38,000 listings (~21,512 after filtering)
- **Limitation**: Excluded from vaastu analysis. Features/amenities stored as numeric codes without decoder; vaastu detectable only in free-text descriptions (~5.5% rate vs ~50% in comparable 99acres data from CampusX)

### 99acres Multi-City Data (`data/raw/99acres/<city>/`)

- **Source**: 99acres.com listings (houses + flats)
- **Collected via**: `scripts/01_collect_99acres.py` (Playwright-based scraper)
- **Cities**: Bangalore, Chennai, Hyderabad, Pune, Mumbai, Delhi, Noida, Gurgaon, etc.
- **Preserves**: Raw HTML + text for reproducibility

### Housing.com Multi-City Data (`data/raw/housingcom/<city>/`)

- **Source**: Housing.com independent-house listings
- **Collected via**: `scripts/01_collect_housingcom.py` (Playwright-based scraper)
- **Cities**: Bangalore, Chennai, Hyderabad, Pune, Mumbai, Noida, Gurgaon, etc.
- **Preserves**: Raw HTML + text for reproducibility

## Methodology

- Hedonic regression: `log(price) ~ vaastu + controls + sector FE`
- Controls: area, bedrooms, bathrooms, balconies, floor, age, facing, furnishing
- Robustness: Nearest-neighbor matching within sector/property-type

## Repository Layout

```
scripts/
  01_collect_magicbricks.py    # Scrape Magicbricks listings
  01_collect_99acres.py        # Scrape 99acres.com listings
  01_collect_housingcom.py     # Scrape Housing.com listings
  02_parse_magicbricks.py      # Parse Magicbricks HTML to parquet
  02_parse_99acres.py          # Parse 99acres HTML to parquet
  02_parse_housingcom.py       # Parse Housing.com HTML to parquet
  04_analyze_magicbricks.py    # Magicbricks hedonic regressions
  04_analyze_housingcom.py     # Housing.com hedonic regressions
  05_validate_kaggle.py        # Kaggle data quality checks
  utils/                       # Shared utilities (parsing, analysis)
data/
  raw/magicbricks/<city>/   # Magicbricks scraped data
  raw/99acres_campusx/      # 99acres data from campusx repo (Gurugram)
  raw/99acres/<city>/       # 99acres scraped data (multi-city)
  raw/housingcom/<city>/    # Housing.com scraped data
  raw/99acres_kaggle/       # Kaggle dataset (arvanshul)
  config/                   # City URL configurations
  derived/                  # Analysis samples
ms/                         # LaTeX manuscript
tabs/                       # Generated tables
figs/                       # Generated figures
```

## Quick Start

```bash
# Using uv (recommended)
uv sync
uv run playwright install chromium

# Collect Magicbricks data
uv run python scripts/01_collect_magicbricks.py --city delhi-ncr --max-pages 5

# Parse collected data
uv run python scripts/02_parse_magicbricks.py --all-cities --force

# Run analysis
uv run python scripts/04_analyze_magicbricks.py

# Validate Kaggle data quality
uv run python scripts/05_validate_kaggle.py
```

## Caveats

- List prices, not transaction prices
- Vaastu = text mention, not structural certification
- Within-sector support for houses is thin
- Kaggle/arvanshul data excluded from vaastu analysis: features stored as numeric codes without decoder; vaastu detectable only in free-text descriptions (~5% vs ~50% true rate)
