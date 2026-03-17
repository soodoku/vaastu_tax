# Vaastu WTP

Estimating willingness to pay for Vaastu compliance in Indian residential real estate.

## Research Question

Do buyers pay a premium for Vaastu-compliant homes? How much?

## Key Findings

Hedonic regression results from Gurugram prototype (n=3,916):

| Specification | N | Vaastu Premium | 95% CI | Significance |
|--------------|---|----------------|--------|--------------|
| + sector FE (average) | 3,916 | +2.7% | [-0.9%, 6.4%] | n.s. |
| Flat (type-specific) | 2,995 | -0.0% | [-3.4%, 3.4%] | n.s. |
| **House (type-specific)** | 921 | **+11.9%** | [1.1%, 23.9%] | ** |
| Matching robustness | 392 | +6.0% | [0.4%, 12.4%] | * |

Key finding: Independent houses show a significant 11.9% Vaastu premium (~44.7 lakh at median price), while flats show no effect.

## Data Sources

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
  01_collect_99acres.py        # Scrape 99acres.com listings (houses + flats)
  01_collect_housingcom.py     # Scrape Housing.com listings
  02_analyze.py                # Run hedonic regressions, generate tables/figures
  03_extract_vaastu.py         # Build unified analysis dataset from all sources
  04_rationalize_covariates.py # Check covariate coverage by source
  05_analyze_by_source.py      # Source-stratified hedonic regressions
data/
  raw/99acres_campusx/      # 99acres data from campusx repo (Gurugram)
  raw/99acres/<city>/       # 99acres scraped data (multi-city)
  raw/housingcom/<city>/    # Housing.com scraped data
  config/                   # City URL configurations
  derived/                  # Analysis samples
ms/                         # LaTeX manuscript
tabs/                        # Generated tables and macros
figs/                       # Generated figures
```

## Quick Start

```bash
pip install -r requirements.txt
playwright install chromium

# Collect 99acres data
python scripts/01_collect_99acres.py --city gurgaon --property-type both --max-pages 5

# Run analysis on legacy data
python scripts/02_analyze.py --mode legacy_gurugram
```

## Caveats

- List prices, not transaction prices
- Vaastu = text mention, not structural certification
- Within-sector support for houses is thin
