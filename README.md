# Vaastu WTP

Do buyers pay a premium for Vaastu-compliant homes? How much?

## Key Findings

### Summary Across Data Sources

| Source | N | Specification | Premium | 95% CI | Sig. |
|--------|---|---------------|---------|--------|------|
| CampusX (Gurgaon) | 3,173 | + sector FE | +2.0% | [-0.5%, 4.6%] | n.s. |
| Magicbricks (multi-city) | 30,981 | + city FE | +3.8% | [2.9%, 4.8%] | *** |
| Housing.com (multi-city) | 1,596 | + city FE | +5.6% | [-3.9%, 16.0%] | n.s. |

### Aggregate Pooled Analysis

| Specification | N | Premium | 95% CI | Sig. |
|--------------|---|---------|--------|------|
| Pooled (raw) | 59,895 | +6.2% | [4.4%, 8.0%] | *** |
| + bhk | 59,895 | +0.5% | [-0.8%, 1.7%] | n.s. |
| + city FE | 59,895 | +5.6% | [4.4%, 6.9%] | *** |
| + source FE | 59,895 | +15.4% | [13.9%, 16.9%] | *** |
| + city + source FE | 59,895 | +11.5% | [10.2%, 12.9%] | *** |

### Specification Robustness by Source

**CampusX (Gurgaon only):**
| Specification | N | Premium | Sig. |
|--------------|---|---------|------|
| Raw | 3,836 | +45.0% | *** |
| + bhk | 3,836 | +37.9% | *** |
| + structural | 3,836 | +6.1% | *** |
| + sector FE | 3,173 | +2.0% | n.s. |

**Magicbricks (multi-city):**
| Specification | N | Premium | Sig. |
|--------------|---|---------|------|
| Raw | 32,940 | +32.4% | *** |
| + bhk | 32,940 | +12.6% | *** |
| + structural | 30,981 | +7.5% | *** |
| + city FE | 30,981 | +3.8% | *** |

**Magicbricks Apartments Only** (with developer FE):
| Specification | N | Premium | Sig. |
|--------------|---|---------|------|
| + structural | 26,149 | +6.4% | *** |
| + city FE | 26,149 | +2.9% | *** |
| + developer FE | 26,149 | +3.8% | *** |
| + city + developer FE | 26,149 | +3.0% | *** |

Developer FE is only meaningful for apartments/flats (90% of MagicBricks data) since houses/villas typically don't have developers listed. Within-developer comparisons still show a significant +3% Vaastu premium.

**Housing.com (multi-city):**
| Specification | N | Premium | Sig. |
|--------------|---|---------|------|
| Raw | 1,599 | +27.4% | *** |
| + bhk | 1,599 | +12.3% | ** |
| + city FE | 1,596 | +5.6% | n.s. |

**Key finding**: Raw correlations show 27-45% premiums, but these attenuate substantially with controls and location FE. With city/sector FE, source-specific premiums range from 2-6%. Magicbricks shows a significant +3.8% premium with city FE, and even with the most stringent within-developer comparisons (apartments only), the premium remains +3.0% (***). Pooled analysis with city + source FE yields +11.5% (***p<0.01, N=59,895).

## Data Sources

### Magicbricks Multi-City Data (`data/raw/magicbricks/<city>/`)

- **Source**: magicbricks.com listings (apartments, houses, villas)
- **Collected via**: `scripts/magicbricks/` 6-step pipeline (Playwright-based scraper)
- **Cities**: Delhi-NCR, Pune, Navi Mumbai, Bangalore, Jaipur, Lucknow, Patna, Chandigarh, Rajkot
- **Sample**: 49,845 listings (30,981 after quality filtering for analysis)
- **Composition**: 90% apartments, 10% houses/villas
- **Preserves**: Raw HTML for reproducibility

### 99acres CampusX Data (`data/raw/99acres_campusx/`)

- **Source**: 99acres.com listings
- **Obtained via**: Public GitHub repository [campusx-official/dsmp-capstone-project](https://github.com/campusx-official/dsmp-capstone-project)
- **Collection method**: Unknown (no documentation in source repo)
- **Files**: `flats_cleaned.csv`, `house_cleaned.csv`, `gurgaon_properties_cleaned_v2.csv`
- **Sample**: ~6,943 listings (Gurugram only)

### 99acres Kaggle Data (`data/raw/99acres_kaggle/`)

- **Source**: 99acres.com listings
- **Obtained via**: Kaggle dataset [arvanshul/real-estate-dataset-99acres](https://www.kaggle.com/datasets/arvanshul/real-estate-dataset-99acres)
- **Cities**: Gurgaon, Mumbai, Hyderabad, Kolkata
- **Sample**: ~38,000 listings (~21,512 after filtering)
- **Limitation**: Excluded from vaastu analysis. Features/amenities stored as numeric codes without decoder; vaastu detectable only in free-text descriptions (~5.5% rate vs ~50% in comparable 99acres data from CampusX)

### 99acres Multi-City Data (`data/raw/99acres/<city>/`)

- **Source**: 99acres.com listings (houses + flats)
- **Collected via**: `scripts/99acres/` 4-step pipeline (Playwright-based scraper)
- **Cities**: Bangalore, Chennai, Hyderabad, Pune, Mumbai, Delhi, Noida, Gurgaon, etc.
- **Preserves**: Raw HTML + text for reproducibility

### Housing.com Multi-City Data (`data/raw/housingcom/<city>/`)

- **Source**: Housing.com independent-house listings
- **Collected via**: `scripts/housingcom/` 4-step pipeline (Playwright-based scraper)
- **Cities**: Bangalore, Chennai, Hyderabad, Pune, Mumbai, Noida, Gurgaon, etc.
- **Preserves**: Raw HTML + text for reproducibility

## Methodology

- Hedonic regression: `log(price) ~ vaastu + controls + sector FE`
- Controls: area, bedrooms, bathrooms, balconies, floor, age, facing, furnishing
- Robustness: Nearest-neighbor matching within sector/property-type

## Data Cleaning

### Price Filters

Raw listing data contains significant noise requiring filtering:

1. **Price range filter**: 0.1 - 100 crore
   - Removes ~14,000 rows with price < 0.1 crore (likely rentals mislabeled as sales or data entry errors)
   - Removes ultra-luxury outliers > 100 crore

2. **Price per sqft filter**: 1,000 - 100,000 INR/sqft
   - Removes ~500 rows with implausible price/area ratios
   - Catches data entry errors where price or area is wrong

3. **BHK filter**: 1 - 10 bedrooms

### Outlier Examples

Some egregious outliers removed by these filters:

| Price (Cr) | BHK | Area (sqft) | Price/sqft | Issue |
|-----------|-----|-------------|------------|-------|
| 357.0 | 2 | 850 | 42 lakh/sqft | Price entry error (should be ~0.357 Cr?) |
| 0.00105 | 2 | 700 | 15/sqft | Rental mislabeled as sale |
| 216.0 | 4 | 2,455 | 8.8 lakh/sqft | Extreme outlier |

These filters reduce MagicBricks from 49,845 to ~31,000 analysis-ready rows.

## Repository Layout

```
scripts/
├── 02_analyze.py                # Main analysis script (99acres/CampusX)
├── 02_download_kaggle.py        # Download Kaggle dataset
├── 03_unify_99acres.py          # Unify 99acres data sources
├── 04_analyze_magicbricks.py    # MagicBricks hedonic regressions
├── 04_analyze_housingcom.py     # Housing.com hedonic regressions
├── 05_analyze_by_source.py      # Cross-source comparison
├── 05_validate_kaggle.py        # Kaggle data quality checks
├── export_dataverse.py          # Export data for Dataverse
│
├── 99acres/                     # 99acres 4-step pipeline
│   ├── 01_collect_search.py
│   ├── 02_extract_urls.py
│   ├── 03_collect_detail.py
│   └── 04_parse.py
│
├── housingcom/                  # Housing.com 4-step pipeline
│   ├── 01_collect_search.py
│   ├── 02_extract_urls.py
│   ├── 03_collect_detail.py
│   └── 04_parse.py
│
├── magicbricks/                 # MagicBricks 6-step pipeline
│   ├── 01_collect_search.py
│   ├── 02_extract_urls.py
│   ├── 03_collect_projects.py
│   ├── 04_extract_listing_urls.py
│   ├── 05_collect_listings.py
│   ├── 06_parse.py
│   └── run_pipeline.py
│
└── utils/                       # Shared utilities
    ├── parsing.py
    ├── scraping.py
    ├── feature_extraction.py
    └── analysis.py

data/
├── raw/
│   ├── magicbricks/<city>/      # MagicBricks scraped data
│   ├── 99acres/<city>/          # 99acres scraped data
│   ├── 99acres_campusx/         # CampusX Gurgaon data
│   ├── 99acres_kaggle/          # Kaggle dataset
│   └── housingcom/<city>/       # Housing.com scraped data
├── v1/                          # Consolidated export (parquet)
├── config/                      # City URL configurations
└── derived/                     # Analysis outputs

ms/                              # LaTeX manuscript
tabs/                            # Generated tables
figs/                            # Generated figures
tests/                           # Unit tests
```

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
uv sync
uv run playwright install chromium
```

### Data Collection

**MagicBricks** (6-step pipeline, or use run_pipeline.py):
```bash
uv run python scripts/magicbricks/run_pipeline.py --city delhi
```

**99acres**:
```bash
uv run python scripts/99acres/01_collect_search.py --city bangalore
uv run python scripts/99acres/02_extract_urls.py --city bangalore
uv run python scripts/99acres/03_collect_detail.py --city bangalore
uv run python scripts/99acres/04_parse.py --city bangalore
```

**Housing.com**:
```bash
uv run python scripts/housingcom/01_collect_search.py --city mumbai
uv run python scripts/housingcom/02_extract_urls.py --city mumbai
uv run python scripts/housingcom/03_collect_detail.py --city mumbai
uv run python scripts/housingcom/04_parse.py --city mumbai
```

### Analysis

```bash
uv run python scripts/04_analyze_magicbricks.py
uv run python scripts/04_analyze_housingcom.py
uv run python scripts/05_analyze_by_source.py
```

## Caveats

- List prices, not transaction prices
- Vaastu = text mention, not structural certification
- Within-sector support for houses is thin
- Kaggle data excluded from Vaastu analysis: features stored as numeric codes without decoder; Vaastu detectable only in free-text descriptions (~5% vs ~50% true rate)
