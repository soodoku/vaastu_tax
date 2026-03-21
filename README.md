# Vaastu WTP

Estimating willingness to pay for Vaastu compliance in Indian residential real estate.

## Research Question

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

**Housing.com (multi-city):**
| Specification | N | Premium | Sig. |
|--------------|---|---------|------|
| Raw | 1,599 | +27.4% | *** |
| + bhk | 1,599 | +12.3% | ** |
| + city FE | 1,596 | +5.6% | n.s. |

**Key finding**: Raw correlations show 27-45% premiums, but these attenuate substantially with controls and location FE. With city/sector FE, source-specific premiums range from 2-6%. Magicbricks shows a significant +3.8% premium with city FE. Pooled analysis with city + source FE yields +11.5% (***p<0.01, N=59,895).

## Data Sources

### Magicbricks Multi-City Data (`data/raw/magicbricks/<city>/`)

- **Source**: magicbricks.com listings (apartments, houses, villas)
- **Collected via**: `scripts/magicbricks/` pipeline (Playwright-based scraper)
- **Cities**: Delhi-NCR, Pune, Navi Mumbai, Bangalore, Jaipur, Lucknow, Patna, Chandigarh, Rajkot
- **Sample**: 49,845 listings (33,421 after quality filtering for analysis)
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
├── 01_collect_*.py              # Data collection (99acres, Housing.com, Magicbricks)
├── 02_parse_*.py                # Parse HTML to parquet
├── 02_analyze.py                # Main analysis script (99acres/CampusX)
├── 02_download_kaggle.py        # Download Kaggle dataset
├── 03_unify_99acres.py          # Unify 99acres data sources
├── 04_analyze_*.py              # Source-specific hedonic regressions
├── 04_rationalize_covariates.py # Covariate harmonization
├── 05_analyze_by_source.py      # Cross-source comparison
├── 05_validate_kaggle.py        # Kaggle data quality checks
├── utils/                       # Shared utilities module
│   ├── __init__.py
│   ├── parsing.py               # File I/O, regex patterns, data extraction
│   ├── scraping.py              # Browser automation, proxy, retry logic
│   ├── feature_extraction.py    # Vaastu detection, sector/city parsing
│   └── analysis.py              # Hedonic regression utilities
└── magicbricks/                 # MagicBricks 4-step pipeline
    ├── 01_collect_search.py     # Collect search result pages
    ├── 02_extract_urls.py       # Extract property URLs from search
    ├── 03_collect_detail.py     # Collect individual property pages
    └── 04_parse.py              # Parse HTML to parquet

data/
├── raw/
│   ├── magicbricks/<city>/      # Magicbricks scraped data
│   ├── 99acres/<city>/          # 99acres scraped data (multi-city)
│   ├── 99acres_campusx/         # 99acres data from CampusX repo (Gurugram)
│   ├── housingcom/<city>/       # Housing.com scraped data
│   └── kaggle_arvanshul/        # Kaggle dataset (arvanshul)
├── config/                      # City URL configurations
└── derived/                     # Analysis samples, unified datasets

ms/                              # LaTeX manuscript
tabs/                            # Generated tables
figs/                            # Generated figures
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

**Magicbricks** (4-step pipeline):
```bash
uv run python scripts/magicbricks/01_collect_search.py --city delhi
uv run python scripts/magicbricks/02_extract_urls.py --city delhi
uv run python scripts/magicbricks/03_collect_detail.py --city delhi
uv run python scripts/magicbricks/04_parse.py --city delhi
```

**99acres**:
```bash
uv run python scripts/01_collect_99acres.py --city bangalore --max-pages 10
uv run python scripts/02_parse_99acres.py --city bangalore
```

**Housing.com**:
```bash
uv run python scripts/01_collect_housingcom.py --city mumbai --max-pages 10
uv run python scripts/02_parse_housingcom.py --city mumbai
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
- Kaggle/arvanshul data excluded from vaastu analysis: features stored as numeric codes without decoder; vaastu detectable only in free-text descriptions (~5% vs ~50% true rate)
