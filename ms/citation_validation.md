# Citation validation log

Validated on 2026-03-15.

This file records the bibliography entries used in `vaastu_wtp_refs.bib` and the metadata fields that were cross-checked before inclusion in the manuscript. The goal is mundane but crucial: no zombie citations, no mashed-up page ranges, no DOI spaghetti.

| Key | Source type | Fields validated | Notes |
|---|---|---|---|
| `rosen1974` | Journal article | author, title, journal, year, volume, issue, pages, DOI | Canonical hedonic-pricing reference. |
| `bourassa_peng1999` | Journal article | author, title, journal, year, volume, issue, pages, DOI | Auckland lucky-house-number paper. |
| `lin_chen_twu2012` | Journal article | author, title, journal, year, volume, issue, pages, DOI | Taiwan feng shui quantile-regression paper. |
| `shum_sun_ye2014` | Journal article | author, title, journal, year, volume, issue, pages, DOI | Chinese transaction-level lucky-apartment paper. |
| `fortin_hill_huang2014` | Journal article | author, title, journal, year, volume, issue, pages, DOI | Vancouver superstition paper; the bibliography uses the journal version rather than the earlier IZA discussion paper. |
| `lu2018` | Journal article | author, title, journal, year, volume, pages, DOI | Shanghai south-facing-orientation paper. |
| `campusx_dsmp_capstone` | Repository | repository name, URL, access date | Public GitHub repository containing the Gurugram listing files bundled here. |
| `housing_hyderabad_2026` | Web page | URL, access date, displayed city inventory count | Housing.com Hyderabad independent-house search page. |
| `housing_bangalore_2026` | Web page | URL, access date, displayed city inventory count | Housing.com Bangalore independent-house search page. |
| `housing_chennai_2026` | Web page | URL, access date, displayed city inventory count | Housing.com Chennai independent-house search page. |
| `housing_pune_2026` | Web page | URL, access date, displayed city inventory count | Housing.com Pune independent-house search page. |

## Practical notes

1. The manuscript intentionally cites the journal version of Fortin, Hill, and Huang rather than only the IZA discussion paper.
2. The portal citations are used only to document data-collection feasibility and visible inventory scale, not as scientific evidence about causal effects.
3. The Gurugram analysis itself is fully reproducible from the local CSV files in `data/raw/gurugram_legacy/` and the analysis script in `scripts/analyze_vaastu.py`.
