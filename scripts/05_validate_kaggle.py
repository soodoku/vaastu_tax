#!/usr/bin/env python3
"""
Validate Kaggle 99acres URLs by checking which listings still exist.

For each listing in the Kaggle dataset:
1. Construct full URL from PD_URL
2. Use Playwright to check if page exists (handles JS/bot detection)
3. Save validated subset

Usage:
    python scripts/05_validate_kaggle.py --test         # Test with 10 URLs
    python scripts/05_validate_kaggle.py --sample 100   # Sample 100 URLs
    python scripts/05_validate_kaggle.py                # Full validation
"""

import argparse
import asyncio
import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
KAGGLE_DIR = DATA_DIR / "raw" / "99acres_kaggle" / "arvanshul"
OUTPUT_PATH = DATA_DIR / "derived" / "kaggle_validated.csv"

BASE_URL = "https://www.99acres.com"


async def check_url_playwright(url: str, page) -> tuple[str, int, bool]:
    """Check if a URL returns valid property page using Playwright."""
    try:
        response = await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        status = response.status if response else 0

        if status == 200:
            content = await page.content()
            is_valid = (
                "Property Not Available" not in content
                and "Page not found" not in content
                and "property-heading" in content.lower()
                or "price" in content.lower()
            )
            return url, status, is_valid
        return url, status, False
    except Exception:
        return url, 0, False


async def check_urls_playwright(urls: list[str], concurrency: int = 5) -> dict[str, tuple[int, bool]]:
    """Check URLs using Playwright browser."""
    from playwright.async_api import async_playwright

    results = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )

        pages = [await context.new_page() for _ in range(concurrency)]

        for i in range(0, len(urls), concurrency):
            batch = urls[i : i + concurrency]
            tasks = []
            for j, url in enumerate(batch):
                page = pages[j % len(pages)]
                tasks.append(check_url_playwright(url, page))

            batch_results = await asyncio.gather(*tasks)
            for url, status, is_valid in batch_results:
                results[url] = (status, is_valid)

            done = min(i + concurrency, len(urls))
            valid_so_far = sum(1 for v in results.values() if v[1])
            print(f"  Checked {done}/{len(urls)} URLs ({valid_so_far} valid)...", end="\r")

            if i + concurrency < len(urls):
                await asyncio.sleep(0.5)

        await browser.close()

    print()
    return results


def load_kaggle_data() -> pd.DataFrame:
    """Load all Kaggle city files."""
    all_data = []

    for csv_path in KAGGLE_DIR.glob("*.csv"):
        if csv_path.name == "facets":
            continue
        if csv_path.is_dir():
            continue

        try:
            df = pd.read_csv(csv_path, low_memory=False)
            if "PD_URL" not in df.columns:
                continue

            city = csv_path.stem.replace("_10k", "").title()
            df["source_city"] = city
            df["source_file"] = csv_path.name
            all_data.append(df)
            print(f"  Loaded {len(df):,} listings from {csv_path.name}")
        except Exception as e:
            print(f"  Error loading {csv_path.name}: {e}")

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Validate Kaggle 99acres URLs")
    parser.add_argument("--test", action="store_true", help="Test with 10 URLs")
    parser.add_argument(
        "--sample", type=int, default=0, help="Sample N URLs (0 = all)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Concurrent requests"
    )
    args = parser.parse_args()

    print("Loading Kaggle data...")
    df = load_kaggle_data()
    print(f"Total: {len(df):,} listings from {df['source_city'].nunique()} cities")

    df = df[df["PD_URL"].notna() & (df["PD_URL"] != "")]
    print(f"With valid PD_URL: {len(df):,}")

    if args.test:
        df = df.head(10)
        print("Test mode: checking 10 URLs")
    elif args.sample > 0:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print(f"Sample mode: checking {len(df)} URLs")

    df = df.copy()
    df["full_url"] = BASE_URL + df["PD_URL"]

    print(f"\nChecking {len(df):,} URLs using Playwright...")
    start = time.time()

    url_results = asyncio.run(check_urls_playwright(df["full_url"].tolist(), args.batch_size))

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s ({len(df)/elapsed:.1f} URLs/s)")

    df.loc[:, "http_status"] = df["full_url"].apply(lambda u: url_results.get(u, (0, False))[0])
    df.loc[:, "is_valid"] = df["full_url"].apply(lambda u: url_results.get(u, (0, False))[1])

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    valid_count = df["is_valid"].sum()
    total = len(df)
    print(f"\nOverall: {valid_count:,}/{total:,} ({100*valid_count/total:.1f}%) still live")

    print("\nBy city:")
    city_stats = (
        df.groupby("source_city")
        .agg(
            total=("is_valid", "count"),
            valid=("is_valid", "sum"),
        )
        .assign(pct=lambda x: 100 * x["valid"] / x["total"])
    )
    print(city_stats.to_string())

    print("\nBy HTTP status:")
    print(df["http_status"].value_counts().to_string())

    if not args.test:
        valid_df = df[df["is_valid"]].copy()
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        valid_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved {len(valid_df):,} validated listings to {OUTPUT_PATH}")

    if "EXPIRY_DATE" in df.columns:
        print("\nExpiry date distribution:")
        df["expiry_year"] = pd.to_datetime(df["EXPIRY_DATE"], errors="coerce").dt.year
        print(df["expiry_year"].value_counts().sort_index().to_string())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Validation rate: {100*valid_count/total:.1f}%
Valid listings: {valid_count:,}
Invalid listings: {total - valid_count:,}

Note: The Kaggle dataset contains listings from 2023 that have since expired.
99acres listing URLs are time-limited and become invalid after expiry.
For fresh data, use the scraper: scripts/01_collect_99acres.py
""")


if __name__ == "__main__":
    main()
