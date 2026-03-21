#!/usr/bin/env python3
"""Run the full MagicBricks scraping pipeline for specified cities.

Executes all 4 steps sequentially for each city:
1. Collect search pages
2. Extract URLs
3. Collect detail pages
4. Parse to parquet

Example
-------
python scripts/magicbricks/run_pipeline.py --cities delhi-ncr jaipur --max-search-pages 50
python scripts/magicbricks/run_pipeline.py --city-prefix delhi-ncr chandigarh lucknow patna
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils import project_root
from scripts.utils.scraping import logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    city_group = parser.add_mutually_exclusive_group(required=True)
    city_group.add_argument(
        "--cities",
        nargs="+",
        help="Full city keys (e.g., delhi-ncr_apartment)",
    )
    city_group.add_argument(
        "--city-prefix",
        nargs="+",
        help="City prefixes to match (e.g., delhi-ncr matches all delhi-ncr_* keys)",
    )
    parser.add_argument(
        "--max-search-pages",
        type=int,
        default=50,
        help="Max search pages per city (default: 50)",
    )
    parser.add_argument(
        "--max-detail-pages",
        type=int,
        default=500,
        help="Max detail pages per city (default: 500)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from where we left off",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def load_city_config() -> dict:
    config_path = project_root() / "data" / "config" / "magicbricks_cities.json"
    with open(config_path) as f:
        return json.load(f)


def get_cities_to_process(args: argparse.Namespace, config: dict) -> list[str]:
    if args.cities:
        return args.cities

    cities = []
    for prefix in args.city_prefix:
        for key in config:
            if key.startswith(prefix + "_"):
                cities.append(key)
    return sorted(cities)


def run_step(script: str, city: str, extra_args: list[str]) -> bool:
    script_path = project_root() / "scripts" / "magicbricks" / script
    cmd = [sys.executable, str(script_path), "--city", city] + extra_args

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_pipeline_for_city(city: str, args: argparse.Namespace) -> dict:
    logger.info("=" * 60)
    logger.info("Starting pipeline for %s", city)
    logger.info("=" * 60)

    results = {"city": city, "steps": {}}

    common_args = []
    if args.headless:
        common_args.append("--headless")
    if args.verbose:
        common_args.append("-v")

    # Step 1: Collect search pages
    step1_args = common_args + ["--max-pages", str(args.max_search_pages)]
    if args.resume:
        step1_args.append("--resume")
    success = run_step("01_collect_search.py", city, step1_args)
    results["steps"]["collect_search"] = success
    if not success:
        logger.error("[%s] Step 1 (collect_search) failed", city)
        return results

    # Step 2: Extract URLs
    success = run_step("02_extract_urls.py", city, ["-v"] if args.verbose else [])
    results["steps"]["extract_urls"] = success
    if not success:
        logger.error("[%s] Step 2 (extract_urls) failed", city)
        return results

    # Step 3: Collect detail pages
    step3_args = common_args + ["--max-pages", str(args.max_detail_pages)]
    if args.resume:
        step3_args.append("--resume")
    success = run_step("03_collect_detail.py", city, step3_args)
    results["steps"]["collect_detail"] = success
    if not success:
        logger.error("[%s] Step 3 (collect_detail) failed", city)
        return results

    # Step 4: Parse to parquet
    success = run_step("04_parse.py", city, ["-v"] if args.verbose else [])
    results["steps"]["parse"] = success
    if not success:
        logger.error("[%s] Step 4 (parse) failed", city)
        return results

    logger.info("[%s] Pipeline completed successfully", city)
    return results


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    config = load_city_config()
    cities = get_cities_to_process(args, config)

    if not cities:
        logger.error("No cities matched the specified criteria")
        sys.exit(1)

    logger.info("Processing %d cities: %s", len(cities), cities)

    all_results = []
    for city in cities:
        if city not in config:
            logger.warning("City %s not found in config, skipping", city)
            continue

        result = run_pipeline_for_city(city, args)
        all_results.append(result)
        print(json.dumps(result, indent=2))

    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    successful = sum(1 for r in all_results if all(r["steps"].values()))
    logger.info("Cities processed: %d", len(all_results))
    logger.info("Fully successful: %d", successful)
    logger.info("With errors: %d", len(all_results) - successful)


if __name__ == "__main__":
    main()
