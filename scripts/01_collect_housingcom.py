#!/usr/bin/env python3
"""Collect Housing.com independent-house listings with Vaastu-related text fields.

The script is designed for a cautious, reproducible workflow:

1. Check robots.txt before each fetch.
2. Save raw HTML and raw body text for every page.
3. Parse a conservative set of fields from listing-detail text.
4. Emit a normalized CSV/JSONL that can feed the analysis pipeline.

Notes
-----
- Housing portals change their DOMs often. This collector therefore saves raw files first
  and uses mostly text-based parsing rather than brittle CSS selectors.
- The default city URLs live in data/config/cities.json and can be edited without
  touching the code.
- The script prefers Playwright because some pages are rendered dynamically.

Example
-------
python scripts/01_collect_housingcom.py \
    --city hyderabad \
    --max-pages 5 \
    --max-detail-pages 200 \
    --output-dir data/raw/housingcom/hyderabad

python scripts/01_collect_housingcom.py --all-cities --max-pages 3 --max-detail-pages 100
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

try:
    from playwright.sync_api import sync_playwright
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Playwright is required. Install dependencies and run 'playwright install chromium'."
    ) from exc


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DOMAIN = "https://housing.com"
SEARCH_STOP_HEADS = {
    "property location",
    "amenities",
    "about this property",
    "overview",
    "project details",
    "nearby places",
    "specifications",
    "furnishing",
    "configuration",
    "ratings",
    "locality",
    "calculator",
    "sale and rent trends",
    "home loan offers",
    "similar properties",
    "ratings by features",
    "what is good here",
    "what can be better",
    "reviews by residents",
    "faq",
    "read more",
}


@dataclass
class ListingRecord:
    property_id: str
    url: str
    city: str
    search_page: int
    title: Optional[str]
    locality_line: Optional[str]
    price_display: Optional[str]
    price_crore: Optional[float]
    builtup_area_sqft: Optional[float]
    avg_price_per_sqft: Optional[float]
    bhk: Optional[float]
    bathrooms: Optional[float]
    balconies: Optional[float]
    furnishing: Optional[str]
    facing: Optional[str]
    possession_status: Optional[str]
    special_highlights: Optional[str]
    amenities: Optional[str]
    about_this_property: Optional[str]
    vaastu_mentioned: int
    vaastu_mentions_text: Optional[str]
    last_updated: Optional[str]
    raw_text_path: str
    raw_html_path: str


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root_from_here()
    parser = argparse.ArgumentParser(description=__doc__)
    city_group = parser.add_mutually_exclusive_group(required=True)
    city_group.add_argument("--city", help="City key from data/config/cities.json")
    city_group.add_argument(
        "--all-cities",
        action="store_true",
        help="Collect data for all cities in the config file",
    )
    parser.add_argument(
        "--config",
        default=str(root / "data" / "config" / "housingcom_cities.json"),
        help="Path to city-URL config JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for raw pages and parsed outputs. Defaults to data/raw/housingcom/<city>",
    )
    parser.add_argument("--max-pages", type=int, default=5, help="Number of search-result pages to visit")
    parser.add_argument(
        "--max-detail-pages",
        type=int,
        default=250,
        help="Hard cap on listing detail pages fetched after deduplication",
    )
    parser.add_argument("--min-sleep", type=float, default=1.0, help="Minimum sleep between requests")
    parser.add_argument("--max-sleep", type=float, default=2.5, help="Maximum sleep between requests")
    parser.add_argument("--timeout-ms", type=int, default=60000, help="Playwright page timeout in ms")
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run Chromium with a visible window (helpful when debugging anti-bot issues)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip detail pages whose parsed rows already exist in parsed_listings.csv",
    )
    return parser.parse_args()


def normalize_ws(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "na"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_city_config(path: Path) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a JSON object of city -> url.")
    return data


class RobotsGuard:
    def __init__(self, root_url: str, user_agent: str = USER_AGENT) -> None:
        self.root_url = root_url.rstrip("/")
        self.user_agent = user_agent
        self.rp = RobotFileParser()
        self.rp.set_url(f"{self.root_url}/robots.txt")
        self.rp.read()

    def assert_allowed(self, url: str) -> None:
        allowed = self.rp.can_fetch(self.user_agent, url)
        if not allowed:
            raise PermissionError(f"robots.txt disallows fetch: {url}")


RE_SHOWING = re.compile(r"Showing\s+\d+\s*-\s*\d+\s+of\s+([\d,]+)", re.I)
RE_PRICE_UNIT = re.compile(r"([0-9][\d,]*(?:\.\d+)?)\s*(Cr|Crore|L|Lac|Lakh|Lakhs|K)", re.I)
RE_TITLE = re.compile(r"^(\d+(?:\.\d+)?)\s*BHK\s+.+?(Independent House|Villa|Row House)\b", re.I)
RE_BHK = re.compile(r"(\d+(?:\.\d+)?)\s*BHK\b", re.I)
RE_SQFT = re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft|square feet)\b", re.I)
RE_PPSF = re.compile(r"₹\s*([\d,]+(?:\.\d+)?)\s*/\s*(?:sq\.?\s*ft|sqft)", re.I)
RE_BATH = re.compile(r"(\d+(?:\.\d+)?)\s*(?:Bath|Bathroom)s?\b", re.I)
RE_BALCONY = re.compile(r"(\d+(?:\.\d+)?)\s*Balcon(?:y|ies)\b", re.I)
RE_LAST_UPDATED = re.compile(r"Last updated:?\s*(.+)$", re.I)
RE_POSSESSION = re.compile(
    r"\b(Ready to move|Ready to Move|Under Construction|Resale|New Launch|Immediately|Possession Started)\b",
    re.I,
)
RE_FACING = re.compile(
    r"\b(North(?:[- ]East|[- ]West)?|South(?:[- ]East|[- ]West)?|East|West)\s*Facing\b|"
    r"\bFacing\s*:?\s*(North(?:[- ]East|[- ]West)?|South(?:[- ]East|[- ]West)?|East|West)\b",
    re.I,
)
RE_FURNISHING = re.compile(r"\b(Unfurnished|Semi[- ]Furnished|Fully Furnished|Furnished)\b", re.I)
RE_VAASTU = re.compile(r"\bvaa?stu\b|\bvastu\b", re.I)
RE_PROPERTY_ID = re.compile(r"(?:/page/|/)(\d{5,})[^/]*$")


def jitter_sleep(lo: float, hi: float) -> None:
    if hi <= 0:
        return
    time.sleep(random.uniform(max(0.0, lo), max(lo, hi)))


def price_to_crore(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    raw = normalize_ws(raw)
    if "price on request" in raw.lower():
        return None
    m = RE_PRICE_UNIT.search(raw)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    unit = m.group(2).lower()
    if unit in {"cr", "crore"}:
        return value
    if unit in {"l", "lac", "lakh", "lakhs"}:
        return value / 100.0
    if unit == "k":
        return value / 100000.0
    return None


def number_from_match(pattern: re.Pattern[str], text: str) -> Optional[float]:
    m = pattern.search(text)
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def extract_lines(text: str) -> List[str]:
    lines = [normalize_ws(x) for x in text.splitlines()]
    return [x for x in lines if x]


def find_title(lines: Sequence[str]) -> Optional[str]:
    for line in lines:
        if RE_TITLE.search(line):
            return line
    return None


def find_locality_line(lines: Sequence[str], title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    try:
        idx = list(lines).index(title)
    except ValueError:
        return None
    for cand in lines[idx + 1 : idx + 6]:
        lower = cand.lower()
        if not cand:
            continue
        if "₹" in cand:
            continue
        if RE_SHOWING.search(cand):
            continue
        if any(tok in lower for tok in ["emi", "sq.ft", "special highlights", "amenities", "about this property"]):
            continue
        return cand
    return None


def extract_section(lines: Sequence[str], heading: str) -> Optional[str]:
    heading_l = heading.lower().strip()
    start = None
    for idx, line in enumerate(lines):
        if line.lower() == heading_l:
            start = idx + 1
            break
    if start is None:
        return None

    buffer: List[str] = []
    for line in lines[start:]:
        line_l = line.lower().strip()
        if line_l in SEARCH_STOP_HEADS:
            break
        buffer.append(line)
    out = normalize_ws(" | ".join(buffer))
    return out or None


def find_price_display(lines: Sequence[str], title_idx: Optional[int]) -> Optional[str]:
    start = title_idx + 1 if title_idx is not None else 0
    for line in lines[start : start + 10]:
        if "₹" in line or "price on request" in line.lower():
            return line
    for line in lines:
        if "₹" in line or "price on request" in line.lower():
            return line
    return None


def find_first_matching_line(lines: Sequence[str], pattern: re.Pattern[str]) -> Optional[str]:
    for line in lines:
        if pattern.search(line):
            return line
    return None


def normalize_direction(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = text.replace(" ", "-")
    mapping = {
        "North-East": "North-East",
        "North-West": "North-West",
        "South-East": "South-East",
        "South-West": "South-West",
        "North": "North",
        "South": "South",
        "East": "East",
        "West": "West",
    }
    for key, val in mapping.items():
        if key.lower() in text.lower():
            return val
    return normalize_ws(text)


def property_id_from_url(url: str) -> str:
    m = RE_PROPERTY_ID.search(url)
    if m:
        return m.group(1)
    stem = Path(urlparse(url).path).name
    return slugify(stem)


def inventory_total_from_text(text: str) -> Optional[int]:
    m = RE_SHOWING.search(text)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def extract_detail_links(hrefs: Iterable[str], root_url: str) -> List[str]:
    out: List[str] = []
    for href in hrefs:
        if not href:
            continue
        url = urljoin(root_url, href)
        parsed = urlparse(url)
        if parsed.netloc and "housing.com" not in parsed.netloc:
            continue
        path = parsed.path.lower()
        if "/in/buy/" not in path:
            continue
        if any(bad in path for bad in ["/price-trends", "/locality-insights", "/builder/", "/projects/"]):
            continue
        # Keep property-detail-like URLs. Housing.com patterns can vary a bit, so we allow
        # either an explicit /page/<id> path fragment or the '-for-rs-' slug style.
        if not ("/page/" in path or "-for-rs-" in path):
            continue
        if url not in out:
            out.append(url)
    return out


def parse_detail_text(*, text: str, url: str, city: str, search_page: int, raw_text_path: str, raw_html_path: str) -> ListingRecord:
    lines = extract_lines(text)
    title = find_title(lines)
    title_idx = lines.index(title) if title in lines else None
    locality_line = find_locality_line(lines, title)
    price_display = find_price_display(lines, title_idx)
    price_crore = price_to_crore(price_display)

    title_and_body = "\n".join(lines)
    bhk = number_from_match(RE_BHK, title or title_and_body)
    bathrooms = number_from_match(RE_BATH, title_and_body)
    balconies = number_from_match(RE_BALCONY, title_and_body)
    builtup_area_sqft = number_from_match(RE_SQFT, title_and_body)
    avg_price_per_sqft = number_from_match(RE_PPSF, title_and_body)

    facing_line = find_first_matching_line(lines, RE_FACING)
    furnishing_line = find_first_matching_line(lines, RE_FURNISHING)
    possession_line = find_first_matching_line(lines, RE_POSSESSION)

    facing = None
    if facing_line:
        m = RE_FACING.search(facing_line)
        if m:
            facing = normalize_direction(m.group(1) or m.group(2))

    furnishing = None
    if furnishing_line:
        m = RE_FURNISHING.search(furnishing_line)
        furnishing = normalize_ws(m.group(1)) if m else None

    possession_status = None
    if possession_line:
        m = RE_POSSESSION.search(possession_line)
        possession_status = normalize_ws(m.group(1)) if m else None

    special_highlights = extract_section(lines, "Special Highlights")
    amenities = extract_section(lines, "Amenities")
    about_this_property = extract_section(lines, "About this property")

    vaastu_chunks = []
    for chunk in [title, special_highlights, amenities, about_this_property, title_and_body]:
        if chunk and RE_VAASTU.search(chunk):
            vaastu_chunks.append(chunk)
    vaastu_mentions_text = " || ".join(dict.fromkeys(vaastu_chunks)) if vaastu_chunks else None
    vaastu_mentioned = int(bool(vaastu_chunks))

    last_updated = None
    for line in lines:
        m = RE_LAST_UPDATED.search(line)
        if m:
            last_updated = normalize_ws(m.group(1))
            break

    return ListingRecord(
        property_id=property_id_from_url(url),
        url=url,
        city=city,
        search_page=search_page,
        title=title,
        locality_line=locality_line,
        price_display=price_display,
        price_crore=price_crore,
        builtup_area_sqft=builtup_area_sqft,
        avg_price_per_sqft=avg_price_per_sqft,
        bhk=bhk,
        bathrooms=bathrooms,
        balconies=balconies,
        furnishing=furnishing,
        facing=facing,
        possession_status=possession_status,
        special_highlights=special_highlights,
        amenities=amenities,
        about_this_property=about_this_property,
        vaastu_mentioned=vaastu_mentioned,
        vaastu_mentions_text=vaastu_mentions_text,
        last_updated=last_updated,
        raw_text_path=raw_text_path,
        raw_html_path=raw_html_path,
    )


def save_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_existing_property_ids(parsed_csv: Path) -> set[str]:
    if not parsed_csv.exists():
        return set()
    import pandas as pd

    try:
        df = pd.read_csv(parsed_csv, dtype={"property_id": str})
    except Exception:
        return set()
    return set(df["property_id"].dropna().astype(str))


def get_fieldnames() -> List[str]:
    return list(asdict(ListingRecord(
        property_id="",
        url="",
        city="",
        search_page=0,
        title=None,
        locality_line=None,
        price_display=None,
        price_crore=None,
        builtup_area_sqft=None,
        avg_price_per_sqft=None,
        bhk=None,
        bathrooms=None,
        balconies=None,
        furnishing=None,
        facing=None,
        possession_status=None,
        special_highlights=None,
        amenities=None,
        about_this_property=None,
        vaastu_mentioned=0,
        vaastu_mentions_text=None,
        last_updated=None,
        raw_text_path="",
        raw_html_path="",
    )).keys())


def collect_city(
    city: str,
    base_url: str,
    outdir: Path,
    args: argparse.Namespace,
    guard: RobotsGuard,
    fieldnames: List[str],
) -> dict:
    raw_search = ensure_dir(outdir / "raw" / "search")
    raw_detail = ensure_dir(outdir / "raw" / "detail")
    meta_dir = ensure_dir(outdir / "meta")
    ensure_dir(outdir)

    parsed_csv = outdir / "parsed_listings.csv"
    parsed_jsonl = outdir / "parsed_listings.jsonl"
    search_manifest = outdir / "search_pages.jsonl"

    existing_ids = read_existing_property_ids(parsed_csv) if args.resume else set()

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=not args.headful)
        except Exception as exc:
            raise SystemExit(
                "Chromium did not launch. Install it with: playwright install chromium"
            ) from exc

        context = browser.new_context(user_agent=USER_AGENT, viewport={"width": 1440, "height": 2200})
        page = context.new_page()
        page.set_default_timeout(args.timeout_ms)

        detail_links: List[Tuple[str, int]] = []
        inventory_totals: List[Optional[int]] = []

        for pageno in range(1, args.max_pages + 1):
            url = f"{base_url}?page={pageno}"
            guard.assert_allowed(url)
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)
            html = page.content()
            text = page.locator("body").inner_text()
            hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")

            html_path = raw_search / f"{slugify(city)}_p{pageno}.html"
            text_path = raw_search / f"{slugify(city)}_p{pageno}.txt"
            save_text(html_path, html)
            save_text(text_path, text)

            total = inventory_total_from_text(text)
            inventory_totals.append(total)
            links = extract_detail_links(hrefs, base_url)
            detail_links.extend((u, pageno) for u in links)

            append_jsonl(
                search_manifest,
                [
                    {
                        "city": city,
                        "page": pageno,
                        "search_url": url,
                        "inventory_total": total,
                        "n_detail_links_found": len(links),
                        "raw_html_path": str(html_path.relative_to(outdir)),
                        "raw_text_path": str(text_path.relative_to(outdir)),
                    }
                ],
            )
            jitter_sleep(args.min_sleep, args.max_sleep)

        seen: set[str] = set()
        unique_links: List[Tuple[str, int]] = []
        for url, pageno in detail_links:
            if url in seen:
                continue
            seen.add(url)
            unique_links.append((url, pageno))

        if args.max_detail_pages > 0:
            unique_links = unique_links[: args.max_detail_pages]

        parsed_rows: List[dict] = []
        for idx, (url, search_page) in enumerate(unique_links, start=1):
            pid = property_id_from_url(url)
            if args.resume and pid in existing_ids:
                continue
            guard.assert_allowed(url)
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
            html = page.content()
            text = page.locator("body").inner_text()

            html_path = raw_detail / f"{slugify(city)}_{pid}.html"
            text_path = raw_detail / f"{slugify(city)}_{pid}.txt"
            save_text(html_path, html)
            save_text(text_path, text)

            row = asdict(
                parse_detail_text(
                    text=text,
                    url=url,
                    city=city,
                    search_page=search_page,
                    raw_text_path=str(text_path.relative_to(outdir)),
                    raw_html_path=str(html_path.relative_to(outdir)),
                )
            )
            parsed_rows.append(row)
            append_csv(parsed_csv, [row], fieldnames=fieldnames)
            append_jsonl(parsed_jsonl, [row])

            if idx % 25 == 0:
                print(f"[{city}] Parsed {idx} detail pages...", file=sys.stderr)
            jitter_sleep(args.min_sleep, args.max_sleep)

        browser.close()

    summary = {
        "city": city,
        "base_url": base_url,
        "n_search_pages_fetched": args.max_pages,
        "search_inventory_totals": inventory_totals,
        "n_detail_urls_deduplicated": len(unique_links),
        "n_rows_written_this_run": len(parsed_rows),
        "parsed_csv": str(parsed_csv.relative_to(outdir)),
        "parsed_jsonl": str(parsed_jsonl.relative_to(outdir)),
    }
    save_json(meta_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    random.seed(42)

    root = project_root_from_here()
    config = load_city_config(Path(args.config))
    guard = RobotsGuard(DOMAIN)
    fieldnames = get_fieldnames()

    if args.all_cities:
        unique_urls = {}
        for city, url in config.items():
            if url not in unique_urls.values():
                unique_urls[city] = url
        cities_to_collect = list(unique_urls.keys())
        print(f"Collecting {len(cities_to_collect)} cities: {cities_to_collect}", file=sys.stderr)

        all_summaries = []
        for city in cities_to_collect:
            base_url = config[city]
            outdir = root / "data" / "raw" / "housingcom" / slugify(city)
            print(f"\n=== Collecting {city} ===", file=sys.stderr)
            summary = collect_city(city, base_url, outdir, args, guard, fieldnames)
            all_summaries.append(summary)
            print(json.dumps(summary, indent=2))

        total_rows = sum(s["n_rows_written_this_run"] for s in all_summaries)
        print(f"\n=== Collection complete ===", file=sys.stderr)
        print(f"Cities: {len(all_summaries)}, Total rows: {total_rows}", file=sys.stderr)
    else:
        if args.city not in config:
            raise SystemExit(f"City '{args.city}' not found in {args.config}. Known keys: {sorted(config)}")

        base_url = config[args.city]
        outdir = Path(args.output_dir) if args.output_dir else root / "data" / "raw" / "housingcom" / slugify(args.city)
        summary = collect_city(args.city, base_url, outdir, args, guard, fieldnames)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
