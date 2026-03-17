#!/usr/bin/env python3
"""Collect 99acres.com listings for independent houses and flats.

The script is designed for a cautious, reproducible workflow:

1. Check robots.txt before each fetch.
2. Save raw HTML and raw body text for every page.
3. Extract embedded JSON data (__NEXT_DATA__) when available.
4. Parse a conservative set of fields from listing-detail text.
5. Emit a normalized CSV/JSONL that can feed the analysis pipeline.

Notes
-----
- 99acres uses dynamic rendering, so Playwright is required.
- The default city URLs live in data/config/99acres_cities.json and can be edited without
  touching the code.
- Supports both houses (independent-house-villa) and flats.

Example
-------
python scripts/01_collect_99acres.py \
    --city gurgaon \
    --property-type both \
    --max-pages 5 \
    --max-detail-pages 200

python scripts/01_collect_99acres.py --all-cities --property-type house --max-pages 3
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

try:
    from playwright.sync_api import sync_playwright
except Exception as exc:
    raise SystemExit(
        "Playwright is required. Install dependencies and run 'playwright install chromium'."
    ) from exc


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DOMAIN = "https://www.99acres.com"


@dataclass
class ListingRecord:
    property_id: str
    url: str
    city: str
    property_type: str
    search_page: int
    title: Optional[str]
    locality: Optional[str]
    price_display: Optional[str]
    price_crore: Optional[float]
    builtup_area_sqft: Optional[float]
    carpet_area_sqft: Optional[float]
    bhk: Optional[float]
    bathrooms: Optional[float]
    balconies: Optional[float]
    furnishing: Optional[str]
    facing: Optional[str]
    possession_status: Optional[str]
    floor_number: Optional[int]
    total_floors: Optional[int]
    property_age: Optional[str]
    amenities: Optional[str]
    description: Optional[str]
    vaastu_mentioned: int
    vaastu_mentions_text: Optional[str]
    seller_type: Optional[str]
    posted_date: Optional[str]
    last_updated: Optional[str]
    raw_text_path: str
    raw_html_path: str


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root_from_here()
    parser = argparse.ArgumentParser(description=__doc__)
    city_group = parser.add_mutually_exclusive_group(required=True)
    city_group.add_argument("--city", help="City key from data/config/99acres_cities.json")
    city_group.add_argument(
        "--all-cities",
        action="store_true",
        help="Collect data for all cities in the config file",
    )
    parser.add_argument(
        "--config",
        default=str(root / "data" / "config" / "99acres_cities.json"),
        help="Path to city-URL config JSON",
    )
    parser.add_argument(
        "--property-type",
        choices=["house", "flat", "both"],
        default="both",
        help="Property type to collect",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for raw pages and parsed outputs. Defaults to data/raw/99acres/<city>",
    )
    parser.add_argument("--max-pages", type=int, default=5, help="Number of search-result pages to visit")
    parser.add_argument(
        "--max-detail-pages",
        type=int,
        default=250,
        help="Hard cap on listing detail pages fetched after deduplication",
    )
    parser.add_argument("--min-sleep", type=float, default=2.0, help="Minimum sleep between requests")
    parser.add_argument("--max-sleep", type=float, default=4.0, help="Maximum sleep between requests")
    parser.add_argument("--timeout-ms", type=int, default=60000, help="Playwright page timeout in ms")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium in headless mode (99acres blocks headless, so headful is default)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip detail pages whose parsed rows already exist in parsed_listings.csv",
    )
    parser.add_argument(
        "--skip-kaggle-ids",
        action="store_true",
        help="Skip property IDs already present in Kaggle datasets (arvanshul, campusx)",
    )
    parser.add_argument(
        "--proxy",
        help="Proxy server URL (e.g., http://user:pass@host:port)",
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


def load_city_config(path: Path) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a JSON object of city -> {{house, flat}} URLs.")
    return data


class RobotsGuard:
    def __init__(
        self, root_url: str, user_agent: str = USER_AGENT, proxy: Optional[str] = None
    ) -> None:
        self.root_url = root_url.rstrip("/")
        self.user_agent = user_agent
        self.proxy = proxy
        self.rp = RobotFileParser()
        self.rp.set_url(f"{self.root_url}/robots.txt")
        self.robots_available = False
        self._fetch_robots_with_playwright()

    def _fetch_robots_with_playwright(self) -> None:
        """Fetch robots.txt using Playwright since some sites block urllib."""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=False,
                    args=["--disable-blink-features=AutomationControlled"]
                )
                proxy_config = {"server": self.proxy} if self.proxy else None
                context = browser.new_context(
                    user_agent=self.user_agent,
                    locale="en-IN",
                    timezone_id="Asia/Kolkata",
                    proxy=proxy_config,
                )
                page = context.new_page()
                page.goto(f"{self.root_url}/robots.txt", wait_until="domcontentloaded")
                page.wait_for_timeout(2000)
                text = page.locator("body").inner_text()
                browser.close()

                if "User-agent" in text or "Disallow" in text or "Allow" in text:
                    self.rp.parse(text.splitlines())
                    self.robots_available = True
        except Exception:
            pass

    def is_allowed(self, url: str) -> bool:
        if not self.robots_available:
            return True
        return self.rp.can_fetch(self.user_agent, url)

    def assert_allowed(self, url: str) -> None:
        allowed = self.is_allowed(url)
        if not allowed:
            raise PermissionError(f"robots.txt disallows fetch: {url}")


RE_PRICE_UNIT = re.compile(r"₹?\s*([0-9][\d,]*(?:\.\d+)?)\s*(Cr|Crore|L|Lac|Lakh|Lakhs|K)?\b", re.I)
RE_BHK = re.compile(r"(\d+(?:\.\d+)?)\s*BHK\b", re.I)
RE_SQFT = re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft|square feet)\b", re.I)
RE_BATH = re.compile(r"(\d+(?:\.\d+)?)\s*(?:Bath|Bathroom)s?\b", re.I)
RE_BALCONY = re.compile(r"(\d+(?:\.\d+)?)\s*Balcon(?:y|ies)\b", re.I)
RE_FLOOR = re.compile(r"(?:Floor|Storey)\s*:?\s*(\d+)\s*(?:of|out of|/)?\s*(\d+)?", re.I)
RE_LAST_UPDATED = re.compile(r"(?:Posted|Updated|Last updated):?\s*(.+?)(?:\||$)", re.I)
RE_POSTED_DATE = re.compile(r"Posted\s*(?:on)?\s*:?\s*(.+?)(?:\||$)", re.I)
RE_POSSESSION = re.compile(
    r"\b(Ready to move|Ready to Move|Under Construction|Resale|New Launch|Immediately|Possession Started|New Property)\b",
    re.I,
)
RE_FACING = re.compile(
    r"\b(North(?:[- ]?East|[- ]?West)?|South(?:[- ]?East|[- ]?West)?|East|West)\s*(?:Facing)?\b",
    re.I,
)
RE_FURNISHING = re.compile(r"\b(Unfurnished|Semi[- ]?Furnished|Fully[- ]?Furnished|Furnished)\b", re.I)
RE_VAASTU = re.compile(r"\bvaa?stu\b|\bvastu\b", re.I)
RE_PROPERTY_ID = re.compile(r"spid-([A-Z]?\d{7,})")
RE_SELLER_TYPE = re.compile(r"\b(Owner|Agent|Builder|Dealer)\b", re.I)
RE_PROPERTY_AGE = re.compile(r"(?:Age|Property Age|Years Old)\s*:?\s*([^|,\n]+)", re.I)
RE_AMENITIES_SECTION = re.compile(r"(?:Amenities|Features)\s*:?\s*", re.I)


def jitter_sleep(lo: float, hi: float) -> None:
    if hi <= 0:
        return
    time.sleep(random.uniform(max(0.0, lo), max(lo, hi)))


MAX_RETRIES = 3
BACKOFF_BASE = 2.0


def is_blocked_response(html: str, text: str) -> bool:
    """Detect 429/403 or blocking patterns in page content."""
    blocked_patterns = [
        "access denied",
        "too many requests",
        "rate limit",
        "please try again later",
        "captcha",
        "verify you are human",
        "blocked",
        "403 forbidden",
        "429 too many",
    ]
    combined = (html + text).lower()
    return any(pat in combined for pat in blocked_patterns)


def fetch_with_retry(
    page,
    url: str,
    wait_ms: int = 3000,
    max_retries: int = MAX_RETRIES,
) -> Tuple[str, str, bool]:
    """Fetch a page with retry logic and backoff on failures/blocks.

    Returns (html, text, success).
    """
    for attempt in range(max_retries):
        try:
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(wait_ms)
            html = page.content()
            text = page.locator("body").inner_text()

            if is_blocked_response(html, text):
                backoff = BACKOFF_BASE ** (attempt + 1) + random.uniform(0, 2)
                print(f"    [BLOCKED] Attempt {attempt + 1}/{max_retries}, backing off {backoff:.1f}s", file=sys.stderr)
                time.sleep(backoff)
                continue

            return html, text, True
        except Exception as e:
            backoff = BACKOFF_BASE ** (attempt + 1) + random.uniform(0, 2)
            print(f"    [RETRY] Attempt {attempt + 1}/{max_retries} failed: {e}, backing off {backoff:.1f}s", file=sys.stderr)
            time.sleep(backoff)

    return "", "", False


def price_to_crore(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    raw = normalize_ws(raw)
    if "price on request" in raw.lower() or "por" in raw.lower():
        return None
    m = RE_PRICE_UNIT.search(raw)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit in {"cr", "crore"}:
        return value
    if unit in {"l", "lac", "lakh", "lakhs"}:
        return value / 100.0
    if unit == "k":
        return value / 100000.0
    if value > 10000000:
        return value / 10000000.0
    return None


def number_from_match(pattern: re.Pattern[str], text: str) -> Optional[float]:
    m = pattern.search(text)
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def extract_lines(text: str) -> List[str]:
    lines = [normalize_ws(x) for x in text.splitlines()]
    return [x for x in lines if x]


def normalize_direction(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = text.replace(" ", "-").replace("--", "-")
    mapping = {
        "North-East": "North-East",
        "NorthEast": "North-East",
        "North-West": "North-West",
        "NorthWest": "North-West",
        "South-East": "South-East",
        "SouthEast": "South-East",
        "South-West": "South-West",
        "SouthWest": "South-West",
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


def extract_next_data(html: str) -> Optional[Dict[str, Any]]:
    """Extract __NEXT_DATA__ JSON from page if present."""
    pattern = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
    m = pattern.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def extract_detail_links(hrefs: Iterable[str], root_url: str) -> List[str]:
    out: List[str] = []
    for href in hrefs:
        if not href:
            continue
        url = urljoin(root_url, href)
        parsed = urlparse(url)
        if parsed.netloc and "99acres.com" not in parsed.netloc:
            continue
        path = parsed.path.lower()
        if "spid-" not in path:
            continue
        if any(bad in path for bad in ["/projects/", "/builder/", "/price-trends", "/search/"]):
            continue
        if url not in out:
            out.append(url)
    return out


def parse_detail_from_next_data(
    next_data: Dict[str, Any],
    url: str,
    city: str,
    property_type: str,
    search_page: int,
    raw_text_path: str,
    raw_html_path: str,
) -> Optional[ListingRecord]:
    """Parse listing from __NEXT_DATA__ if available."""
    try:
        props = next_data.get("props", {})
        page_props = props.get("pageProps", {})
        listing_data = page_props.get("listingData", {}) or page_props.get("propertyData", {}) or {}
        if not listing_data:
            return None

        property_id = str(listing_data.get("id", "")) or property_id_from_url(url)
        title = listing_data.get("title") or listing_data.get("heading")
        locality = listing_data.get("locality") or listing_data.get("localityName")
        price_display = listing_data.get("price") or listing_data.get("priceDisplay")
        price_crore = None
        if isinstance(price_display, (int, float)):
            price_crore = price_display / 10000000.0
        elif isinstance(price_display, str):
            price_crore = price_to_crore(price_display)

        builtup = listing_data.get("builtUpArea") or listing_data.get("superBuiltupArea")
        carpet = listing_data.get("carpetArea")
        bhk = listing_data.get("bhk") or listing_data.get("bedroom")
        bathrooms = listing_data.get("bathroom")
        balconies = listing_data.get("balcony")
        furnishing = listing_data.get("furnishing")
        facing = listing_data.get("facing")
        floor_num = listing_data.get("floor") or listing_data.get("floorNumber")
        total_floors = listing_data.get("totalFloor")
        possession = listing_data.get("possession") or listing_data.get("possessionStatus")
        age = listing_data.get("propertyAge") or listing_data.get("age")
        amenities = listing_data.get("amenities")
        if isinstance(amenities, list):
            amenities = ", ".join(str(a) for a in amenities)
        description = listing_data.get("description") or listing_data.get("about")
        seller = listing_data.get("sellerType") or listing_data.get("postedBy")
        posted = listing_data.get("postedDate") or listing_data.get("createdAt")
        updated = listing_data.get("updatedAt") or listing_data.get("lastUpdated")

        all_text = json.dumps(listing_data, ensure_ascii=False).lower()
        vaastu_mentioned = 1 if RE_VAASTU.search(all_text) else 0
        vaastu_text = None
        if vaastu_mentioned:
            chunks = []
            if description and RE_VAASTU.search(description):
                chunks.append(description)
            if title and RE_VAASTU.search(title):
                chunks.append(title)
            vaastu_text = " || ".join(chunks) if chunks else "vaastu mentioned in listing data"

        return ListingRecord(
            property_id=property_id,
            url=url,
            city=city,
            property_type=property_type,
            search_page=search_page,
            title=title,
            locality=locality,
            price_display=str(price_display) if price_display else None,
            price_crore=price_crore,
            builtup_area_sqft=float(builtup) if builtup else None,
            carpet_area_sqft=float(carpet) if carpet else None,
            bhk=float(bhk) if bhk else None,
            bathrooms=float(bathrooms) if bathrooms else None,
            balconies=float(balconies) if balconies else None,
            furnishing=furnishing,
            facing=normalize_direction(facing) if facing else None,
            possession_status=possession,
            floor_number=int(floor_num) if floor_num else None,
            total_floors=int(total_floors) if total_floors else None,
            property_age=str(age) if age else None,
            amenities=amenities,
            description=description,
            vaastu_mentioned=vaastu_mentioned,
            vaastu_mentions_text=vaastu_text,
            seller_type=seller,
            posted_date=str(posted) if posted else None,
            last_updated=str(updated) if updated else None,
            raw_text_path=raw_text_path,
            raw_html_path=raw_html_path,
        )
    except Exception:
        return None


def parse_detail_from_text(
    text: str,
    url: str,
    city: str,
    property_type: str,
    search_page: int,
    raw_text_path: str,
    raw_html_path: str,
) -> ListingRecord:
    """Fallback parser using text extraction."""
    lines = extract_lines(text)
    all_text = "\n".join(lines)

    property_id = property_id_from_url(url)

    title = None
    for line in lines:
        line_lower = line.lower()
        if RE_BHK.search(line) and ("house" in line_lower or "flat" in line_lower or "villa" in line_lower or "apartment" in line_lower):
            title = line
            break

    locality = None
    for line in lines:
        line_lower = line.lower()
        if "address" in line_lower:
            continue
        if ("sector" in line_lower or ", " in line) and not "₹" in line:
            if not RE_PRICE_UNIT.search(line) and len(line) < 100:
                locality = line
                break

    price_display = None
    price_crore = None
    for line in lines:
        if "₹" in line and ("cr" in line.lower() or "lac" in line.lower() or "lakh" in line.lower()):
            price_display = line
            price_crore = price_to_crore(line)
            break
        if "price on request" in line.lower():
            price_display = "Price on Request"
            break

    bhk = number_from_match(RE_BHK, all_text)
    bathrooms = number_from_match(RE_BATH, all_text)
    balconies = number_from_match(RE_BALCONY, all_text)

    builtup = None
    carpet = None
    sqft_matches = RE_SQFT.findall(all_text)
    if sqft_matches:
        builtup = float(sqft_matches[0].replace(",", ""))
        if len(sqft_matches) > 1:
            carpet = float(sqft_matches[1].replace(",", ""))

    facing = None
    m = RE_FACING.search(all_text)
    if m:
        facing = normalize_direction(m.group(1))

    furnishing = None
    m = RE_FURNISHING.search(all_text)
    if m:
        furnishing = normalize_ws(m.group(1))

    possession = None
    m = RE_POSSESSION.search(all_text)
    if m:
        possession = normalize_ws(m.group(1))

    floor_num = None
    total_floors = None
    m = RE_FLOOR.search(all_text)
    if m:
        floor_num = int(m.group(1))
        if m.group(2):
            total_floors = int(m.group(2))

    age = None
    m = RE_PROPERTY_AGE.search(all_text)
    if m:
        age = normalize_ws(m.group(1))

    seller = None
    m = RE_SELLER_TYPE.search(all_text)
    if m:
        seller = normalize_ws(m.group(1))

    posted = None
    m = RE_POSTED_DATE.search(all_text)
    if m:
        posted = normalize_ws(m.group(1))

    updated = None
    m = RE_LAST_UPDATED.search(all_text)
    if m:
        updated = normalize_ws(m.group(1))

    description = None
    for i, line in enumerate(lines):
        if "about" in line.lower() and "property" in line.lower():
            desc_lines = []
            for j in range(i + 1, min(i + 20, len(lines))):
                if any(stop in lines[j].lower() for stop in ["amenities", "features", "specifications", "overview"]):
                    break
                desc_lines.append(lines[j])
            if desc_lines:
                description = " ".join(desc_lines)
            break

    amenities = None
    for i, line in enumerate(lines):
        if RE_AMENITIES_SECTION.search(line):
            amen_lines = []
            for j in range(i + 1, min(i + 30, len(lines))):
                if any(stop in lines[j].lower() for stop in ["about", "description", "overview", "contact"]):
                    break
                amen_lines.append(lines[j])
            if amen_lines:
                amenities = " | ".join(amen_lines)
            break

    vaastu_mentioned = 1 if RE_VAASTU.search(all_text) else 0
    vaastu_text = None
    if vaastu_mentioned:
        chunks = []
        for line in lines:
            if RE_VAASTU.search(line):
                chunks.append(line)
        vaastu_text = " || ".join(dict.fromkeys(chunks)) if chunks else None

    return ListingRecord(
        property_id=property_id,
        url=url,
        city=city,
        property_type=property_type,
        search_page=search_page,
        title=title,
        locality=locality,
        price_display=price_display,
        price_crore=price_crore,
        builtup_area_sqft=builtup,
        carpet_area_sqft=carpet,
        bhk=bhk,
        bathrooms=bathrooms,
        balconies=balconies,
        furnishing=furnishing,
        facing=facing,
        possession_status=possession,
        floor_number=floor_num,
        total_floors=total_floors,
        property_age=age,
        amenities=amenities,
        description=description,
        vaastu_mentioned=vaastu_mentioned,
        vaastu_mentions_text=vaastu_text,
        seller_type=seller,
        posted_date=posted,
        last_updated=updated,
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


def load_kaggle_property_ids(root: Path) -> set[str]:
    """Load property IDs from Kaggle datasets to avoid re-scraping."""
    import pandas as pd

    ids: set[str] = set()
    kaggle_dir = root / "data" / "raw" / "99acres_kaggle"
    if not kaggle_dir.exists():
        return ids

    arvanshul_dir = kaggle_dir / "arvanshul"
    if arvanshul_dir.exists():
        for csv_file in arvanshul_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, usecols=["PROP_ID"], dtype={"PROP_ID": str})
                ids.update(df["PROP_ID"].dropna().astype(str))
            except Exception:
                pass

    campusx_dir = root / "data" / "raw" / "99acres_campusx"
    if campusx_dir.exists():
        for csv_file in campusx_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                if "property_id" in df.columns:
                    ids.update(df["property_id"].dropna().astype(str))
            except Exception:
                pass

    return ids


def get_fieldnames() -> List[str]:
    return list(asdict(ListingRecord(
        property_id="",
        url="",
        city="",
        property_type="",
        search_page=0,
        title=None,
        locality=None,
        price_display=None,
        price_crore=None,
        builtup_area_sqft=None,
        carpet_area_sqft=None,
        bhk=None,
        bathrooms=None,
        balconies=None,
        furnishing=None,
        facing=None,
        possession_status=None,
        floor_number=None,
        total_floors=None,
        property_age=None,
        amenities=None,
        description=None,
        vaastu_mentioned=0,
        vaastu_mentions_text=None,
        seller_type=None,
        posted_date=None,
        last_updated=None,
        raw_text_path="",
        raw_html_path="",
    )).keys())


def collect_property_type(
    city: str,
    base_url: str,
    property_type: str,
    outdir: Path,
    args: argparse.Namespace,
    guard: RobotsGuard,
    fieldnames: List[str],
    page,
    existing_ids: set[str],
) -> Tuple[List[dict], int]:
    """Collect listings for a single property type."""
    raw_search = ensure_dir(outdir / "raw" / "search")
    raw_detail = ensure_dir(outdir / "raw" / "detail")

    parsed_csv = outdir / "parsed_listings.csv"
    parsed_jsonl = outdir / "parsed_listings.jsonl"
    search_manifest = outdir / "search_pages.jsonl"

    detail_links: List[Tuple[str, int]] = []

    for pageno in range(1, args.max_pages + 1):
        url = f"{base_url}&page={pageno}" if "?" in base_url else f"{base_url}?page={pageno}"
        if not guard.is_allowed(url):
            print(f"  [SKIP] robots.txt disallows: {url}", file=sys.stderr)
            continue

        print(f"  [{property_type}] Fetching search page {pageno}...", file=sys.stderr)
        html, text, success = fetch_with_retry(page, url, wait_ms=3000)
        if not success:
            print(f"  [ERROR] Failed to load search page {pageno} after retries", file=sys.stderr)
            continue
        hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")

        html_path = raw_search / f"{slugify(city)}_{property_type}_p{pageno}.html"
        text_path = raw_search / f"{slugify(city)}_{property_type}_p{pageno}.txt"
        save_text(html_path, html)
        save_text(text_path, text)

        links = extract_detail_links(hrefs, base_url)
        detail_links.extend((u, pageno) for u in links)

        append_jsonl(
            search_manifest,
            [
                {
                    "city": city,
                    "property_type": property_type,
                    "page": pageno,
                    "search_url": url,
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

        if not guard.is_allowed(url):
            print(f"  [SKIP] robots.txt disallows: {url}", file=sys.stderr)
            continue

        print(f"  [{property_type}] Fetching detail {idx}/{len(unique_links)}: {pid}", file=sys.stderr)
        html, text, success = fetch_with_retry(page, url, wait_ms=2000)
        if not success:
            print(f"  [ERROR] Failed to load detail page {pid} after retries", file=sys.stderr)
            continue

        html_path = raw_detail / f"{slugify(city)}_{property_type}_{pid}.html"
        text_path = raw_detail / f"{slugify(city)}_{property_type}_{pid}.txt"
        save_text(html_path, html)
        save_text(text_path, text)

        next_data = extract_next_data(html)
        record = None
        if next_data:
            record = parse_detail_from_next_data(
                next_data=next_data,
                url=url,
                city=city,
                property_type=property_type,
                search_page=search_page,
                raw_text_path=str(text_path.relative_to(outdir)),
                raw_html_path=str(html_path.relative_to(outdir)),
            )

        if not record:
            record = parse_detail_from_text(
                text=text,
                url=url,
                city=city,
                property_type=property_type,
                search_page=search_page,
                raw_text_path=str(text_path.relative_to(outdir)),
                raw_html_path=str(html_path.relative_to(outdir)),
            )

        row = asdict(record)
        parsed_rows.append(row)
        append_csv(parsed_csv, [row], fieldnames=fieldnames)
        append_jsonl(parsed_jsonl, [row])

        if idx % 10 == 0:
            print(f"  [{property_type}] Parsed {idx} detail pages...", file=sys.stderr)

        jitter_sleep(args.min_sleep, args.max_sleep)

    return parsed_rows, len(unique_links)


def collect_city(
    city: str,
    city_urls: Dict[str, str],
    outdir: Path,
    args: argparse.Namespace,
    guard: RobotsGuard,
    fieldnames: List[str],
    kaggle_ids: set[str],
) -> dict:
    meta_dir = ensure_dir(outdir / "meta")
    ensure_dir(outdir)

    parsed_csv = outdir / "parsed_listings.csv"
    existing_ids = read_existing_property_ids(parsed_csv) if args.resume else set()

    if args.skip_kaggle_ids:
        existing_ids.update(kaggle_ids)
        print(f"  Skipping {len(kaggle_ids)} Kaggle property IDs", file=sys.stderr)

    property_types = []
    if args.property_type in ("house", "both"):
        property_types.append("house")
    if args.property_type in ("flat", "both"):
        property_types.append("flat")

    all_rows: List[dict] = []
    totals_by_type: Dict[str, int] = {}

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                headless=args.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ]
            )
        except Exception as exc:
            raise SystemExit(
                "Chromium did not launch. Install it with: playwright install chromium"
            ) from exc

        proxy_config = {"server": args.proxy} if args.proxy else None
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 900},
            locale="en-IN",
            timezone_id="Asia/Kolkata",
            proxy=proxy_config,
        )
        page = context.new_page()
        page.set_default_timeout(args.timeout_ms)

        for prop_type in property_types:
            if prop_type not in city_urls:
                print(f"  [WARN] No URL for {prop_type} in {city}, skipping", file=sys.stderr)
                continue

            base_url = city_urls[prop_type]
            print(f"  Collecting {prop_type} listings from {city}...", file=sys.stderr)

            rows, n_detail = collect_property_type(
                city=city,
                base_url=base_url,
                property_type=prop_type,
                outdir=outdir,
                args=args,
                guard=guard,
                fieldnames=fieldnames,
                page=page,
                existing_ids=existing_ids,
            )
            all_rows.extend(rows)
            totals_by_type[prop_type] = n_detail
            existing_ids.update(r["property_id"] for r in rows)

        browser.close()

    summary = {
        "city": city,
        "property_types": property_types,
        "n_search_pages_fetched": args.max_pages,
        "detail_urls_by_type": totals_by_type,
        "n_rows_written_this_run": len(all_rows),
        "parsed_csv": str((outdir / "parsed_listings.csv").relative_to(outdir)),
        "parsed_jsonl": str((outdir / "parsed_listings.jsonl").relative_to(outdir)),
    }
    save_json(meta_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    random.seed(42)

    root = project_root_from_here()
    config = load_city_config(Path(args.config))
    guard = RobotsGuard(DOMAIN, proxy=args.proxy)
    fieldnames = get_fieldnames()

    kaggle_ids: set[str] = set()
    if args.skip_kaggle_ids:
        print("Loading Kaggle property IDs for deduplication...", file=sys.stderr)
        kaggle_ids = load_kaggle_property_ids(root)
        print(f"  Found {len(kaggle_ids)} existing property IDs", file=sys.stderr)

    if args.all_cities:
        cities_to_collect = list(config.keys())
        unique_cities = []
        seen_urls = set()
        for city in cities_to_collect:
            url_tuple = tuple(sorted(config[city].items()))
            if url_tuple not in seen_urls:
                seen_urls.add(url_tuple)
                unique_cities.append(city)

        print(f"Collecting {len(unique_cities)} cities: {unique_cities}", file=sys.stderr)

        all_summaries = []
        for city in unique_cities:
            city_urls = config[city]
            outdir = root / "data" / "raw" / "99acres" / slugify(city)
            print(f"\n=== Collecting {city} ===", file=sys.stderr)
            summary = collect_city(city, city_urls, outdir, args, guard, fieldnames, kaggle_ids)
            all_summaries.append(summary)
            print(json.dumps(summary, indent=2))

        total_rows = sum(s["n_rows_written_this_run"] for s in all_summaries)
        print(f"\n=== Collection complete ===", file=sys.stderr)
        print(f"Cities: {len(all_summaries)}, Total rows: {total_rows}", file=sys.stderr)
    else:
        if args.city not in config:
            raise SystemExit(f"City '{args.city}' not found in {args.config}. Known keys: {sorted(config)}")

        city_urls = config[args.city]
        outdir = Path(args.output_dir) if args.output_dir else root / "data" / "raw" / "99acres" / slugify(args.city)
        summary = collect_city(args.city, city_urls, outdir, args, guard, fieldnames, kaggle_ids)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
