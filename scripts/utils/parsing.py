"""Shared parsing utilities for property listing extraction."""

from __future__ import annotations

import csv
import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def read_html_gz(path: Path) -> str:
    """Read gzip-compressed HTML file."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return fh.read()


def write_html_gz(path: Path, html: str) -> None:
    """Write HTML to gzip-compressed file."""
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(html)


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Append rows to JSONL file."""
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# Common regex patterns
RE_PRICE_UNIT = re.compile(
    r"₹?\s*([0-9][\d,]*(?:\.\d+)?)\s*(Cr|Crore|L|Lac|Lakh|Lakhs|K)?\b", re.I
)
RE_BHK = re.compile(r"(\d+(?:\.\d+)?)\s*BHK\b", re.I)
RE_SQFT = re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft|square feet)\b", re.I)
RE_CARPET = re.compile(r"carpet\s*(?:area)?[:\s]*([\d,]+(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft)?", re.I)
RE_BUILTUP = re.compile(
    r"(?:built[- ]?up|super built[- ]?up)\s*(?:area)?[:\s]*"
    r"([\d,]+(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft)?",
    re.I
)
RE_PPSF = re.compile(r"₹\s*([\d,]+(?:\.\d+)?)\s*/\s*(?:sq\.?\s*ft|sqft)", re.I)
RE_BATH = re.compile(r"(\d+(?:\.\d+)?)\s*(?:Bath|Bathroom)s?\b", re.I)
RE_BALCONY = re.compile(r"(\d+(?:\.\d+)?)\s*Balcon(?:y|ies)\b", re.I)
RE_FLOOR = re.compile(r"(?:Floor|Storey)\s*:?\s*(\d+)\s*(?:of|out of|/)?\s*(\d+)?", re.I)
RE_LAST_UPDATED = re.compile(r"(?:Last updated|Updated):?\s*(.+?)(?:\||$)", re.I)
RE_POSTED_DATE = re.compile(r"Posted\s*(?:on)?:?\s*(.+?)(?:\||$)", re.I)
RE_POSSESSION = re.compile(
    r"\b(Ready to move|Ready to Move|Under Construction|Resale|New Launch|"
    r"Immediately|Possession Started|New Property)\b",
    re.I,
)
RE_FACING = re.compile(
    r"\b(North(?:[- ]?East|[- ]?West)?|South(?:[- ]?East|[- ]?West)?|East|West)\s*(?:Facing)?\b",
    re.I,
)
RE_FURNISHING = re.compile(
    r"\b(Unfurnished|Semi[- ]?Furnished|Fully[- ]?Furnished|Furnished)\b", re.I
)
RE_PROPERTY_AGE = re.compile(
    r"(?:property age|age of property|age)[:\s]*(\d+(?:\s*-\s*\d+)?\s*(?:year|yr)s?)", re.I
)
RE_SELLER_TYPE = re.compile(r"\b(Owner|Dealer|Builder|Agent)\b", re.I)
RE_AMENITIES_SECTION = re.compile(r"(?:Amenities|Features)\s*:?\s*", re.I)
RE_SHOWING = re.compile(r"Showing\s+\d+\s*-\s*\d+\s+of\s+([\d,]+)", re.I)
RE_RESULTS_COUNT = re.compile(r"(\d[\d,]*)\s+results?\b", re.I)


def normalize_ws(text: Optional[str]) -> str:
    """Normalize whitespace in text."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "na"


def extract_lines(text: str) -> List[str]:
    """Extract non-empty lines from text."""
    lines = [normalize_ws(x) for x in text.splitlines()]
    return [x for x in lines if x]


def price_to_crore(raw: Optional[str]) -> Optional[float]:
    """Convert price string to crore value."""
    if not raw:
        return None
    raw = normalize_ws(raw)
    if "price on request" in raw.lower() or "por" in raw.lower() or "call for price" in raw.lower():
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
    """Extract first number from regex match."""
    m = pattern.search(text)
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def normalize_direction(text: Optional[str]) -> Optional[str]:
    """Normalize compass direction text."""
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


def find_price_display(lines: Sequence[str], title_idx: Optional[int] = None) -> Optional[str]:
    """Find price display line in listing text."""
    start = title_idx + 1 if title_idx is not None else 0
    for line in lines[start : start + 10]:
        if "₹" in line or "price on request" in line.lower() or "call for price" in line.lower():
            return line
    for line in lines[:30]:
        if "₹" in line:
            return line
    return None


def find_first_matching_line(lines: Sequence[str], pattern: re.Pattern[str]) -> Optional[str]:
    """Find first line matching a regex pattern."""
    for line in lines:
        if pattern.search(line):
            return line
    return None


def extract_section(
    lines: Sequence[str],
    heading: str,
    stop_heads: Optional[set[str]] = None,
    max_lines: int = 20,
) -> Optional[str]:
    """Extract text section following a heading."""
    if stop_heads is None:
        stop_heads = {
            "property location", "amenities", "about this property", "overview",
            "project details", "nearby places", "specifications", "furnishing",
            "configuration", "ratings", "locality", "calculator", "price trends",
            "home loan", "similar properties", "emi calculator", "faq", "read more",
            "about the builder", "about builder", "more about", "contact",
            "safety tips", "report this listing",
        }

    heading_l = heading.lower().strip()
    start = None
    for idx, line in enumerate(lines):
        line_lower = line.lower().strip()
        if line_lower == heading_l or line_lower.startswith(heading_l + " "):
            start = idx + 1
            break
    if start is None:
        return None

    buffer: List[str] = []
    for line in lines[start:]:
        line_l = line.lower().strip()
        if line_l in stop_heads:
            break
        buffer.append(line)
        if len(buffer) >= max_lines:
            break
    out = normalize_ws(" | ".join(buffer))
    return out or None


def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    """Write rows to CSV file."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write rows to JSONL file."""
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(manifest_path: Path) -> List[dict]:
    """Load scrape manifest JSONL file."""
    if not manifest_path.exists():
        return []
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def read_existing_property_ids(parsed_csv: Path) -> set[str]:
    """Read property IDs already parsed to avoid re-parsing."""
    if not parsed_csv.exists():
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(parsed_csv, dtype={"property_id": str})
        return set(df["property_id"].dropna().astype(str))
    except Exception:
        return set()


def extract_next_data(html: str) -> Optional[Dict]:
    """Extract __NEXT_DATA__ JSON from Next.js pages."""
    pattern = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
    m = pattern.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def normalize_price_to_crore(price: Optional[float]) -> Optional[float]:
    """Normalize price value to crore.

    Handles both already-normalized values (< 100) and raw rupee values (> 100).
    """
    if price is None or pd.isna(price):
        return None
    try:
        price = float(price)
    except (ValueError, TypeError):
        return None
    if price <= 0:
        return None
    if price > 100:
        return price / 10000000.0
    return price


def write_parquet(path: Path, rows: List[dict]) -> None:
    """Write rows to Parquet file with zstd compression."""
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False, compression="zstd")


def read_parquet(path: Path) -> pd.DataFrame:
    """Read Parquet file into DataFrame."""
    return pd.read_parquet(path)


def read_existing_property_ids_parquet(parsed_parquet: Path) -> set[str]:
    """Read property IDs already parsed from parquet file."""
    if not parsed_parquet.exists():
        return set()
    try:
        df = pd.read_parquet(parsed_parquet, columns=["property_id"])
        return set(df["property_id"].dropna().astype(str))
    except Exception:
        return set()
