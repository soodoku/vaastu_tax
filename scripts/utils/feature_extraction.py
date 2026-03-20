"""Shared feature extraction utilities for vaastu detection and location parsing."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import pandas as pd

RE_VAASTU = re.compile(r"\bvaa?stu\b", re.IGNORECASE)
RE_SECTOR = re.compile(r"sector\s*(\d+[a-z]?)", re.IGNORECASE)

FALSE_POSITIVE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bvastushodh\b", re.IGNORECASE),
]


def _has_true_vaastu_match(text: str) -> bool:
    """
    Check if text has a true vaastu match (not just a false positive like Vastushodh).

    The logic is:
    1. Check if RE_VAASTU matches (standalone vaastu/vastu)
    2. If no match, check if any false positive pattern matches
    3. Return True only if we have a real vaastu match
    """
    if RE_VAASTU.search(text):
        return True
    return False


def extract_vaastu_mentions(
    text: str, max_sentences: int = 3
) -> Tuple[bool, Optional[str]]:
    """
    Extract vaastu mentions from text at sentence-level.

    Returns:
        (vaastu_mentioned: bool, vaastu_mentions_text: Optional[str])

    Examples:
        >>> extract_vaastu_mentions("Vastu compliant flat available")
        (True, 'Vastu compliant flat available')
        >>> extract_vaastu_mentions("By Vastushodh Developers")
        (False, None)
        >>> extract_vaastu_mentions("No mention here")
        (False, None)
    """
    if not text:
        return False, None

    sentences = re.split(r"[.!?\n|]", text)

    matching_sentences: List[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not _has_true_vaastu_match(sentence):
            continue
        matching_sentences.append(sentence)

    if not matching_sentences:
        return False, None

    unique_sentences = list(dict.fromkeys(matching_sentences))
    return True, " || ".join(unique_sentences[:max_sentences])


def extract_sector_from_text(text: Optional[str]) -> Optional[str]:
    """Extract sector number from address or location text."""
    if not text or pd.isna(text):
        return None
    text = str(text)
    match = RE_SECTOR.search(text)
    if match:
        return f"sector_{match.group(1).lower()}"
    return None


CITY_PATTERNS = {
    "gurgaon": ["gurgaon", "gurugram"],
    "mumbai": ["mumbai"],
    "hyderabad": ["hyderabad"],
    "kolkata": ["kolkata", "calcutta"],
    "delhi": ["delhi", "new delhi"],
    "bangalore": ["bangalore", "bengaluru"],
    "chennai": ["chennai"],
    "pune": ["pune"],
    "noida": ["noida"],
    "greater noida": ["greater noida"],
    "faridabad": ["faridabad"],
    "ghaziabad": ["ghaziabad"],
    "thane": ["thane"],
    "navi mumbai": ["navi mumbai"],
}


def extract_city_from_address(address: Optional[str]) -> Optional[str]:
    """Extract city name from address string."""
    if not address or pd.isna(address):
        return None
    address = str(address).lower()
    for city, patterns in CITY_PATTERNS.items():
        for pattern in patterns:
            if pattern in address:
                return city
    return None
