"""Tests for feature_extraction module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils.feature_extraction import (
    extract_vaastu_mentions,
    extract_sector_from_text,
    extract_city_from_address,
)


class TestExtractVaastuMentions:
    def test_basic_vaastu_match(self):
        assert extract_vaastu_mentions("Vastu compliant flat")[0] is True
        assert extract_vaastu_mentions("vaastu perfect home")[0] is True
        assert extract_vaastu_mentions("This is a vastu friendly layout")[0] is True

    def test_false_positive_vastushodh(self):
        assert extract_vaastu_mentions("By Vastushodh Developers")[0] is False
        assert extract_vaastu_mentions("Vastushodh Urban Gram")[0] is False
        assert extract_vaastu_mentions("Project by Vastushodh Realty")[0] is False

    def test_mixed_content(self):
        text = "This property by Vastushodh is vastu compliant"
        result = extract_vaastu_mentions(text)
        assert result[0] is True
        assert "vastu compliant" in result[1].lower()

    def test_no_match(self):
        assert extract_vaastu_mentions("Beautiful apartment for sale")[0] is False
        assert extract_vaastu_mentions("")[0] is False

    def test_sentence_extraction(self):
        text = "Nice flat. Vastu compliant property. Good location. Well maintained."
        result = extract_vaastu_mentions(text)
        assert result[0] is True
        assert "Vastu compliant property" in result[1]
        assert "Good location" not in result[1]

    def test_multiple_sentences(self):
        text = "Vastu facing. East facing vastu. Perfect vaastu alignment."
        result = extract_vaastu_mentions(text)
        assert result[0] is True
        sentences = result[1].split(" || ")
        assert len(sentences) <= 3

    def test_max_sentences_limit(self):
        text = "Vastu 1. Vastu 2. Vastu 3. Vastu 4. Vastu 5."
        result = extract_vaastu_mentions(text, max_sentences=2)
        assert result[0] is True
        sentences = result[1].split(" || ")
        assert len(sentences) == 2


class TestExtractSectorFromText:
    def test_basic_sector(self):
        assert extract_sector_from_text("Sector 45, Gurgaon") == "sector_45"
        assert extract_sector_from_text("sector 12a Noida") == "sector_12a"
        assert extract_sector_from_text("SECTOR 100") == "sector_100"

    def test_no_sector(self):
        assert extract_sector_from_text("Bandra West, Mumbai") is None
        assert extract_sector_from_text("") is None
        assert extract_sector_from_text(None) is None


class TestExtractCityFromAddress:
    def test_basic_cities(self):
        assert extract_city_from_address("Sector 45, Gurgaon") == "gurgaon"
        assert extract_city_from_address("Bandra, Mumbai") == "mumbai"
        assert extract_city_from_address("Whitefield, Bangalore") == "bangalore"
        assert extract_city_from_address("Whitefield, Bengaluru") == "bangalore"

    def test_no_city(self):
        assert extract_city_from_address("Some random address") is None
        assert extract_city_from_address("") is None
        assert extract_city_from_address(None) is None
