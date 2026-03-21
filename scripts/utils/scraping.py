"""Shared utilities for browser-based scraping."""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

try:
    from playwright.sync_api import sync_playwright
except Exception as exc:
    raise SystemExit(
        "Playwright is required. Install dependencies and run 'playwright install chromium'."
    ) from exc

from .parsing import slugify

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DOMAIN = "https://www.magicbricks.com"
MAX_RETRIES = 3
BACKOFF_BASE = 2.0

logger = logging.getLogger(__name__)


def setup_logging(
    verbose: bool = False, quiet: bool = False, log_file: Optional[str] = None
) -> None:
    """Configure logging for scraping scripts."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handlers: List[logging.Handler] = []

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handlers.append(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)


def parse_proxy_url(proxy_url: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse proxy URL into Playwright proxy config."""
    if not proxy_url:
        return None
    parsed = urlsplit(proxy_url)
    scheme = parsed.scheme or "http"
    config: Dict[str, Any] = {"server": f"{scheme}://{parsed.hostname}:{parsed.port}"}
    if parsed.username:
        config["username"] = parsed.username
    if parsed.password:
        config["password"] = parsed.password
    return config


def get_proxy(proxy_arg: Optional[str] = None) -> Optional[str]:
    """Get proxy URL from argument or environment."""
    return proxy_arg or os.environ.get("BRIGHT_DATA_PROXY_URL")


class RobotsGuard:
    """Check robots.txt compliance for URLs."""

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
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=False,
                    args=["--disable-blink-features=AutomationControlled"],
                )
                proxy_config = parse_proxy_url(self.proxy)
                context = browser.new_context(
                    user_agent=self.user_agent,
                    locale="en-IN",
                    timezone_id="Asia/Kolkata",
                    proxy=proxy_config,
                    ignore_https_errors=bool(proxy_config),
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
        """Check if URL is allowed by robots.txt."""
        if not self.robots_available:
            return True
        return self.rp.can_fetch(self.user_agent, url)

    def assert_allowed(self, url: str) -> None:
        """Raise PermissionError if URL is disallowed."""
        if not self.is_allowed(url):
            raise PermissionError(f"robots.txt disallows fetch: {url}")


def jitter_sleep(lo: float, hi: float) -> None:
    """Sleep for a random duration between lo and hi seconds."""
    if hi <= 0:
        return
    time.sleep(random.uniform(max(0.0, lo), max(lo, hi)))


def is_blocked_response(text: str) -> bool:
    """Detect if response indicates blocking."""
    blocked_patterns = [
        "access denied",
        "too many requests",
        "rate limit",
        "please try again later",
        "captcha",
        "verify you are human",
        "403 forbidden",
        "429 too many",
    ]
    text_lower = text.lower()
    if any(pat in text_lower for pat in blocked_patterns):
        return True
    if len(text.strip()) < 500 and ("error" in text_lower or "denied" in text_lower):
        return True
    return False


def fetch_with_retry(
    page,
    url: str,
    wait_ms: int = 3000,
    max_retries: int = MAX_RETRIES,
) -> Tuple[str, str, bool]:
    """Fetch URL with retry logic on failure or blocking."""
    for attempt in range(max_retries):
        try:
            response = page.goto(url, wait_until="domcontentloaded")

            if response and response.status >= 400:
                backoff = BACKOFF_BASE ** (attempt + 1) + random.uniform(0, 2)
                logger.warning(
                    "HTTP %d on attempt %d/%d, backing off %.1fs",
                    response.status,
                    attempt + 1,
                    max_retries,
                    backoff,
                )
                time.sleep(backoff)
                continue

            page.wait_for_timeout(wait_ms)
            html = page.content()
            text = page.locator("body").inner_text()

            if is_blocked_response(text):
                backoff = BACKOFF_BASE ** (attempt + 1) + random.uniform(0, 2)
                logger.warning(
                    "Blocked on attempt %d/%d, backing off %.1fs",
                    attempt + 1,
                    max_retries,
                    backoff,
                )
                time.sleep(backoff)
                continue

            return html, text, True
        except Exception as e:
            backoff = BACKOFF_BASE ** (attempt + 1) + random.uniform(0, 2)
            logger.warning(
                "Attempt %d/%d failed: %s, backing off %.1fs",
                attempt + 1,
                max_retries,
                e,
                backoff,
            )
            time.sleep(backoff)

    return "", "", False


def load_city_config(path: Path) -> Dict[str, str]:
    """Load city configuration JSON."""
    import json

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a JSON object of city -> url.")
    return data


def city_outdir(city: str, base_dir: Optional[Path] = None) -> Path:
    """Get output directory for a city."""
    from .parsing import project_root

    if base_dir:
        return base_dir / slugify(city)
    return project_root() / "data" / "raw" / "magicbricks" / slugify(city)


def create_browser_context(
    playwright,
    headless: bool = False,
    proxy: Optional[str] = None,
    timeout_ms: int = 60000,
):
    """Create Playwright browser context with anti-detection settings."""
    try:
        browser = playwright.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
    except Exception as exc:
        raise SystemExit(
            "Chromium did not launch. Install it with: playwright install chromium"
        ) from exc

    proxy_config = parse_proxy_url(proxy)
    context = browser.new_context(
        user_agent=USER_AGENT,
        viewport={"width": 1440, "height": 2200},
        locale="en-IN",
        timezone_id="Asia/Kolkata",
        proxy=proxy_config,
        ignore_https_errors=bool(proxy_config),
    )
    page = context.new_page()
    page.set_default_timeout(timeout_ms)

    return browser, context, page
