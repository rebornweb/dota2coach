
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import monotonic

import httpx

CACHE = Path(__file__).resolve().parent / "hero_stats_cache.json"
API_URL = "https://api.opendota.com/api/heroStats"

CACHE_TTL_SECONDS = 3600
RETRY_DELAY_SECONDS = 60
REQUEST_TIMEOUT = httpx.Timeout(10, connect=5)
logger = logging.getLogger(__name__)
_lock = Lock()
_retry_after = 0.0
_memory_cache = None


def _validate_heroes(heroes):
    if (not isinstance(heroes, list) or not heroes
            or any(not isinstance(hero, dict)
                   or not isinstance(hero.get("localized_name"), str)
                   or not hero["localized_name"].strip() for hero in heroes)):
        raise ValueError("Invalid hero data")
    return heroes


def _read_cache():
    try:
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        heroes = _validate_heroes(data["heroes"])
        fetched_at = datetime.fromisoformat(data["fetched_at"])
        if fetched_at.tzinfo is None:
            raise ValueError("Cache timestamp must include a timezone")
        return heroes, fetched_at
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(heroes, fetched_at):
    temporary = None
    try:
        # Atomic replacement prevents partially written JSON being read.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE.parent,
                                         prefix="hero_stats_", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump({"fetched_at": fetched_at.isoformat(), "heroes": heroes}, file)
        os.replace(temporary, CACHE)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def get_heroes():
    """Use a one-hour cache; fall back to stale data during OpenDota outages."""
    global _retry_after, _memory_cache
    # Serialize refreshes within this worker to avoid duplicate API requests.
    with _lock:
        cached = _memory_cache or _read_cache()
        if cached:
            age = (datetime.now(timezone.utc) - cached[1]).total_seconds()
            if 0 <= age < CACHE_TTL_SECONDS:
                return cached[0]

        if monotonic() < _retry_after:
            if cached:
                return cached[0]
            raise RuntimeError("OpenDota unavailable and no valid local cache exists. Retry shortly.")

        try:
            response = httpx.get(API_URL, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            heroes = _validate_heroes(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            _retry_after = monotonic() + RETRY_DELAY_SECONDS
            if cached:
                logger.warning("OpenDota unavailable; using hero data from %s.", cached[1].isoformat())
                return cached[0]
            raise RuntimeError("OpenDota unavailable and no valid local cache exists.") from exc

        fetched_at = datetime.now(timezone.utc)
        _memory_cache = (heroes, fetched_at)
        _retry_after = 0.0
        try:
            _write_cache(heroes, fetched_at)
        except OSError:
            logger.warning("Could not save hero cache; keeping data in memory.", exc_info=True)
        return heroes
