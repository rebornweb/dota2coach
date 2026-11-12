"""OpenDota purchase counts, cached per hero with explicit freshness metadata."""
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import monotonic

import httpx

ROOT = Path(__file__).resolve().parent / "item_cache"
API = "https://api.opendota.com/api"
STAGES = ("start_game_items", "early_game_items", "mid_game_items", "late_game_items")
TTL = 300
_lock = Lock()
_retry = {}
logger = logging.getLogger(__name__)


def validate_popularity(data):
    if not isinstance(data, dict):
        raise ValueError("Invalid popularity response")
    for stage in STAGES:
        counts = data.get(stage)
        if not isinstance(counts, dict) or any(
            not str(key).isdigit() or type(value) is not int or value < 0
            for key, value in counts.items()
        ):
            raise ValueError("Invalid item purchase counts")
    return {stage: data[stage] for stage in STAGES}


def validate_items(data):
    if not isinstance(data, dict) or not data:
        raise ValueError("Invalid item constants")
    if any(not isinstance(item, dict) or type(item.get("id")) is not int
           for item in data.values()):
        raise ValueError("Invalid item metadata")
    return data


def cached_fetch(key, url, validate, ttl=TTL):
    with _lock:
        path = ROOT / f"{key}.json"
        cached = None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            data = validate(payload["data"])
            stamp = datetime.fromisoformat(payload["fetched_at"])
            if stamp.tzinfo is None:
                raise ValueError("Missing timezone")
            cached = (data, stamp)
            age = (datetime.now(timezone.utc) - stamp).total_seconds()
            if 0 <= age < ttl:
                return data, stamp.isoformat(), False
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if monotonic() >= _retry.get(key, 0):
            try:
                response = httpx.get(url, timeout=httpx.Timeout(15, connect=5))
                response.raise_for_status()
                data = validate(response.json())
                stamp = datetime.now(timezone.utc).isoformat()
            except (httpx.HTTPError, ValueError) as exc:
                _retry[key] = monotonic() + 60
                if not cached:
                    raise RuntimeError("OpenDota item data unavailable; retry shortly.") from exc
            else:
                temporary = None
                try:
                    ROOT.mkdir(exist_ok=True)
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=ROOT,
                                                     delete=False, suffix=".tmp") as file:
                        temporary = Path(file.name)
                        json.dump({"data": data, "fetched_at": stamp}, file)
                    os.replace(temporary, path)
                except OSError:
                    logger.warning("Could not persist item cache", exc_info=True)
                finally:
                    if temporary:
                        temporary.unlink(missing_ok=True)
                return data, stamp, False
        if cached:
            return cached[0], cached[1].isoformat(), True
        raise RuntimeError("OpenDota item data unavailable; retry shortly.")


def get_item_popularity(hero_id):
    if type(hero_id) is not int or hero_id <= 0:
        raise ValueError("Invalid hero ID")
    url = f"{API}/heroes/{hero_id}/itemPopularity"
    counts, fetched_at, stale = cached_fetch(str(hero_id), url, validate_popularity)
    try:
        items, items_at, items_stale = cached_fetch(
            "items", f"{API}/constants/items", validate_items, 86400)
    except RuntimeError:
        items, items_at, items_stale = {}, None, True
    by_id = {str(item["id"]): item for item in items.values()}
    stages = {}
    for stage in STAGES:
        total = sum(counts[stage].values())
        stages[stage] = [
            {"item_id": int(item_id), "name": by_id.get(item_id, {}).get("dname") or f"Item #{item_id}",
             "count": count, "purchase_share_pct": round(count / total * 100, 2) if total else 0}
            for item_id, count in sorted(counts[stage].items(), key=lambda pair: (-pair[1], int(pair[0])))
        ]
    return {"hero_id": hero_id, "source": url, "fetched_at": fetched_at, "stale": stale,
            "items_fetched_at": items_at, "items_stale": items_stale, "stages": stages,
            "limitation": "Aggregate purchases, not win rates or item effects. Shares are of purchases within each stage, not matches. No game-mode filter; not Turbo-specific. Upstream updates may lag."}
