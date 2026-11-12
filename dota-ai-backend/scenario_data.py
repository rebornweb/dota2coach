"""Validated aggregate outcomes from OpenDota scenarios."""
from item_data import API, cached_fetch

KINDS = ("itemTimings", "laneRoles", "misc")
LIMITATION = ("Observed associations, not causal item recommendations or match forecasts. "
              "No Turbo filter. Timing thresholds overlap; do not sum their games. "
              "Small samples are uncertain. Item timings use seconds before purchase cutoff; "
              "lane-role time is a maximum match duration. Lane roles are not support positions.")


def integer(value):
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    raise ValueError("Invalid scenario count")


def validate_rows(data, kind, hero_id=None):
    if not isinstance(data, list):
        raise ValueError("Invalid scenario response")
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("Invalid scenario row")
        games, wins = integer(row.get("games")), integer(row.get("wins"))
        if wins > games:
            raise ValueError("Wins exceed games")
        if kind != "misc":
            if type(row.get("hero_id")) is not int or row["hero_id"] <= 0:
                raise ValueError("Invalid hero ID")
            if hero_id is not None and row["hero_id"] != hero_id:
                raise ValueError("Scenario hero differs from selection")
            integer(row.get("time"))
        if kind == "itemTimings" and (not isinstance(row.get("item"), str) or not row["item"].strip()):
            raise ValueError("Invalid item name")
        if kind == "laneRoles" and (type(row.get("lane_role")) is not int or row["lane_role"] not in (1, 2, 3, 4)):
            raise ValueError("Invalid lane role")
        if kind == "misc" and (not isinstance(row.get("scenario"), str)
                               or type(row.get("is_radiant")) is not bool):
            raise ValueError("Invalid team scenario")
    return data


def get_scenario(kind, hero_id=None):
    if kind not in KINDS:
        raise ValueError("Unknown scenario")
    if kind != "misc" and (type(hero_id) is not int or hero_id <= 0):
        raise ValueError("Hero ID required")
    suffix = f"?hero_id={hero_id}" if kind != "misc" else ""
    url = f"{API}/scenarios/{kind}{suffix}"
    data, stamp, stale = cached_fetch(f"scenario_{kind}_{hero_id if kind != 'misc' else 'all'}",
        url, lambda rows: validate_rows(rows, kind, hero_id), ttl=3600)
    rows = []
    for raw in data:
        games, wins = integer(raw["games"]), integer(raw["wins"])
        row = {**raw, "games": games, "wins": wins,
               "win_rate_pct": round(100 * wins / games, 2) if games else None,
               "small_sample": games < 100}
        if "time" in row:
            row["time"] = integer(row["time"])
            row["time_minutes"] = row["time"] / 60
        rows.append(row)
    return {"source": url, "fetched_at": stamp, "stale": stale,
            "rows": sorted(rows, key=lambda row: -row["games"]), "limitation": LIMITATION}


def get_scenarios(hero_id, kinds=KINDS):
    result = {}
    for kind in kinds:
        try:
            result[kind] = get_scenario(kind, hero_id)
        except (RuntimeError, ValueError):
            result[kind] = {"error": f"OpenDota {kind} unavailable; retry later."}
    return result
