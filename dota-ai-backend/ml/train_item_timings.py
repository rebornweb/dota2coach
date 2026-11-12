"""Optional outcome estimator: collect, train offline, or predict a timing association."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from item_data import API, cached_fetch
from scenario_data import LIMITATION, integer, validate_rows

HERE = Path(__file__).resolve().parent
FEATURES = ["hero_id", "item", "time"]


def prepare(rows, min_games):
    validate_rows(rows, "itemTimings")
    records = []
    seen = set()
    for row in rows:
        games, wins = integer(row["games"]), integer(row["wins"])
        key = (row["hero_id"], row["item"], integer(row["time"]))
        if key in seen:
            raise ValueError("Duplicate hero/item/time rows; use one snapshot")
        seen.add(key)
        if games and games >= min_games:
            records.append({"hero_id": str(row["hero_id"]), "item": row["item"],
                            "time": integer(row["time"]), "games": games, "rate": wins / games})
    return pd.DataFrame(records)


def train(rows, output, min_games=100):
    frame = prepare(rows, min_games)
    if frame.empty:
        raise ValueError("No rows meet the minimum sample size")
    groups = frame.hero_id + ":" + frame.item
    if groups.nunique() < 5:
        raise ValueError("At least five hero/item pairs are required")
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(frame, groups=groups))
    model = make_pipeline(ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), ["hero_id", "item"]),
        ("seconds", "passthrough", ["time"])]),
        RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1))
    training, test = frame.iloc[train_idx], frame.iloc[test_idx]
    model.fit(training[FEATURES], training.rate, randomforestregressor__sample_weight=training.games)
    baseline = float(np.average(training.rate, weights=training.games))
    metrics = {
        "weighted_mae_percentage_points": 100 * mean_absolute_error(test.rate, model.predict(test[FEATURES]), sample_weight=test.games),
        "baseline_weighted_mae_percentage_points": 100 * mean_absolute_error(test.rate, np.full(len(test), baseline), sample_weight=test.games),
        "training_pairs": int(groups.iloc[train_idx].nunique()), "test_pairs": int(groups.iloc[test_idx].nunique()),
        "rows": len(frame), "min_games": min_games,
    }
    model.fit(frame[FEATURES], frame.rate, randomforestregressor__sample_weight=frame.games)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics,
                 "heroes": sorted(frame.hero_id.unique()), "items": sorted(frame.item.unique()),
                 "min_time": int(frame.time.min()), "max_time": int(frame.time.max()),
                 "limitation": LIMITATION}, output)
    print(json.dumps(metrics, indent=2))
    print("Saved:", output)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Offline JSON array of raw itemTimings rows")
    parser.add_argument("--output", type=Path, default=HERE / "item_timings.joblib")
    parser.add_argument("--min-games", type=int, default=100)
    parser.add_argument("--predict", type=int, metavar="HERO_ID")
    parser.add_argument("--item", help="Internal item name, e.g. spirit_vessel")
    parser.add_argument("--time", type=int, help="Purchase cutoff in seconds, e.g. 900 for 15 minutes")
    args = parser.parse_args()
    if args.min_games < 1:
        parser.error("--min-games must be positive")
    if args.predict is not None:
        if not args.item or args.time is None:
            parser.error("--predict requires --item and --time")
        bundle = joblib.load(args.output)
        if str(args.predict) not in bundle["heroes"] or args.item not in bundle["items"]:
            parser.error("Hero or item not represented in training data")
        if not bundle["min_time"] <= args.time <= bundle["max_time"]:
            parser.error("Time is outside the training range")
        frame = pd.DataFrame([{"hero_id": str(args.predict), "item": args.item, "time": args.time}])
        rate = float(np.clip(bundle["model"].predict(frame[FEATURES])[0], 0, 1))
        print(f"Experimental associated win-rate estimate: {100 * rate:.2f}%")
        print(bundle["limitation"])
        return
    if args.dataset:
        rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    else:
        rows, stamp, stale = cached_fetch("scenario_itemTimings_all", f"{API}/scenarios/itemTimings",
                                         lambda value: validate_rows(value, "itemTimings"), ttl=3600)
        if stale:
            raise RuntimeError("Fresh training data unavailable; use an explicit offline dataset")
        path = HERE / "item_timings_training.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        print("Fetched:", stamp, "Dataset:", path)
    train(rows, args.output, args.min_games)
    print(LIMITATION)
    print("Timing thresholds overlap; sample weights reflect counts, not independent matches. Evaluation is not future-patch validation.")


if __name__ == "__main__":
    main()
