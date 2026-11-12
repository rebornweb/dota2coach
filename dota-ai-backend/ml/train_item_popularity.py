"""Train an experimental purchase-share estimator; run --help for usage."""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dota_data import get_heroes
from item_data import STAGES, get_item_popularity

HERE = Path(__file__).resolve().parent
LIMITATION = "Experimental estimate of aggregate purchase share, not win probability, item strength, or Turbo-specific advice. Held-out heroes test transfer, not future patches."


def training_rows(snapshots):
    rows = []
    for snapshot in snapshots:
        hero_id = snapshot["hero_id"]
        for stage in STAGES:
            entries = snapshot["stages"][stage]
            total = sum(entry["count"] for entry in entries)
            if not total:
                continue
            # Include zero purchases for items seen in another stage for this hero.
            counts = {entry["item_id"]: entry["count"] for entry in entries}
            candidates = {entry["item_id"] for values in snapshot["stages"].values() for entry in values}
            for item_id in sorted(candidates):
                rows.append({"hero_id": str(hero_id), "stage": stage, "item_id": str(item_id),
                             "share": counts.get(item_id, 0) / total * 100})
    return pd.DataFrame(rows)


def train(snapshots, output):
    frame = training_rows(snapshots)
    if frame.empty or frame.hero_id.nunique() < 5:
        raise ValueError("Collect at least five heroes with nonempty purchase data.")
    features = ["hero_id", "stage", "item_id"]
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
                               .split(frame, groups=frame.hero_id))
    model = make_pipeline(
        ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore"), features)]),
        RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
    model.fit(frame.iloc[train_idx][features], frame.iloc[train_idx]["share"])
    test = frame.iloc[test_idx]
    predicted = model.predict(test[features])
    # Compare against mean purchase share for each item/stage from training heroes.
    means = frame.iloc[train_idx].groupby(["stage", "item_id"])["share"].mean()
    baseline = [means.get((row.stage, row.item_id), 0) for row in test.itertuples()]
    metrics = {"held_out_hero_mae_percentage_points": float(mean_absolute_error(test.share, predicted)),
               "baseline_mae_percentage_points": float(mean_absolute_error(test.share, baseline)),
               "training_heroes": int(frame.iloc[train_idx].hero_id.nunique()),
               "test_heroes": int(test.hero_id.nunique()), "rows": len(frame)}
    model.fit(frame[features], frame.share)
    joblib.dump({"model": model, "features": features, "metrics": metrics,
                 "heroes": sorted(frame.hero_id.unique()), "limitation": LIMITATION}, output)
    print(json.dumps(metrics, indent=2))
    print(LIMITATION)
    print("Saved:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Train offline from a previously saved snapshot JSON")
    parser.add_argument("--hero-ids", nargs="+", type=int, help="Collect only these hero IDs (default: all)")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between hero requests (minimum 1)")
    parser.add_argument("--output", type=Path, default=HERE / "item_popularity.joblib")
    parser.add_argument("--predict", type=int, metavar="HERO_ID", help="Use saved model to rank items for a hero")
    parser.add_argument("--stage", choices=STAGES, default="mid_game_items")
    args = parser.parse_args()
    if args.predict is not None:
        bundle = joblib.load(args.output)
        if str(args.predict) not in bundle["heroes"]:
            parser.error("Hero was not in the training dataset; collect it and retrain first.")
        snapshot = get_item_popularity(args.predict)
        items = {entry["item_id"]: entry["name"] for values in snapshot["stages"].values() for entry in values}
        if not items:
            parser.error("No candidate items available for this hero")
        frame = pd.DataFrame([{"hero_id": str(args.predict), "stage": args.stage, "item_id": str(item)} for item in items])
        scores = np.clip(bundle["model"].predict(frame[bundle["features"]]), 0, 100)
        for (item, name), score in sorted(zip(items.items(), scores), key=lambda pair: -pair[1])[:10]:
            print(f"{name} ({item}): {score:.2f}% estimated purchase share")
        print("Estimates are independent and may not sum to 100%.")
        print(bundle["limitation"])
        return
    if args.dataset:
        snapshots = json.loads(args.dataset.read_text(encoding="utf-8"))
    else:
        heroes = get_heroes()
        ids = sorted({hero["id"] for hero in heroes})
        if args.hero_ids:
            if not set(args.hero_ids).issubset(ids):
                parser.error("Unknown hero ID")
            ids = sorted(set(args.hero_ids))
        snapshots = []
        for index, hero_id in enumerate(ids):
            if index:
                time.sleep(max(1, args.delay))
            snapshot = get_item_popularity(hero_id)
            if snapshot["stale"]:
                raise RuntimeError("Collection returned stale data; retry later or train an explicit offline dataset.")
            snapshots.append(snapshot)
            print(f"Collected {hero_id} ({index + 1}/{len(ids)})", flush=True)
        dataset = HERE / "item_popularity_training.json"
        dataset.write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
        print("Dataset:", dataset)
    train(snapshots, args.output)


if __name__ == "__main__":
    main()
