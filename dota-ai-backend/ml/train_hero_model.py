
from pathlib import Path

import httpx
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

API_URL = "https://api.opendota.com/api/heroStats"

# 1. Download training data
response = httpx.get(API_URL, timeout=30)
response.raise_for_status()

heroes = pd.DataFrame(response.json())

# 2. Select the model's input features
features = [
    "base_health",
    "base_armor",
    "base_attack_min",
    "base_attack_max",
    "base_str",
    "base_agi",
    "base_int",
    "str_gain",
    "agi_gain",
    "int_gain",
    "attack_range",
    "move_speed"
]

# 3. Calculate our training label (target)
heroes = heroes[heroes["turbo_picks"] >= 100].copy()

heroes["turbo_win_rate"] = (
    heroes["turbo_wins"] / heroes["turbo_picks"] * 100
)

# Exclude incomplete records
heroes = heroes.dropna(
    subset=features + ["turbo_win_rate"]
)

X = heroes[features]
y = heroes["turbo_win_rate"]

# 4. Separate training and test heroes
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 5. Train ML model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# 6. Evaluate predictions on held-out heroes
predictions = model.predict(X_test)

print("Heroes used:", len(heroes))
print("Mean absolute error:",
      mean_absolute_error(y_test, predictions))

# 7. Save the model and feature order
output = Path(__file__).resolve().parent / "hero_model.joblib"

joblib.dump({
    "model": model,
    "features": features
}, output)

print("Saved:", output)
