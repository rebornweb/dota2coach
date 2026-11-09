
from sklearn.tree import DecisionTreeClassifier
import joblib

# Features: [HP percentage, enemy burst threat]
X = [
    [20, 9],
    [30, 8],
    [40, 7],
    [80, 2],
    [90, 1],
    [75, 3],
    [25, 5],
    [85, 7],
]

# Labels: 0 = Defense, 1 = Damage
y = [0, 0, 0, 1, 1, 1, 0, 1]

model = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "dota_ml.joblib")

print("ML model trained and saved!")
