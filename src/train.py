from sklearn.model_selection import train_test_split
import joblib

from preprocessing import load_and_preprocess_data
from model_builder import build_model
from evaluator import evaluate_model


# Load Dataset
X, y, encoder = load_and_preprocess_data("data/sensor_data.csv")

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build Model
model = build_model()

# Train
model.fit(X_train, y_train)

# Evaluate
predictions, accuracy = evaluate_model(model, X_test, y_test)

print("=" * 50)
print("Model Accuracy:", f"{accuracy*100:.2f}%")

print("=" * 50)
print("Predictions")
print(predictions)

print("=" * 50)
print("Actual")
print(y_test.values)

# Save Model
joblib.dump(model, "model/model.pkl")

print("=" * 50)
print("Model Saved Successfully!")