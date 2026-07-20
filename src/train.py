import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/sensor_data.csv")

# Create Label Encoder
encoder = LabelEncoder()

# Convert text labels into numbers
df["AirQuality"] = encoder.fit_transform(df["AirQuality"])

print("Encoded Dataset")
print(df.head())

print("\nLabel Mapping")
for index, label in enumerate(encoder.classes_):
    print(f"{label} -> {index}")

# Features (Input)
X = df[["Temperature", "Humidity", "MQ7", "MQ136"]]

# Target (Output)
y = df["AirQuality"]

print("=" * 50)
print("Features (X)")
print(X.head())

print("=" * 50)
print("Target (y)")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("=" * 50)
print("Training Data:", len(X_train))

print("Testing Data:", len(X_test))

print("=" * 50)
print("Training Decision Tree Model..")

#Create Model
model = DecisionTreeClassifier(random_state=42)


# Train Model
model.fit(X_train, y_train)

print("Model Training Completed Successfuly!")

print("=" * 50)
print("Making Decision")

predictions = model.predict(X_test)

print("Predicted Values : ")
print(predictions)

print("Actual Values :")
print(y_test.values)


accuracy = accuracy_score(y_test, predictions)

print("=" * 50)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save Model
saved_path = joblib.dump(model, "model/model.pkl")

print(saved_path)

print("=" * 50)
print("Model Saved Successfully!")
print("Location: model/model.pkl")