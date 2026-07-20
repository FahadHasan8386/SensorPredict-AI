import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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