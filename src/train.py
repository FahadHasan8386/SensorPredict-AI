import pandas as pd

# Load dataset
df = pd.read_csv("data/sensor_data.csv")

print("=" * 50)
print("Dataset Shape")
print(df.shape)

print("=" * 50)
print("Column Names")
print(df.columns)

print("=" * 50)
print("First 5 Rows")
print(df.head())

print("=" * 50)
print("Missing Values")
print(df.isnull().sum())

print("=" * 50)
print("Air Quality Count")
print(df["AirQuality"].value_counts())

print("=" * 50)
print("Statistics")
print(df.describe())