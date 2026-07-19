import pandas as pd

# Load dataset
df = pd.read_csv("data/sensor_data.csv")

print(df.head())
print(df.info())
print(df.describe())