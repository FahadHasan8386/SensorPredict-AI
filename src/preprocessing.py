import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    encoder = LabelEncoder()
    df["AirQuality"] = encoder.fit_transform(df["AirQuality"])

    X = df[["Temperature", "Humidity", "MQ7", "MQ136"]]
    y = df["AirQuality"]

    return X, y, encoder