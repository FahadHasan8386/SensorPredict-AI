from fastapi import FastAPI
from api.schemas import PredictionRequest

import pandas as pd
import joblib

app = FastAPI(
    title="SensorPredict AI API"
)

# Load Model
model = joblib.load("model/model.pkl")

labels = {
    0: "Good",
    1: "Moderate",
    2: "Poor"
}

@app.get("/")
def home():
    return {
        "message": "SensorPredict AI API Running"
    }

@app.post("/predict")
def predict(data: PredictionRequest):

    input_data = pd.DataFrame(
        [[
            data.temperature,
            data.humidity,
            data.mq7,
            data.mq136
        ]],
        columns=[
            "Temperature",
            "Humidity",
            "MQ7",
            "MQ136"
        ]
    )

    prediction = model.predict(input_data)

    return {
        "prediction": labels[prediction[0]]
    }