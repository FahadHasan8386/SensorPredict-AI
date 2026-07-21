import joblib
import pandas as pd

# Load Model
model = joblib.load("model/model.pkl")

labels = {
    0: "Good",
    1: "Moderate",
    2: "Poor"
}

print("=" * 50)
print("Air Quality Prediction")
print("=" * 50)

temperature = float(input("Temperature: "))
humidity = float(input("Humidity: "))
mq7 = int(input("MQ7: "))
mq136 = int(input("MQ136: "))

input_data = pd.DataFrame(
    [[temperature, humidity, mq7, mq136]],
    columns=["Temperature", "Humidity", "MQ7", "MQ136"]
)

prediction = model.predict(input_data)

print("\nPrediction Result")
print("=" * 50)
print("Predicted Air Quality:", labels[prediction[0]])