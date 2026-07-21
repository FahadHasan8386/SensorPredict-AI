import joblib

model = joblib.load("model/model.pkl")

temperature = float(input("Temperature: "))
humidity = float(input("Humidity: "))
mq7 = int(input("MQ7: "))
mq136 = int(input("MQ136: "))

prediction = model.predict([[temperature, humidity, mq7, mq136]])

labels = {
    0: "Good",
    1: "Moderate",
    2: "Poor"
}

print("=" * 50)
print("Predicted Air Quality:", labels[prediction[0]])