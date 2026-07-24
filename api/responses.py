from datetime import datetime


def success_response(prediction: str):
    return {
        "success": True,
        "message": "Prediction completed successfully.",
        "data": {
            "prediction": prediction
        },
        "timestamp": datetime.utcnow().isoformat()
    }