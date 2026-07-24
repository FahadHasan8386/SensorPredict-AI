from pydantic import BaseModel

class PredictionRequest(BaseModel):
    temperature: float
    humidity: float
    mq7 : int
    mq136 : int