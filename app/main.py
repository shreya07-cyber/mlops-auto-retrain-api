import pickle
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "model.pkl"
METRICS_PATH = "metrics.json"

app = FastAPI(
    title="House Price Prediction API",
    description="Automated MLOps pipeline — model retrains on new data via CI/CD.",
    version="1.0.0",
)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run model/train.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Request / Response schemas ──────────────────────────────────────────────

class HouseFeatures(BaseModel):
    area: int = Field(..., gt=0, description="Area of the house in sq ft")
    bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    age: int = Field(..., ge=0, description="Age of the house in years")

    model_config = {
        "json_schema_extra": {
            "examples": [{"area": 1500, "bedrooms": 3, "age": 10}]
        }
    }


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "INR"
    model_version: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "running", "message": "House Price Prediction API is live 🚀"}


@app.get("/health", tags=["Health"])
def health():
    model_exists = os.path.exists(MODEL_PATH)
    return {
        "status": "healthy" if model_exists else "model_missing",
        "model_loaded": model_exists,
    }


@app.get("/metrics", tags=["Model Info"])
def get_metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: HouseFeatures):
    try:
        model = load_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    input_data = [[features.area, features.bedrooms, features.age]]
    price = model.predict(input_data)[0]

    return PredictionResponse(
        predicted_price=round(price, 2),
        model_version="latest",
    )
