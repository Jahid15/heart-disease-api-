import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HealthResponse, HeartFeatures, InfoResponse, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "heart_model.joblib"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the presence of heart disease from clinical features.",
    version="1.0.0",
)

model = None
metadata = {}

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
if METADATA_PATH.exists():
    metadata = json.loads(METADATA_PATH.read_text())


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Heart Disease Prediction API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.get("/info", response_model=InfoResponse)
def info():
    if not metadata:
        raise HTTPException(status_code=503, detail="model metadata is not available")
    return InfoResponse(
        model_type=metadata["model_type"],
        pipeline=metadata["pipeline"],
        features=metadata["features"],
        classes=metadata["classes"],
        test_accuracy=metadata["test_accuracy"],
        dataset=metadata["dataset"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="model is not loaded")

    features = metadata.get("features", list(payload.model_dump().keys()))
    row = pd.DataFrame([payload.model_dump()])[features]

    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])

    return PredictionResponse(
        heart_disease=bool(prediction),
        probability=round(probability, 4),
        label=metadata["classes"][str(prediction)],
    )
