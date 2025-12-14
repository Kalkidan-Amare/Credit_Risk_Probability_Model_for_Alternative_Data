"""FastAPI application to serve credit risk predictions."""
from __future__ import annotations

import os

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException

from .pydantic_models import PredictionRequest, PredictionResponse

MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/credit-risk-model/1")

app = FastAPI(title="Credit Risk Scoring API", version="0.1.0")


def load_model(model_uri: str = MODEL_URI):
    try:
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:  # pragma: no cover - depends on runtime env
        msg = f"Failed to load model from {model_uri}: {exc}"
        raise RuntimeError(msg)


model = load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if not request.features:
        raise HTTPException(status_code=400, detail="No features provided")
    payload = pd.DataFrame([request.features])
    try:
        pred = model.predict(payload)
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(payload)[:, 1][0]
            except Exception:
                proba = None
        risk_probability = float(proba if proba is not None else pred[0])
        label = (
            int(round(risk_probability))
            if proba is not None
            else int(pred[0])
        )
        return PredictionResponse(
            risk_probability=risk_probability,
            label=label,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
