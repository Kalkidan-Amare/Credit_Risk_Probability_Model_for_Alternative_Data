"""Inference utility to score new customers."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

DEFAULT_MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/credit-risk-model/1",
)


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found at {path}")
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    if path.suffix.lower() in {".json"}:
        return pd.read_json(path)
    raise ValueError("Unsupported file format; use CSV or JSON")


def predict(model_uri: str, data: pd.DataFrame) -> pd.DataFrame:
    model = mlflow.pyfunc.load_model(model_uri)
    preds = model.predict(data)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(data)[:, 1]
        except Exception:
            proba = None
    result = data.copy()
    result["prediction"] = preds
    if proba is not None:
        result["probability"] = proba
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Score customers with trained model",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to CSV or JSON with feature columns",
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        default=DEFAULT_MODEL_URI,
        help="MLflow model URI",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = load_input(Path(args.input))
    scored = predict(args.model_uri, df)
    output_path = Path(args.output)
    scored.to_csv(output_path, index=False)
    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
