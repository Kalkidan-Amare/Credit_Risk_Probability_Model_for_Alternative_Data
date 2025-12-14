"""Model training with experiment tracking via MLflow."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.data_processing import build_preprocessing_pipeline

TARGET_COL = "is_high_risk"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}")
    return pd.read_csv(path)


def split_features_labels(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col} missing")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def infer_column_types(df: pd.DataFrame) -> Tuple[list, list]:
    categorical = [c for c in df.columns if df[c].dtype == "object"]
    numeric = [c for c in df.columns if c not in categorical]
    return categorical, numeric


def build_models() -> Dict[str, Tuple[object, dict]]:
    models = {
        "log_reg": (
            LogisticRegression(max_iter=500),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 8],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
            },
        ),
    }
    return models


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    return metrics


def train_and_log(df: pd.DataFrame, experiment_name: str = "credit-risk"):
    mlflow.set_experiment(experiment_name)
    X, y = split_features_labels(df)
    categorical, numeric = infer_column_types(X)
    preprocess = build_preprocessing_pipeline(categorical, numeric)
    models = build_models()

    best_auc = -np.inf
    best_run_uri = ""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    for name, (estimator, param_grid) in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", estimator),
            ]
        )
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
        )
        with mlflow.start_run(run_name=name) as run:
            search.fit(X_train, y_train)
            metrics = evaluate_model(search.best_estimator_, X_test, y_test)
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                search.best_estimator_,
                artifact_path="model",
            )
            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_run_uri = run.info.artifact_uri + "/model"

    if best_run_uri:
        model_name = "credit-risk-model"
        mlflow.register_model(best_run_uri, model_name)
        return best_run_uri
    raise RuntimeError("No model runs completed")


def main():
    parser = argparse.ArgumentParser(description="Train credit risk models")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/features.csv",
        help="Path to processed feature table",
    )
    args = parser.parse_args()
    df = load_dataset(Path(args.data))
    best_uri = train_and_log(df)
    print(f"Best model logged at {best_uri}")


if __name__ == "__main__":
    main()
