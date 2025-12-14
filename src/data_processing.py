"""Data processing utilities for credit risk modeling."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence, Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xverse.transformer import WOE
except ImportError:  # pragma: no cover - optional dependency
    WOE = None


@dataclass
class RFMConfig:
    customer_col: str = "CustomerId"
    amount_col: str = "Amount"
    datetime_col: str = "TransactionStartTime"
    snapshot_date: Optional[datetime] = None
    n_clusters: int = 3
    random_state: int = 42


def add_time_parts(
    df: pd.DataFrame,
    time_col: str = "TransactionStartTime",
) -> pd.DataFrame:
    """Extract hour, day, month, and year from the transaction timestamp."""
    if time_col not in df.columns:
        raise KeyError(f"Expected column '{time_col}' in dataframe")
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data["transaction_hour"] = data[time_col].dt.hour
    data["transaction_day"] = data[time_col].dt.day
    data["transaction_month"] = data[time_col].dt.month
    data["transaction_year"] = data[time_col].dt.year
    return data


def aggregate_customer_features(
    df: pd.DataFrame,
    customer_col: str = "CustomerId",
    amount_col: str = "Amount",
) -> pd.DataFrame:
    """Aggregate basic transaction statistics per customer."""
    if customer_col not in df.columns or amount_col not in df.columns:
        raise KeyError("Customer or amount column missing from dataframe")
    grouped = (
        df.groupby(customer_col)[amount_col]
        .agg(
            total_amount="sum",
            avg_amount="mean",
            transaction_count="count",
            std_amount="std",
        )
        .reset_index()
    )
    grouped["std_amount"].fillna(0.0, inplace=True)
    return grouped


def compute_rfm(
    df: pd.DataFrame,
    config: Optional[RFMConfig] = None,
) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary metrics per customer."""
    cfg = config or RFMConfig()
    if cfg.customer_col not in df.columns:
        raise KeyError(f"Missing customer column {cfg.customer_col}")
    if cfg.amount_col not in df.columns:
        raise KeyError(f"Missing amount column {cfg.amount_col}")
    if cfg.datetime_col not in df.columns:
        raise KeyError(f"Missing datetime column {cfg.datetime_col}")

    data = df.copy()
    data[cfg.datetime_col] = pd.to_datetime(data[cfg.datetime_col])
    snapshot = cfg.snapshot_date or (
        data[cfg.datetime_col].max() + timedelta(days=1)
    )
    recency = (
        data.groupby(cfg.customer_col)[cfg.datetime_col]
        .max()
        .apply(lambda dt: (snapshot - dt).days)
    )
    frequency = data.groupby(cfg.customer_col)[cfg.amount_col].count()
    monetary = data.groupby(cfg.customer_col)[cfg.amount_col].sum()
    rfm = pd.DataFrame(
        {
            cfg.customer_col: recency.index,
            "Recency": recency.values,
            "Frequency": frequency.values,
            "Monetary": monetary.values,
        }
    )
    return rfm


def label_high_risk_cluster(
    rfm: pd.DataFrame,
    config: Optional[RFMConfig] = None,
) -> pd.DataFrame:
    """Label customers in the lowest engagement cluster as high risk."""
    cfg = config or RFMConfig()
    required_cols = ["Recency", "Frequency", "Monetary"]
    for col in required_cols:
        if col not in rfm.columns:
            raise KeyError(f"Missing RFM column {col}")

    features = rfm[required_cols].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = KMeans(
        n_clusters=cfg.n_clusters,
        random_state=cfg.random_state,
        n_init=10,
    )
    clusters = model.fit_predict(scaled)
    rfm = rfm.copy()
    rfm["cluster"] = clusters

    cluster_stats = (
        rfm.groupby("cluster")[["Frequency", "Monetary"]]
        .mean()
        .sum(axis=1)
        .sort_values()
    )
    high_risk_cluster = cluster_stats.index[0]
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)
    return rfm


def merge_rfm_labels(
    df: pd.DataFrame,
    config: Optional[RFMConfig] = None,
) -> pd.DataFrame:
    """Merge RFMderived high-risk labels back to the transaction dataset"""
    cfg = config or RFMConfig()
    rfm = compute_rfm(df, cfg)
    labeled = label_high_risk_cluster(rfm, cfg)
    merged = df.merge(
        labeled[[cfg.customer_col, "is_high_risk"]],
        on=cfg.customer_col,
        how="left",
    )
    return merged


def build_preprocessing_pipeline(
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
) -> ColumnTransformer:
    """Create a ColumnTransformer with imputation,scaling,one-hot encoding"""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ]
    )
    return preprocessor


def apply_woe(
    train_df: pd.DataFrame,
    target_col: str,
    categorical_cols: Sequence[str],
) -> Tuple[pd.DataFrame, Optional[WOE]]:
    """Apply Weight of Evidence encoding to categorical columns."""
    if WOE is None:
        raise ImportError("xverse is not installed; install it to use WoE.")
    if target_col not in train_df.columns:
        raise KeyError(f"Target column {target_col} not found")
    woe = WOE()
    woe.fit(train_df[categorical_cols], train_df[target_col])
    transformed = woe.transform(train_df[categorical_cols])
    result = train_df.copy()
    result.drop(columns=list(categorical_cols), inplace=True)
    result = pd.concat([result, transformed], axis=1)
    return result, woe


def build_feature_table(
    df: pd.DataFrame,
    config: Optional[RFMConfig] = None,
) -> pd.DataFrame:
    """Create a model-ready feature table with aggregates, datetime"""
    cfg = config or RFMConfig()
    enriched = add_time_parts(df, cfg.datetime_col)
    aggregated = aggregate_customer_features(
        enriched,
        cfg.customer_col,
        cfg.amount_col,
    )
    labeled = label_high_risk_cluster(
        compute_rfm(enriched, cfg),
        cfg,
    )
    feature_table = aggregated.merge(
        labeled[[cfg.customer_col, "is_high_risk"]],
        on=cfg.customer_col,
        how="left",
    )
    return feature_table


__all__ = [
    "RFMConfig",
    "add_time_parts",
    "aggregate_customer_features",
    "compute_rfm",
    "label_high_risk_cluster",
    "merge_rfm_labels",
    "build_preprocessing_pipeline",
    "apply_woe",
    "build_feature_table",
]
