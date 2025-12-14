import pandas as pd

from src.data_processing import (
    RFMConfig,
    aggregate_customer_features,
    compute_rfm,
    label_high_risk_cluster,
)


def sample_df():
    return pd.DataFrame(
        {
            "CustomerId": [1, 1, 2, 3],
            "Amount": [100.0, 50.0, 20.0, 5.0],
            "TransactionStartTime": [
                "2025-01-01",
                "2025-01-05",
                "2025-01-03",
                "2025-01-02",
            ],
        }
    )


def test_aggregate_customer_features_columns():
    df = sample_df()
    aggregated = aggregate_customer_features(df)
    expected_cols = {
        "CustomerId",
        "total_amount",
        "avg_amount",
        "transaction_count",
        "std_amount",
    }
    assert expected_cols.issubset(set(aggregated.columns))
    assert (
        aggregated.loc[
            aggregated["CustomerId"] == 1,
            "transaction_count",
        ].item()
        == 2
    )


def test_label_high_risk_cluster_outputs_flag():
    df = sample_df()
    rfm = compute_rfm(df, RFMConfig())
    labeled = label_high_risk_cluster(rfm)
    assert "is_high_risk" in labeled.columns
    assert set(labeled["is_high_risk"]).issubset({0, 1})
