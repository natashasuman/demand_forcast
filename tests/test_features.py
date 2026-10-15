import numpy as np
import pandas as pd
import pytest

from src.features.build_features import TimeSeriesFeatureEngine


@pytest.fixture
def sample_sales_data():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    sales = np.arange(1, 61, dtype=float)
    return pd.DataFrame({"date": dates, "sales": sales})


def test_calendar_features(sample_sales_data):
    features_df = TimeSeriesFeatureEngine.create_calendar_features(sample_sales_data)
    expected_cols = [
        "dayofweek",
        "month",
        "quarter",
        "dayofyear",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "fourier_sin_1",
        "fourier_cos_1",
    ]
    for col in expected_cols:
        assert col in features_df.columns, f"Missing feature {col}"
    assert len(features_df) == len(sample_sales_data)


def test_lag_and_rolling_features(sample_sales_data):
    processed = TimeSeriesFeatureEngine.create_lag_and_rolling_features(
        sample_sales_data,
        target_col="sales",
        date_col="date",
        lags=[1, 7],
        windows=[7],
    )
    assert not processed.empty
    assert "sales_lag_1" in processed.columns
    assert "sales_lag_7" in processed.columns
    assert "sales_rolling_mean_7" in processed.columns
    assert "sales_rolling_std_7" in processed.columns
    # Check no null values left after dropping NA
    assert processed.isnull().sum().sum() == 0
    # Rolling mean at t must only use values prior to t (shifted by 1)
    # The first row of processed has index 7 (since max lag/window is 7)
    first_row_sales = processed.iloc[0]["sales"]
    assert first_row_sales == 8.0
    assert processed.iloc[0]["sales_lag_1"] == 7.0
