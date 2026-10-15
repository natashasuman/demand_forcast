import numpy as np
import pandas as pd
import pytest

from src.models.metrics import calculate_metrics
from src.models.train_arima import ARIMAForecaster
from src.models.train_lightgbm import LightGBMForecaster
from src.models.train_prophet import ProphetForecaster


@pytest.fixture
def dummy_train_test_series():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    sales = 50.0 + np.sin(np.linspace(0, 10, 100)) * 10.0 + np.arange(100) * 0.1
    df = pd.DataFrame({"date": dates, "sales": sales})
    train_df = df.iloc[:80].copy()
    test_df = df.iloc[80:].copy()
    return train_df, test_df


def test_calculate_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    metrics = calculate_metrics(y_true, y_pred)

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "WAPE" in metrics
    assert metrics["MAE"] == 10.0
    # WAPE = (|10| + |10| + |10|) / 600 * 100 = 30 / 600 * 100 = 5.0%
    assert metrics["WAPE"] == 5.0


def test_arima_forecaster(dummy_train_test_series, tmp_path):
    train_df, test_df = dummy_train_test_series
    tracking_uri = f"sqlite:///{tmp_path / 'test_arima.db'}"

    arima = ARIMAForecaster(order=(1, 1, 0))
    model, metrics = arima.train_and_log(train_df, test_df, target_col="sales", tracking_uri=tracking_uri)
    assert model is not None
    assert metrics["WAPE"] >= 0.0


def test_lightgbm_forecaster(dummy_train_test_series, tmp_path):
    train_df, test_df = dummy_train_test_series
    tracking_uri = f"sqlite:///{tmp_path / 'test_lgb.db'}"

    # Add simple lag features
    for df in [train_df, test_df]:
        df["lag_1"] = df["sales"].shift(1).bfill()
        df["dayofweek"] = df["date"].dt.dayofweek

    lgb_model = LightGBMForecaster(params={"n_estimators": 10, "verbose": -1})
    model, metrics = lgb_model.train_and_log(
        train_df,
        test_df,
        feature_cols=["lag_1", "dayofweek"],
        target_col="sales",
        tracking_uri=tracking_uri,
    )
    assert model is not None
    assert metrics["WAPE"] >= 0.0


def test_prophet_forecaster(dummy_train_test_series, tmp_path):
    train_df, test_df = dummy_train_test_series
    tracking_uri = f"sqlite:///{tmp_path / 'test_prophet.db'}"

    prophet = ProphetForecaster(yearly_seasonality=False, weekly_seasonality=False)
    model, metrics = prophet.train_and_log(
        train_df,
        test_df,
        date_col="date",
        target_col="sales",
        tracking_uri=tracking_uri,
    )
    assert model is not None
    assert metrics["WAPE"] >= 0.0
