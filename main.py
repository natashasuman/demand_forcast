import os
import numpy as np
import pandas as pd

from src.config import settings
from src.features.build_features import TimeSeriesFeatureEngine
from src.models.train_arima import ARIMAForecaster
from src.models.train_lightgbm import LightGBMForecaster
from src.models.train_prophet import ProphetForecaster
from src.tracking.mlflow_registry import SageMakerRegistrar


def load_or_fetch_data(csv_path: str = "data/raw_sales_data.csv") -> pd.DataFrame:
    """Loads existing raw sales CSV or downloads and processes real UCI Online Retail data if missing."""
    if os.path.exists(csv_path):
        print(f"[+] Ingesting data from {csv_path}")
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    print(f"[!] Raw data file '{csv_path}' not found.")
    print("[+] Extracting and processing real-world UCI Online Retail dataset...")
    from data.download_uci_retail import download_and_process_uci_retail
    return download_and_process_uci_retail(output_path=csv_path)


def run_pipeline(dry_run_registry: bool = True):
    print("=" * 60)
    print(f"=== {settings.PROJECT_NAME} ===")
    print("=" * 60)

    # 1. Ingestion
    raw_df = load_or_fetch_data()

    # 2. Sequential Train/Test Split (80/20 chronological split)
    train_size = int(len(raw_df) * 0.8)
    raw_train = raw_df.iloc[:train_size].copy()
    raw_test = raw_df.iloc[train_size:].copy()

    print(f"Dataset summary: Total={len(raw_df)} | Train={len(raw_train)} | Test={len(raw_test)}")

    leaderboard: dict[str, dict[str, float]] = {}

    # 3. Benchmark Model 1: ARIMA
    print("\n--- Training Model 1: ARIMA ---")
    try:
        arima = ARIMAForecaster(order=(2, 1, 1), seasonal_order=(1, 0, 1, 7))
        _, arima_metrics = arima.train_and_log(raw_train, raw_test, target_col="sales")
        leaderboard["ARIMA"] = arima_metrics
    except Exception as e:
        print(f"[-] ARIMA failed: {e}")

    # 4. Benchmark Model 2: Prophet
    print("\n--- Training Model 2: Prophet ---")
    try:
        prophet = ProphetForecaster(yearly_seasonality=True, weekly_seasonality=True)
        _, prophet_metrics = prophet.train_and_log(raw_train, raw_test, date_col="date", target_col="sales")
        leaderboard["Prophet"] = prophet_metrics
    except Exception as e:
        print(f"[-] Prophet failed: {e}")

    # 5. Benchmark Model 3: LightGBM
    print("\n--- Training Model 3: LightGBM (with Feature Store) ---")
    feature_engine = TimeSeriesFeatureEngine()
    processed_df = feature_engine.create_lag_and_rolling_features(raw_df, target_col="sales", date_col="date")

    # Split tabular feature dataframe sequentially
    feat_train_size = int(len(processed_df) * 0.8)
    feat_train = processed_df.iloc[:feat_train_size].copy()
    feat_test = processed_df.iloc[feat_train_size:].copy()
    feature_cols = [c for c in processed_df.columns if c not in ["date", "sales"]]

    lgb_forecaster = LightGBMForecaster()
    _, lgb_metrics = lgb_forecaster.train_and_log(
        train_df=feat_train,
        test_df=feat_test,
        feature_cols=feature_cols,
        target_col="sales",
    )
    leaderboard["LightGBM"] = lgb_metrics

    # 6. Evaluation & Model Selection Leaderboard
    print("\n" + "=" * 60)
    print("=== MODEL BENCHMARK LEADERBOARD (Primary Metric: WAPE) ===")
    print("=" * 60)
    print(f"{'Model':<15} | {'WAPE (%)':<10} | {'RMSE':<10} | {'MAE':<10}")
    print("-" * 55)

    best_model = None
    lowest_wape = float("inf")

    for model_name, metrics in leaderboard.items():
        wape = metrics["WAPE"]
        rmse = metrics["RMSE"]
        mae = metrics["MAE"]
        print(f"{model_name:<15} | {wape:<10.2f} | {rmse:<10.2f} | {mae:<10.2f}")
        if wape < lowest_wape:
            lowest_wape = wape
            best_model = model_name

    print("=" * 60)
    print(f"[CHAMPION] Champion Model Selected: {best_model} with WAPE: {lowest_wape:.2f}%")

    # 7. SageMaker Model Registry Hook
    print("\n--- Registering Champion Model ---")
    registrar = SageMakerRegistrar()
    model_s3_uri = f"s3://{settings.S3_BUCKET_NAME}/models/{best_model.lower()}/model.tar.gz"
    model_name = f"demand-forecaster-{best_model.lower()}-champion"

    arn = registrar.register_model_artifact(
        model_s3_uri=model_s3_uri,
        model_name=model_name,
        dry_run=dry_run_registry,
    )
    print(f"[SUCCESS] Registration completed: {arn}")
    return leaderboard, best_model


if __name__ == "__main__":
    run_pipeline()
