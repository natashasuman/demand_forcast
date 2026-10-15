import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from src.config import settings
from src.models.metrics import calculate_metrics


class LightGBMForecaster:
    """Gradient boosted tree model for tabular time series forecasting."""

    def __init__(self, params: dict | None = None):
        self.params = params or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
            "random_state": 42,
        }
        self.model: lgb.LGBMRegressor | None = None

    def evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return calculate_metrics(y_true, y_pred)

    def train_and_log(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "sales",
        experiment_name: str = "Demand_Forecasting_Benchmark",
        tracking_uri: str | None = None,
    ) -> tuple[lgb.LGBMRegressor, dict[str, float]]:
        mlflow.set_tracking_uri(tracking_uri or settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]

        with mlflow.start_run(run_name="LightGBM_Forecaster") as run:
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(X_train, y_train)

            predictions = self.model.predict(X_test)
            metrics = self.evaluate_metrics(y_test.values, predictions)

            # Log hyperparameters & metrics to MLflow
            mlflow.log_params(self.params)
            mlflow.log_param("model_type", "LightGBM")
            mlflow.log_param("num_features", len(feature_cols))
            mlflow.log_metrics(metrics)
            mlflow.lightgbm.log_model(self.model, artifact_path="lightgbm_model")

            print(
                f"[+] LightGBM Training Complete. Run ID: {run.info.run_id} | "
                f"WAPE: {metrics['WAPE']:.2f}% | RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f}"
            )
            return self.model, metrics
