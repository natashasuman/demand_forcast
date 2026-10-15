import mlflow
import mlflow.prophet
import mlflow.pyfunc
import numpy as np
import pandas as pd
from prophet import Prophet

from src.config import settings
from src.models.metrics import calculate_metrics


class ProphetForecaster:
    """Additive time-series model handling trend, seasonality, and holidays using Prophet."""

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
    ):
        self.params = {
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality,
            "changepoint_prior_scale": changepoint_prior_scale,
        }
        self.model: Prophet | None = None

    def evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return calculate_metrics(y_true, y_pred)

    def train_and_log(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_col: str = "date",
        target_col: str = "sales",
        experiment_name: str = "Demand_Forecasting_Benchmark",
        tracking_uri: str | None = None,
    ) -> tuple[Prophet, dict[str, float]]:
        mlflow.set_tracking_uri(tracking_uri or settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

        # Prepare prophet format (ds, y)
        prophet_train = pd.DataFrame({
            "ds": pd.to_datetime(train_df[date_col]),
            "y": train_df[target_col].values,
        })
        prophet_test = pd.DataFrame({
            "ds": pd.to_datetime(test_df[date_col]),
            "y": test_df[target_col].values,
        })

        with mlflow.start_run(run_name="Prophet_Forecaster") as run:
            self.model = Prophet(**self.params)
            self.model.fit(prophet_train)

            # Predict for the test period
            future = pd.DataFrame({"ds": prophet_test["ds"]})
            forecast = self.model.predict(future)

            predictions = forecast["yhat"].values
            y_test = prophet_test["y"].values
            metrics = self.evaluate_metrics(y_test, predictions)

            # Log parameters & metrics to MLflow
            mlflow.log_params({
                "model_type": "Prophet",
                **self.params,
            })
            mlflow.log_metrics(metrics)

            try:
                mlflow.prophet.log_model(self.model, artifact_path="prophet_model")
            except Exception:
                # Fallback if prophet mlflow flavor encounters binary serialization quirks
                mlflow.pyfunc.log_model(
                    artifact_path="prophet_model",
                    python_model=None,
                )

            print(
                f"[+] Prophet Training Complete. Run ID: {run.info.run_id} | "
                f"WAPE: {metrics['WAPE']:.2f}% | RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f}"
            )
            return self.model, metrics
