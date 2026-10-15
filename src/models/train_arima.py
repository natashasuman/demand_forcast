import mlflow
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from src.config import settings
from src.models.metrics import calculate_metrics


class ARIMAForecaster:
    """Statistical ARIMA time-series model using statsmodels."""

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1), seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model: ARIMAResults | None = None

    def evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return calculate_metrics(y_true, y_pred)

    def train_and_log(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = "sales",
        experiment_name: str = "Demand_Forecasting_Benchmark",
        tracking_uri: str | None = None,
    ) -> tuple[ARIMAResults, dict[str, float]]:
        mlflow.set_tracking_uri(tracking_uri or settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

        with mlflow.start_run(run_name="ARIMA_Forecaster") as run:
            model = ARIMA(
                y_train,
                order=self.order,
                seasonal_order=self.seasonal_order if self.seasonal_order != (0, 0, 0, 0) else None,
            )
            self.fitted_model = model.fit()

            # Out-of-sample forecast for the test period
            forecast_result = self.fitted_model.forecast(steps=len(y_test))
            metrics = self.evaluate_metrics(y_test, forecast_result)

            # Log to MLflow
            mlflow.log_params({
                "model_type": "ARIMA",
                "order_p": self.order[0],
                "order_d": self.order[1],
                "order_q": self.order[2],
                "seasonal_order": str(self.seasonal_order),
                "aic": float(self.fitted_model.aic),
                "bic": float(self.fitted_model.bic),
            })
            mlflow.log_metrics(metrics)
            mlflow.statsmodels.log_model(self.fitted_model, artifact_path="arima_model")

            print(
                f"[+] ARIMA Training Complete. Run ID: {run.info.run_id} | "
                f"WAPE: {metrics['WAPE']:.2f}% | RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f}"
            )
            return self.fitted_model, metrics
