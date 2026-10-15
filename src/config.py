import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure MLflow allows file-based stores in tests and local environments
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise Demand Forecasting Engine"
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    AWS_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "demand-forecasting-mlops-data")
    SAGEMAKER_EXECUTION_ROLE: str = os.getenv(
        "SAGEMAKER_EXECUTION_ROLE",
        "arn:aws:iam::123456789012:role/SageMakerRole",
    )
    PRIMARY_METRIC: str = "WAPE"  # Weighted Absolute Percentage Error

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
