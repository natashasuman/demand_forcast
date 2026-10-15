from datetime import date
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise Multi-Model Demand Forecasting Inference API",
    version="1.0.0",
)


class ForecastRequest(BaseModel):
    model_name: str = Field(default="lightgbm", description="Forecasting model to use: lightgbm, arima, prophet")
    horizon_days: int = Field(default=7, ge=1, le=90, description="Forecast horizon in days")


class ForecastPoint(BaseModel):
    date: date
    predicted_sales: float


class ForecastResponse(BaseModel):
    model_name: str
    horizon_days: int
    forecast: list[ForecastPoint]


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}


@app.post("/predict", response_model=ForecastResponse)
def predict_demand(request: ForecastRequest):
    allowed_models = ["lightgbm", "arima", "prophet"]
    if request.model_name.lower() not in allowed_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_name}' not supported. Choose from {allowed_models}",
        )

    # Placeholder inference response reflecting model contract
    import pandas as pd
    today = pd.Timestamp.now().normalize()
    dates = [today + pd.Timedelta(days=i) for i in range(1, request.horizon_days + 1)]

    # Example baseline prediction points
    predictions = [
        ForecastPoint(date=d.date(), predicted_sales=round(100.0 + i * 2.5, 2))
        for i, d in enumerate(dates)
    ]

    return ForecastResponse(
        model_name=request.model_name.lower(),
        horizon_days=request.horizon_days,
        forecast=predictions,
    )
