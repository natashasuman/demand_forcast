import numpy as np
import pandas as pd


class TimeSeriesFeatureEngine:
    """Feature engineering pipeline for tabular time-series models (e.g. LightGBM)."""

    @staticmethod
    def create_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Extracts temporal and calendar features."""
        df = df.copy()
        date_series = pd.to_datetime(df[date_col])

        df["dayofweek"] = date_series.dt.dayofweek
        df["month"] = date_series.dt.month
        df["quarter"] = date_series.dt.quarter
        df["dayofyear"] = date_series.dt.dayofyear
        df["is_weekend"] = date_series.dt.dayofweek.isin([5, 6]).astype(int)
        df["is_month_start"] = date_series.dt.is_month_start.astype(int)
        df["is_month_end"] = date_series.dt.is_month_end.astype(int)

        # Fourier terms for smooth annual seasonality (365.25 days)
        day_of_year = date_series.dt.dayofyear
        for order in [1, 2]:
            df[f"fourier_sin_{order}"] = np.sin(2 * np.pi * order * day_of_year / 365.25)
            df[f"fourier_cos_{order}"] = np.cos(2 * np.pi * order * day_of_year / 365.25)

        return df

    @staticmethod
    def create_lag_and_rolling_features(
        df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        lags: list[int] | None = None,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Engineers temporal, lag, rolling window, and Fourier features for LightGBM.
        All rolling windows are shifted by 1 to strictly prevent data leakage.
        """
        if lags is None:
            lags = [1, 7, 14, 30]
        if windows is None:
            windows = [7, 14, 30]

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        # Calendar features
        df = TimeSeriesFeatureEngine.create_calendar_features(df, date_col=date_col)

        # Lag features
        for lag in lags:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

        # Rolling window features (shifted by 1 to prevent target leakage)
        target_shifted = df[target_col].shift(1)
        for window in windows:
            df[f"{target_col}_rolling_mean_{window}"] = target_shifted.rolling(window=window).mean()
            df[f"{target_col}_rolling_std_{window}"] = target_shifted.rolling(window=window).std()

        return df.dropna().reset_index(drop=True)
