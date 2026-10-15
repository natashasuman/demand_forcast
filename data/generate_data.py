import os
import numpy as np
import pandas as pd


def generate_and_save_data(output_path: str = "data/raw_sales_data.csv", n_days: int = 730):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    np.random.seed(42)

    # Base trend + annual seasonality + weekly seasonality + noise
    trend = np.linspace(50, 120, n_days)
    annual_seasonality = np.sin(np.linspace(0, 4 * np.pi, n_days)) * 25
    weekly_seasonality = (dates.dayofweek >= 5).astype(int) * 15  # weekend bump
    noise = np.random.normal(0, 6, n_days)

    sales = np.maximum(10, trend + annual_seasonality + weekly_seasonality + noise)

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "sales": np.round(sales, 2),
    })

    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows in {output_path}")


if __name__ == "__main__":
    generate_and_save_data()
