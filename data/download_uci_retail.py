import io
import os
import zipfile
import requests
import pandas as pd


UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
OUTPUT_CSV = "data/raw_sales_data.csv"


def download_and_process_uci_retail(output_path: str = OUTPUT_CSV) -> pd.DataFrame:
    """
    Downloads the real-life UCI Online Retail dataset (541k+ transactions),
    filters cancellations/returns, aggregates to daily sales demand,
    ensures a complete daily time series, and saves to data/raw_sales_data.csv.
    """
    print(f"[+] Downloading UCI Online Retail dataset from {UCI_URL} ...")
    response = requests.get(UCI_URL, timeout=60)
    response.raise_for_status()

    print("[+] Extracting dataset from zip...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        excel_name = [name for name in z.namelist() if name.endswith(".xlsx")][0]
        print(f"[+] Found Excel file: {excel_name}. Loading into Pandas (this may take a few seconds)...")
        with z.open(excel_name) as f:
            df = pd.read_excel(f)

    print(f"[+] Raw transactions loaded: {len(df):,} rows.")

    # Data Cleaning:
    # 1. Filter out cancellations (InvoiceNo starting with 'C' or negative Quantity)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    valid_df = df[~df["InvoiceNo"].str.startswith("C") & (df["Quantity"] > 0)].copy()

    # 2. Extract Date (daily resolution)
    valid_df["date"] = pd.to_datetime(valid_df["InvoiceDate"]).dt.date

    # 3. Aggregate daily sales volume (total units demanded)
    daily_sales = (
        valid_df.groupby("date")["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Quantity": "sales"})
    )
    daily_sales["date"] = pd.to_datetime(daily_sales["date"])

    # 4. Fill calendar gaps with 0 (e.g. holidays / Sundays when store wasn't processing orders)
    daily_sales = daily_sales.set_index("date").asfreq("D", fill_value=0.0).reset_index()

    # Round sales to 2 decimal places
    daily_sales["sales"] = daily_sales["sales"].round(2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    daily_sales.to_csv(output_path, index=False)

    print(f"[+] Successfully processed real-world dataset!")
    print(f"    - Date Range: {daily_sales['date'].min().strftime('%Y-%m-%d')} to {daily_sales['date'].max().strftime('%Y-%m-%d')}")
    print(f"    - Total Days: {len(daily_sales)}")
    print(f"    - Mean Daily Demand: {daily_sales['sales'].mean():.2f} units")
    print(f"    - Saved to: {output_path}")

    return daily_sales


if __name__ == "__main__":
    download_and_process_uci_retail()
