"""Data collection module for NSE stock volatility forecasting.

Downloads daily OHLCV data for five NSE stocks:
- RELIANCE.NS
- HDFCBANK.NS
- INFY.NS
- ICICIBANK.NS
- TCS.NS

Period: 2015-01-01 to 2026-11-17
Saves original downloaded data to data/raw/
"""

import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
import yfinance as yf

# Target tickers and date range
TICKERS: List[str] = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "TCS.NS"
]

START_DATE: str = "2015-01-01"
END_DATE: str = "2026-11-18"  # End is exclusive in yfinance; ensures coverage through 2026-11-17

RAW_DIR: Path = Path("data/raw")


def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data for a single stock from yfinance.
    
    Ensures standard columns: Open, High, Low, Close, Adj Close, Volume.
    """
    print(f"Downloading {ticker} from {start} to {end}...")
    # Fetch ticker object and historical data
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, auto_adjust=False, actions=True)
    
    if df.empty:
        # Fallback to yf.download if history returns empty
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-index columns if returned
            df.columns = [col[0] for col in df.columns]
    
    # Reset index to make Date a standard column
    df = df.reset_index()
    
    # Normalize Date to timezone-naive YYYY-MM-DD
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    
    # Ensure expected columns are present
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available_cols = [c for c in expected_cols if c in df.columns]
    df = df[available_cols].copy()
    
    # Add Ticker column
    df["Ticker"] = ticker
    
    # Sort strictly by date ascending
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def download_all_stocks() -> Dict[str, pd.DataFrame]:
    """Download and save raw daily data for all specified tickers."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stock_dfs: Dict[str, pd.DataFrame] = {}
    combined_records = []
    
    for ticker in TICKERS:
        df = download_stock_data(ticker, START_DATE, END_DATE)
        
        # Save raw per-stock CSV
        file_path = RAW_DIR / f"{ticker}.csv"
        df.to_csv(file_path, index=False)
        print(f"  -> Saved {len(df)} records to {file_path}")
        
        stock_dfs[ticker] = df
        combined_records.append(df)
        
    # Save combined raw dataset
    combined_df = pd.concat(combined_records, ignore_index=True)
    combined_path = RAW_DIR / "all_stocks_raw.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"All raw data saved successfully. Combined records: {len(combined_df)}")
    
    return stock_dfs


if __name__ == "__main__":
    download_all_stocks()
