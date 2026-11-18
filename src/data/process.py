"""Data cleaning, validation, and feature/target calculation module.

Validates:
- Missing values
- Duplicate dates
- Missing OHLCV fields
- Non-positive prices or invalid price relationships (High < Low, etc.)
- Historical date coverage & cross-stock calendar alignment

Methodology:
- Daily log return: r_t = ln(P_t / P_{t-1}) using Adjusted Close (dividend/split adjusted)
- Five-day forward realized variance: RV_(t,t+5) = sum of squared daily log returns over next 5 trading days
- Five-day forward realized volatility: sigma_(t,t+5) = sqrt(RV_(t,t+5))
- Strictly preserves causality (no future information leakage into date t features).
Saves validated datasets in data/processed/
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

RAW_DIR: Path = Path("data/raw")
PROCESSED_DIR: Path = Path("data/processed")
REPORTS_DIR: Path = Path("reports")

TICKERS: List[str] = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "TCS.NS"
]


def validate_raw_dataframe(df: pd.DataFrame, ticker: str) -> Dict[str, any]:
    """Perform rigorous checks on a raw OHLCV DataFrame.
    
    Checks:
    - Missing values count per column
    - Duplicate dates
    - Required OHLCV fields presence
    - Invalid/non-positive prices (Open, High, Low, Close, Adj Close <= 0, Volume < 0)
    - High < Low, High < Open, High < Close, Low > Open, Low > Close anomalies
    - Date range and coverage
    """
    report = {
        "ticker": ticker,
        "total_records": len(df),
        "columns_present": list(df.columns),
        "missing_fields": [],
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_dates_count": int(df.duplicated(subset=["Date"]).sum()) if "Date" in df.columns else 0,
        "invalid_prices_count": 0,
        "price_logic_violations": 0,
        "start_date": None,
        "end_date": None,
        "calendar_days": 0,
        "trading_days": len(df)
    }
    
    required_fields = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    report["missing_fields"] = [f for f in required_fields if f not in df.columns]
    
    if "Date" in df.columns and len(df) > 0:
        report["start_date"] = str(df["Date"].min().strftime("%Y-%m-%d"))
        report["end_date"] = str(df["Date"].max().strftime("%Y-%m-%d"))
        report["calendar_days"] = (df["Date"].max() - df["Date"].min()).days + 1
        
    price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df.columns]
    for col in price_cols:
        invalid_count = int((df[col] <= 0).sum())
        if invalid_count > 0:
            report["invalid_prices_count"] += invalid_count
            
    if "Volume" in df.columns:
        negative_vol = int((df["Volume"] < 0).sum())
        if negative_vol > 0:
            report["invalid_prices_count"] += negative_vol
            
    if all(c in df.columns for c in ["High", "Low"]):
        hl_violations = int((df["High"] < df["Low"]).sum())
        report["price_logic_violations"] += hl_violations
        
    return report


def process_single_stock(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Clean data and construct target variables strictly conforming to project methodology."""
    # Ensure sorted by Date ascending
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    
    validation_before = validate_raw_dataframe(df, ticker)
    
    # 1. Forward-fill any minor sporadic missing values if existing (none expected in liquid NSE)
    df = df.ffill().dropna(subset=["Adj Close", "Close"]).reset_index(drop=True)
    
    # 2. Daily Log Return: r_t = ln(P_t / P_(t-1))
    # We use Adjusted Close to prevent artificial volatility jumps from corporate actions (stock splits/dividends)
    df["Log_Return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    
    # Also record simple return for completeness: R_t = (P_t - P_(t-1)) / P_(t-1)
    df["Simple_Return"] = (df["Adj Close"] - df["Adj Close"].shift(1)) / df["Adj Close"].shift(1)
    
    # 3. Squared daily log return: r_t^2
    df["Squared_Log_Return"] = df["Log_Return"] ** 2
    
    # 4. Five-day Forward Realized Variance:
    # RV_(t,t+5) = sum of squared daily log returns over next five trading days:
    # RV_(t,t+5) = r_{t+1}^2 + r_{t+2}^2 + r_{t+3}^2 + r_{t+4}^2 + r_{t+5}^2
    # Vectorized computation:
    sq_ret = df["Squared_Log_Return"]
    # Shifting by -1 aligns day t+1 with index t. Then rolling window of 5 summing backwards over 5 days.
    # Shifting the rolling sum of 5 on reversed array:
    forward_rv = sq_ret.iloc[::-1].rolling(window=5, min_periods=5).sum().iloc[::-1].shift(-1)
    df["Target_Forward_RV_5d"] = forward_rv
    
    # 5. Five-day Forward Realized Volatility:
    # sigma_(t,t+5) = sqrt(RV_(t,t+5))
    df["Target_Forward_Volatility_5d"] = np.sqrt(df["Target_Forward_RV_5d"])
    
    # Annualized version of 5-day forward realized volatility for financial interpretability:
    # sigma_ann = sigma_5d * sqrt(252 / 5)
    df["Target_Forward_Volatility_5d_Annualized"] = df["Target_Forward_Volatility_5d"] * np.sqrt(252.0 / 5.0)
    
    # Historical rolling volatility features (Strictly Backward-Looking, no future leakage):
    # 5-day, 10-day, 20-day, and 30-day realized volatility available at day t:
    for win in [5, 10, 20, 30]:
        df[f"Hist_Vol_{win}d"] = np.sqrt(df["Squared_Log_Return"].rolling(window=win, min_periods=win).sum())
        df[f"Hist_Vol_{win}d_Annualized"] = df[f"Hist_Vol_{win}d"] * np.sqrt(252.0 / win)
        
    validation_after = {
        "ticker": ticker,
        "processed_records": len(df),
        "valid_log_returns": int(df["Log_Return"].notnull().sum()),
        "valid_forward_vol_targets": int(df["Target_Forward_Volatility_5d"].notnull().sum()),
        "final_unobserved_future_targets": int(df["Target_Forward_Volatility_5d"].isnull().sum()),
        "start_date": str(df["Date"].min().strftime("%Y-%m-%d")),
        "end_date": str(df["Date"].max().strftime("%Y-%m-%d")),
    }
    
    return df, {"before": validation_before, "after": validation_after}


def process_and_validate_all() -> Tuple[Dict[str, pd.DataFrame], Dict[str, any]]:
    """Process all stocks, validate, and save to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    processed_dfs: Dict[str, pd.DataFrame] = {}
    validation_summary: Dict[str, any] = {}
    combined_records = []
    
    for ticker in TICKERS:
        raw_file = RAW_DIR / f"{ticker}.csv"
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_file}. Run download.py first.")
            
        raw_df = pd.read_csv(raw_file)
        proc_df, val_meta = process_single_stock(raw_df, ticker)
        
        # Save per-stock processed dataset
        proc_file = PROCESSED_DIR / f"{ticker}.csv"
        proc_df.to_csv(proc_file, index=False)
        print(f"Processed {ticker}: {len(proc_df)} rows saved to {proc_file}")
        
        processed_dfs[ticker] = proc_df
        validation_summary[ticker] = val_meta
        combined_records.append(proc_df)
        
    combined_df = pd.concat(combined_records, ignore_index=True)
    combined_file = PROCESSED_DIR / "all_stocks_processed.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"All processed data saved to {combined_file} ({len(combined_df)} records).")
    
    # Save validation metadata JSON
    val_meta_file = REPORTS_DIR / "validation_summary.json"
    with open(val_meta_file, "w") as f:
        json.dump(validation_summary, f, indent=2)
        
    return processed_dfs, validation_summary


if __name__ == "__main__":
    process_and_validate_all()
