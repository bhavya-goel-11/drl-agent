"""
Data Pipeline — Fetch, Engineer, Save, Load.

This module is the single entry point for all data operations.
It fetches raw OHLCV from yfinance, applies the full 31-feature
engineering pipeline, and persists the result as a compressed
Parquet file. Downstream modules (train, evaluate) call load_data()
to read from the local .parquet — zero database dependencies.
"""

import os
from pathlib import Path

import yfinance as yf
import pandas as pd
from loguru import logger
from typing import List, Optional, Tuple, Union

from data_pipeline.features import FeatureEngineer

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_PATH = DATA_DIR / "processed_data.parquet"

# ── Full Nifty-50 Ticker Universe ──────────────────────────────────────────────
TICKERS = [
    'TECHM.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS', 'INFY.NS', 'TRENT.NS',
    'BAJAJ-AUTO.NS', 'HCLTECH.NS', 'RELIANCE.NS', 'TITAN.NS', 'EICHERMOT.NS',
    'NTPC.NS', 'TCS.NS', 'DRREDDY.NS', 'CIPLA.NS', 'SUNPHARMA.NS',
    'WIPRO.NS', 'NESTLEIND.NS', 'APOLLOHOSP.NS', 'ULTRACEMCO.NS', 'BHARTIARTL.NS',
    'M&M.NS', 'SBIN.NS', 'ADANIPORTS.NS', 'ASIANPAINT.NS', 'HINDUNILVR.NS',
    'GRASIM.NS', 'POWERGRID.NS', 'ITC.NS', 'TATACONSUM.NS', 'BAJAJFINSV.NS',
    'MARUTI.NS', 'LT.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'BAJFINANCE.NS',
    'KOTAKBANK.NS', 'ADANIENT.NS', 'BEL.NS', 'ONGC.NS', 'SHRIRAMFIN.NS',
    'HDFCBANK.NS', 'HINDALCO.NS', 'BRITANNIA.NS', 'INDUSINDBK.NS',
    'HEROMOTOCO.NS', 'BPCL.NS',
    '^NSEI',  # Nifty 50 benchmark (used for RS_Momentum feature)
]


# ═══════════════════════════════════════════════════════════════════════════════
#  FETCH & PROCESS — Run once to build the .parquet file
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_and_process(
    tickers: List[str] = None,
    start: str = "2008-01-01",
    end: str = "2026-01-01",
    output_path: str = None,
) -> str:
    """Fetch raw OHLCV via yfinance, engineer features, save to Parquet.

    Parameters
    ----------
    tickers : list[str], optional
        Stock symbols. Defaults to the full 46+1 Nifty universe.
    start, end : str
        Date range for yfinance download.
    output_path : str, optional
        Override the default Parquet save location.

    Returns
    -------
    str
        Absolute path to the saved Parquet file.
    """
    tickers = tickers or TICKERS
    output = Path(output_path) if output_path else PARQUET_PATH

    # ── 1. Download raw OHLCV ─────────────────────────────────────────────
    logger.info(f"Downloading OHLCV for {len(tickers)} tickers ({start} → {end})…")
    all_frames = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                logger.warning(f"No data returned for {ticker} — skipping.")
                continue

            # yfinance sometimes returns MultiIndex columns for single tickers
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # Standardise column names to lowercase
            df.columns = [c.lower() for c in df.columns]
            df["symbol"] = ticker
            all_frames.append(df)
            logger.debug(f"  {ticker}: {len(df)} rows")

        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")

    if not all_frames:
        raise RuntimeError("No data could be downloaded. Check network / tickers.")

    raw_df = pd.concat(all_frames)
    raw_df.index.name = "date"
    raw_df.index = pd.to_datetime(raw_df.index)

    # Strip timezone if present (yfinance sometimes adds tz)
    if raw_df.index.tz is not None:
        raw_df.index = raw_df.index.tz_localize(None)

    logger.info(f"Raw dataset: {len(raw_df)} rows × {raw_df.shape[1]} cols across {raw_df['symbol'].nunique()} symbols.")

    # ── 2. Feature Engineering ────────────────────────────────────────────
    engineer = FeatureEngineer()
    processed_df = engineer.add_technical_indicators(raw_df)
    logger.info(f"After feature engineering: {processed_df.shape[0]} rows × {processed_df.shape[1]} cols.")

    # ── 3. Rename symbol → ticker (standard column name for rest of pipeline)
    if "symbol" in processed_df.columns:
        processed_df.rename(columns={"symbol": "ticker"}, inplace=True)

    # ── 4. Save to Parquet ────────────────────────────────────────────────
    os.makedirs(output.parent, exist_ok=True)
    processed_df.to_parquet(str(output), engine="pyarrow", compression="snappy")

    size_mb = os.path.getsize(output) / (1024 * 1024)
    logger.success(
        f"Saved → {output}  "
        f"({processed_df.shape[0]} rows × {processed_df.shape[1]} cols, {size_mb:.1f} MB)"
    )

    return str(output)


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA — Used by train.py and evaluate.py
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(
    ticker: Optional[str] = None,
    split_date: Optional[str] = None,
    parquet_path: Optional[str] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load processed data from the local Parquet file.

    Parameters
    ----------
    ticker : str, optional
        If provided, filter to this single ticker. If None, return all.
    split_date : str, optional
        If provided (e.g. "2023-01-01"), return a (train_df, test_df) tuple
        split at this date boundary.
    parquet_path : str, optional
        Override the default Parquet file location.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, pd.DataFrame)
        The loaded data, optionally filtered and/or split.
    """
    pq = Path(parquet_path) if parquet_path else PARQUET_PATH

    if not pq.exists():
        raise FileNotFoundError(
            f"Parquet file not found at {pq}. "
            f"Run `python3 -m data_pipeline.loader` first to create it."
        )

    logger.info(f"Loading data from {pq}…")
    df = pd.read_parquet(str(pq))

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Filter by ticker
    if ticker is not None:
        df = df[df["ticker"] == ticker].copy()
        if df.empty:
            logger.warning(f"Ticker '{ticker}' not found in parquet.")

    logger.info(f"Loaded {len(df)} rows ({df['ticker'].nunique() if 'ticker' in df.columns else 1} tickers).")

    # Split by date
    if split_date is not None:
        boundary = pd.to_datetime(split_date)
        train_df = df[df.index < boundary].copy()
        test_df = df[df.index >= boundary].copy()
        logger.info(f"Split at {split_date}: train={len(train_df)}, test={len(test_df)}")
        return train_df, test_df

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT — python3 -m data_pipeline.loader
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("═" * 60)
    logger.info("  DATA PIPELINE — Fetch → Engineer → Save to Parquet")
    logger.info("═" * 60)

    output = fetch_and_process()

    # Attempt git sync (non-fatal if not in a configured repo)
    try:
        from utils.git_sync import sync_data
        sync_data()
    except Exception as e:
        logger.warning(f"Git sync skipped: {e}")

    # Quick sanity printout
    df = pd.read_parquet(output)
    print(f"\n── Tickers ({df['ticker'].nunique()}): {sorted(df['ticker'].unique())}")
    print(f"── Shape: {df.shape}")
    print(f"── Columns: {list(df.columns)}")
    print(f"── Date range: {df.index.min()} → {df.index.max()}")