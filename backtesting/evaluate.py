"""
Out-of-Sample Backtesting Pipeline — Parquet-native.

Loads processed features from the local .parquet file and the best
trained model weights from models/best_model.pth, runs the full
multi-asset backtest, and pushes reports to the GitHub repository.
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from loguru import logger
import warnings

# Suppress pandas-ta / runtime warnings for clean logs
warnings.filterwarnings("ignore", category=RuntimeWarning)

from data_pipeline.loader import load_data
from drl_models.agent import DRLAgent
from backtesting.engine import BacktestEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"


def main():
    logger.info("═" * 60)
    logger.info("  OUT-OF-SAMPLE BACKTESTING — Parquet-Native")
    logger.info("═" * 60)

    # ── 1. Load Full Dataset from Parquet ──────────────────────────────────
    logger.info("Loading engineered features from Parquet…")
    df = load_data()

    if df.empty:
        logger.error("DataFrame is empty. Run `python3 -m data_pipeline.loader` first.")
        return

    # Ensure tz-naive DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # ── 2. Group by Ticker & Isolate Numeric Features ─────────────────────
    ticker_col = "ticker" if "ticker" in df.columns else "symbol"
    multi_ticker_data = {}

    for ticker, group in df.groupby(ticker_col):
        num_df = group.copy()
        numeric_cols = num_df.select_dtypes(include=[np.number]).columns
        multi_ticker_data[ticker] = num_df[numeric_cols]

    # ── 3. Dynamic State Dimension Alignment ──────────────────────────────
    first_ticker = list(multi_ticker_data.keys())[0]
    num_features = multi_ticker_data[first_ticker].shape[1]
    state_dim = num_features + 3  # +3 for balance, shares, net_worth
    action_dim = 3  # Hold, Buy, Sell

    logger.info(f"Detected {num_features} indicators → State Dim: {state_dim}")

    # ── 4. Load Trained Model Weights ─────────────────────────────────────
    agent = DRLAgent(state_dim=state_dim, action_dim=action_dim)

    if not MODEL_PATH.exists():
        logger.error(f"Model weights not found at {MODEL_PATH}. Train first.")
        return

    try:
        agent.load_state_dict(torch.load(str(MODEL_PATH), map_location=torch.device('cpu')))
        agent.eval()
        logger.info(f"Model weights loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        return

    # ── 5. Execute Multi-Asset Backtest ───────────────────────────────────
    engine = BacktestEngine(
        multi_ticker_data=multi_ticker_data,
        model=agent,
        start_date="2023-01-01",
        per_stock_budget=100000.0,
        commission=0.002,
    )
    engine.run()

    # ── 6. Git-Sync Reports ───────────────────────────────────────────────
    logger.info("Pushing backtest reports to GitHub…")
    try:
        from utils.git_sync import sync_reports
        sync_reports()
    except Exception as e:
        logger.warning(f"Git sync skipped: {e}")

    logger.success("Backtesting pipeline complete.")


if __name__ == "__main__":
    main()