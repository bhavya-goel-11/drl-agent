import pandas as pd
import numpy as np
import torch
from loguru import logger
import warnings

# Suppress pandas-ta warnings for clean logs
warnings.filterwarnings("ignore", category=RuntimeWarning) 

from database.connection import SessionLocal
from database.models import HistoricalData
from data_pipeline.features import FeatureEngineer, align_multi_ticker_data
from drl_models.agent import DRLAgent
from backtesting.engine import BacktestEngine

TRAIN_CUTOFF = "2023-01-01"


def main():
    logger.info("=" * 80)
    logger.info("Initialising Vectorised Portfolio Evaluation Pipeline")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # 1. Fetch ALL data (train + test) from TimescaleDB
    # ------------------------------------------------------------------
    db = SessionLocal()
    try:
        logger.info("Fetching complete historical data for all tickers...")
        records = (db.query(HistoricalData)
                     .order_by(HistoricalData.date.asc())
                     .all())

        if not records:
            logger.error("No data found in database!")
            return

        data = [{
            'symbol': r.symbol,
            'date':   r.date,
            'open':   float(r.open),
            'high':   float(r.high),
            'low':    float(r.low),
            'close':  float(r.close),
            'volume': float(r.volume),
        } for r in records]

        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)

        # Strip timezone if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    except Exception as e:
        logger.error(f"Database error: {e}")
        return
    finally:
        db.close()

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    engineer = FeatureEngineer()
    engineered_df = engineer.add_technical_indicators(df)

    # ------------------------------------------------------------------
    # 3. Group by ticker & align
    # ------------------------------------------------------------------
    multi_ticker_data = {}
    for ticker, group in engineered_df.groupby('symbol'):
        multi_ticker_data[ticker] = group.copy()

    data_3d, tickers, dates, columns = align_multi_ticker_data(multi_ticker_data)
    n_stocks = len(tickers)
    n_features = len(columns)

    logger.info(f"Aligned {n_stocks} tickers × {n_features} features "
                f"× {data_3d.shape[0]} timesteps")

    # ------------------------------------------------------------------
    # 4. Find train/test split index
    # ------------------------------------------------------------------
    cutoff = pd.to_datetime(TRAIN_CUTOFF)
    train_end_idx = int(np.searchsorted(dates, cutoff))
    logger.info(f"Train period: {dates[0].date()} → {dates[train_end_idx-1].date()} "
                f"({train_end_idx} steps)")
    logger.info(f"Test period:  {dates[train_end_idx].date()} → {dates[-1].date()} "
                f"({len(dates) - train_end_idx} steps)")

    # ------------------------------------------------------------------
    # 5. Build agent and load weights
    # ------------------------------------------------------------------
    # Observation dim must match training env exactly
    obs_dim = n_stocks * n_features + n_stocks + 3
    action_dim = 3

    logger.info(f"Initialising D3QN Network (obs_dim={obs_dim}, "
                f"actions=3×{n_stocks})...")
    agent = DRLAgent(state_dim=obs_dim, action_dim=action_dim,
                     n_stocks=n_stocks)

    model_path = "drl_models/best_universal_dqn_trader.pth"
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
            agent.load_state_dict(checkpoint['policy_net'])
        else:
            agent.load_state_dict(checkpoint)
        agent.eval()
        logger.info("D3QN model weights loaded and locked.")
    except Exception as e:
        logger.error(f"Failed to load weights from {model_path}: {e}")
        return

    # ------------------------------------------------------------------
    # 6. Run vectorised backtest
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("Running Vectorised OOS Backtest")
    logger.info("=" * 80)

    engine = BacktestEngine(
        data_3d=data_3d,
        tickers=tickers,
        dates=dates,
        columns=columns,
        model=agent,
        n_stocks=n_stocks,
        train_end_idx=train_end_idx,
        initial_balance=10_000_000.0,
        commission=0.002,
    )
    engine.run()

    logger.info("\n✅ Evaluation pipeline complete. Check 'reports/' for outputs.")


if __name__ == "__main__":
    main()