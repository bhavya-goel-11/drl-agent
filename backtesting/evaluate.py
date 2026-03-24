import pandas as pd
import numpy as np
import torch
from loguru import logger
import warnings

# Suppress pandas-ta warnings for clean logs
warnings.filterwarnings("ignore", category=RuntimeWarning) 

from database.connection import SessionLocal
from database.models import HistoricalData
from data_pipeline.features import FeatureEngineer
from drl_models.agent import DRLAgent
from backtesting.engine import BacktestEngine

def main():
    logger.info("Initializing Universal Out-of-Sample Backtest Pipeline...")
    
    # 1. Fetch data from TimescaleDB
    db = SessionLocal()
    try:
        logger.info("Fetching complete historical data for all tickers...")
        records = db.query(HistoricalData).order_by(HistoricalData.date.asc()).all()
        
        if not records:
            logger.error("No data found in database!")
            return
            
        data = [{
            'symbol': r.symbol,
            'date': r.date, 
            'open': float(r.open), 
            'high': float(r.high), 
            'low': float(r.low), 
            'close': float(r.close), 
            'volume': float(r.volume)
        } for r in records]
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        # --- FIX: Strip Timezone to make it tz-naive ---
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
    except Exception as e:
        logger.error(f"Database error: {e}")
        return
    finally:
        db.close()

    # 2. Engineer Features
    engineer = FeatureEngineer()
    engineered_df = engineer.add_technical_indicators(df)
    
    # 3. Group by Symbol and isolate Neural Network inputs
    multi_ticker_data = {}
    for ticker, group in engineered_df.groupby('symbol'):
        num_cols_df = group.copy()
        numeric_cols = num_cols_df.select_dtypes(include=[np.number]).columns
        multi_ticker_data[ticker] = num_cols_df[numeric_cols]
    
    # 4. Dynamic State Dimension Alignment
    first_ticker = list(multi_ticker_data.keys())[0]
    num_features = multi_ticker_data[first_ticker].shape[1]
    state_dim = num_features + 3 
    action_dim = 3  
    
    logger.info(f"Detected {num_features} indicators. Initializing Universal Network (State Dim: {state_dim})...")
    agent = DRLAgent(state_dim=state_dim, action_dim=action_dim)
    
    # 5. Load the Final Weights
    model_path = "drl_models/best_universal_dqn_trader.pth"
    try:
        agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        agent.eval() 
        logger.info("Universal model weights loaded and locked successfully.")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        return

    # 6. Execute "Sticky" Multi-Asset Allocation Backtest
    engine = BacktestEngine(
        multi_ticker_data=multi_ticker_data, 
        model=agent, 
        start_date="2023-01-01", 
        per_stock_budget=100000.0,
        commission=0.002, 

    )
    engine.run()

if __name__ == "__main__":
    main()