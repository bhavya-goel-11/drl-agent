import pandas as pd
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
    logger.info("Initializing Out-of-Sample Backtest Pipeline...")
    
    # 1. Fetch SPY data from TimescaleDB
    db = SessionLocal()
    try:
        logger.info("Fetching complete SPY historical data...")
        records = db.query(HistoricalData).filter(HistoricalData.symbol == "SPY").order_by(HistoricalData.date.asc()).all()
        
        if not records:
            logger.error("No data found in database!")
            return
            
        data = [{
            'date': r.date, 'open': r.open, 'high': r.high, 
            'low': r.low, 'close': r.close, 'volume': r.volume
        } for r in records]
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        return
    finally:
        db.close()

    # 2. Engineer Features
    engineer = FeatureEngineer()
    engineered_df = engineer.add_technical_indicators(df)
    engineered_df.dropna(inplace=True)
    
    # 3. Slice the Out-of-Sample Data (with a 200-day Burn-In Runway)
    # We grab data starting from March 2024. 
    # 200 trading days later lands us right around January 2025 to begin trading.
    runway_start = "2024-03-01"
    test_df = engineered_df[engineered_df.index >= runway_start].copy()
    logger.info(f"Out-of-Sample Data Prepared: {len(test_df)} rows (Starting {runway_start} for burn-in)")

    # 4. Initialize Model Architecture & Dimensions
    state_dim = test_df.shape[1] + 3 # Features + (balance, shares, net_worth)
    action_dim = 3                   # Hold, Buy, Sell
    
    logger.info(f"Loading trained neural network weights (State Dim: {state_dim})...")
    agent = DRLAgent(state_dim=state_dim, action_dim=action_dim)
    
    # 5. Load the High-Water Mark Weights
    model_path = "drl_models/best_dqn_trader.pth"
    try:
        # Load directly to CPU to avoid any device mismatch errors during backtesting
        agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        agent.eval() # Freeze weights, disable dropout/batchnorm for pure inference
        logger.info("Model weights loaded and locked successfully.")
    except Exception as e:
        logger.error(f"Failed to load model weights at {model_path}: {e}")
        return

    # 6. Execute the Backtest
    # Initial balance matches the training environment ($10,000)
    engine = BacktestEngine(data=test_df, model=agent, symbol="SPY", initial_balance=10000.0)
    engine.run()

if __name__ == "__main__":
    main()