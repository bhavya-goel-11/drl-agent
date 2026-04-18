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
from backtesting.ml_baselines import MLBaselineTrader

def main():
    logger.info("=" * 80)
    logger.info("Initializing Comprehensive Model Evaluation Pipeline")
    logger.info("  DRL (D3QN) vs Traditional ML (GradientBoosting, RandomForest)")
    logger.info("=" * 80)
    
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
    
    logger.info(f"Detected {num_features} indicators. Initializing D3QN Network (State Dim: {state_dim})...")
    agent = DRLAgent(state_dim=state_dim, action_dim=action_dim)
    
    # 5. Load the Final Weights
    model_path = "drl_models/best_universal_dqn_trader.pth"
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
            agent.load_state_dict(checkpoint['policy_net'])
        else:
            agent.load_state_dict(checkpoint)
        agent.eval() 
        logger.info("D3QN model weights loaded and locked successfully.")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        return

    # =========================================================================
    # PART A: DRL Agent Backtest
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PART A: DRL Agent (D3QN) Out-of-Sample Backtest")
    logger.info("=" * 80)
    
    engine = BacktestEngine(
        multi_ticker_data=multi_ticker_data, 
        model=agent, 
        start_date="2023-01-01", 
        per_stock_budget=100000.0,
        commission=0.002, 
    )
    engine.run()

    # =========================================================================
    # PART B: Traditional ML Baseline Comparison
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PART B: Traditional ML Baseline Models")
    logger.info("=" * 80)
    
    ml_trader = MLBaselineTrader(train_cutoff="2023-01-01")
    ml_trader.train(multi_ticker_data)
    
    ml_results = []
    for model_name in ['gradient_boosting', 'random_forest']:
        logger.info(f"\nBacktesting {model_name.replace('_', ' ').title()}...")
        result = ml_trader.backtest_single_model(
            model_name=model_name,
            multi_ticker_data=multi_ticker_data,
            start_date="2023-01-01",
            per_stock_budget=100000.0,
            commission=0.002,
        )
        ml_results.append(result)
        
        logger.info(f"  {model_name}: Return={result['portfolio_return']:.2f}% | "
                     f"Sharpe={result['portfolio_sharpe']:.2f} | "
                     f"MaxDD={result['portfolio_max_dd']:.2f}% | "
                     f"Trades={result['trades']}")
    
    # =========================================================================
    # PART C: Comparative Report
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PART C: Comparative Analysis — DRL vs Traditional ML")
    logger.info("=" * 80)
    
    engine.generate_comparative_report(ml_results)
    
    logger.info("\n✅ Full evaluation pipeline complete. Check 'reports/' for all outputs.")

if __name__ == "__main__":
    main()