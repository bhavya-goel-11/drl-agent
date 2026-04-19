import pandas as pd
import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import warnings

# Suppress pandas-ta warnings for clean logs
warnings.filterwarnings("ignore", category=RuntimeWarning) 

from database.connection import SessionLocal
from database.models import HistoricalData
from data_pipeline.features import FeatureEngineer, align_multi_ticker_data
from drl_models.agent import DRLAgent
from backtesting.engine import BacktestEngine
from backtesting.ml_baselines import MLBaselineTrader
from backtesting.ensemble_engine import EnsembleBacktestEngine

TRAIN_CUTOFF = "2023-01-01"

def _generate_comparison_report(all_results: dict, output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Metrics Summary Table
    rows = []
    for name, res in all_results.items():
        rows.append({
            'Model': name,
            'Return (%)': res['total_return'],
            'Alpha (%)': res['alpha'],
            'Sharpe': res['sharpe'],
            'Max DD (%)': res['max_drawdown'],
            'Calmar': res['calmar'],
            'Sortino': res['sortino'],
            'Win Rate (%)': res['win_rate'],
            'Volatility (%)': res['volatility'],
            'Trades': res['trades'],
        })
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(f"{output_dir}/model_comparison_metrics.csv", index=False)

    metrics_json = {name: {k: v for k, v in res.items() if k not in ('equity_curve', 'bench_curve', 'dates')}
                    for name, res in all_results.items()}
    with open(f"{output_dir}/model_comparison_metrics.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)

    logger.info("\n" + "=" * 90)
    logger.info("COMPARATIVE MODEL PERFORMANCE (Out-of-Sample)")
    logger.info("=" * 90)
    for _, row in metrics_df.iterrows():
        logger.info(
            f"  {row['Model']:<18} | Return: {row['Return (%)']:>7.2f}% | "
            f"Alpha: {row['Alpha (%)']:>7.2f}% | Sharpe: {row['Sharpe']:>5.2f} | "
            f"MaxDD: {row['Max DD (%)']:>7.2f}% | Trades: {row['Trades']:>4}"
        )
    logger.info("=" * 90)

    # 2. Cumulative Equity Curve
    sns.set_theme(style="darkgrid")
    colors = {
        'D3QN_Agent': '#00ff88',
        'XGBoost': '#ff9900',
        'RandomForest': '#3399ff',
        'Ensemble': '#ffd700',
    }

    fig, ax = plt.subplots(figsize=(16, 8))

    first_res = list(all_results.values())[0]
    if 'bench_curve' in first_res and first_res['bench_curve'] is not None:
        ax.plot(first_res['dates'], first_res['bench_curve'],
                label='Buy & Hold', color='#ff3366', linewidth=1.5, alpha=0.7, linestyle='--')

    for name, res in all_results.items():
        if 'equity_curve' not in res: continue
        color = colors.get(name, '#888888')
        lw = 3.0 if name == 'Ensemble' else 2.0
        ax.plot(res['dates'], res['equity_curve'], label=name, color=color, linewidth=lw)

    ax.set_title('OOS Equity Curves — DRL Agent vs ML Baselines vs Ensemble', fontsize=15, fontweight='bold')
    ax.set_ylabel('Portfolio Value (₹)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x/1e7:.1f}Cr'))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparative_equity_curves.png", dpi=300)
    plt.close()

def main():
    logger.info("=" * 80)
    logger.info("Initialising Vectorised Portfolio Evaluation Pipeline")
    logger.info("=" * 80)

    db = SessionLocal()
    try:
        logger.info("Fetching complete historical data for all tickers...")
        records = (db.query(HistoricalData).order_by(HistoricalData.date.asc()).all())
        if not records:
            logger.error("No data found in database!")
            return
        data = [{'symbol': r.symbol, 'date': r.date, 'open': float(r.open),
                 'high': float(r.high), 'low': float(r.low), 'close': float(r.close),
                 'volume': float(r.volume)} for r in records]
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    except Exception as e:
        logger.error(f"Database error: {e}")
        return
    finally:
        db.close()

    engineer = FeatureEngineer()
    engineered_df = engineer.add_technical_indicators(df)

    multi_ticker_data = {}
    for ticker, group in engineered_df.groupby('symbol'):
        multi_ticker_data[ticker] = group.copy()
    data_3d, tickers, dates, columns = align_multi_ticker_data(multi_ticker_data)
    n_stocks = len(tickers)
    n_features = len(columns)
    logger.info(f"Aligned {n_stocks} tickers × {n_features} features × {data_3d.shape[0]} timesteps")

    cutoff = pd.to_datetime(TRAIN_CUTOFF)
    train_end_idx = int(np.searchsorted(dates, cutoff))

    # 1. D3QN Backtest
    obs_dim = n_stocks * n_features + n_stocks + 3
    agent = DRLAgent(state_dim=obs_dim, action_dim=3, n_stocks=n_stocks)
    
    try:
        checkpoint = torch.load("drl_models/best_universal_dqn_trader.pth", map_location='cpu')
        agent.load_state_dict(checkpoint['policy_net'] if 'policy_net' in checkpoint else checkpoint)
        agent.eval()
    except Exception as e:
        logger.error(f"Failed to load D3QN weights: {e}")
        return

    logger.info("\n" + "=" * 80)
    logger.info("Running Vectorised OOS Backtest (D3QN Agent)")
    logger.info("=" * 80)
    d3qn_engine = BacktestEngine(
        data_3d=data_3d, tickers=tickers, dates=dates, columns=columns,
        model=agent, n_stocks=n_stocks, train_end_idx=train_end_idx,
        initial_balance=10_000_000.0, commission=0.002,
    )
    d3qn_engine.run()

    # 2. ML Baselines
    logger.info("\n" + "=" * 80)
    logger.info("Training & Backtesting ML Baselines (XGBoost & Random Forest)")
    logger.info("=" * 80)
    ml_trader = MLBaselineTrader(
        data_3d=data_3d, tickers=tickers, dates=dates, columns=columns,
        train_end_idx=train_end_idx, initial_balance=10_000_000.0, commission=0.002,
    )
    all_results = ml_trader.run_full_comparison(
        drl_equity=np.array(d3qn_engine.portfolio_history), drl_dates=d3qn_engine.oos_dates
    )

    # 3. Ensemble Engine
    logger.info("\n" + "=" * 80)
    logger.info("Running Capital-Split Ensemble Engine")
    logger.info("=" * 80)
    
    # Ensure all_results has D3QN_Agent
    if 'D3QN_Agent' not in all_results:
        drl_metrics = d3qn_engine.get_metrics() if hasattr(d3qn_engine, 'get_metrics') else {}
        drl_metrics['equity_curve'] = d3qn_engine.portfolio_history
        drl_metrics['bench_curve'] = d3qn_engine.benchmark_history
        drl_metrics['dates'] = d3qn_engine.oos_dates
        all_results['D3QN_Agent'] = drl_metrics
        
    ensemble_engine = EnsembleBacktestEngine(
        all_results=all_results,
        initial_balance=10_000_000.0,
    )
    ensemble_engine.run()

    ensemble_metrics = ensemble_engine.get_metrics()
    ensemble_metrics['equity_curve'] = ensemble_engine.portfolio_history
    ensemble_metrics['bench_curve'] = ensemble_engine.benchmark_history
    ensemble_metrics['dates'] = ensemble_engine.oos_dates
    all_results['Ensemble'] = ensemble_metrics

    _generate_comparison_report(all_results)
    logger.info("\n✅ Full evaluation pipeline complete. Check 'reports/' for outputs.")

if __name__ == "__main__":
    main()