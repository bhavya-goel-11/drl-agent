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

TRAIN_CUTOFF = "2023-01-01"


def _generate_comparison_report(all_results: dict, output_dir: str = "reports"):
    """
    Generate comparative equity curve, metrics table, and per-model reports.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Metrics Summary Table ──────────────────────────────────────
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

    # Save as JSON too for programmatic access
    metrics_json = {name: {k: v for k, v in res.items()
                           if k not in ('equity_curve', 'bench_curve', 'dates')}
                    for name, res in all_results.items()}
    with open(f"{output_dir}/model_comparison_metrics.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)

    # ── 2. Print summary ──────────────────────────────────────────────
    logger.info("\n" + "=" * 90)
    logger.info("COMPARATIVE MODEL PERFORMANCE (Out-of-Sample)")
    logger.info("=" * 90)
    for _, row in metrics_df.iterrows():
        logger.info(
            f"  {row['Model']:<16} | Return: {row['Return (%)']:>7.2f}% | "
            f"Alpha: {row['Alpha (%)']:>7.2f}% | Sharpe: {row['Sharpe']:>5.2f} | "
            f"MaxDD: {row['Max DD (%)']:>7.2f}% | Trades: {row['Trades']:>4}"
        )
    logger.info("=" * 90)

    # ── 3. Cumulative Equity Curve (all models) ───────────────────────
    sns.set_theme(style="darkgrid")
    colors = {
        'D3QN_Agent': '#00ff88',
        'XGBoost': '#ff9900',
        'RandomForest': '#3399ff',
        'Buy_Hold': '#ff3366',
    }

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot Buy & Hold baseline first (from any model's bench_curve)
    first_res = list(all_results.values())[0]
    if 'bench_curve' in first_res and first_res['bench_curve'] is not None:
        bench_dates = first_res['dates']
        bench_eq = first_res['bench_curve']
        ax.plot(bench_dates, bench_eq,
                label='Buy & Hold', color=colors.get('Buy_Hold', '#ff3366'),
                linewidth=1.5, alpha=0.7, linestyle='--')

    # Plot each model
    for name, res in all_results.items():
        if 'equity_curve' not in res:
            continue
        dates = res['dates']
        eq = res['equity_curve']
        color = colors.get(name, '#888888')
        lw = 2.5 if name == 'D3QN_Agent' else 1.8
        ax.plot(dates, eq, label=name, color=color, linewidth=lw)

    ax.set_title('OOS Equity Curves — DRL Agent vs ML Baselines vs Buy & Hold',
                 fontsize=15, fontweight='bold')
    ax.set_ylabel('Portfolio Value (₹)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f'₹{x/1e7:.1f}Cr'))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparative_equity_curves.png", dpi=300)
    plt.close()

    # ── 4. Per-model individual reports ───────────────────────────────
    for name, res in all_results.items():
        if 'equity_curve' not in res:
            continue

        eq = np.array(res['equity_curve'])
        dates = res['dates']

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Equity curve
        axes[0].plot(dates, eq, color=colors.get(name, '#00ff88'), linewidth=2)
        axes[0].set_title(f'{name} — OOS Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value (₹)')
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f'₹{x/1e7:.1f}Cr'))

        # Drawdown curve
        s = pd.Series(eq, index=dates)
        rolling_max = s.cummax()
        drawdown = (s - rolling_max) / rolling_max * 100
        axes[1].fill_between(dates, drawdown, 0, color='red', alpha=0.3)
        axes[1].plot(dates, drawdown, color='red', linewidth=1)
        axes[1].set_title('Drawdown (%)', fontsize=12)
        axes[1].set_ylabel('Drawdown %')

        plt.tight_layout()
        safe_name = name.replace(' ', '_')
        plt.savefig(f"{output_dir}/{safe_name}_detailed_report.png", dpi=300)
        plt.close()

    logger.info(f"All reports saved to '{output_dir}/' directory.")


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
    # 5. Run DRL Agent Backtest
    # ------------------------------------------------------------------
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

    logger.info("\n" + "=" * 80)
    logger.info("Running Vectorised OOS Backtest (DRL Agent)")
    logger.info("=" * 80)

    engine = BacktestEngine(
        data_3d=data_3d, tickers=tickers, dates=dates, columns=columns,
        model=agent, n_stocks=n_stocks, train_end_idx=train_end_idx,
        initial_balance=10_000_000.0, commission=0.002,
    )
    engine.run()

    # Extract DRL equity for comparison
    drl_equity = np.array(engine.portfolio_history)
    drl_dates = engine.oos_dates

    # ------------------------------------------------------------------
    # 6. Train & Backtest ML Baselines
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("Training & Backtesting ML Baselines")
    logger.info("=" * 80)

    ml_trader = MLBaselineTrader(
        data_3d=data_3d, tickers=tickers, dates=dates,
        columns=columns, train_end_idx=train_end_idx,
        initial_balance=10_000_000.0, commission=0.002,
    )

    all_results = ml_trader.run_full_comparison(
        drl_equity=drl_equity, drl_dates=drl_dates
    )

    # ------------------------------------------------------------------
    # 7. Generate Comparative Reports
    # ------------------------------------------------------------------
    _generate_comparison_report(all_results)

    logger.info("\n✅ Full evaluation pipeline complete. Check 'reports/' for outputs.")


if __name__ == "__main__":
    main()