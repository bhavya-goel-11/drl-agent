"""
Traditional ML Baselines for Comparative Analysis.

Implements Gradient Boosting (XGBoost) and Random Forest classifiers
to generate Buy/Sell/Hold signals, then backtests them on the same
out-of-sample period as the DRL agent.

This demonstrates where DRL shines (non-stationary, chaotic regimes)
vs. where traditional ML may suffice (stable trending markets).
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from loguru import logger
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. Using sklearn GradientBoosting as fallback.")


class MLBaselineTrader:
    """
    Trains traditional ML models on the same aligned (T, N, F) tensor
    used by the DRL agent and backtests them on the identical OOS period.
    """

    def __init__(self, data_3d: np.ndarray, tickers: list,
                 dates: pd.DatetimeIndex, columns: list,
                 train_end_idx: int,
                 initial_balance: float = 10_000_000.0,
                 commission: float = 0.002,
                 forward_window: int = 5,
                 buy_threshold: float = 0.02,
                 sell_threshold: float = -0.02):
        self.data = data_3d          # (T, N, F)
        self.tickers = tickers
        self.dates = dates
        self.columns = columns
        self.train_end_idx = train_end_idx
        self.initial_balance = initial_balance
        self.commission = commission
        self.forward_window = forward_window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.n_stocks = len(tickers)

        self.close_idx = columns.index('close') if 'close' in columns else -1
        self.open_idx = columns.index('open') if 'open' in columns else self.close_idx

        self.models = {}
        self.scaler = None
        self.feature_indices = None  # indices into columns to use as features

    # ------------------------------------------------------------------
    # Label & Feature Construction
    # ------------------------------------------------------------------
    def _build_training_data(self):
        """
        Build a pooled (samples, features) matrix from the training portion
        of the 3D tensor, with forward-return labels.
        """
        T_train = self.train_end_idx
        close_data = self.data[:T_train, :, self.close_idx]  # (T_train, N)

        # Forward returns per stock
        fwd = np.full_like(close_data, np.nan)
        w = self.forward_window
        for t in range(T_train - w):
            fwd[t] = (close_data[t + w] - close_data[t]) / (close_data[t] + 1e-12)

        # Labels: 0=Hold, 1=Buy, 2=Sell
        labels = np.zeros_like(fwd, dtype=np.int32)
        labels[fwd > self.buy_threshold] = 1
        labels[fwd < self.sell_threshold] = 2

        # Features: use all F columns at each (t, stock) as one sample
        # Skip first 200 rows for indicator warm-up and last `w` for labels
        start = 200
        end = T_train - w

        all_X = []
        all_y = []
        for t in range(start, end):
            for j in range(self.n_stocks):
                row = self.data[t, j, :]  # (F,)
                if np.any(np.isnan(row)):
                    continue
                all_X.append(row)
                all_y.append(labels[t, j])

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int32)

        logger.info(f"Training data: {X.shape[0]:,} samples × {X.shape[1]} features")
        unique, counts = np.unique(y, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        logger.info(f"Label distribution: {dist}")

        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self):
        """Train Gradient Boosting and Random Forest on pre-cutoff data."""
        logger.info("=" * 70)
        logger.info("Training Traditional ML Baselines...")
        logger.info("=" * 70)

        X, y = self._build_training_data()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # --- Model 1: Gradient Boosting ---
        if HAS_XGBOOST:
            logger.info("Training XGBoost Gradient Boosting...")
            gb_model = xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                reg_alpha=0.1, reg_lambda=1.0, eval_metric='mlogloss',
                random_state=42, n_jobs=-1, verbosity=0,
                tree_method='hist', device='cuda',
            )
        else:
            logger.info("Training Sklearn Gradient Boosting...")
            gb_model = GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=20, random_state=42,
            )

        gb_model.fit(X_scaled, y)
        self.models['XGBoost'] = gb_model
        acc = (gb_model.predict(X_scaled) == y).mean()
        logger.info(f"  ✓ Gradient Boosting trained — train accuracy: {acc:.2%}")

        # --- Model 2: Random Forest ---
        logger.info("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=20,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1,
        )
        rf_model.fit(X_scaled, y)
        self.models['RandomForest'] = rf_model
        acc = (rf_model.predict(X_scaled) == y).mean()
        logger.info(f"  ✓ Random Forest trained — train accuracy: {acc:.2%}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _predict_actions(self, model_name: str, step: int) -> np.ndarray:
        """Predict actions for all N stocks at a given timestep."""
        model = self.models[model_name]
        features = self.data[step, :, :]  # (N, F)

        X = self.scaler.transform(features)
        preds = model.predict(X)  # (N,) with values in {0, 1, 2}
        return preds.astype(np.int64)

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------
    def backtest(self, model_name: str) -> dict:
        """
        Run a vectorised backtest identical to BacktestEngine but driven
        by ML predictions instead of DRL Q-values.
        """
        logger.info(f"Backtesting {model_name}...")

        oos_start = self.train_end_idx
        T = self.data.shape[0]

        cash = self.initial_balance
        holdings = np.zeros(self.n_stocks, dtype=np.float64)

        # Buy & Hold benchmark
        bench_holdings = np.zeros(self.n_stocks, dtype=np.float64)
        first_prices = self.data[oos_start, :, self.close_idx]
        cash_per = self.initial_balance / self.n_stocks
        for i in range(self.n_stocks):
            if first_prices[i] > 0:
                bench_holdings[i] = int(cash_per // (
                    first_prices[i] * (1 + self.commission)))

        portfolio_history = []
        benchmark_history = []
        oos_dates = []
        trade_count = 0

        for step in range(oos_start, T - 1):
            oos_dates.append(self.dates[step])

            # Valuation
            close_prices = self.data[step, :, self.close_idx]
            pv = float(cash + np.sum(holdings * close_prices))
            portfolio_history.append(pv)
            bench_val = float(np.sum(bench_holdings * close_prices))
            benchmark_history.append(bench_val)

            # Predict
            actions = self._predict_actions(model_name, step)

            # Execute at next-day open
            next_step = step + 1
            exec_prices = self.data[next_step, :, self.open_idx]

            buy_mask = (actions == 1) & (holdings == 0)
            sell_mask = (actions == 2) & (holdings > 0)

            # Sell first
            for i in np.where(sell_mask)[0]:
                revenue = holdings[i] * exec_prices[i]
                fee = revenue * self.commission
                cash += (revenue - fee)
                holdings[i] = 0
                trade_count += 1

            # Buy with equal split
            num_buys = int(buy_mask.sum())
            if num_buys > 0:
                cash_per_stock = cash / num_buys
                for i in np.where(buy_mask)[0]:
                    max_cost = exec_prices[i] * (1 + self.commission)
                    if max_cost <= 0:
                        continue
                    shares = int(cash_per_stock // max_cost)
                    if shares > 0:
                        cost = shares * exec_prices[i] * (1 + self.commission)
                        cash -= cost
                        holdings[i] = shares
                        trade_count += 1

        # Final valuation
        final_close = self.data[T - 1, :, self.close_idx]
        final_pv = float(cash + np.sum(holdings * final_close))
        portfolio_history.append(final_pv)
        final_bench = float(np.sum(bench_holdings * final_close))
        benchmark_history.append(final_bench)
        oos_dates.append(self.dates[T - 1])

        # Compute metrics
        eq = np.array(portfolio_history)
        bh = np.array(benchmark_history)
        metrics = self._compute_metrics(eq, bh, model_name, trade_count)
        metrics['equity_curve'] = eq
        metrics['bench_curve'] = bh
        metrics['dates'] = oos_dates

        return metrics

    def _compute_metrics(self, equity: np.ndarray, bench: np.ndarray,
                         model_name: str, trades: int) -> dict:
        """Compute comprehensive risk/return metrics."""
        s = pd.Series(equity)
        pct = s.pct_change().dropna()

        total_return = ((equity[-1] / equity[0]) - 1) * 100
        bench_return = ((bench[-1] / bench[0]) - 1) * 100
        alpha = total_return - bench_return

        # Sharpe
        sharpe = 0.0
        if len(pct) > 1 and pct.std() > 0:
            sharpe = (pct.mean() / pct.std()) * np.sqrt(252)

        # Max Drawdown
        rolling_max = s.cummax()
        dd = (s - rolling_max) / rolling_max
        max_dd = dd.min() * 100

        # Calmar Ratio
        annual_return = total_return / 3.0  # ~3 year OOS
        calmar = annual_return / abs(max_dd) if abs(max_dd) > 0.01 else 0.0

        # Sortino Ratio
        downside = pct[pct < 0]
        sortino = 0.0
        if len(downside) > 0 and downside.std() > 0:
            sortino = (pct.mean() / downside.std()) * np.sqrt(252)

        # Win Rate (daily)
        win_rate = (pct > 0).mean() * 100

        # Volatility (annualised)
        volatility = pct.std() * np.sqrt(252) * 100

        return {
            'model_name': model_name,
            'total_return': round(total_return, 2),
            'bench_return': round(bench_return, 2),
            'alpha': round(alpha, 2),
            'sharpe': round(sharpe, 2),
            'max_drawdown': round(max_dd, 2),
            'calmar': round(calmar, 2),
            'sortino': round(sortino, 2),
            'win_rate': round(win_rate, 2),
            'volatility': round(volatility, 2),
            'trades': trades,
        }

    # ------------------------------------------------------------------
    # Full comparison pipeline
    # ------------------------------------------------------------------
    def run_full_comparison(self, drl_equity: np.ndarray = None,
                            drl_dates: list = None) -> dict:
        """
        Train all ML models, backtest each, and return results dict
        including equity curves for the comparative chart.
        """
        self.train()

        results = {}
        for model_name in self.models:
            results[model_name] = self.backtest(model_name)

        # Attach DRL results if provided
        if drl_equity is not None and drl_dates is not None:
            bench = results[list(results.keys())[0]]['bench_curve']
            drl_trade_count = 0
            drl_ledger_path = "reports/detailed_trade_ledger.csv"
            if os.path.exists(drl_ledger_path):
                try:
                    drl_trades_df = pd.read_csv(drl_ledger_path)
                    if 'Action' in drl_trades_df.columns:
                        drl_trade_count = int(
                            len(drl_trades_df[drl_trades_df['Action'] != 'HOLD'])
                        )
                    else:
                        drl_trade_count = int(len(drl_trades_df))
                except Exception as e:
                    logger.warning(f"Failed to load DRL trade ledger: {e}")

            drl_metrics = self._compute_metrics(
                drl_equity, bench, 'D3QN_Agent',
                trades=drl_trade_count
            )
            drl_metrics['equity_curve'] = drl_equity
            drl_metrics['bench_curve'] = bench
            drl_metrics['dates'] = drl_dates
            results['D3QN_Agent'] = drl_metrics

        return results
