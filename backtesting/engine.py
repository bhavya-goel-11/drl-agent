import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


class BacktestEngine:
    """
    Vectorised Portfolio Backtesting Engine.

    Runs the trained multi-asset D3QN agent through the out-of-sample period,
    building the same state representation as VectorizedTradingEnv and
    executing the action vector (one action per stock) at each timestep.

    Compares aggregate portfolio performance against a Buy & Hold benchmark.
    """

    def __init__(
        self,
        data_3d: np.ndarray,
        tickers: list,
        dates: pd.DatetimeIndex,
        columns: list,
        model,
        n_stocks: int,
        train_end_idx: int,
        initial_balance: float = 10_000_000.0,
        commission: float = 0.002,
    ):
        """
        Args:
            data_3d:        Full aligned (T, N, F) array (train + test).
            tickers:        Ordered list of N ticker strings.
            dates:          Full DatetimeIndex of T dates.
            columns:        Feature column names.
            model:          Trained DRLAgent (in eval mode).
            n_stocks:       Number of tradeable stocks.
            train_end_idx:  Index in `dates` where OOS period begins.
            initial_balance: Portfolio starting capital.
            commission:     Round-trip trading cost fraction.
        """
        self.data = data_3d
        self.tickers = tickers
        self.dates = dates
        self.columns = columns
        self.model = model
        self.n_stocks = n_stocks
        self.train_end_idx = train_end_idx
        self.initial_balance = initial_balance
        self.commission = commission

        self.close_idx = columns.index('close') if 'close' in columns else -1
        self.open_idx  = columns.index('open')  if 'open'  in columns else self.close_idx

        # Portfolio state
        self.cash = initial_balance
        self.holdings = np.zeros(n_stocks, dtype=np.float64)

        # Tracking
        self.portfolio_history = []
        self.benchmark_history = []
        self.trade_ledger = []
        self.oos_dates = []

        # Per-stock tracking for detailed reporting
        self.per_stock_pnl = {t: [] for t in tickers}

        os.makedirs("reports", exist_ok=True)

    def _build_state(self, step: int) -> np.ndarray:
        """Build the same observation vector as VectorizedTradingEnv."""
        frame = self.data[step]  # (N, F)

        lookback_start = max(0, step - 200)
        window = self.data[lookback_start: step + 1]

        rolling_mean = window.mean(axis=0)
        rolling_std  = window.std(axis=0)
        rolling_std  = np.where(rolling_std < 1e-8, 1e-8, rolling_std)

        normalised = (frame - rolling_mean) / rolling_std
        normalised = np.nan_to_num(normalised, nan=0.0)

        market_flat = normalised.flatten()

        close_prices = self.data[step, :, self.close_idx]
        position_vals = (self.holdings * close_prices) / self.initial_balance

        total_equity = self.cash + np.sum(self.holdings * close_prices)
        peak = max(self.initial_balance,
                   max(self.portfolio_history) if self.portfolio_history
                   else self.initial_balance)
        drawdown = (peak - total_equity) / peak if peak > 0 else 0.0

        portfolio_state = np.array([
            self.cash / self.initial_balance,
            total_equity / self.initial_balance,
            drawdown,
        ])

        return np.concatenate([market_flat, position_vals,
                               portfolio_state]).astype(np.float32)

    def _portfolio_value(self, step: int) -> float:
        close_prices = self.data[step, :, self.close_idx]
        return float(self.cash + np.sum(self.holdings * close_prices))

    def _calc_metrics(self, equity_curve):
        """Annualised Sharpe and percentage Max Drawdown."""
        s = pd.Series(equity_curve)
        pct = s.pct_change().dropna()
        if len(pct) == 0 or pct.std() == 0:
            return 0.0, 0.0
        sharpe = (pct.mean() / pct.std()) * np.sqrt(252)
        rolling_max = s.cummax()
        dd = (s - rolling_max) / rolling_max
        max_dd = dd.min() * 100
        return round(sharpe, 2), round(max_dd, 2)

    def run(self):
        """Step through the OOS period and simulate the portfolio."""
        oos_start = self.train_end_idx
        T = self.data.shape[0]

        logger.info(f"Starting vectorised OOS backtest from "
                     f"{self.dates[oos_start].date()} to "
                     f"{self.dates[T-1].date()} "
                     f"({T - oos_start} trading days)")

        # --- Benchmark: Buy & Hold on Day 1 ---
        bench_cash = self.initial_balance
        bench_holdings = np.zeros(self.n_stocks, dtype=np.float64)
        first_prices = self.data[oos_start, :, self.close_idx]
        cash_per_stock = bench_cash / self.n_stocks
        for i in range(self.n_stocks):
            if first_prices[i] > 0:
                shares = int(cash_per_stock // (
                    first_prices[i] * (1 + self.commission)))
                bench_holdings[i] = shares

        # --- OOS simulation ---
        for step in range(oos_start, T - 1):
            today = self.dates[step]
            self.oos_dates.append(today)

            # Portfolio valuation
            pv = self._portfolio_value(step)
            self.portfolio_history.append(pv)

            # Benchmark valuation
            close_prices = self.data[step, :, self.close_idx]
            bench_val = float(np.sum(bench_holdings * close_prices))
            self.benchmark_history.append(bench_val)

            # Inference
            state = self._build_state(step)
            with torch.no_grad():
                device = next(self.model.parameters()).device
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_vals = self.model(state_tensor)  # (1, N, 3)
                actions = q_vals.squeeze(0).argmax(dim=1).cpu().numpy()  # (N,)

            # Execution at next-day open
            next_step = step + 1
            exec_prices = self.data[next_step, :, self.open_idx]

            buy_mask  = (actions == 1) & (self.holdings == 0)
            sell_mask = (actions == 2) & (self.holdings > 0)

            # Sells first
            for i in np.where(sell_mask)[0]:
                revenue = self.holdings[i] * exec_prices[i]
                fee = revenue * self.commission
                self.cash += (revenue - fee)
                self.trade_ledger.append({
                    'Date': self.dates[next_step].date(),
                    'Ticker': self.tickers[i],
                    'Action': 'SELL',
                    'Price': float(exec_prices[i]),
                    'Shares': int(self.holdings[i]),
                })
                self.holdings[i] = 0

            # Buys with equal split
            num_buys = int(buy_mask.sum())
            if num_buys > 0:
                cash_per = self.cash / num_buys
                for i in np.where(buy_mask)[0]:
                    max_cost = exec_prices[i] * (1 + self.commission)
                    if max_cost <= 0:
                        continue
                    shares = int(cash_per // max_cost)
                    if shares > 0:
                        cost = shares * exec_prices[i] * (1 + self.commission)
                        self.cash -= cost
                        self.holdings[i] = shares
                        self.trade_ledger.append({
                            'Date': self.dates[next_step].date(),
                            'Ticker': self.tickers[i],
                            'Action': 'BUY',
                            'Price': float(exec_prices[i]),
                            'Shares': shares,
                        })

        # Final valuation
        final_pv = self._portfolio_value(T - 1)
        self.portfolio_history.append(final_pv)
        final_bench = float(np.sum(
            bench_holdings * self.data[T-1, :, self.close_idx]))
        self.benchmark_history.append(final_bench)
        self.oos_dates.append(self.dates[T - 1])

        self._generate_reports()

    def _generate_reports(self):
        logger.info("Compiling analytics and generating reports...")

        pd.DataFrame(self.trade_ledger).to_csv(
            "reports/detailed_trade_ledger.csv", index=False)

        ai_eq  = np.array(self.portfolio_history)
        bh_eq  = np.array(self.benchmark_history)

        ai_ret  = ((ai_eq[-1] / ai_eq[0]) - 1) * 100
        bh_ret  = ((bh_eq[-1] / bh_eq[0]) - 1) * 100
        ai_sharpe, ai_mdd   = self._calc_metrics(ai_eq)
        bh_sharpe, bh_mdd   = self._calc_metrics(bh_eq)

        self.ai_portfolio_results = {
            'return': round(ai_ret, 2),
            'sharpe': ai_sharpe,
            'max_dd': ai_mdd,
            'bench_return': round(bh_ret, 2),
            'bench_sharpe': bh_sharpe,
            'bench_max_dd': bh_mdd,
        }

        logger.info("\n" + "=" * 80)
        logger.info("PORTFOLIO SUMMARY (Out-of-Sample)")
        logger.info("=" * 80)
        logger.info(f"AI Portfolio Return:     {ai_ret:.2f}%")
        logger.info(f"Buy & Hold Return:       {bh_ret:.2f}%")
        logger.info(f"Alpha Generated:         {ai_ret - bh_ret:.2f}%")
        logger.info("-" * 80)
        logger.info(f"AI Sharpe Ratio:         {ai_sharpe}")
        logger.info(f"B&H Sharpe Ratio:        {bh_sharpe}")
        logger.info("-" * 80)
        logger.info(f"AI Max Drawdown:         {ai_mdd}%")
        logger.info(f"B&H Max Drawdown:        {bh_mdd}%")
        logger.info("-" * 80)
        logger.info(f"Total Trades:            {len(self.trade_ledger)}")
        logger.info("=" * 80 + "\n")

        # --- Plots ---
        sns.set_theme(style="darkgrid")

        plt.figure(figsize=(14, 7))
        plt.plot(self.oos_dates, ai_eq, label='D3QN Portfolio',
                 color='#00ff88', linewidth=2)
        plt.plot(self.oos_dates, bh_eq, label='Buy & Hold',
                 color='#ff3366', linewidth=1.5, alpha=0.8)
        plt.title('OOS Equity Curve — Vectorised D3QN vs Buy & Hold',
                   fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value (₹)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig("reports/equity_curve.png", dpi=300)
        plt.close()

        logger.info("Reports saved to 'reports/' directory.")