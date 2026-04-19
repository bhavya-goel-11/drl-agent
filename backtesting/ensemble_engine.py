"""
Capital-Split Ensemble Engine.

Each constituent model independently manages its own sub-portfolio.
The ensemble equity curve is the sum of all sub-portfolios, which
provides natural diversification across heterogeneous model architectures.

This is the standard academic approach for ensembling trading strategies
with fundamentally different trading frequencies and signal horizons.
"""

import os
import pandas as pd
import numpy as np
from loguru import logger


class EnsembleBacktestEngine:
    """
    Runs each model on an independent capital allocation and aggregates
    the resulting equity curves into a single diversified portfolio.

    Default weights:
        - D3QN Agent:     33.3%
        - XGBoost:        33.3%
        - RandomForest:   33.3%
    """

    def __init__(
        self,
        all_results: dict,
        initial_balance: float = 10_000_000.0,
        weights: dict = None,
    ):
        self.all_results = all_results
        self.initial_balance = initial_balance
        self.model_name = "Ensemble"

        # Models that participate in the ensemble
        self.member_models = [k for k in all_results if k != 'Buy_Hold']

        if weights is None:
            n = len(self.member_models)
            self.weights = {m: 1.0 / n for m in self.member_models}
        else:
            self.weights = weights

        logger.info(f"Ensemble members: {self.member_models}")
        for m, w in self.weights.items():
            logger.info(f"  {m}: {w:.1%} capital (₹{initial_balance * w:,.0f})")

        self.portfolio_history = None
        self.benchmark_history = None
        self.oos_dates = None

        os.makedirs("reports", exist_ok=True)

    def run(self):
        """
        Combine the independently-run equity curves from each model
        into a single diversified portfolio.

        Each model's equity curve is normalised to start at 1.0, then
        scaled by its capital weight, so the ensemble starts at
        initial_balance and reflects the weighted sum of returns.
        """
        ref = self.all_results[self.member_models[0]]
        n_points = len(ref['equity_curve'])
        self.oos_dates = ref['dates']
        self.benchmark_history = list(ref['bench_curve'])

        ensemble_eq = np.zeros(n_points, dtype=np.float64)

        for model_name in self.member_models:
            res = self.all_results[model_name]
            eq = np.array(res['equity_curve'], dtype=np.float64)

            # Normalise to fractional returns from starting value
            normalised = eq / eq[0]

            # Scale by allocated capital
            allocated = self.initial_balance * self.weights[model_name]
            ensemble_eq += normalised * allocated

        self.portfolio_history = list(ensemble_eq)

        # Compute total trades as sum of constituent trades
        self.total_trades = sum(
            self.all_results[m].get('trades', 0) for m in self.member_models
        )

        self._log_summary()

    def _log_summary(self):
        eq = np.array(self.portfolio_history)
        bh = np.array(self.benchmark_history)

        ai_ret = ((eq[-1] / eq[0]) - 1) * 100
        bh_ret = ((bh[-1] / bh[0]) - 1) * 100

        logger.info("\n" + "=" * 80)
        logger.info("ENSEMBLE PORTFOLIO SUMMARY (Capital-Split Diversification)")
        logger.info("=" * 80)
        logger.info(f"Ensemble Return:    {ai_ret:.2f}%")
        logger.info(f"Buy & Hold Return:  {bh_ret:.2f}%")
        logger.info(f"Alpha Generated:    {ai_ret - bh_ret:.2f}%")
        logger.info("-" * 80)
        for m in self.member_models:
            r = self.all_results[m]
            logger.info(f"  {m:<16} → {r['total_return']:>7.2f}% return, "
                        f"weight {self.weights[m]:.1%}")
        logger.info("=" * 80)

    def get_metrics(self):
        eq = np.array(self.portfolio_history)
        bh = np.array(self.benchmark_history)
        s = pd.Series(eq)
        pct = s.pct_change().dropna()

        total_return = ((eq[-1] / eq[0]) - 1) * 100
        bench_return = ((bh[-1] / bh[0]) - 1) * 100
        alpha = total_return - bench_return

        sharpe = 0.0
        if len(pct) > 1 and pct.std() > 0:
            sharpe = (pct.mean() / pct.std()) * np.sqrt(252)

        rolling_max = s.cummax()
        dd = (s - rolling_max) / rolling_max
        max_dd = dd.min() * 100

        annual_return = total_return / 3.0
        calmar = annual_return / abs(max_dd) if abs(max_dd) > 0.01 else 0.0

        downside = pct[pct < 0]
        sortino = 0.0
        if len(downside) > 0 and downside.std() > 0:
            sortino = (pct.mean() / downside.std()) * np.sqrt(252)

        win_rate = (pct > 0).mean() * 100
        volatility = pct.std() * np.sqrt(252) * 100

        return {
            'model_name': 'Ensemble',
            'total_return': round(total_return, 2),
            'bench_return': round(bench_return, 2),
            'alpha': round(alpha, 2),
            'sharpe': round(sharpe, 2),
            'max_drawdown': round(max_dd, 2),
            'calmar': round(calmar, 2),
            'sortino': round(sortino, 2),
            'win_rate': round(win_rate, 2),
            'volatility': round(volatility, 2),
            'trades': self.total_trades,
        }
