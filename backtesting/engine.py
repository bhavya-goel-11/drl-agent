import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

class BacktestEngine:
    """
    Comprehensive Quantitative Reporting Suite.
    Compares Independent AI Agent performance against a Buy & Hold Benchmark 
    across all 47 tickers with risk-adjusted metrics (Sharpe, MaxDD).
    """
    
    def __init__(self, multi_ticker_data: dict, model, start_date: str = "2023-01-01", 
                 per_stock_budget=1000000.0, commission=0.002):
        self.data_dict = multi_ticker_data
        self.model = model
        self.budget = per_stock_budget
        self.commission = commission
        self.start_date = pd.to_datetime(start_date)
        
        # Tracking independent accounts and training "Data Density"
        self.accounts = {t: {"balance": per_stock_budget, "shares": 0, "history": []} for t in multi_ticker_data.keys()}
        self.benchmarks = {t: {"shares": 0, "history": []} for t in multi_ticker_data.keys()}
        self.data_density = {t: len(df[df.index < self.start_date]) for t, df in multi_ticker_data.items()}
        
        self.trade_ledger = []
        self.dates = []

        # Build master timeline
        all_dates = pd.DatetimeIndex([])
        for df in self.data_dict.values():
            filtered = df[df.index >= self.start_date]
            all_dates = all_dates.union(filtered.index)
        self.trading_dates = all_dates.sort_values().unique()
        
        # Ensure reporting directory exists
        os.makedirs("reports", exist_ok=True)

    def _build_state(self, ticker: str, current_date: pd.Timestamp, total_net_worth: float, current_price: float) -> np.ndarray:
        df = self.data_dict[ticker]
        if current_date not in df.index: return None
        
        try:
            idx = df.index.get_loc(current_date)
            if isinstance(idx, (slice, np.ndarray)):
                idx = np.where(df.index == current_date)[0][0]
                
            frame = df.iloc[idx]
            # Ensure 200-day window for Z-Score normalization
            if idx < 200: return None
            
            window = df.iloc[idx - 200 : idx + 1]
            norm_indicators = np.nan_to_num(((frame - window.mean()) / window.std().replace(0, 1e-8)).values, nan=0.0)
            
            # Match training state exactly
            norm_shares = (self.accounts[ticker]["shares"] * current_price) / self.budget
            return np.append(norm_indicators, [self.accounts[ticker]["balance"]/self.budget, norm_shares, total_net_worth/self.budget]).astype(np.float32)
        except:
            return None

    def _calc_metrics(self, equity_curve):
        """Calculates Annualized Sharpe and Percentage Max Drawdown."""
        s = pd.Series(equity_curve)
        pct_change = s.pct_change().dropna()
        if len(pct_change) == 0 or pct_change.std() == 0: return 0.0, 0.0
        
        sharpe = (pct_change.mean() / pct_change.std()) * np.sqrt(252)
        rolling_max = s.cummax()
        drawdown = (s - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        return round(sharpe, 2), round(max_dd, 2)

    def run(self):
        logger.info(f"Starting Detailed Independent Backtest (₹{self.budget} per stock)...")
        
        # Initialize Benchmark: Buy and Hold on Day 1
        for t, df_t in self.data_dict.items():
            first_valid_date = df_t[df_t.index >= self.start_date].index.min()
            if pd.notna(first_valid_date):
                first_price = float(df_t.loc[first_valid_date, 'close'])
                self.benchmarks[t]["shares"] = self.budget / (first_price * (1 + self.commission))

        for i, today in enumerate(self.trading_dates):
            if i >= len(self.trading_dates) - 1: break 
            tomorrow = self.trading_dates[i+1]
            self.dates.append(today)
            
            for t in self.data_dict.keys():
                acc = self.accounts[t]
                bench = self.benchmarks[t]
                df_t = self.data_dict[t]
                
                if today not in df_t.index: 
                    # Forward fill last known value for consistent plotting
                    last_val = acc["history"][-1] if acc["history"] else self.budget
                    acc["history"].append(last_val)
                    last_bench = bench["history"][-1] if bench["history"] else self.budget
                    bench["history"].append(last_bench)
                    continue
                
                # 1. Daily Valuation
                price_today = float(df_t.loc[today, 'close'])
                net_worth = acc["balance"] + (acc["shares"] * price_today)
                acc["history"].append(net_worth)
                
                bench_worth = bench["shares"] * price_today if bench["shares"] > 0 else self.budget
                bench["history"].append(bench_worth)
                
                # 2. Inference
                state = self._build_state(t, today, net_worth, price_today)
                if state is None: continue
                
                with torch.no_grad():
                    q_vals = self.model(torch.FloatTensor(state).unsqueeze(0)).squeeze(0)
                    action = torch.argmax(q_vals).item()

                # 3. Execution
                if tomorrow not in df_t.index: continue
                price_tmrw = float(df_t.loc[tomorrow, 'open'])

                if action == 2 and acc["shares"] > 0: # SELL
                    val = acc["shares"] * price_tmrw
                    comm = val * self.commission
                    acc["balance"] += (val - comm)
                    self.trade_ledger.append({'Date': tomorrow.date(), 'Ticker': t, 'Action': 'SELL', 'Price': price_tmrw, 'Shares': acc["shares"]})
                    acc["shares"] = 0
                    
                elif action == 1 and acc["shares"] == 0: # BUY
                    shares = int(acc["balance"] // (price_tmrw * (1 + self.commission)))
                    if shares > 0:
                        cost = (shares * price_tmrw) * (1 + self.commission)
                        acc["balance"] -= cost
                        acc["shares"] = shares
                        self.trade_ledger.append({'Date': tomorrow.date(), 'Ticker': t, 'Action': 'BUY', 'Price': price_tmrw, 'Shares': shares})

        self._generate_reports()

    def _generate_reports(self):
        logger.info("Compiling final analytics and generating visual reports...")
        
        # 1. Export Trade Ledger
        pd.DataFrame(self.trade_ledger).to_csv("reports/detailed_trade_ledger.csv", index=False)
        
        results = []
        agg_ai_equity = np.zeros(len(self.dates))
        agg_bench_equity = np.zeros(len(self.dates))
        
        for t in self.data_dict.keys():
            # Exclude Index from individual return stats
            if t == '^NSEI' or not self.accounts[t]["history"]: continue
            
            ai_hist = self.accounts[t]["history"]
            bench_hist = self.benchmarks[t]["history"]
            
            # Align history lengths
            if len(ai_hist) < len(self.dates):
                ai_hist = [self.budget] * (len(self.dates) - len(ai_hist)) + ai_hist
                bench_hist = [self.budget] * (len(self.dates) - len(bench_hist)) + bench_hist

            agg_ai_equity += np.array(ai_hist)
            agg_bench_equity += np.array(bench_hist)
            
            ai_ret = ((ai_hist[-1] / self.budget) - 1) * 100
            bench_ret = ((bench_hist[-1] / self.budget) - 1) * 100
            
            ai_sharpe, ai_mdd = self._calc_metrics(ai_hist)
            bench_sharpe, bench_mdd = self._calc_metrics(bench_hist)
            
            results.append({
                "Ticker": t, 
                "AI_Return (%)": round(ai_ret, 2), 
                "Bench_Return (%)": round(bench_ret, 2), 
                "Alpha (%)": round(ai_ret - bench_ret, 2),
                "AI_Sharpe": ai_sharpe, 
                "Bench_Sharpe": bench_sharpe,
                "AI_MaxDD (%)": ai_mdd, 
                "Bench_MaxDD (%)": bench_mdd,
                "Train_Rows": self.data_density[t]
            })

        # 2. Console Summary & CSV Metrics
        res_df = pd.DataFrame(results).sort_values(by="Alpha (%)", ascending=False)
        res_df.to_csv("reports/ticker_comparative_metrics.csv", index=False)
        
        logger.info("\n" + "="*100 + "\nINDIVIDUAL TICKER PERFORMANCE: AI vs BENCHMARK\n" + "="*100)
        logger.info(f"\n{res_df.to_string(index=False)}")

        total_ai_ret = ((agg_ai_equity[-1] / agg_ai_equity[0]) - 1) * 100
        total_bench_ret = ((agg_bench_equity[-1] / agg_bench_equity[0]) - 1) * 100
        port_sharpe, port_mdd = self._calc_metrics(agg_ai_equity)
        bench_port_sharpe, bench_port_mdd = self._calc_metrics(agg_bench_equity)
        
        logger.info("\n" + "="*80 + "\nMACRO PORTFOLIO SUMMARY\n" + "="*80)
        logger.info(f"Total Portfolio AI Return:   {total_ai_ret:.2f}%")
        logger.info(f"Total Portfolio Bench Return:{total_bench_ret:.2f}%")
        logger.info(f"Aggregate Alpha Generated:   {total_ai_ret - total_bench_ret:.2f}%")
        logger.info("-" * 80)
        logger.info(f"AI Sharpe Ratio:             {port_sharpe}")
        logger.info(f"Benchmark Sharpe Ratio:      {bench_port_sharpe}")
        logger.info("-" * 80)
        logger.info(f"AI Max Drawdown:             {port_mdd}%")
        logger.info(f"Benchmark Max Drawdown:      {bench_port_mdd}%")
        logger.info("=" * 80 + "\n")

        # 3. Visualization
        sns.set_theme(style="darkgrid")
        
        # Plot 1: Cumulative Equity Curve
        plt.figure(figsize=(12, 6))
        plt.plot(self.dates, agg_ai_equity, label='AI Prop Fund (Total)', color='#00ff88', linewidth=2)
        plt.plot(self.dates, agg_bench_equity, label='Buy & Hold Benchmark (Total)', color='#ff3366', linewidth=1.5, alpha=0.8)
        plt.title('Total Portfolio Equity Curve (AI vs Benchmark)', fontsize=14, fontweight='bold')
        plt.ylabel('Total Net Worth (₹)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig("reports/equity_curve.png", dpi=300)
        plt.close()
        
        # Plot 2: Alpha Distribution Bar Chart
        plt.figure(figsize=(14, 8))
        colors = ['#00ff88' if x > 0 else '#ff3366' for x in res_df['Alpha (%)']]
        sns.barplot(x='Ticker', y='Alpha (%)', data=res_df, hue='Ticker', palette=colors, legend=False)
        plt.title('Alpha Generated per Stock', fontsize=14, fontweight='bold')
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        plt.savefig("reports/alpha_distribution.png", dpi=300)
        plt.close()

        logger.info("Reports saved to the 'reports/' directory.")