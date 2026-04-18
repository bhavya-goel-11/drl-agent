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
    
    Now includes ML baseline comparison support.
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
        
        # Store aggregate equity for ML comparison
        self.agg_ai_equity = None
        self.agg_bench_equity = None
        self.ai_portfolio_results = {}
        
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

    def _generate_reports(self, ml_results: list = None):
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
        
        # Store for comparison
        self.agg_ai_equity = agg_ai_equity
        self.agg_bench_equity = agg_bench_equity

        # 2. Console Summary & CSV Metrics
        res_df = pd.DataFrame(results).sort_values(by="Alpha (%)", ascending=False)
        res_df.to_csv("reports/ticker_comparative_metrics.csv", index=False)
        
        logger.info("\n" + "="*100 + "\nINDIVIDUAL TICKER PERFORMANCE: AI vs BENCHMARK\n" + "="*100)
        logger.info(f"\n{res_df.to_string(index=False)}")

        total_ai_ret = ((agg_ai_equity[-1] / agg_ai_equity[0]) - 1) * 100
        total_bench_ret = ((agg_bench_equity[-1] / agg_bench_equity[0]) - 1) * 100
        port_sharpe, port_mdd = self._calc_metrics(agg_ai_equity)
        bench_port_sharpe, bench_port_mdd = self._calc_metrics(agg_bench_equity)
        
        # Store portfolio-level results
        self.ai_portfolio_results = {
            'return': total_ai_ret,
            'sharpe': port_sharpe,
            'max_dd': port_mdd,
            'bench_return': total_bench_ret,
            'bench_sharpe': bench_port_sharpe,
            'bench_max_dd': bench_port_mdd,
        }
        
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
        plt.figure(figsize=(14, 7))
        plt.plot(self.dates, agg_ai_equity, label='D3QN AI Agent (Total)', color='#00ff88', linewidth=2)
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
    
    def generate_comparative_report(self, ml_results: list):
        """
        Generate a comprehensive DRL vs Traditional ML comparison report.
        Call this AFTER run() and after ML baselines have been backtested.
        
        Args:
            ml_results: list of dicts from MLBaselineTrader.backtest_single_model()
        """
        logger.info("\n" + "=" * 100)
        logger.info("COMPREHENSIVE MODEL COMPARISON: DRL vs TRADITIONAL ML")
        logger.info("=" * 100)
        
        comparison_rows = [{
            'Model': 'D3QN (DRL Agent)',
            'Portfolio Return (%)': self.ai_portfolio_results.get('return', 0),
            'Sharpe Ratio': self.ai_portfolio_results.get('sharpe', 0),
            'Max Drawdown (%)': self.ai_portfolio_results.get('max_dd', 0),
            'Alpha vs B&H (%)': round(
                self.ai_portfolio_results.get('return', 0) - 
                self.ai_portfolio_results.get('bench_return', 0), 2
            ),
            'Total Trades': len(self.trade_ledger),
        }]
        
        for ml_res in ml_results:
            comparison_rows.append({
                'Model': ml_res['model_name'].replace('_', ' ').title(),
                'Portfolio Return (%)': ml_res['portfolio_return'],
                'Sharpe Ratio': ml_res['portfolio_sharpe'],
                'Max Drawdown (%)': ml_res['portfolio_max_dd'],
                'Alpha vs B&H (%)': round(
                    ml_res['portfolio_return'] - 
                    self.ai_portfolio_results.get('bench_return', 0), 2
                ),
                'Total Trades': ml_res['trades'],
            })
        
        comparison_rows.append({
            'Model': 'Buy & Hold (Benchmark)',
            'Portfolio Return (%)': self.ai_portfolio_results.get('bench_return', 0),
            'Sharpe Ratio': self.ai_portfolio_results.get('bench_sharpe', 0),
            'Max Drawdown (%)': self.ai_portfolio_results.get('bench_max_dd', 0),
            'Alpha vs B&H (%)': 0.0,
            'Total Trades': 0,
        })
        
        comp_df = pd.DataFrame(comparison_rows)
        comp_df.to_csv("reports/model_comparison.csv", index=False)
        
        logger.info(f"\n{comp_df.to_string(index=False)}")
        logger.info("")
        
        # Generate comparison equity curve plot
        plt.figure(figsize=(16, 8))
        plt.plot(self.dates, self.agg_ai_equity, label='D3QN (DRL)', color='#00ff88', linewidth=2.5)
        plt.plot(self.dates, self.agg_bench_equity, label='Buy & Hold', color='#ff3366', linewidth=1.5, alpha=0.7)
        
        ml_colors = ['#4488ff', '#ffaa00', '#aa44ff', '#44ffaa']
        for i, ml_res in enumerate(ml_results):
            if len(ml_res['equity_curve']) == len(self.dates):
                color = ml_colors[i % len(ml_colors)]
                label = ml_res['model_name'].replace('_', ' ').title()
                plt.plot(self.dates, ml_res['equity_curve'], label=label, 
                         color=color, linewidth=1.5, linestyle='--', alpha=0.9)
        
        plt.title('Model Comparison: DRL vs Traditional ML vs Buy & Hold', fontsize=14, fontweight='bold')
        plt.ylabel('Total Net Worth (₹)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig("reports/model_comparison_equity.png", dpi=300)
        plt.close()
        
        # Sharpe comparison bar chart
        plt.figure(figsize=(10, 6))
        models = comp_df['Model']
        sharpes = comp_df['Sharpe Ratio']
        colors = ['#00ff88' if s > 0 else '#ff3366' for s in sharpes]
        plt.barh(models, sharpes, color=colors, edgecolor='white', linewidth=0.5)
        plt.title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Sharpe Ratio', fontsize=12)
        plt.axvline(x=0, color='white', linewidth=0.8, alpha=0.5)
        plt.tight_layout()
        plt.savefig("reports/sharpe_comparison.png", dpi=300)
        plt.close()
        
        logger.info("Comparative reports saved to 'reports/' directory.")
        logger.info("=" * 100)