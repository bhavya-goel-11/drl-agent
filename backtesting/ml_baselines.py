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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
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
    Trains traditional ML models to predict next-day direction and generates
    Buy/Sell/Hold signals comparable to the DRL agent.
    """
    
    def __init__(self, train_cutoff: str = "2023-01-01"):
        self.train_cutoff = pd.to_datetime(train_cutoff)
        self.models = {}
        self.scalers = {}
        self.feature_cols = None
    
    def _create_labels(self, df: pd.DataFrame, forward_window: int = 5,
                       buy_threshold: float = 0.02, sell_threshold: float = -0.02) -> pd.Series:
        """
        Create classification labels based on forward returns:
        - BUY (1): forward return > buy_threshold
        - SELL (2): forward return < sell_threshold  
        - HOLD (0): otherwise
        """
        forward_return = df['close'].pct_change(forward_window).shift(-forward_window)
        labels = pd.Series(0, index=df.index)  # Default: HOLD
        labels[forward_return > buy_threshold] = 1   # BUY
        labels[forward_return < sell_threshold] = 2   # SELL
        return labels
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract numeric feature columns for ML, excluding target-leaking columns."""
        exclude_cols = {'symbol', 'date', 'label'}
        numeric_df = df.select_dtypes(include=[np.number])
        feature_cols = [c for c in numeric_df.columns if c not in exclude_cols]
        return numeric_df[feature_cols]
    
    def train(self, multi_ticker_data: dict):
        """Train Gradient Boosting and Random Forest on pre-cutoff data."""
        logger.info("=" * 70)
        logger.info("Training Traditional ML Baselines...")
        logger.info("=" * 70)
        
        # Concatenate all training data
        all_train_X = []
        all_train_y = []
        
        for ticker, df in multi_ticker_data.items():
            if ticker == '^NSEI':
                continue
                
            train_df = df[df.index < self.train_cutoff].copy()
            if len(train_df) < 100:
                continue
            
            # Create labels
            labels = self._create_labels(train_df)
            features = self._prepare_features(train_df)
            
            # Drop rows with NaN labels (last few rows due to forward shift)
            valid_mask = labels.notna() & ~features.isna().any(axis=1)
            all_train_X.append(features[valid_mask])
            all_train_y.append(labels[valid_mask])
        
        if not all_train_X:
            logger.error("No training data available for ML baselines!")
            return
        
        X_train = pd.concat(all_train_X)
        y_train = pd.concat(all_train_y)
        
        self.feature_cols = X_train.columns.tolist()
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['main'] = scaler
        
        logger.info(f"Training data shape: {X_train_scaled.shape} | "
                     f"Label distribution: {dict(y_train.value_counts().sort_index())}")
        
        # --- Model 1: Gradient Boosting ---
        if HAS_XGBOOST:
            logger.info("Training XGBoost Gradient Boosting Classifier...")
            gb_model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        else:
            logger.info("Training Sklearn Gradient Boosting Classifier...")
            gb_model = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=20,
                random_state=42,
            )
        
        gb_model.fit(X_train_scaled, y_train.astype(int))
        self.models['gradient_boosting'] = gb_model
        logger.info("  ✓ Gradient Boosting trained successfully.")
        
        # --- Model 2: Random Forest ---
        logger.info("Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=20,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        rf_model.fit(X_train_scaled, y_train.astype(int))
        self.models['random_forest'] = rf_model
        logger.info("  ✓ Random Forest trained successfully.")
        
        # Training accuracy report
        for name, model in self.models.items():
            train_pred = model.predict(X_train_scaled)
            accuracy = (train_pred == y_train.astype(int).values).mean()
            logger.info(f"  {name} training accuracy: {accuracy:.2%}")
    
    def predict(self, model_name: str, features: np.ndarray) -> int:
        """Predict action (0=Hold, 1=Buy, 2=Sell) for a single observation."""
        if model_name not in self.models:
            return 0
        
        scaler = self.scalers.get('main')
        if scaler is not None:
            features = scaler.transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)
        
        return int(self.models[model_name].predict(features)[0])
    
    def backtest_single_model(self, model_name: str, multi_ticker_data: dict,
                               start_date: str = "2023-01-01",
                               per_stock_budget: float = 100000.0,
                               commission: float = 0.002) -> dict:
        """
        Run a full backtest for a single ML model, identical to the DRL backtest.
        Returns per-ticker and aggregate results.
        """
        start_dt = pd.to_datetime(start_date)
        accounts = {t: {"balance": per_stock_budget, "shares": 0, "history": []} 
                     for t in multi_ticker_data.keys()}
        benchmarks = {t: {"shares": 0, "history": []} for t in multi_ticker_data.keys()}
        trade_ledger = []
        
        # Build master timeline
        all_dates = pd.DatetimeIndex([])
        for df in multi_ticker_data.values():
            filtered = df[df.index >= start_dt]
            all_dates = all_dates.union(filtered.index)
        trading_dates = all_dates.sort_values().unique()
        dates = []
        
        # Init benchmark
        for t, df_t in multi_ticker_data.items():
            first_valid = df_t[df_t.index >= start_dt].index.min()
            if pd.notna(first_valid):
                price = float(df_t.loc[first_valid, 'close'])
                benchmarks[t]["shares"] = per_stock_budget / (price * (1 + commission))
        
        # Run simulation
        for i, today in enumerate(trading_dates):
            if i >= len(trading_dates) - 1:
                break
            tomorrow = trading_dates[i + 1]
            dates.append(today)
            
            for t in multi_ticker_data.keys():
                acc = accounts[t]
                bench = benchmarks[t]
                df_t = multi_ticker_data[t]
                
                if today not in df_t.index:
                    last_val = acc["history"][-1] if acc["history"] else per_stock_budget
                    acc["history"].append(last_val)
                    last_bench = bench["history"][-1] if bench["history"] else per_stock_budget
                    bench["history"].append(last_bench)
                    continue
                
                # Valuation
                price_today = float(df_t.loc[today, 'close'])
                net_worth = acc["balance"] + (acc["shares"] * price_today)
                acc["history"].append(net_worth)
                
                bench_worth = bench["shares"] * price_today if bench["shares"] > 0 else per_stock_budget
                bench["history"].append(bench_worth)
                
                # Feature extraction for ML prediction
                features = self._prepare_features(df_t)
                if today not in features.index:
                    continue
                    
                idx = features.index.get_loc(today)
                if isinstance(idx, (slice, np.ndarray)):
                    idx = np.where(features.index == today)[0][0]
                
                if idx < 200:
                    continue
                
                feature_row = features.iloc[idx].values
                if np.any(np.isnan(feature_row)):
                    continue
                
                # Predict
                action = self.predict(model_name, feature_row)
                
                # Execute
                if tomorrow not in df_t.index:
                    continue
                price_tmrw = float(df_t.loc[tomorrow, 'open'])
                
                if action == 2 and acc["shares"] > 0:  # SELL
                    val = acc["shares"] * price_tmrw
                    comm = val * commission
                    acc["balance"] += (val - comm)
                    trade_ledger.append({
                        'Date': tomorrow.date(), 'Ticker': t, 
                        'Action': 'SELL', 'Price': price_tmrw,
                        'Shares': acc["shares"], 'Model': model_name
                    })
                    acc["shares"] = 0
                    
                elif action == 1 and acc["shares"] == 0:  # BUY
                    shares = int(acc["balance"] // (price_tmrw * (1 + commission)))
                    if shares > 0:
                        cost = (shares * price_tmrw) * (1 + commission)
                        acc["balance"] -= cost
                        acc["shares"] = shares
                        trade_ledger.append({
                            'Date': tomorrow.date(), 'Ticker': t,
                            'Action': 'BUY', 'Price': price_tmrw,
                            'Shares': shares, 'Model': model_name
                        })
        
        # Calculate aggregate equity
        agg_equity = np.zeros(len(dates))
        agg_bench = np.zeros(len(dates))
        results = []
        
        for t in multi_ticker_data.keys():
            if t == '^NSEI' or not accounts[t]["history"]:
                continue
            
            ai_hist = accounts[t]["history"]
            bench_hist = benchmarks[t]["history"]
            
            if len(ai_hist) < len(dates):
                ai_hist = [per_stock_budget] * (len(dates) - len(ai_hist)) + ai_hist
                bench_hist = [per_stock_budget] * (len(dates) - len(bench_hist)) + bench_hist
            
            agg_equity += np.array(ai_hist[:len(dates)])
            agg_bench += np.array(bench_hist[:len(dates)])
            
            ai_ret = ((ai_hist[-1] / per_stock_budget) - 1) * 100
            bench_ret = ((bench_hist[-1] / per_stock_budget) - 1) * 100
            
            # Sharpe
            s = pd.Series(ai_hist)
            pct = s.pct_change().dropna()
            sharpe = (pct.mean() / pct.std()) * np.sqrt(252) if len(pct) > 1 and pct.std() > 0 else 0.0
            
            # Max DD
            rolling_max = s.cummax()
            dd = (s - rolling_max) / rolling_max
            max_dd = dd.min() * 100
            
            results.append({
                "Ticker": t,
                f"{model_name}_Return (%)": round(ai_ret, 2),
                "Bench_Return (%)": round(bench_ret, 2),
                f"{model_name}_Alpha (%)": round(ai_ret - bench_ret, 2),
                f"{model_name}_Sharpe": round(sharpe, 2),
                f"{model_name}_MaxDD (%)": round(max_dd, 2),
            })
        
        # Portfolio-level metrics
        if len(agg_equity) > 0 and agg_equity[0] > 0:
            port_ret = ((agg_equity[-1] / agg_equity[0]) - 1) * 100
            s = pd.Series(agg_equity)
            pct = s.pct_change().dropna()
            port_sharpe = (pct.mean() / pct.std()) * np.sqrt(252) if len(pct) > 1 and pct.std() > 0 else 0.0
            rolling_max = s.cummax()
            dd = (s - rolling_max) / rolling_max
            port_mdd = dd.min() * 100
        else:
            port_ret = port_sharpe = port_mdd = 0.0
        
        return {
            'model_name': model_name,
            'per_ticker': pd.DataFrame(results),
            'portfolio_return': round(port_ret, 2),
            'portfolio_sharpe': round(port_sharpe, 2),
            'portfolio_max_dd': round(port_mdd, 2),
            'equity_curve': agg_equity,
            'bench_curve': agg_bench,
            'dates': dates,
            'trades': len(trade_ledger),
            'trade_ledger': trade_ledger,
        }
