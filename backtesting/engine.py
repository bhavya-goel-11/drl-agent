import pandas as pd
import numpy as np
import torch
from loguru import logger
from .metrics import calculate_sharpe_ratio, calculate_max_drawdown
from execution_engine.broker import MockBrokerAPI
from execution_engine.oms import OrderManagementSystem
from typing import Any

class BacktestEngine:
    """An Event-Driven Backtest Engine for evaluating DRL strategies out-of-sample."""
    
    def __init__(self, data: pd.DataFrame, model: Any, symbol: str = "SPY", initial_balance=10000.0):
        self.data = data.sort_index()
        self.model = model
        self.symbol = symbol
        self.initial_balance = initial_balance
        
        # Initialize Execution Infrastructure
        self.broker = MockBrokerAPI(initial_balance=initial_balance)
        self.oms = OrderManagementSystem(self.broker)
        
        self.portfolio_values = []
        self.dates = []

    def _get_broker_balance(self) -> float:
        """Safely extracts balance regardless of how the agent coded MockBrokerAPI."""
        if hasattr(self.broker, 'get_account_summary'):
            return self.broker.get_account_summary().get("balance", self.initial_balance)
        return getattr(self.broker, 'balance', self.initial_balance)

    def _get_broker_shares(self) -> float:
        """Safely extracts shares regardless of dictionary structure."""
        positions = getattr(self.broker, 'positions', {})
        if callable(positions):
            positions = positions()
            
        pos = positions.get(self.symbol, 0)
        if isinstance(pos, dict):
            return pos.get("shares", 0)
        return pos
        
    def _build_state(self, current_step: int) -> np.ndarray:
        frame = self.data.iloc[current_step]
        
        # 1. Rolling 200-day Window Normalization
        start_idx = max(0, current_step - 200)
        window = self.data.iloc[start_idx : current_step + 1]
        
        rolling_mean = window.mean()
        rolling_std = window.std().replace(0, 1e-8)
        
        normalized_frame = (frame - rolling_mean) / rolling_std
        normalized_array = np.nan_to_num(np.array(normalized_frame.values), nan=0.0)
        
        # 2. Portfolio State Integration
        current_balance = self._get_broker_balance()
        shares_held = self._get_broker_shares()
        current_price = frame.get("close", frame.values[-1])
        
        net_worth = current_balance + (shares_held * current_price)
        
        norm_balance = current_balance / self.initial_balance
        norm_net_worth = net_worth / self.initial_balance
        norm_shares = (shares_held * current_price) / self.initial_balance
        
        obs = np.append(normalized_array, [norm_balance, norm_shares, norm_net_worth])
        return obs.astype(np.float32)

    def run(self):
        logger.info(f"Starting Out-of-Sample Backtest for {self.symbol}...")
        
        for i in range(200, len(self.data) - 1):
            date = self.data.index[i]
            
            # 1. Observe the market at Today's Close
            state_array = self._build_state(current_step=i)
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
            
            # 2. Agent decides action based on current state (100% Exploitation)
            with torch.no_grad():
                action = self.model(state_tensor).argmax().item()
            
            # 3. Translate DRL action (0=Hold, 1=Buy, 2=Sell) for the OMS
            tomorrow_row = self.data.iloc[i + 1]
            execution_price = tomorrow_row.get('open', tomorrow_row.get('close'))
            
            balance = self._get_broker_balance()
            shares_held = self._get_broker_shares()

            if action == 1 and balance >= execution_price: # Buy
                quantity = int(balance // execution_price)
                if quantity > 0:
                    # FIX: Pass integer 1 instead of string "BUY"
                    self.oms.execute_signal(self.symbol, 1, execution_price, quantity=quantity)
                    
            elif action == 2 and shares_held > 0: # Sell
                # FIX: Pass integer 2 instead of string "SELL"
                self.oms.execute_signal(self.symbol, 2, execution_price, quantity=shares_held)
            
            # 4. Record EOD Portfolio Value
            post_trade_balance = self._get_broker_balance()
            new_shares = self._get_broker_shares()
            eod_net_worth = post_trade_balance + (new_shares * execution_price)
            
            self.portfolio_values.append(eod_net_worth)
            self.dates.append(tomorrow_row.name)
            
        logger.info("Backtest Completed.")
        self._calculate_performance()
        
    def _calculate_performance(self):
        if not self.portfolio_values:
            logger.error("No portfolio values recorded. Backtest failed.")
            return
            
        vals = pd.Series(self.portfolio_values, index=self.dates)
        returns = vals.pct_change().dropna()
        
        sharpe = calculate_sharpe_ratio(returns)
        max_dd = calculate_max_drawdown(self.portfolio_values)
        
        final_balance = self.portfolio_values[-1]
        profit = final_balance - self.initial_balance
        
        initial_price = self.data.iloc[201].get('open', self.data.iloc[201].get('close'))
        final_price = self.data.iloc[-1].get('close')
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100
        strategy_return = (profit / self.initial_balance) * 100
        
        logger.info("====================================")
        logger.info("      OUT-OF-SAMPLE PERFORMANCE     ")
        logger.info("====================================")
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Final Balance:   ${final_balance:,.2f}")
        logger.info(f"Total Profit:    ${profit:,.2f} ({strategy_return:.2f}%)")
        logger.info(f"Benchmark (B&H): {buy_hold_return:.2f}%")
        logger.info("------------------------------------")
        logger.info(f"Sharpe Ratio:    {sharpe:.2f}")
        logger.info(f"Max Drawdown:    {max_dd*100:.2f}%")
        logger.info("====================================")