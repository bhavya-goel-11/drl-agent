import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from loguru import logger
from collections import deque
import random

class TradingEnv(gym.Env):
    """
    Institutional-grade stock trading environment with risk-adjusted rewards.
    
    Anti-Convergence Features:
    1. Rolling Sharpe reward — agent optimizes risk-adjusted returns, not raw PnL
    2. Drawdown penalty — prevents catastrophic concentration in single stocks
    3. Volatility-scaled rewards — normalizes reward signal across market regimes
    4. Inactivity penalty — discourages the "hold cash forever" local optimum
    5. Trade cost awareness — realistic friction prevents overtrading convergence
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_balance=10000, commission=0.002,
                 sharpe_window: int = 20, drawdown_penalty_coeff: float = 0.5,
                 inactivity_penalty: float = -0.0005):
        super(TradingEnv, self).__init__()
        
        # Handle different input data formats (dictionary, groupby object, or plain DataFrame)
        if isinstance(data, dict):
            self.tickers = list(data.keys())
            self.data_dict = {ticker: df.copy().reset_index(drop=True) for ticker, df in data.items()}
        elif hasattr(data, 'groups'): # Covers DataFrameGroupBy
            self.tickers = list(data.groups.keys())
            self.data_dict = {ticker: group.copy().reset_index(drop=True) for ticker, group in data}
        elif isinstance(data, pd.DataFrame):
            if 'symbol' in data.columns:
                grouped = data.groupby('symbol')
                self.tickers = list(grouped.groups.keys())
                self.data_dict = {ticker: group.copy().reset_index(drop=True) for ticker, group in grouped}
            else:
                self.tickers = ['DEFAULT']
                self.data_dict = {'DEFAULT': data.copy().reset_index(drop=True)}
        else:
            raise ValueError("Unsupported data format. Please provide a dict or grouped DataFrame.")
            
        # Ensure pure numeric state space to match required feature constraints without data format crashing
        for ticker, df in self.data_dict.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.data_dict[ticker] = df[numeric_cols]
        
        self.initial_balance = initial_balance
        self.commission = commission
        
        # --- Risk-Adjusted Reward Configuration ---
        self.sharpe_window = sharpe_window
        self.drawdown_penalty_coeff = drawdown_penalty_coeff
        self.inactivity_penalty = inactivity_penalty
        
        self.active_ticker = random.choice(self.tickers)
        self.df = self.data_dict[self.active_ticker]
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        num_features = len(self.df.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features + 3,), dtype=np.float32
        )
        
        # Reward tracking for Sharpe calculation
        self._returns_buffer = deque(maxlen=sharpe_window)
        self._peak_net_worth = initial_balance
        self._hold_counter = 0
        
        logger.info(f"Initialized TradingEnv with {len(self.tickers)} tickers. "
                     f"Target feature shape {self.observation_space.shape} | "
                     f"Sharpe window: {sharpe_window}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Episodic Ticker Randomization
        self.active_ticker = random.choice(self.tickers)
        self.df = self.data_dict[self.active_ticker]
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        
        # Reset reward tracking
        self._returns_buffer.clear()
        self._peak_net_worth = self.initial_balance
        self._hold_counter = 0
        
        # Gym v26+ requires reset to return (observation, info)
        return self._next_observation(), {}

    def _next_observation(self):
        frame = self.df.iloc[self.current_step]
        
        # Rolling 200-day Window Normalization (No Future Data!)
        start_idx = max(0, self.current_step - 200)
        window = self.df.iloc[start_idx : self.current_step + 1]
        
        rolling_mean = window.mean()
        rolling_std = window.std().replace(0, 1e-8)
        
        normalized_frame = (frame - rolling_mean) / rolling_std
        normalized_array = np.nan_to_num(np.array(normalized_frame.values), nan=0.0)
        
        norm_balance = self.balance / self.initial_balance
        norm_net_worth = self.net_worth / self.initial_balance
        
        # USE CURRENT PRICE HERE: The agent evaluates its current state using today's price.
        current_price = frame.get("close", frame.values[-1])
        norm_shares = (self.shares_held * current_price) / self.initial_balance
        
        obs = np.append(normalized_array, [norm_balance, norm_shares, norm_net_worth])
        return obs.astype(np.float32)

    def _calculate_reward(self, step_return: float, action: int) -> float:
        """
        Multi-component risk-adjusted reward function.
        
        This is the KEY anti-premature-convergence mechanism:
        - Raw PnL reward chases momentum → agent converges on high-beta stocks
        - Sharpe reward balances return vs risk → agent learns robust strategies
        - Drawdown penalty prevents catastrophic positions
        - Inactivity penalty prevents the "do nothing" local optimum
        
        Reference: Papers S0952197625032270 (EAAI) and S2667305325001486 (ISA)
        """
        # 1. Base: step-by-step return (scaled)
        self._returns_buffer.append(step_return)
        
        # 2. Rolling Sharpe component (only when we have enough data)
        sharpe_reward = 0.0
        if len(self._returns_buffer) >= 5:
            returns_arr = np.array(self._returns_buffer)
            mean_ret = returns_arr.mean()
            std_ret = returns_arr.std()
            if std_ret > 1e-8:
                # Differential Sharpe: reward improvement in risk-adjusted terms
                sharpe_reward = mean_ret / std_ret
            else:
                sharpe_reward = mean_ret * 10.0  # Pure positive returns with zero vol → reward
        
        # 3. Drawdown penalty — penalize being far below peak
        self._peak_net_worth = max(self._peak_net_worth, self.net_worth)
        drawdown = (self._peak_net_worth - self.net_worth) / self._peak_net_worth
        dd_penalty = -self.drawdown_penalty_coeff * (drawdown ** 2) if drawdown > 0.02 else 0.0
        
        # 4. Inactivity penalty — prevent "hold cash" local optimum
        if action == 0:  # Hold
            self._hold_counter += 1
        else:
            self._hold_counter = 0
        
        inactivity = self.inactivity_penalty * max(0, self._hold_counter - 20) if self.shares_held == 0 and self.balance > self.initial_balance * 0.95 else 0.0
        
        # 5. Combine components with balanced weights
        reward = (
            0.4 * step_return +          # Base PnL signal 
            0.4 * sharpe_reward * 0.1 +   # Risk-adjusted quality 
            dd_penalty +                   # Drawdown punishment
            inactivity                     # Anti-inactivity
        )
        
        return float(np.clip(reward, -1.0, 1.0))

    def step(self, action):
        # 1. Record the net worth BEFORE the market moves
        prev_net_worth = self.net_worth
        
        # Execute at TOMORROW'S Open (or next available price)
        execution_step = min(self.current_step + 1, len(self.df) - 1)
        # Fallback to 'close' if your dataset doesn't have an 'open' column
        execution_price = self.df.iloc[execution_step].get("open", self.df.iloc[execution_step].get("close"))
        
        # 3. Execute Trade (with transaction friction)
        if action == 1 and self.balance >= execution_price: # Buy
            shares_bought = self.balance // execution_price
            cost = shares_bought * execution_price
            fee = cost * self.commission
            self.balance -= (cost + fee)
            self.shares_held += shares_bought
            
        elif action == 2 and self.shares_held > 0: # Sell
            revenue = self.shares_held * execution_price
            fee = revenue * self.commission
            self.balance += (revenue - fee)
            self.shares_held = 0
            
        # 4. Move time forward to tomorrow!
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # 5. Calculate NEW net worth using TOMORROW'S price
        if not done:
            new_price = self.df.iloc[self.current_step].get("close", self.df.iloc[self.current_step].values[-1])
        else:
            new_price = execution_price
            
        self.net_worth = self.balance + (self.shares_held * new_price)
        
        # 6. Step-by-step return (not raw PnL)
        step_return = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0.0
        
        # 7. Risk-adjusted reward
        reward = self._calculate_reward(step_return, action)
        
        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            'net_worth': self.net_worth, 
            'step_profit': self.net_worth - prev_net_worth,
            'step_return': step_return,
            'ticker': self.active_ticker,
        }
        
        # Gym v26+ requires step to return (obs, reward, terminated, truncated, info)
        return obs, reward, done, False, info

    def render(self, mode='human', close=False):
        profit = self.net_worth - self.initial_balance
        print(f"Step: {self.current_step}, Net Worth: ${self.net_worth:.2f}, Total Profit: ${profit:.2f}")