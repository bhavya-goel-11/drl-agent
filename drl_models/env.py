import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from loguru import logger

class TradingEnv(gym.Env):
    """An institutional-grade stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance=10000, commission=0.001):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission # 0.1% transaction fee to simulate slippage
        
        # Calculate statistics for Z-score normalization
        self.feature_means = self.df.mean()
        self.feature_stds = self.df.std().replace(0, 1e-8)
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        num_features = len(self.df.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features + 3,), dtype=np.float32
        )
        
        logger.info(f"Initialized TradingEnv with shape {self.observation_space.shape}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        
        # Gym v26+ requires reset to return (observation, info)
        return self._next_observation(), {}

    def _next_observation(self):
        frame = self.df.iloc[self.current_step]
        
        # FIX: Rolling 200-day Window Normalization (No Future Data!)
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

    def step(self, action):
        # 1. Record the net worth BEFORE the market moves
        prev_net_worth = self.net_worth
        
        # FIX: Execute at TOMORROW'S Open (or next available price)
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
        
        # 6. The Reward Fix: Step-by-step PnL
        step_profit = self.net_worth - prev_net_worth
        
        # 7. Scale the reward for PyTorch stability 
        # (e.g. a $100 profit on a $10k account = 0.01 reward)
        reward = step_profit / self.initial_balance
        
        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {'net_worth': self.net_worth, 'step_profit': step_profit}
        
        # Gym v26+ requires step to return (obs, reward, terminated, truncated, info)
        return obs, reward, done, False, info

    def render(self, mode='human', close=False):
        profit = self.net_worth - self.initial_balance
        print(f"Step: {self.current_step}, Net Worth: ${self.net_worth:.2f}, Total Profit: ${profit:.2f}")