import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from loguru import logger
from collections import deque


class VectorizedTradingEnv(gym.Env):
    """
    Vectorized multi-asset portfolio trading environment.
    
    At every timestep the agent observes the full market state for ALL N stocks
    simultaneously and outputs a vector of N actions (one per stock).  The reward
    is computed at the *portfolio* level using risk-adjusted metrics so the agent
    learns cross-asset allocation, not just single-stock timing.
    
    Key Design Choices:
        • MultiDiscrete([3]*N) action space — compatible with D3QN architecture.
        • Equal-split capital allocation across simultaneous buy signals.
        • Portfolio-level Sharpe reward with concentration & diversification terms.
        • Stratified random sub-windows (252 trading days ≈ 1 year) so every
          calendar year in the training set is sampled proportionately.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        data_3d: np.ndarray,
        tickers: list,
        dates: pd.DatetimeIndex,
        columns: list,
        initial_balance: float = 10_000_000.0,
        commission: float = 0.002,
        window_size: int = 252,
        sharpe_window: int = 20,
        drawdown_penalty_coeff: float = 0.5,
        concentration_penalty_coeff: float = 0.1,
        diversification_bonus: float = 0.0002,
    ):
        """
        Args:
            data_3d:   (T, N, F) aligned market data.
            tickers:   list of N ticker strings.
            dates:     DatetimeIndex of T dates.
            columns:   list of F feature column names.
            initial_balance:  Starting cash for the portfolio.
            commission:       Round-trip trading cost fraction.
            window_size:      Number of trading days per episode sub-window.
            sharpe_window:    Rolling window for Sharpe reward calculation.
            drawdown_penalty_coeff:  Weight of the drawdown penalty term.
            concentration_penalty_coeff:  Herfindahl index penalty weight.
            diversification_bonus:  Bonus per step for holding multiple names.
        """
        super().__init__()

        self.data = data_3d                       # (T, N, F)
        self.tickers = tickers
        self.dates = dates
        self.columns = columns
        self.total_steps, self.n_stocks, self.n_features = data_3d.shape

        # Locate special columns for execution
        self.close_idx = columns.index('close') if 'close' in columns else -1
        self.open_idx  = columns.index('open')  if 'open'  in columns else self.close_idx

        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = min(window_size, self.total_steps - 1)

        # Reward parameters
        self.sharpe_window = sharpe_window
        self.dd_coeff = drawdown_penalty_coeff
        self.conc_coeff = concentration_penalty_coeff
        self.div_bonus = diversification_bonus

        # --- Spaces ---
        # Action: one discrete action per stock  (0=Hold, 1=Buy, 2=Sell)
        self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)

        # Observation: flattened [market_features | positions | portfolio_state]
        self.obs_dim = self.n_stocks * self.n_features + self.n_stocks + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # --- Stratified year sampler ---
        self._build_year_index()

        # --- Per-episode state (initialized in reset) ---
        self.cash = initial_balance
        self.holdings = np.zeros(self.n_stocks, dtype=np.float64)
        self.start_step = 0
        self.end_step = 0
        self.current_step = 0
        self._returns_buffer = deque(maxlen=sharpe_window)
        self._peak_value = initial_balance

        logger.info(
            f"VectorizedTradingEnv initialised — "
            f"{self.n_stocks} stocks × {self.n_features} features | "
            f"obs_dim={self.obs_dim} | window={self.window_size} days | "
            f"T={self.total_steps}"
        )

    # ------------------------------------------------------------------
    # Stratified year sampling
    # ------------------------------------------------------------------
    def _build_year_index(self):
        """Map each timestep to its calendar year for stratified sampling."""
        years = pd.DatetimeIndex(self.dates).year.values
        self.year_of_step = years
        unique_years = np.unique(years)

        self.year_to_indices = {}
        for y in unique_years:
            self.year_to_indices[y] = np.where(years == y)[0]

        # Track sample counts to enforce proportionate coverage
        self.year_sample_counts = {y: 0 for y in unique_years}
        logger.debug(f"Year strata: {list(unique_years)} "
                     f"({len(unique_years)} distinct years)")

    def _sample_start_index(self) -> int:
        """Pick a start index using least-sampled-year strategy."""
        min_count = min(self.year_sample_counts.values())
        candidates = [y for y, c in self.year_sample_counts.items()
                      if c == min_count]
        year = int(np.random.choice(candidates))

        indices = self.year_to_indices[year]
        # Must leave room for window_size steps
        valid = indices[indices <= self.total_steps - self.window_size]

        if len(valid) == 0:
            # Fallback: pick any valid start
            start = np.random.randint(
                0, max(1, self.total_steps - self.window_size))
        else:
            start = int(np.random.choice(valid))

        self.year_sample_counts[year] += 1
        return start

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.start_step = self._sample_start_index()
        self.end_step = min(self.start_step + self.window_size,
                            self.total_steps - 1)
        self.current_step = self.start_step

        self.cash = self.initial_balance
        self.holdings = np.zeros(self.n_stocks, dtype=np.float64)
        self._returns_buffer.clear()
        self._peak_value = self.initial_balance

        return self._build_observation(), {}

    def step(self, actions: np.ndarray):
        """
        Execute one time-step across all N stocks.

        Args:
            actions: np.ndarray of shape (N,) with values in {0, 1, 2}.

        Returns:
            obs, reward, terminated, truncated, info
        """
        actions = np.asarray(actions, dtype=np.int64)
        prev_value = self._portfolio_value()

        # --- Execution at next-day open ---
        exec_step = self.current_step + 1
        if exec_step >= self.end_step:
            # Episode over — return terminal
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            return obs, 0.0, True, False, self._info(prev_value)

        exec_prices = self.data[exec_step, :, self.open_idx]   # (N,)

        buy_mask  = (actions == 1) & (self.holdings == 0)
        sell_mask = (actions == 2) & (self.holdings > 0)

        # --- Sell first (frees cash) ---
        for i in np.where(sell_mask)[0]:
            revenue = self.holdings[i] * exec_prices[i]
            fee = revenue * self.commission
            self.cash += (revenue - fee)
            self.holdings[i] = 0

        # --- Buy with equal-split ---
        num_buys = int(buy_mask.sum())
        if num_buys > 0:
            cash_per_stock = self.cash / num_buys
            for i in np.where(buy_mask)[0]:
                max_cost_per_share = exec_prices[i] * (1 + self.commission)
                if max_cost_per_share <= 0:
                    continue
                shares = int(cash_per_stock // max_cost_per_share)
                if shares > 0:
                    cost = shares * exec_prices[i] * (1 + self.commission)
                    self.cash -= cost
                    self.holdings[i] = shares

        # --- Advance time ---
        self.current_step = exec_step
        terminated = self.current_step >= self.end_step - 1

        new_value = self._portfolio_value()
        step_return = ((new_value - prev_value) / prev_value
                       if prev_value > 0 else 0.0)

        reward = self._calculate_reward(step_return, actions)

        obs = (self._build_observation() if not terminated
               else np.zeros(self.obs_dim, dtype=np.float32))

        return obs, reward, terminated, False, self._info(new_value)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _build_observation(self) -> np.ndarray:
        frame = self.data[self.current_step]  # (N, F)

        # 200-day rolling Z-score normalisation (no future data)
        lookback_start = max(self.start_step, self.current_step - 200)
        window = self.data[lookback_start: self.current_step + 1]  # (W, N, F)

        rolling_mean = window.mean(axis=0)   # (N, F)
        rolling_std  = window.std(axis=0)    # (N, F)
        rolling_std  = np.where(rolling_std < 1e-8, 1e-8, rolling_std)

        normalised = (frame - rolling_mean) / rolling_std
        normalised = np.nan_to_num(normalised, nan=0.0)

        market_flat = normalised.flatten()  # (N*F,)

        # Per-stock position value (normalised)
        close_prices = self.data[self.current_step, :, self.close_idx]
        position_vals = (self.holdings * close_prices) / self.initial_balance

        # Portfolio summary
        total_equity = self.cash + np.sum(self.holdings * close_prices)
        self._peak_value = max(self._peak_value, total_equity)
        drawdown = ((self._peak_value - total_equity) / self._peak_value
                    if self._peak_value > 0 else 0.0)

        portfolio_state = np.array([
            self.cash / self.initial_balance,
            total_equity / self.initial_balance,
            drawdown,
        ])

        obs = np.concatenate([market_flat, position_vals, portfolio_state])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _portfolio_value(self) -> float:
        close_prices = self.data[self.current_step, :, self.close_idx]
        return float(self.cash + np.sum(self.holdings * close_prices))

    def _calculate_reward(self, step_return: float, actions: np.ndarray) -> float:
        """
        Portfolio-level risk-adjusted reward.

        Components:
            1. Step return (base PnL signal).
            2. Rolling Sharpe (risk-adjusted quality).
            3. Drawdown penalty (catastrophic loss prevention).
            4. Concentration penalty (Herfindahl — punish single-stock bets).
            5. Diversification bonus (reward holding multiple names).
        """
        self._returns_buffer.append(step_return)

        # 1 — Rolling Sharpe
        sharpe_reward = 0.0
        if len(self._returns_buffer) >= 5:
            arr = np.array(self._returns_buffer)
            mu, sigma = arr.mean(), arr.std()
            if sigma > 1e-8:
                sharpe_reward = mu / sigma
            else:
                sharpe_reward = mu * 10.0

        # 2 — Drawdown penalty
        pv = self._portfolio_value()
        self._peak_value = max(self._peak_value, pv)
        dd = (self._peak_value - pv) / self._peak_value if self._peak_value > 0 else 0.0
        dd_penalty = -self.dd_coeff * (dd ** 2) if dd > 0.02 else 0.0

        # 3 — Concentration penalty (Herfindahl index of position weights)
        close_prices = self.data[self.current_step, :, self.close_idx]
        position_values = self.holdings * close_prices
        total_pos = position_values.sum()
        if total_pos > 0:
            weights = position_values / total_pos
            hhi = float(np.sum(weights ** 2))
            conc_penalty = -self.conc_coeff * hhi
        else:
            conc_penalty = 0.0

        # 4 — Diversification bonus
        num_held = int(np.sum(self.holdings > 0))
        div_bonus = (self.div_bonus * (num_held / self.n_stocks)
                     if num_held >= 3 else 0.0)

        reward = (
            0.4 * step_return
            + 0.4 * sharpe_reward * 0.1
            + dd_penalty
            + conc_penalty
            + div_bonus
        )

        return float(np.clip(reward, -5.0, 5.0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _info(self, portfolio_value: float) -> dict:
        return {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'num_positions': int(np.sum(self.holdings > 0)),
            'window': f"{self.dates[self.start_step].date()}→"
                      f"{self.dates[self.end_step].date()}",
        }

    def render(self, mode='human', close=False):
        pv = self._portfolio_value()
        profit = pv - self.initial_balance
        held = int(np.sum(self.holdings > 0))
        print(f"Step {self.current_step} | PV: ₹{pv:,.0f} | "
              f"Profit: ₹{profit:,.0f} | Positions: {held}/{self.n_stocks}")