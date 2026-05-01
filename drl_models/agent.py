import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random
from collections import deque
from loguru import logger


def mask_invalid_q_values(
    q_values: torch.Tensor,
    states: torch.Tensor,
    n_stocks: int,
) -> torch.Tensor:
    """
    Mask actions that cannot execute under the long-only portfolio rules.

    Action layout is 0=Hold, 1=Buy, 2=Sell. The state layout is
    [N*F market features | N position values | 3 portfolio scalars].
    """
    market_dim = states.shape[1] - n_stocks - 3
    position_start = market_dim
    position_end = position_start + n_stocks
    position_vals = states[:, position_start:position_end]
    has_position = position_vals > 1e-8

    masked = q_values.clone()
    masked[:, :, 1] = masked[:, :, 1].masked_fill(has_position, -1e9)
    masked[:, :, 2] = masked[:, :, 2].masked_fill(~has_position, -1e9)
    return masked


def _position_values_from_state(state: np.ndarray, n_stocks: int) -> np.ndarray:
    market_dim = state.shape[0] - n_stocks - 3
    position_start = market_dim
    position_end = position_start + n_stocks
    return state[position_start:position_end]


def enforce_minimum_positions(
    actions: np.ndarray,
    q_values: np.ndarray,
    state: np.ndarray,
    n_stocks: int,
    min_positions: int,
) -> np.ndarray:
    """Ensure the long-only policy keeps at least a small portfolio deployed."""
    if min_positions <= 0:
        return actions

    adjusted = np.asarray(actions, dtype=np.int64).copy()
    position_vals = _position_values_from_state(state, n_stocks)
    has_position = position_vals > 1e-8
    projected_positions = int(has_position.sum() + np.sum((adjusted == 1) & (~has_position)))
    deficit = min_positions - projected_positions
    if deficit <= 0:
        return adjusted

    flat_candidates = np.where((~has_position) & (adjusted != 1))[0]
    if len(flat_candidates) == 0:
        return adjusted

    buy_scores = q_values[flat_candidates, 1] - q_values[flat_candidates, 0]
    ranked = flat_candidates[np.argsort(buy_scores)[::-1]]
    adjusted[ranked[:deficit]] = 1
    return adjusted


# ---------------------------------------------------------------------------
# NoisyLinear Layer (Factorised Gaussian Noise)
# Replaces epsilon-greedy with learned, state-dependent exploration.
# The noise parameters decay naturally as the agent becomes confident,
# eliminating the need for manual epsilon scheduling.
# Reference: "Noisy Networks for Exploration" (Fortunato et al., 2018)
# ---------------------------------------------------------------------------
class NoisyLinear(nn.Module):
    """A linear layer with learnable noise for exploration."""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# Dueling Double DQN Network — Shared Asset Encoder Edition
#
# Input:  flattened portfolio state  (batch, N*F + N + 3)
# Output: per-stock Q-values         (batch, N, 3)
#
# Architecture:
#   1) Parse state into:
#      - per-stock market features (N, F)
#      - per-stock position scalar (N, 1)
#      - portfolio state (3), broadcast to every stock
#   2) SharedEncoder processes each stock independently -> (N, 64)
#   3) PortfolioContext = mean over stocks -> (1, 64), broadcast to all stocks
#   4) Concatenate stock embedding + context -> (N, 128)
#   5) Per-stock dueling heads output Q-values (N, 3)
# ---------------------------------------------------------------------------
class DRLAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, n_stocks: int,
                 noise_std: float = 0.7):
        """
        Args:
            state_dim:   N*F + N + 3  (flattened observation dimension).
            action_dim:  3  (Hold / Buy / Sell).
            n_stocks:    N  (number of tradeable tickers).
            noise_std:   Initial NoisyNet standard deviation (higher = more
                         exploration, important for the combinatorial action
                         space).
        """
        super(DRLAgent, self).__init__()
        self.n_stocks = n_stocks
        self.action_dim = action_dim

        # State layout is [N*F market features | N position values | 3 portfolio scalars]
        market_dim = state_dim - n_stocks - 3
        if market_dim <= 0 or market_dim % n_stocks != 0:
            raise ValueError(
                f"Invalid state_dim={state_dim} for n_stocks={n_stocks}. "
                "Expected state_dim = N*F + N + 3."
            )
        self.market_features_per_stock = market_dim // n_stocks
        self.features_per_stock = self.market_features_per_stock + 1 + 3

        # Shared encoder: same MLP applied independently to every stock
        self.shared_encoder = nn.Sequential(
            nn.Linear(self.features_per_stock, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        combined_dim = 128  # [stock_embedding(64) | portfolio_context(64)]

        # Per-stock value stream V_i(s): how favorable current state is for stock i
        self.value_stream = nn.Sequential(
            NoisyLinear(combined_dim, 128, std_init=noise_std),
            nn.ReLU(),
            NoisyLinear(128, 1, std_init=noise_std),
        )

        # Per-stock advantage stream A_i(s, a): action preference for stock i
        self.advantage_stream = nn.Sequential(
            NoisyLinear(combined_dim, 128, std_init=noise_std),
            nn.ReLU(),
            NoisyLinear(128, action_dim, std_init=noise_std),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, state_dim)
        Returns:
            Q-values: (batch, N, 3)
        """
        batch_size = x.size(0)

        market_end = self.n_stocks * self.market_features_per_stock
        position_end = market_end + self.n_stocks

        market_flat = x[:, :market_end]                   # (B, N*F)
        position_vals = x[:, market_end:position_end]     # (B, N)
        portfolio_state = x[:, position_end:]             # (B, 3)

        market_features = market_flat.reshape(
            batch_size, self.n_stocks, self.market_features_per_stock
        )                                                 # (B, N, F)
        position_feature = position_vals.unsqueeze(-1)    # (B, N, 1)
        portfolio_broadcast = portfolio_state.unsqueeze(1).expand(
            -1, self.n_stocks, -1
        )                                                 # (B, N, 3)

        per_stock_features = torch.cat(
            [market_features, position_feature, portfolio_broadcast],
            dim=2
        )                                                 # (B, N, F+4)

        encoded_flat = self.shared_encoder(
            per_stock_features.reshape(batch_size * self.n_stocks, self.features_per_stock)
        )                                                 # (B*N, 64)
        encoded_stocks = encoded_flat.reshape(batch_size, self.n_stocks, 64)  # (B, N, 64)

        portfolio_context = encoded_stocks.mean(dim=1, keepdim=True)  # (B, 1, 64)
        context_expanded = portfolio_context.expand(-1, self.n_stocks, -1)  # (B, N, 64)

        combined = torch.cat([encoded_stocks, context_expanded], dim=2)  # (B, N, 128)
        combined_flat = combined.reshape(batch_size * self.n_stocks, 128)  # (B*N, 128)

        value = self.value_stream(combined_flat).reshape(batch_size, self.n_stocks, 1)
        advantage = self.advantage_stream(combined_flat).reshape(
            batch_size, self.n_stocks, self.action_dim
        )

        # Per-stock dueling aggregation
        q_values = value + advantage - advantage.mean(dim=2, keepdim=True)
        return q_values                                    # (batch, N, 3)

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers for fresh exploration each step."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ---------------------------------------------------------------------------
# Prioritized Experience Replay (PER) with Sum-Tree
# Standard uniform replay undersamples rare but critical transitions
# (market crashes, regime changes, breakouts). PER samples proportional
# to TD-error magnitude — the agent learns most from transitions where
# it was "most wrong", accelerating escape from local optima.
# Reference: "Prioritized Experience Replay" (Schaul et al., 2016)
# ---------------------------------------------------------------------------
class SumTree:
    """Binary tree data structure for O(log n) prioritized sampling."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        return self.tree[0]
    
    def add(self, priority: float, data):
        idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Experience replay buffer with priority-based sampling using a Sum-Tree."""
    
    PER_e = 1e-6   # Small constant to ensure no transition has zero priority
    PER_a = 0.6    # Priority exponent: 0=uniform, 1=full prioritization
    PER_b = 0.4    # Importance sampling exponent (annealed to 1.0 during training)
    PER_b_increment = 0.001  # How fast beta anneals to 1.0
    
    def __init__(self, capacity: int = 500000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Store a transition. `action` and `reward` can be vectorised arrays.
        """
        reward = np.asarray(reward, dtype=np.float32)
        # New transitions get maximum priority so they are sampled at least once
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        self.tree.add(max_priority, (state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        self.PER_b = min(1.0, self.PER_b + self.PER_b_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            if data is None or (isinstance(data, (int, float)) and data == 0):
                # Fallback: sample a random valid entry
                s = random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
                if data is None or (isinstance(data, (int, float)) and data == 0):
                    continue
            
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)
        
        if len(batch) == 0:
            return None
        
        # Importance sampling weights to correct for the non-uniform sampling bias
        sampling_probabilities = np.array(priorities) / (self.tree.total() + 1e-8)
        is_weights = np.power(self.tree.n_entries * sampling_probabilities + 1e-8, -self.PER_b)
        is_weights /= is_weights.max()  # Normalize
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(idxs),
            np.array(is_weights, dtype=np.float32),
        )
    
    def update_priorities(self, idxs, td_errors):
        """Update priorities based on new TD-errors after a training step."""
        for idx, td_error in zip(idxs, td_errors):
            priority = (np.abs(td_error) + self.PER_e) ** self.PER_a
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries


# ---------------------------------------------------------------------------
# Legacy ReplayBuffer kept for backward compatibility
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Standard uniform replay buffer (kept for legacy/comparison)."""
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        reward = np.asarray(reward, dtype=np.float32)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32), 
            np.array(actions, dtype=np.int64), 
            np.array(rewards, dtype=np.float32), 
            np.array(next_states, dtype=np.float32), 
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# D3QN Trainer: Dueling Double DQN — Vectorised Multi-Asset Edition
#
# Key adaptations for the vectorised environment:
# - Actions are now (batch, N) integer arrays instead of scalars.
# - Q-values are (batch, N, 3); we gather per-stock Q for the chosen action.
# - Rewards are now per-stock vectors (batch, N), one reward per stock head.
# - TD errors are averaged across stocks per transition for PER priorities.
# ---------------------------------------------------------------------------
class DQNTrainer:
    def __init__(self, state_dim: int, action_dim: int, n_stocks: int,
                 lr: float = 1e-4, gamma: float = 0.99, tau: float = 0.005,
                 epsilon_start: float = 0.35, epsilon_end: float = 0.05,
                 epsilon_decay_steps: int = 100_000,
                 min_positions: int = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.n_stocks = n_stocks
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.action_steps = 0
        self.min_positions = (
            max(1, self.n_stocks // 10) if min_positions is None else min_positions
        )
        
        # Policy network (actively trains)
        self.policy_net = DRLAgent(state_dim, action_dim, n_stocks).to(self.device)
        
        # Target network (provides stable Q-value targets via Polyak averaging)
        self.target_net = DRLAgent(state_dim, action_dim, n_stocks).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, eps=1.5e-4)
        
        # Huber loss is more robust to reward outliers than MSE
        # In volatile markets, occasional large rewards/losses won't destabilize training
        self.criterion = nn.SmoothL1Loss(reduction='none')
        
        logger.info(f"Initialized Vectorised D3QN Trainer on {self.device} | "
                     f"State: {state_dim} | Actions: {action_dim}×{n_stocks} | τ: {tau}")

    def _current_epsilon(self) -> float:
        progress = min(1.0, self.action_steps / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def _sample_valid_random_actions(self, state: np.ndarray) -> np.ndarray:
        position_vals = _position_values_from_state(state, self.n_stocks)
        has_position = position_vals > 1e-8
        actions = np.zeros(self.n_stocks, dtype=np.int64)

        flat_idx = np.where(~has_position)[0]
        held_idx = np.where(has_position)[0]

        if len(flat_idx) > 0:
            actions[flat_idx] = np.random.choice([0, 1], size=len(flat_idx))
        if len(held_idx) > 0:
            actions[held_idx] = np.random.choice([0, 2], size=len(held_idx))
        return enforce_minimum_positions(
            actions=actions,
            q_values=np.zeros((self.n_stocks, self.action_dim), dtype=np.float32),
            state=state,
            n_stocks=self.n_stocks,
            min_positions=self.min_positions,
        )

    def train_step(self, states, actions, rewards, next_states, dones,
                   is_weights=None):
        """
        Double DQN training step for vectorised multi-asset actions.

        Actions shape:      (batch, N)
        Q-values shape:     (batch, N, 3)
        Rewards shape:      (batch, N)  — per-stock reward vectors
        Dones shape:        (batch,)
        """
        states      = torch.FloatTensor(states).to(self.device)       # (B, state_dim)
        actions     = torch.LongTensor(actions).to(self.device)       # (B, N)
        rewards     = torch.FloatTensor(rewards).to(self.device)      # (B, N)
        next_states = torch.FloatTensor(next_states).to(self.device)  # (B, state_dim)
        dones       = torch.FloatTensor(dones).to(self.device)        # (B,)

        # Current Q-values: pick Q of chosen action per stock
        q_all = self.policy_net(states)                               # (B, N, 3)
        current_q = q_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (B, N)

        with torch.no_grad():
            # DOUBLE DQN: policy net picks the action, target net evaluates
            next_q_policy = self.policy_net(next_states)              # (B, N, 3)
            next_q_policy = mask_invalid_q_values(
                next_q_policy, next_states, self.n_stocks)
            next_actions = next_q_policy.argmax(dim=2)                # (B, N)
            next_q_target = self.target_net(next_states) \
                .gather(2, next_actions.unsqueeze(-1)).squeeze(-1)    # (B, N)

            target_q = (rewards
                        + self.gamma * next_q_target
                        * (1 - dones.unsqueeze(1)))                   # (B, N)

        # Per-stock Huber loss
        td_errors = current_q - target_q                              # (B, N)
        loss = self.criterion(current_q, target_q)                    # (B, N)
        
        # Apply importance sampling weights from PER (if available)
        if is_weights is not None:
            is_weights_t = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)
            loss = (loss * is_weights_t).mean()
        else:
            loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients during volatile market data
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Reset NoisyNet noise for next forward pass
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        
        # Mean |TD-error| per transition for PER priority updates
        mean_td = td_errors.detach().abs().cpu().numpy().mean(axis=1)  # (B,)
        return loss.item(), mean_td

    def soft_sync_target_network(self):
        """
        Polyak averaging: θ_target = τ * θ_policy + (1 - τ) * θ_target
        
        Unlike hard sync (copy all weights every N steps), soft sync provides
        smooth, continuous updates that prevent the training instability
        caused by sudden target shifts — a major source of oscillation
        and premature convergence.
        """
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def sync_target_network(self):
        """Hard sync (legacy). Prefer soft_sync_target_network()."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        logger.debug("Target network hard-synced with Policy network.")

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select actions for all N stocks simultaneously.
        
        Returns:
            np.ndarray of shape (N,) with values in {0, 1, 2}.
        """
        with torch.no_grad():
            if self.policy_net.training:
                self.action_steps += 1
                if np.random.random() < self._current_epsilon():
                    return self._sample_valid_random_actions(state)
                self.policy_net.reset_noise()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)      # (1, N, 3)
            q_values = mask_invalid_q_values(q_values, state_tensor, self.n_stocks)
            q_np = q_values.squeeze(0).cpu().numpy()
            actions = q_np.argmax(axis=1)
            return enforce_minimum_positions(
                actions=actions,
                q_values=q_np,
                state=state,
                n_stocks=self.n_stocks,
                min_positions=self.min_positions,
            )

    def save_checkpoint(self, filepath: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'n_stocks': self.n_stocks,
        }, filepath)
        logger.info(f"Model checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            # Legacy format: just state_dict
            self.policy_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)
        logger.info(f"Model loaded from {filepath}")
