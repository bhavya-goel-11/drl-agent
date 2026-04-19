import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random
from collections import deque
from loguru import logger


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

    def reset_sigma(self):
        """Reset only the noise scales, preserving learned weights."""
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def get_sigma_mean(self) -> float:
        """Return mean absolute sigma for monitoring noise collapse."""
        return float(self.weight_sigma.data.abs().mean().item())


# ---------------------------------------------------------------------------
# Dueling Double DQN Network — Vectorised Multi-Asset Edition
#
# Input:  flattened portfolio state  (batch, N*F + N + 3)
# Output: per-stock Q-values         (batch, N, 3)
#
# Architecture:
#   Shared backbone  →  Value stream V(s)        (batch, 1)
#                    →  Advantage stream A(s, a)  (batch, N*3) → (batch, N, 3)
#   Q(s, a_i) = V(s) + A(s, a_i) - mean_a A(s, ·_i)   per stock
#
# The shared value stream captures "how good is this portfolio state overall?"
# The per-stock advantage captures "which action is marginally best for stock i?"
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
        
        # Shared feature extraction backbone with LayerNorm for training stability
        # Wider layers (768) to handle the high-dimensional multi-stock state
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Value stream: V(s) — how good is this portfolio state?
        self.value_stream = nn.Sequential(
            NoisyLinear(512, 256, std_init=noise_std),
            nn.ReLU(),
            NoisyLinear(256, 1, std_init=noise_std),
        )
        
        # Advantage stream: A(s, a) for all N stocks × 3 actions
        self.advantage_stream = nn.Sequential(
            NoisyLinear(512, 512, std_init=noise_std),
            nn.ReLU(),
            NoisyLinear(512, n_stocks * action_dim, std_init=noise_std),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, state_dim)
        Returns:
            Q-values: (batch, N, 3)
        """
        features = self.feature_layer(x)

        value = self.value_stream(features)                # (batch, 1)
        advantage = self.advantage_stream(features)        # (batch, N*3)
        advantage = advantage.view(-1, self.n_stocks, self.action_dim)  # (batch, N, 3)

        # Q(s,a_i) = V(s) + (A(s,a_i) - mean_a A(s,·_i))
        q_values = (value.unsqueeze(1)
                    + advantage
                    - advantage.mean(dim=2, keepdim=True))
        return q_values                                    # (batch, N, 3)
    
    def reset_noise(self):
        """Reset noise in all NoisyLinear layers for fresh exploration each step."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def reset_sigma(self):
        """Force re-initialization of noise scales if they collapse."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_sigma()

    def get_sigma_stats(self) -> float:
        """Monitor NoisyNet sigma magnitude — early warning for noise collapse."""
        sigmas = []
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                sigmas.append(module.get_sigma_mean())
        return float(np.mean(sigmas)) if sigmas else 0.0


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
        Store a transition.  `action` can be a scalar or an ndarray (vectorised).
        """
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
# - A single portfolio-level scalar reward is broadcast to all N stock heads.
# - TD errors are averaged across stocks per transition for PER priorities.
# ---------------------------------------------------------------------------
class DQNTrainer:
    def __init__(self, state_dim: int, action_dim: int, n_stocks: int,
                 lr: float = 1e-4, gamma: float = 0.99, tau: float = 0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.n_stocks = n_stocks
        
        # Policy network (actively trains)
        self.policy_net = DRLAgent(state_dim, action_dim, n_stocks).to(self.device)
        
        # Target network (provides stable Q-value targets via Polyak averaging)
        self.target_net = DRLAgent(state_dim, action_dim, n_stocks).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr,
                                     eps=1e-5, weight_decay=1e-5)
        
        # Huber loss is more robust to reward outliers than MSE
        # In volatile markets, occasional large rewards/losses won't destabilize training
        self.criterion = nn.SmoothL1Loss(reduction='none')
        
        logger.info(f"Initialized Vectorised D3QN Trainer on {self.device} | "
                     f"State: {state_dim} | Actions: {action_dim}×{n_stocks} | τ: {tau}")

    def train_step(self, states, actions, rewards, next_states, dones,
                   is_weights=None):
        """
        Double DQN training step for vectorised multi-asset actions.

        Actions shape:      (batch, N)
        Q-values shape:     (batch, N, 3)
        Rewards/dones:      (batch,)  — portfolio-level scalars
        """
        states      = torch.FloatTensor(states).to(self.device)       # (B, state_dim)
        actions     = torch.LongTensor(actions).to(self.device)       # (B, N)
        rewards     = torch.FloatTensor(rewards).to(self.device)      # (B,)
        next_states = torch.FloatTensor(next_states).to(self.device)  # (B, state_dim)
        dones       = torch.FloatTensor(dones).to(self.device)        # (B,)

        # Current Q-values: pick Q of chosen action per stock
        q_all = self.policy_net(states)                               # (B, N, 3)
        current_q = q_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (B, N)

        with torch.no_grad():
            # DOUBLE DQN: policy net picks the action, target net evaluates
            next_q_policy = self.policy_net(next_states)              # (B, N, 3)
            next_actions = next_q_policy.argmax(dim=2)                # (B, N)
            next_q_target = self.target_net(next_states) \
                .gather(2, next_actions.unsqueeze(-1)).squeeze(-1)    # (B, N)

            # Broadcast portfolio reward to all stock heads
            target_q = (rewards.unsqueeze(1)
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
        
        # Tighter gradient clipping to prevent volatile market windows from
        # destabilizing representations. With 46 stock heads all receiving the
        # same noisy portfolio reward, gradient variance is inherently high.
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
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

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """
        Select actions for all N stocks simultaneously.
        
        Uses per-stock epsilon perturbation instead of all-or-nothing:
        each stock independently has an `epsilon` chance of a random action.
        This ensures exploration is distributed across the portfolio rather
        than the agent either fully exploring or fully exploiting.
        
        Returns:
            np.ndarray of shape (N,) with values in {0, 1, 2}.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)      # (1, N, 3)
            actions = q_values.squeeze(0).argmax(dim=1)   # (N,)
            actions = actions.cpu().numpy()

        # Per-stock epsilon perturbation
        if epsilon > 0:
            random_mask = np.random.random(self.n_stocks) < epsilon
            random_actions = np.random.randint(0, self.action_dim, size=self.n_stocks)
            actions = np.where(random_mask, random_actions, actions)

        return actions

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