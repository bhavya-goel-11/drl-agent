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


# ---------------------------------------------------------------------------
# Dueling Double DQN Network (D3QN)
# Splits Q(s,a) into V(s) + A(s,a) streams.
# - Value stream: learns which market states are inherently good/bad
# - Advantage stream: learns which action is marginally better
# This prevents premature convergence because in many market states
# the action choice barely matters — the dueling architecture learns this.
# Reference: "Dueling Network Architectures" (Wang et al., 2016)
# ---------------------------------------------------------------------------
class DRLAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(DRLAgent, self).__init__()
        
        # Shared feature extraction backbone with LayerNorm for training stability
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        
        # Value stream: V(s) — how good is this market state?
        self.value_stream = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, 1),
        )
        
        # Advantage stream: A(s,a) — how much better is action a vs average?
        self.advantage_stream = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, action_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        # Subtracting mean advantage ensures identifiability
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values
    
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
    
    def __init__(self, capacity: int = 200000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
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
# D3QN Trainer: Dueling Double DQN with PER, NoisyNets, Huber Loss
# Combines all anti-convergence techniques into a single training loop:
# - Double DQN: policy net selects action, target net evaluates
# - Huber loss: more robust to outliers than MSE (volatile market rewards)
# - Soft target update (Polyak): smooth weight transfer prevents oscillation
# ---------------------------------------------------------------------------
class DQNTrainer:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-4, 
                 gamma: float = 0.99, tau: float = 0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        
        # Policy network (actively trains)
        self.policy_net = DRLAgent(state_dim, action_dim).to(self.device)
        
        # Target network (provides stable Q-value targets via Polyak averaging)
        self.target_net = DRLAgent(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, eps=1.5e-4)
        
        # Huber loss is more robust to reward outliers than MSE
        # In volatile markets, occasional large rewards/losses won't destabilize training
        self.criterion = nn.SmoothL1Loss(reduction='none')
        
        logger.info(f"Initialized D3QN Trainer on {self.device} | "
                     f"State: {state_dim} | Action: {action_dim} | τ: {tau}")

    def train_step(self, states, actions, rewards, next_states, dones,
                   is_weights=None):
        """
        Double DQN training step with importance sampling correction.
        
        Key difference from vanilla DQN:
        - Action SELECTION uses policy_net (which action is best?)
        - Action EVALUATION uses target_net (what's that action's value?)
        This decoupling eliminates the maximization bias that causes
        Q-value overestimation → premature convergence to greedy policies.
        """
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values from policy network
        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            # DOUBLE DQN: policy net picks the action, target net evaluates it
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Element-wise Huber loss
        td_errors = current_q - target_q
        loss = self.criterion(current_q, target_q)
        
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
        
        return loss.item(), td_errors.detach().cpu().numpy().flatten()

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

    def select_action(self, state: np.ndarray) -> int:
        """
        Action selection using NoisyNet (no epsilon needed).
        The noise in NoisyLinear layers provides state-dependent exploration
        that naturally decays as the network becomes confident.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def save_checkpoint(self, filepath: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
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