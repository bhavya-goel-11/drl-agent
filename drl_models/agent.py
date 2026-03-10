import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from loguru import logger

class DRLAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(DRLAgent, self).__init__()
        # Standard Feed-Forward MLP
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Stores historical transitions to prevent catastrophic forgetting and break data autocorrelation."""
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

class DQNTrainer:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.action_dim = action_dim
        
        # 1. The Main Network (actively trains)
        self.policy_net = DRLAgent(state_dim, action_dim).to(self.device)
        
        # 2. The Target Network (provides stable Q-value targets)
        self.target_net = DRLAgent(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net does not train
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        logger.info(f"Initialized DQNTrainer on {self.device} | State: {state_dim} | Action: {action_dim}")

    def train_step(self, states, actions, rewards, next_states, dones):
        """Calculates loss using the Bellman equation and performs one gradient descent step."""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Get current Q values from policy network
        current_q = self.policy_net(states).gather(1, actions)

        # Get max next Q values from target network (detached from graph)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # Bellman Equation: Q(s, a) = r + gamma * max(Q(s', a')) * (1 - done)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # Optimize the model
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients during volatile market data
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

    def sync_target_network(self):
        """Copies weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        logger.debug("Target network synced with Policy network.")

    def save_checkpoint(self, filepath: str):
        torch.save(self.policy_net.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        logger.info(f"Model loaded from {filepath}")