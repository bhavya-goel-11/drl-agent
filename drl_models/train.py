"""
DRL Training Pipeline — Parquet-first, Colab-native.

Loads processed data from the local .parquet file, trains the DQN agent,
saves the best model to models/best_model.pth, and git-syncs checkpoints
every 50 episodes so progress survives Colab disconnects.
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import random
from loguru import logger

from drl_models.env import TradingEnv
from drl_models.agent import DQNTrainer, ReplayBuffer
from data_pipeline.loader import load_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
FINAL_MODEL_PATH = MODEL_DIR / "final_model.pth"


def main():
    logger.info("═" * 60)
    logger.info("  DRL TRAINING PIPELINE — Parquet-Native")
    logger.info("═" * 60)

    # ── 1. Load Data from Parquet (Training Firewall: Pre-2023) ────────────
    logger.info("Loading engineered features from Parquet…")
    train_df, _ = load_data(split_date="2023-01-01")

    if train_df.empty:
        logger.error("Training DataFrame is empty. Run `python3 -m data_pipeline.loader` first.")
        return

    logger.info(f"Training data: {train_df.shape[0]} rows × {train_df.shape[1]} cols")

    # ── 2. Group by Ticker → Multi-Asset Dictionary ───────────────────────
    ticker_col = "ticker" if "ticker" in train_df.columns else "symbol"
    multi_ticker_data = {}
    for ticker, group in train_df.groupby(ticker_col):
        multi_ticker_data[ticker] = group.copy()

    logger.info(f"Grouped into {len(multi_ticker_data)} separate ticker dataframes for training.")

    # ── 3. Initialize Environment ─────────────────────────────────────────
    env = TradingEnv(multi_ticker_data, initial_balance=10000)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # ── 4. Hyperparameter Configuration ───────────────────────────────────
    lr = 1e-4
    gamma = 0.99
    episodes = 500
    batch_size = 64
    sync_target_frames = 1000
    checkpoint_interval = 50  # Git-sync every N episodes

    # Epsilon-Greedy parameters
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    # ── 5. Initialize Agent & Replay Buffer ───────────────────────────────
    trainer = DQNTrainer(state_dim, action_dim, lr=lr, gamma=gamma)
    buffer = ReplayBuffer(capacity=100000)

    # Resume from checkpoint if available
    os.makedirs(MODEL_DIR, exist_ok=True)
    if BEST_MODEL_PATH.exists():
        logger.info(f"Resuming from checkpoint: {BEST_MODEL_PATH}")
        trainer.load_checkpoint(str(BEST_MODEL_PATH))

    logger.info(f"Environment ready. State={state_dim}, Actions={action_dim}")
    logger.info("Commencing Training Loop…")

    frame_count = 0
    best_reward = -float('inf')

    # ── 6. Core Optimization Loop ─────────────────────────────────────────
    for episode in range(episodes):
        state_output = env.reset()
        state = state_output[0] if isinstance(state_output, tuple) else state_output

        total_reward = 0
        done = False

        while not done:
            frame_count += 1

            # Action Selection: Epsilon-Greedy
            if random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(trainer.device)
                    action = trainer.policy_net(state_tensor).argmax().item()  # Exploit

            # Execute Action
            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_output

            total_reward += reward

            # Store transition in Replay Buffer
            buffer.push(state, action, reward, next_state, done)
            state = next_state

            # Train the Neural Network via Batch Sampling
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                loss = trainer.train_step(states, actions, rewards, next_states, dones)

            # Sync Target Network
            if frame_count % sync_target_frames == 0:
                trainer.sync_target_network()

        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Log episode metrics
        if (episode + 1) % 10 == 0:
            profit = env.net_worth - env.initial_balance
            logger.info(
                f"Episode {episode + 1}/{episodes} | "
                f"Total Reward: {total_reward:.3f} | "
                f"Profit: ${profit:.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

        # ── High-Water Mark Checkpoint ────────────────────────────────────
        if total_reward > best_reward and episode > 10:
            logger.info(f"New high-water mark! {best_reward:.3f} → {total_reward:.3f}. Saving…")
            best_reward = total_reward
            trainer.save_checkpoint(str(BEST_MODEL_PATH))

        # ── Periodic Git Sync (every N episodes) ─────────────────────────
        if (episode + 1) % checkpoint_interval == 0:
            logger.info(f"Git-syncing model checkpoint at episode {episode + 1}…")
            try:
                from utils.git_sync import sync_model
                sync_model(episode=episode + 1)
            except Exception as e:
                logger.warning(f"Git sync skipped: {e}")

    # ── 7. Save Final Model ───────────────────────────────────────────────
    trainer.save_checkpoint(str(FINAL_MODEL_PATH))
    logger.success("Training complete. Final model weights saved.")

    # Final git sync
    try:
        from utils.git_sync import sync_model
        sync_model(episode=episodes)
    except Exception as e:
        logger.warning(f"Final git sync skipped: {e}")


if __name__ == "__main__":
    main()