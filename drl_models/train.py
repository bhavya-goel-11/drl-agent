import pandas as pd
import numpy as np
import torch
import math
import random
from drl_models.env import TradingEnv
from drl_models.agent import DQNTrainer, PrioritizedReplayBuffer
from loguru import logger

# ---------------------------------------------------------------------------
# Curriculum Training with Regime-Aware Progression
#
# Premature convergence often happens because the agent is thrown into a
# chaotic multi-stock universe from day 1 and locks onto the first pattern
# that works (momentum, sticky stocks, etc.).
#
# Curriculum learning introduces stocks gradually by volatility:
#   Phase 1: Low-vol trending stocks (learn basic buy/sell timing)
#   Phase 2: Medium-vol stocks (learn risk management)
#   Phase 3: Full universe (generalize across regimes)
#
# Reference: QTNet POMDP framework (2312.15730), practical recommendations
# ---------------------------------------------------------------------------

def _classify_tickers_by_volatility(multi_ticker_data: dict) -> dict:
    """
    Classify tickers into volatility terciles for curriculum training.
    Returns dict with 'low', 'medium', 'high' keys mapping to ticker lists.
    """
    volatilities = {}
    for ticker, df in multi_ticker_data.items():
        if 'close' in df.columns and len(df) > 60:
            returns = df['close'].pct_change().dropna()
            volatilities[ticker] = returns.std() * np.sqrt(252)  # Annualized vol
    
    if not volatilities:
        # Fallback: all tickers in one bucket
        return {'low': list(multi_ticker_data.keys()), 'medium': [], 'high': []}
    
    sorted_tickers = sorted(volatilities.items(), key=lambda x: x[1])
    n = len(sorted_tickers)
    tercile_1 = n // 3
    tercile_2 = 2 * n // 3
    
    classified = {
        'low': [t[0] for t in sorted_tickers[:tercile_1]],
        'medium': [t[0] for t in sorted_tickers[tercile_1:tercile_2]],
        'high': [t[0] for t in sorted_tickers[tercile_2:]],
    }
    
    logger.info(f"Volatility Classification — Low: {len(classified['low'])} | "
                f"Medium: {len(classified['medium'])} | High: {len(classified['high'])}")
    
    for level, tickers in classified.items():
        if tickers:
            vols = [volatilities[t] for t in tickers]
            logger.debug(f"  {level.upper()}: vol range [{min(vols):.2%}, {max(vols):.2%}] — {tickers[:5]}...")
    
    return classified


def _get_curriculum_tickers(classified: dict, episode: int, total_episodes: int) -> list:
    """Return the appropriate ticker subset based on training phase."""
    phase_1_end = total_episodes * 0.30  # First 30%: low vol only
    phase_2_end = total_episodes * 0.65  # Next 35%: low + medium
    
    if episode < phase_1_end:
        pool = classified['low']
    elif episode < phase_2_end:
        pool = classified['low'] + classified['medium']
    else:
        pool = classified['low'] + classified['medium'] + classified['high']
    
    return pool if pool else list(classified.get('low', []))


def _cosine_annealing_lr(optimizer, episode: int, total_episodes: int,
                         lr_max: float, lr_min: float = 1e-6):
    """
    Cosine annealing learning rate schedule.
    Starts at lr_max, smoothly decreases to lr_min following a cosine curve.
    This prevents learning rate from being too high during fine-tuning phases,
    which is a common cause of oscillation and convergence failure.
    """
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * episode / total_episodes))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    logger.info("=" * 80)
    logger.info("Starting D3QN Training Pipeline (Anti-Convergence Edition)")
    logger.info("=" * 80)
    
    # 1. Fetch live historical data from the TimescaleDB instance
    from database import SessionLocal, HistoricalData
    from data_pipeline.features import FeatureEngineer
    
    db = SessionLocal()
    try:
        logger.info("Fetching all available tickers from PostgreSQL (Training Firewall: Pre-2023)...")
        records = db.query(HistoricalData).filter(HistoricalData.date < "2023-01-01").order_by(HistoricalData.date.asc()).all()
        
        if not records:
            logger.error("No data found in database! Make sure 'loader.py' was run first.")
            return
            
        data = [{
            'symbol': r.symbol,
            'date': r.date,
            'open': r.open,
            'high': r.high,
            'low': r.low,
            'close': r.close,
            'volume': r.volume
        } for r in records]
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        # Ensure the index is a proper datetime object for precise slicing
        df.index = pd.to_datetime(df.index) 
        logger.info(f"Loaded {len(df)} base records from db.")
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return
    finally:
        db.close()
        
    # 2. Add algorithmic features (indicators) & drop NaNs
    engineer = FeatureEngineer()
    engineered_df = engineer.add_technical_indicators(df)
    logger.info(f"Engineered DataFrame Shape (after dropping NaNs): {engineered_df.shape}")
    
    # 3. Group the data by ticker into a dictionary
    multi_ticker_data = {}
    for ticker, group in engineered_df.groupby('symbol'):
        multi_ticker_data[ticker] = group.copy()
        
    logger.info(f"Grouped into {len(multi_ticker_data)} separate ticker dataframes for training.")
    
    # 4. Classify tickers by volatility for curriculum learning
    classified_tickers = _classify_tickers_by_volatility(multi_ticker_data)
    
    # Init Env - Passing the MULTI-TICKER data structure
    env = TradingEnv(multi_ticker_data, initial_balance=10000)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n 
    
    # --- Hyperparameter Configuration ---
    lr = 3e-4             # Initial learning rate (will be cosine-annealed)
    gamma = 0.99          # Discount factor
    episodes = 600        # Extended for curriculum phases
    batch_size = 128      # Larger batch for more stable gradients
    tau = 0.005           # Polyak averaging coefficient for soft target sync
    
    # 5. Initialize the D3QN Agent and Prioritized Replay Buffer
    trainer = DQNTrainer(state_dim, action_dim, lr=lr, gamma=gamma, tau=tau)
    buffer = PrioritizedReplayBuffer(capacity=200000)
    
    logger.info(f"Environment ready. State={state_dim}, Actions={action_dim}")
    logger.info(f"Training Config: LR={lr} | γ={gamma} | τ={tau} | Batch={batch_size}")
    logger.info(f"Buffer: PrioritizedReplayBuffer(200K) | Architecture: Dueling Double DQN + NoisyNet")
    logger.info("Commencing Curriculum Training Loop...")
    
    frame_count = 0
    
    # High-Water Mark Tracker
    best_reward = -float('inf')
    best_sharpe_approx = -float('inf')
    
    # Early stopping patience
    patience = 80
    no_improve_count = 0
    
    # Rolling reward tracker for convergence monitoring
    reward_history = []
    
    # 6. The Core Optimization Loop
    for episode in range(episodes):
        # Cosine annealing LR
        current_lr = _cosine_annealing_lr(trainer.optimizer, episode, episodes, lr)
        
        # Curriculum: select appropriate ticker subset
        available_tickers = _get_curriculum_tickers(classified_tickers, episode, episodes)
        
        # Override environment's ticker pool for this episode
        env.tickers = available_tickers
        
        state_output = env.reset()
        state = state_output[0] if isinstance(state_output, tuple) else state_output
        
        total_reward = 0
        episode_returns = []
        done = False
        episode_loss = 0
        loss_count = 0
        
        while not done:
            frame_count += 1
            
            # NoisyNet handles exploration — no epsilon needed!
            action = trainer.select_action(state)
            
            # Execute Action
            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, info = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_output
                
            total_reward += reward
            episode_returns.append(info.get('step_return', 0.0))
            
            # Store transition in Prioritized Replay Buffer
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            
            # Train the Neural Network via Prioritized Batch Sampling
            if len(buffer) >= batch_size:
                sample = buffer.sample(batch_size)
                if sample is not None:
                    states, actions, rewards, next_states, dones, idxs, is_weights = sample
                    loss, td_errors = trainer.train_step(
                        states, actions, rewards, next_states, dones, is_weights
                    )
                    # Update priorities in the replay buffer
                    buffer.update_priorities(idxs, td_errors)
                    episode_loss += loss
                    loss_count += 1
            
            # Soft sync target network every step (Polyak averaging)
            trainer.soft_sync_target_network()
                
        # Episode metrics
        reward_history.append(total_reward)
        avg_loss = episode_loss / max(loss_count, 1)
        
        # Approximate episode Sharpe for model selection
        ep_returns = np.array(episode_returns)
        ep_sharpe = 0.0
        if len(ep_returns) > 5 and ep_returns.std() > 0:
            ep_sharpe = (ep_returns.mean() / ep_returns.std()) * np.sqrt(252)
        
        # Log episode metrics
        if (episode + 1) % 10 == 0:
            profit = env.net_worth - env.initial_balance
            avg_reward_50 = np.mean(reward_history[-50:]) if len(reward_history) >= 50 else np.mean(reward_history)
            phase = "Phase1-LowVol" if episode < episodes * 0.30 else ("Phase2-MedVol" if episode < episodes * 0.65 else "Phase3-FullUniverse")
            
            logger.info(
                f"Ep {episode + 1:>4}/{episodes} | "
                f"R: {total_reward:>8.3f} | "
                f"Avg50: {avg_reward_50:>7.3f} | "
                f"Sharpe: {ep_sharpe:>6.2f} | "
                f"Loss: {avg_loss:>7.5f} | "
                f"P&L: ${profit:>8.2f} | "
                f"LR: {current_lr:.2e} | "
                f"{phase} [{env.active_ticker}]"
            )
            
        # Check High-Water Mark (use Sharpe as primary selection criterion)
        if ep_sharpe > best_sharpe_approx and episode > 30:
            logger.info(f"  ★ New best Sharpe: {best_sharpe_approx:.2f} → {ep_sharpe:.2f}. Saving checkpoint...")
            best_sharpe_approx = ep_sharpe
            trainer.save_checkpoint("drl_models/best_universal_dqn_trader.pth")
            no_improve_count = 0
        elif total_reward > best_reward and episode > 10:
            best_reward = total_reward
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Early stopping check
        if no_improve_count >= patience and episode > episodes * 0.5:
            logger.warning(f"Early stopping triggered at episode {episode + 1} "
                          f"(no improvement for {patience} episodes)")
            break
            
    # 7. Checkpoint the final trained weights
    trainer.save_checkpoint("drl_models/universal_dqn_trader.pth")
    
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info(f"  Best Sharpe (approx): {best_sharpe_approx:.2f}")
    logger.info(f"  Best Reward: {best_reward:.3f}")
    logger.info(f"  Total Frames: {frame_count:,}")
    logger.info("  Model weights secured for backtesting.")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()