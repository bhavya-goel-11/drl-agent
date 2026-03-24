import pandas as pd
import numpy as np
import torch
import random
from drl_models.env import TradingEnv
from drl_models.agent import DQNTrainer, ReplayBuffer
from loguru import logger

def main():
    logger.info("Starting Institutional DRL Training Pipeline...")
    
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
    
    # Init Env - Passing the MULTI-TICKER data structure
    env = TradingEnv(multi_ticker_data, initial_balance=10000)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n 
    
    # --- Hyperparameter Configuration ---
    lr = 1e-4             # Learning Rate
    gamma = 0.99          # Discount Factor
    episodes = 500        # Full convergence run
    batch_size = 64
    sync_target_frames = 1000 
    
    # Epsilon-Greedy parameters
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995 
    
    # 4. Initialize the upgraded Agent and the Experience Replay Buffer
    trainer = DQNTrainer(state_dim, action_dim, lr=lr, gamma=gamma)
    buffer = ReplayBuffer(capacity=100000)
    
    logger.info(f"Environment ready. State={state_dim}, Actions={action_dim}")
    logger.info("Commencing Live Market Training Loop...")
    
    frame_count = 0
    
    # High-Water Mark Tracker
    best_reward = -float('inf')
    
    # 5. The Core Optimization Loop
    for episode in range(episodes):
        state_output = env.reset()
        state = state_output[0] if isinstance(state_output, tuple) else state_output
        
        total_reward = 0
        done = False
        
        while not done:
            frame_count += 1
            
            # Action Selection: Epsilon-Greedy
            if random.random() < epsilon:
                action = env.action_space.sample() # Explore
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(trainer.device)
                    action = trainer.policy_net(state_tensor).argmax().item() # Exploit
            
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
            logger.info(f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward:.3f} | Profit: ${profit:.2f} | Epsilon: {epsilon:.3f}")
            
        # Check High-Water Mark
        if total_reward > best_reward and episode > 10: # Wait a few episodes before saving "bests"
            logger.info(f"New high-water mark! Reward improved from {best_reward:.3f} to {total_reward:.3f}. Saving model...")
            best_reward = total_reward
            trainer.save_checkpoint("drl_models/best_universal_dqn_trader.pth")
            
    # 6. Checkpoint the final trained weights
    trainer.save_checkpoint("drl_models/universal_dqn_trader.pth")
    logger.info("Training simulation completed. Final model weights secured for backtesting.")

if __name__ == "__main__":
    main()