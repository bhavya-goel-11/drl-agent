import pandas as pd
import numpy as np
import torch
import math
from drl_models.env import VectorizedTradingEnv
from drl_models.agent import DQNTrainer, PrioritizedReplayBuffer
from loguru import logger

# ---------------------------------------------------------------------------
# Vectorised Multi-Asset D3QN Training Pipeline
#
# This loop trains a single D3QN agent that simultaneously decides
# Buy / Hold / Sell for ALL N stocks at every timestep.
#
# Key differences from the single-stock curriculum loop:
#   • Data is a date-aligned (T, N, F) tensor — no per-ticker randomisation.
#   • Episodes use stratified random 1-year sub-windows so every calendar
#     year in the training set is sampled proportionately.
#   • The action is a vector of N discrete choices, not a scalar.
#   • Reward is portfolio-level (Sharpe + drawdown + concentration penalties).
#   • 6000 episodes with extended patience to account for the combinatorial
#     action space (3^N possible action vectors per step).
# ---------------------------------------------------------------------------

TRAIN_CUTOFF = "2023-01-01"


def _cosine_annealing_warm_restarts(optimizer, episode: int, total_episodes: int,
                                     lr_max: float, lr_min: float = 1e-6,
                                     n_restarts: int = 3):
    """
    Cosine annealing with warm restarts.
    
    Splits training into `n_restarts` cycles, each with a full cosine anneal.
    At the start of each cycle, LR resets to lr_max, giving the agent a
    "fresh kick" that helps escape plateaus and local minima.
    """
    cycle_len = total_episodes / n_restarts
    cycle_pos = episode % cycle_len
    lr = lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos(math.pi * cycle_pos / cycle_len))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    logger.info("=" * 80)
    logger.info("Starting Vectorised Multi-Asset D3QN Training Pipeline")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # 1.  Fetch live historical data from TimescaleDB
    # ------------------------------------------------------------------
    from database import SessionLocal, HistoricalData
    from data_pipeline.features import FeatureEngineer, align_multi_ticker_data

    db = SessionLocal()
    try:
        logger.info("Fetching all tickers from PostgreSQL "
                     f"(Training Firewall: pre-{TRAIN_CUTOFF})...")
        records = (db.query(HistoricalData)
                     .filter(HistoricalData.date < TRAIN_CUTOFF)
                     .order_by(HistoricalData.date.asc())
                     .all())

        if not records:
            logger.error("No data found in database! Run 'loader.py' first.")
            return

        data = [{
            'symbol': r.symbol,
            'date':   r.date,
            'open':   r.open,
            'high':   r.high,
            'low':    r.low,
            'close':  r.close,
            'volume': r.volume,
        } for r in records]

        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        logger.info(f"Loaded {len(df)} base records from db.")

    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return
    finally:
        db.close()

    # ------------------------------------------------------------------
    # 2.  Feature engineering
    # ------------------------------------------------------------------
    engineer = FeatureEngineer()
    engineered_df = engineer.add_technical_indicators(df)
    logger.info(f"Engineered DataFrame shape: {engineered_df.shape}")

    # ------------------------------------------------------------------
    # 3.  Group by ticker
    # ------------------------------------------------------------------
    multi_ticker_data = {}
    for ticker, group in engineered_df.groupby('symbol'):
        multi_ticker_data[ticker] = group.copy()
    logger.info(f"Grouped into {len(multi_ticker_data)} ticker DataFrames.")

    # ------------------------------------------------------------------
    # 4.  Align into (T, N, F) tensor
    # ------------------------------------------------------------------
    data_3d, tickers, dates, columns = align_multi_ticker_data(multi_ticker_data)
    n_stocks = len(tickers)
    n_features = len(columns)

    logger.info(f"Aligned tensor: T={data_3d.shape[0]}, "
                f"N={n_stocks}, F={n_features}")
    logger.info(f"Tickers ({n_stocks}): {tickers}")

    # ------------------------------------------------------------------
    # 5.  Create vectorised environment
    # ------------------------------------------------------------------
    env = VectorizedTradingEnv(
        data_3d=data_3d,
        tickers=tickers,
        dates=dates,
        columns=columns,
        initial_balance=10_000_000.0,   # ₹1 Cr portfolio
        commission=0.002,
        window_size=126,                # ~6 month sub-windows
    )

    state_dim  = env.observation_space.shape[0]
    action_dim = 3  # Hold / Buy / Sell

    # ------------------------------------------------------------------
    # 6.  Hyperparameters
    # ------------------------------------------------------------------
    lr            = 2e-4          # Lower LR for stability with 46 stocks
    gamma         = 0.99
    episodes      = 6000
    batch_size    = 256
    tau           = 0.001         # Slower target sync for more stable Q-targets
    warmup_episodes = 50          # Fill buffer before training
    target_sync_interval = 4      # Sync target net every N steps (not every step)

    # ------------------------------------------------------------------
    # 7.  Initialise agent & buffer
    # ------------------------------------------------------------------
    trainer = DQNTrainer(state_dim, action_dim, n_stocks, lr=lr,
                         gamma=gamma, tau=tau)
    buffer  = PrioritizedReplayBuffer(capacity=500_000)

    logger.info(f"Env ready.  State={state_dim}  Actions=3×{n_stocks}")
    logger.info(f"Config: LR={lr} | γ={gamma} | τ={tau} | Batch={batch_size}")
    logger.info(f"Buffer: PER(500K) | Architecture: Vectorised Dueling D3QN + NoisyNet")
    logger.info(f"Episodes: {episodes} | Warmup: {warmup_episodes}")
    logger.info("Commencing training loop...")

    # ------------------------------------------------------------------
    # 8.  Training loop
    # ------------------------------------------------------------------
    frame_count = 0
    best_avg_reward = -float('inf')    # Track rolling average, not single episode
    patience     = 500
    no_improve   = 0
    reward_history = []
    avg_reward_history = []            # Track Avg50 for smooth improvement detection

    for episode in range(episodes):
        current_lr = _cosine_annealing_warm_restarts(
            trainer.optimizer, episode, episodes, lr, n_restarts=3)

        state, _ = env.reset()
        total_reward = 0.0
        episode_returns = []
        episode_loss = 0.0
        loss_count = 0
        done = False

        # Per-stock epsilon: decays from 30% to 5% over training
        # Each stock independently has this chance of a random action
        epsilon = max(0.05, 0.30 * (1.0 - episode / (episodes * 0.7)))

        while not done:
            frame_count += 1

            action = trainer.select_action(state, epsilon=epsilon)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step_ret = ((info.get('portfolio_value', env.initial_balance)
                         - env.initial_balance) / env.initial_balance)
            episode_returns.append(step_ret)

            # Store transition
            buffer.push(state, action, reward, next_state, done)
            state = next_state

            # Train after warmup
            if episode >= warmup_episodes and len(buffer) >= batch_size:
                sample = buffer.sample(batch_size)
                if sample is not None:
                    (s, a, r, ns, d, idxs, is_w) = sample
                    loss, td_errors = trainer.train_step(
                        s, a, r, ns, d, is_w)
                    buffer.update_priorities(idxs, td_errors)
                    episode_loss += loss
                    loss_count += 1

            # Soft sync target network at controlled intervals (not every step)
            if frame_count % target_sync_interval == 0:
                trainer.soft_sync_target_network()

        # --- Episode metrics ---
        reward_history.append(total_reward)
        avg_loss = episode_loss / max(loss_count, 1)

        ep_returns = np.array(episode_returns)
        ep_sharpe = 0.0
        if len(ep_returns) > 5 and ep_returns.std() > 0:
            ep_sharpe = (ep_returns.mean() / ep_returns.std()) * np.sqrt(252)

        # --- Rolling average for smooth improvement tracking ---
        avg50 = (np.mean(reward_history[-50:])
                 if len(reward_history) >= 50
                 else np.mean(reward_history))
        avg_reward_history.append(avg50)

        # --- Logging ---
        if (episode + 1) % 10 == 0:
            pv = info.get('portfolio_value', env.initial_balance)
            profit = pv - env.initial_balance
            positions = info.get('num_positions', 0)
            sigma_mean = trainer.policy_net.get_sigma_stats()

            logger.info(
                f"Ep {episode+1:>4}/{episodes} | "
                f"R: {total_reward:>8.3f} | "
                f"Avg50: {avg50:>7.3f} | "
                f"Sharpe: {ep_sharpe:>6.2f} | "
                f"Loss: {avg_loss:>7.5f} | "
                f"P&L: ₹{profit:>12,.0f} | "
                f"LR: {current_lr:.2e} | "
                f"ε: {epsilon:.3f} | "
                f"σ: {sigma_mean:.4f} | "
                f"Pos: {positions}/{n_stocks} | "
                f"{info.get('window', '')}"
            )

        # --- Improvement tracking based on ROLLING AVERAGE ---
        # Using Avg50 instead of single-episode metrics eliminates noise
        # from lucky/unlucky window draws and tracks true policy improvement.
        improved = False

        if (episode > warmup_episodes + 50 and
                avg50 > best_avg_reward + 0.01):  # Require meaningful improvement
            logger.info(f"  ★ New best Avg50 reward: {best_avg_reward:.3f} → "
                        f"{avg50:.3f}.  Saving checkpoint...")
            best_avg_reward = avg50
            trainer.save_checkpoint(
                "drl_models/best_universal_dqn_trader.pth")
            improved = True

        if improved:
            no_improve = 0
        else:
            no_improve += 1

            # --- Adaptive exploration recovery ---
            if no_improve == 150:
                logger.warning(f"No improvement for {no_improve} episodes. "
                               f"Re-injecting NoisyNet sigma...")
                trainer.policy_net.reset_sigma()
                trainer.target_net.reset_sigma()

            elif no_improve == 300:
                logger.warning(f"No improvement for {no_improve} episodes. "
                               f"Sigma reset + hard target sync...")
                trainer.policy_net.reset_sigma()
                trainer.sync_target_network()  # Hard reset target to break cycles

            elif no_improve == 450:
                logger.warning(f"No improvement for {no_improve} episodes. "
                               f"Full exploration reset — re-randomising advantage stream...")
                # Nuclear option: re-initialize the advantage stream weights
                # while preserving the learned feature backbone and value stream
                for module in trainer.policy_net.advantage_stream.modules():
                    if isinstance(module, torch.nn.Linear):
                        torch.nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            module.bias.data.zero_()
                trainer.policy_net.reset_sigma()
                trainer.target_net.load_state_dict(
                    trainer.policy_net.state_dict())

        # --- Early stopping ---
        if no_improve >= patience and episode > episodes * 0.5:
            logger.warning(
                f"Early stopping at episode {episode+1} "
                f"(no improvement for {patience} episodes)")
            break

        # --- Sigma collapse detection ---
        if (episode + 1) % 100 == 0:
            sigma_mean = trainer.policy_net.get_sigma_stats()
            if sigma_mean < 0.001:
                logger.warning(f"⚠ NoisyNet sigma collapsed to {sigma_mean:.6f}! "
                               f"Resetting sigma to prevent exploration death...")
                trainer.policy_net.reset_sigma()
                trainer.target_net.reset_sigma()

    # ------------------------------------------------------------------
    # 9.  Save final weights
    # ------------------------------------------------------------------
    trainer.save_checkpoint("drl_models/universal_dqn_trader.pth")

    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info(f"  Best Avg50 Reward:    {best_avg_reward:.3f}")
    logger.info(f"  Total Frames:         {frame_count:,}")
    logger.info(f"  Tickers:              {n_stocks}")
    logger.info("  Model weights saved for backtesting.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()