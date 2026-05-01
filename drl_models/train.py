import pandas as pd
import numpy as np
import torch
import math
import os
import json
import uuid
from datetime import datetime, timezone
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
#   • 3000 episodes with extended patience to account for the combinatorial
#     action space (3^N possible action vectors per step).
# ---------------------------------------------------------------------------

TRAIN_END = "2020-12-31"
VAL_END = "2022-12-31"


def _cosine_annealing_lr(optimizer, episode: int, total_episodes: int,
                         lr_max: float, lr_min: float = 1e-6):
    """
    Cosine annealing learning rate schedule.
    Starts at lr_max, smoothly decreases to lr_min following a cosine curve.
    """
    lr = lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos(math.pi * episode / total_episodes))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def _build_validation_state(
    data_3d: np.ndarray,
    current_step: int,
    holdings: np.ndarray,
    cash: float,
    initial_balance: float,
    close_idx: int,
) -> np.ndarray:
    frame = data_3d[current_step]  # (N, F)

    lookback_start = max(0, current_step - 200)
    window = data_3d[lookback_start: current_step + 1]

    rolling_mean = window.mean(axis=0)
    rolling_std = window.std(axis=0)
    rolling_std = np.where(rolling_std < 1e-8, 1e-8, rolling_std)

    normalised = (frame - rolling_mean) / rolling_std
    normalised = np.nan_to_num(normalised, nan=0.0)
    market_flat = normalised.flatten()

    close_prices = data_3d[current_step, :, close_idx]
    position_vals = (holdings * close_prices) / initial_balance

    total_equity = cash + np.sum(holdings * close_prices)
    drawdown = 0.0
    portfolio_state = np.array([
        cash / initial_balance,
        total_equity / initial_balance,
        drawdown,
    ])

    obs = np.concatenate([market_flat, position_vals, portfolio_state])
    return obs.astype(np.float32)


def _evaluate_on_validation(
    trainer: DQNTrainer,
    data_3d: np.ndarray,
    tickers: list,
    columns: list,
    initial_balance: float = 10_000_000.0,
    commission: float = 0.002,
):
    if data_3d.shape[0] < 2:
        return {
            'return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'trades': 0,
            'avg_positions': 0.0,
            'action_hold_pct': 0.0,
            'action_buy_pct': 0.0,
            'action_sell_pct': 0.0,
        }

    close_idx = columns.index('close') if 'close' in columns else -1
    open_idx = columns.index('open') if 'open' in columns else close_idx

    n_stocks = len(tickers)
    cash = initial_balance
    holdings = np.zeros(n_stocks, dtype=np.float64)
    portfolio_history = []
    action_counts = np.zeros(3, dtype=np.int64)
    position_counts = []
    trades = 0

    was_training = trainer.policy_net.training
    trainer.policy_net.eval()

    with torch.no_grad():
        for step in range(0, data_3d.shape[0] - 1):
            close_prices = data_3d[step, :, close_idx]
            pv = float(cash + np.sum(holdings * close_prices))
            portfolio_history.append(pv)

            state = _build_validation_state(
                data_3d=data_3d,
                current_step=step,
                holdings=holdings,
                cash=cash,
                initial_balance=initial_balance,
                close_idx=close_idx,
            )
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(trainer.device)
            q_vals = trainer.policy_net(state_tensor)  # (1, N, 3)
            actions = q_vals.squeeze(0).argmax(dim=1).cpu().numpy()
            action_counts += np.bincount(actions, minlength=3)[:3]

            exec_prices = data_3d[step + 1, :, open_idx]
            buy_mask = (actions == 1) & (holdings == 0)
            sell_mask = (actions == 2) & (holdings > 0)
            trades += int(buy_mask.sum() + sell_mask.sum())

            for i in np.where(sell_mask)[0]:
                revenue = holdings[i] * exec_prices[i]
                fee = revenue * commission
                cash += (revenue - fee)
                holdings[i] = 0

            num_buys = int(buy_mask.sum())
            if num_buys > 0:
                cash_per_stock = cash / num_buys
                for i in np.where(buy_mask)[0]:
                    max_cost = exec_prices[i] * (1 + commission)
                    if max_cost <= 0:
                        continue
                    shares = int(cash_per_stock // max_cost)
                    if shares > 0:
                        cost = shares * exec_prices[i] * (1 + commission)
                        cash -= cost
                        holdings[i] = shares
            position_counts.append(int(np.sum(holdings > 0)))

    final_close = data_3d[-1, :, close_idx]
    final_pv = float(cash + np.sum(holdings * final_close))
    portfolio_history.append(final_pv)

    eq = np.array(portfolio_history, dtype=np.float64)
    total_return = ((eq[-1] / eq[0]) - 1) * 100 if eq[0] > 0 else 0.0

    pct = pd.Series(eq).pct_change().dropna()
    sharpe = 0.0
    if len(pct) > 1 and pct.std() > 0:
        sharpe = (pct.mean() / pct.std()) * np.sqrt(252)

    rolling_max = np.maximum.accumulate(eq)
    dd = (eq - rolling_max) / np.where(rolling_max == 0, 1.0, rolling_max)
    max_dd = dd.min() * 100
    total_actions = int(action_counts.sum())
    action_pct = (
        action_counts / total_actions
        if total_actions > 0
        else np.zeros(3, dtype=np.float64)
    )

    if was_training:
        trainer.policy_net.train()

    return {
        'return': float(total_return),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'trades': int(trades),
        'avg_positions': float(np.mean(position_counts)) if position_counts else 0.0,
        'action_hold_pct': float(action_pct[0]),
        'action_buy_pct': float(action_pct[1]),
        'action_sell_pct': float(action_pct[2]),
    }


def _date_range_str(date_index: pd.DatetimeIndex) -> str:
    if len(date_index) == 0:
        return "N/A"
    return f"{date_index[0].date()} → {date_index[-1].date()}"


def _split_boundary(split_date: str, dates: pd.DatetimeIndex) -> pd.Timestamp:
    """Return a split timestamp compatible with the aligned date index timezone."""
    ts = pd.Timestamp(split_date)
    index_tz = pd.DatetimeIndex(dates).tz

    if index_tz is None:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    if ts.tzinfo is None:
        return ts.tz_localize(index_tz)
    return ts.tz_convert(index_tz)


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
                    f"(Split: train<= {TRAIN_END}, val<= {VAL_END}, oos>{VAL_END})...")
        records = (db.query(HistoricalData)
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
    # 5.  Strict temporal split
    # ------------------------------------------------------------------
    train_end_dt = _split_boundary(TRAIN_END, dates)
    val_end_dt = _split_boundary(VAL_END, dates)

    train_mask = dates <= train_end_dt
    val_mask = (dates > train_end_dt) & (dates <= val_end_dt)
    oos_mask = dates > val_end_dt

    train_data_3d = data_3d[train_mask]
    val_data_3d = data_3d[val_mask]
    oos_data_3d = data_3d[oos_mask]

    train_dates = dates[train_mask]
    val_dates = dates[val_mask]
    oos_dates = dates[oos_mask]

    logger.info(
        f"Split sizes | Train: {train_data_3d.shape[0]} steps "
        f"[{_date_range_str(train_dates)}] | "
        f"Val: {val_data_3d.shape[0]} steps "
        f"[{_date_range_str(val_dates)}] | "
        f"OOS: {oos_data_3d.shape[0]} steps "
        f"[{_date_range_str(oos_dates)}]"
    )

    if train_data_3d.shape[0] < 253:
        logger.error("Training split is too short for 252-day windows.")
        return
    if val_data_3d.shape[0] < 2:
        logger.error("Validation split is too short to evaluate.")
        return

    # ------------------------------------------------------------------
    # 6.  Create vectorised environment
    # ------------------------------------------------------------------
    env = VectorizedTradingEnv(
        data_3d=train_data_3d,
        tickers=tickers,
        dates=train_dates,
        columns=columns,
        initial_balance=10_000_000.0,   # ₹1 Cr portfolio
        commission=0.002,
        window_size=252,                # ~1 year sub-windows
    )

    state_dim  = env.observation_space.shape[0]
    action_dim = 3  # Hold / Buy / Sell

    # ------------------------------------------------------------------
    # 7.  Hyperparameters
    # ------------------------------------------------------------------
    lr            = 5e-4
    gamma         = 0.99
    episodes      = 3000
    batch_size    = 256
    tau           = 0.005
    warmup_episodes = 50     # Fill buffer before training
    val_eval_interval = 10

    # ------------------------------------------------------------------
    # 8.  Initialise agent & buffer
    # ------------------------------------------------------------------
    trainer = DQNTrainer(state_dim, action_dim, n_stocks, lr=lr,
                         gamma=gamma, tau=tau)
    buffer  = PrioritizedReplayBuffer(capacity=500_000)

    run_config = {
        'train_end': TRAIN_END,
        'val_end': VAL_END,
        'lr': lr,
        'gamma': gamma,
        'episodes': episodes,
        'batch_size': batch_size,
        'tau': tau,
        'warmup_episodes': warmup_episodes,
        'val_eval_interval': val_eval_interval,
        'buffer_capacity': 500_000,
        'window_size': 252,
        'commission': 0.002,
        'initial_balance': 10_000_000.0,
    }

    logger.info(f"Env ready.  State={state_dim}  Actions=3×{n_stocks}")
    logger.info(f"Config: LR={lr} | γ={gamma} | τ={tau} | Batch={batch_size}")
    logger.info(f"Buffer: PER(500K) | Architecture: Vectorised Dueling D3QN + NoisyNet")
    logger.info(
        f"Episodes: {episodes} | Warmup: {warmup_episodes} | "
        f"Val interval: {val_eval_interval}"
    )
    logger.info("Commencing training loop...")

    # ------------------------------------------------------------------
    # 9.  Training loop
    # ------------------------------------------------------------------
    frame_count = 0
    best_sharpe  = -float('inf')
    best_reward  = -float('inf')
    patience     = 250
    no_improve   = 0
    reward_history = []
    last_val_metrics = {'return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0}

    for episode in range(episodes):
        current_lr = _cosine_annealing_lr(trainer.optimizer, episode,
                                          episodes, lr)

        state, _ = env.reset()
        total_reward = 0.0
        episode_returns = []
        episode_loss = 0.0
        loss_count = 0
        done = False

        while not done:
            frame_count += 1

            # NoisyNet handles exploration — no epsilon needed
            action = trainer.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward_scalar = float(np.mean(reward))
            total_reward += reward_scalar
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

            # Soft sync target network every step
            trainer.soft_sync_target_network()

        # --- Episode metrics ---
        reward_history.append(total_reward)
        avg_loss = episode_loss / max(loss_count, 1)

        ep_returns = np.array(episode_returns)
        ep_sharpe = 0.0
        if len(ep_returns) > 5 and ep_returns.std() > 0:
            ep_sharpe = (ep_returns.mean() / ep_returns.std()) * np.sqrt(252)

        # --- Logging ---
        if (episode + 1) % 10 == 0:
            pv = info.get('portfolio_value', env.initial_balance)
            profit = pv - env.initial_balance
            avg50 = (np.mean(reward_history[-50:])
                     if len(reward_history) >= 50
                     else np.mean(reward_history))
            positions = info.get('num_positions', 0)

            logger.info(
                f"Ep {episode+1:>4}/{episodes} | "
                f"R: {total_reward:>8.3f} | "
                f"Avg50: {avg50:>7.3f} | "
                f"Sharpe: {ep_sharpe:>6.2f} | "
                f"Loss: {avg_loss:>7.5f} | "
                f"P&L: ₹{profit:>12,.0f} | "
                f"LR: {current_lr:.2e} | "
                f"Pos: {positions}/{n_stocks} | "
                f"{info.get('window', '')}"
            )

        # --- Periodic validation ---
        if (episode + 1) % val_eval_interval == 0:
            last_val_metrics = _evaluate_on_validation(
                trainer=trainer,
                data_3d=val_data_3d,
                tickers=tickers,
                columns=columns,
                initial_balance=10_000_000.0,
                commission=0.002,
            )
            logger.info(
                f"Validation Ep {episode+1:>4}/{episodes} | "
                f"Return: {last_val_metrics['return']:>7.2f}% | "
                f"Sharpe: {last_val_metrics['sharpe']:>6.2f} | "
                f"MaxDD: {last_val_metrics['max_drawdown']:>7.2f}%"
            )

        # --- Checkpointing ---
        if ep_sharpe > best_sharpe and episode > warmup_episodes:
            logger.info(f"  ★ New best Sharpe: {best_sharpe:.2f} → "
                        f"{ep_sharpe:.2f}.  Saving checkpoint...")
            best_sharpe = ep_sharpe
            trainer.save_checkpoint(
                "drl_models/best_universal_dqn_trader.pth")
            no_improve = 0
        elif total_reward > best_reward and episode > warmup_episodes:
            best_reward = total_reward
            no_improve = 0
        else:
            no_improve += 1

        # --- Early stopping ---
        if no_improve >= patience and episode > episodes * 0.5:
            logger.warning(
                f"Early stopping at episode {episode+1} "
                f"(no improvement for {patience} episodes)")
            break

    # ------------------------------------------------------------------
    # 10.  Save final weights
    # ------------------------------------------------------------------
    final_checkpoint_path = "drl_models/universal_dqn_trader.pth"
    trainer.save_checkpoint(final_checkpoint_path)

    final_val_metrics = _evaluate_on_validation(
        trainer=trainer,
        data_3d=val_data_3d,
        tickers=tickers,
        columns=columns,
        initial_balance=10_000_000.0,
        commission=0.002,
    )
    if last_val_metrics != final_val_metrics:
        last_val_metrics = final_val_metrics

    # ------------------------------------------------------------------
    # 11. Run logging
    # ------------------------------------------------------------------
    os.makedirs("reports/experiments", exist_ok=True)
    ts = datetime.now(timezone.utc)
    run_id = f"d3qn_{ts.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    experiment_log = {
        'timestamp': ts.isoformat(),
        'run_id': run_id,
        'config': run_config,
        'final_validation_return': round(last_val_metrics['return'], 4),
        'final_validation_sharpe': round(last_val_metrics['sharpe'], 4),
        'final_validation_max_drawdown': round(last_val_metrics['max_drawdown'], 4),
        'checkpoint_path': final_checkpoint_path,
    }
    experiment_log_path = f"reports/experiments/{run_id}.json"
    with open(experiment_log_path, 'w') as f:
        json.dump(experiment_log, f, indent=2)
    logger.info(f"Run log saved: {experiment_log_path}")

    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info(f"  Best Sharpe (approx): {best_sharpe:.2f}")
    logger.info(f"  Best Reward:          {best_reward:.3f}")
    logger.info(
        f"  Final Validation:     Return={last_val_metrics['return']:.2f}% | "
        f"Sharpe={last_val_metrics['sharpe']:.2f} | "
        f"MaxDD={last_val_metrics['max_drawdown']:.2f}%"
    )
    logger.info(f"  Total Frames:         {frame_count:,}")
    logger.info(f"  Tickers:              {n_stocks}")
    logger.info("  Model weights saved for backtesting.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
