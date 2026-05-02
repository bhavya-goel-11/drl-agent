import json
import os
from datetime import datetime, timezone

import numpy as np
import optuna
import pandas as pd
import torch

from loguru import logger

from database import SessionLocal, HistoricalData
from data_pipeline.features import FeatureEngineer, align_multi_ticker_data
from drl_models import train as drl_train
from drl_models.agent import DQNTrainer, PrioritizedReplayBuffer
from drl_models.env import VectorizedTradingEnv


_DATA_CACHE = None
NO_TRADE_SHARPE_PENALTY = 1.0


def _load_split_data():
    logger.info("Loading and preparing data for Optuna tuning...")
    db = SessionLocal()
    try:
        records = (
            db.query(HistoricalData)
            .order_by(HistoricalData.date.asc())
            .all()
        )
        if not records:
            raise RuntimeError("No data found in database. Run loader first.")

        data = [{
            'symbol': r.symbol,
            'date': r.date,
            'open': r.open,
            'high': r.high,
            'low': r.low,
            'close': r.close,
            'volume': r.volume,
        } for r in records]
    finally:
        db.close()

    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)

    engineer = FeatureEngineer()
    engineered_df = engineer.add_technical_indicators(df)

    multi_ticker_data = {}
    for ticker, group in engineered_df.groupby('symbol'):
        multi_ticker_data[ticker] = group.copy()

    data_3d, tickers, dates, columns = align_multi_ticker_data(multi_ticker_data)

    train_end_dt = drl_train._split_boundary(drl_train.TRAIN_END, dates)
    val_end_dt = drl_train._split_boundary(drl_train.VAL_END, dates)

    train_mask = dates <= train_end_dt
    val_mask = (dates > train_end_dt) & (dates <= val_end_dt)
    oos_mask = dates > val_end_dt

    split_data = {
        'tickers': tickers,
        'columns': columns,
        'train_data_3d': data_3d[train_mask],
        'val_data_3d': data_3d[val_mask],
        'oos_data_3d': data_3d[oos_mask],
        'train_dates': dates[train_mask],
        'val_dates': dates[val_mask],
        'oos_dates': dates[oos_mask],
    }

    logger.info(
        f"Tuning data ready | Train={split_data['train_data_3d'].shape[0]} "
        f"Val={split_data['val_data_3d'].shape[0]} "
        f"OOS={split_data['oos_data_3d'].shape[0]} | "
        f"Tickers={len(tickers)}"
    )
    return split_data


def _get_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = _load_split_data()
    return _DATA_CACHE


def _warm_start_policy(
    trainer: DQNTrainer,
    train_data_3d: np.ndarray,
    columns: list,
    initial_balance: float = 10_000_000.0,
    horizon: int = 20,
    buy_threshold: float = 0.03,
    sell_threshold: float = -0.03,
    sample_steps: int = 256,
    epochs: int = 3,
    batch_size: int = 64,
) -> tuple:
    """Give the policy a small supervised prior before RL fine-tuning."""
    if train_data_3d.shape[0] <= horizon + 1:
        return None, None

    close_idx = columns.index('close') if 'close' in columns else -1
    max_step = train_data_3d.shape[0] - horizon - 1
    sample_count = min(sample_steps, max_step)
    steps = np.linspace(0, max_step - 1, num=sample_count, dtype=np.int64)

    states = []
    labels = []
    n_stocks = train_data_3d.shape[1]

    for step in steps:
        close_now = train_data_3d[step, :, close_idx]
        close_future = train_data_3d[step + horizon, :, close_idx]
        future_ret = (close_future - close_now) / (close_now + 1e-12)

        flat_holdings = np.zeros(n_stocks, dtype=np.float64)
        flat_labels = np.zeros(n_stocks, dtype=np.int64)
        flat_labels[future_ret > buy_threshold] = 1
        states.append(drl_train._build_validation_state(
            data_3d=train_data_3d,
            current_step=int(step),
            holdings=flat_holdings,
            cash=initial_balance,
            initial_balance=initial_balance,
            close_idx=close_idx,
        ))
        labels.append(flat_labels)

        held_prices = np.maximum(close_now, 1e-12)
        held_holdings = (initial_balance / n_stocks) / held_prices
        held_labels = np.zeros(n_stocks, dtype=np.int64)
        held_labels[future_ret < sell_threshold] = 2
        states.append(drl_train._build_validation_state(
            data_3d=train_data_3d,
            current_step=int(step),
            holdings=held_holdings,
            cash=0.0,
            initial_balance=initial_balance,
            close_idx=close_idx,
        ))
        labels.append(held_labels)

    states_t = torch.FloatTensor(np.array(states, dtype=np.float32)).to(trainer.device)
    labels_t = torch.LongTensor(np.array(labels, dtype=np.int64)).to(trainer.device)

    was_training = trainer.policy_net.training
    trainer.policy_net.train()
    criterion = torch.nn.CrossEntropyLoss()

    order = torch.arange(states_t.shape[0], device=trainer.device)
    for _ in range(epochs):
        order = order[torch.randperm(order.numel(), device=trainer.device)]
        for start in range(0, order.numel(), batch_size):
            idx = order[start:start + batch_size]
            logits = trainer.policy_net(states_t[idx])
            loss = criterion(
                logits.reshape(-1, trainer.action_dim),
                labels_t[idx].reshape(-1),
            )
            trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.policy_net.parameters(), max_norm=10.0)
            trainer.optimizer.step()
            trainer.policy_net.reset_noise()

    trainer.sync_target_network()
    if not was_training:
        trainer.policy_net.eval()

    logger.info(
        f"Warm-started policy on {states_t.shape[0]} supervised states "
        f"(horizon={horizon}, buy>{buy_threshold:.1%}, sell<{sell_threshold:.1%})"
    )
    return states_t.detach(), labels_t.detach()


def objective(trial: optuna.Trial) -> float:
    window_size = trial.suggest_categorical("window_size", [126, 252, 504])
    lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
    gamma = trial.suggest_categorical("gamma", [0.95, 0.97, 0.99])
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    buffer_capacity = trial.suggest_categorical("buffer_capacity", [100000, 500000])
    reward_type = trial.suggest_categorical("reward_type", ['A', 'B', 'C', 'D'])

    data = _get_data()
    train_data_3d = data['train_data_3d']
    val_data_3d = data['val_data_3d']
    tickers = data['tickers']
    columns = data['columns']
    train_dates = data['train_dates']
    n_stocks = len(tickers)

    if train_data_3d.shape[0] <= window_size + 1:
        raise optuna.TrialPruned(
            f"Train split too short for window_size={window_size}."
        )
    if val_data_3d.shape[0] < 2:
        raise optuna.TrialPruned("Validation split too short.")

    env = VectorizedTradingEnv(
        data_3d=train_data_3d,
        tickers=tickers,
        dates=train_dates,
        columns=columns,
        initial_balance=10_000_000.0,
        commission=0.002,
        window_size=window_size,
        reward_type=reward_type,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = 3

    trainer = DQNTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        n_stocks=n_stocks,
        lr=lr,
        gamma=gamma,
        tau=tau,
    )
    demo_states, demo_labels = _warm_start_policy(
        trainer=trainer,
        train_data_3d=train_data_3d,
        columns=columns,
        initial_balance=10_000_000.0,
    )
    buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

    episodes = 1500
    warmup_episodes = 50
    val_eval_interval = 50
    no_trade_eval_count = 0
    best_objective = -float('inf')
    best_episode = 0
    best_val_metrics = None
    train_updates = 0
    bc_interval = 4
    bc_weight = 0.2

    for episode in range(episodes):
        drl_train._cosine_annealing_lr(
            optimizer=trainer.optimizer,
            episode=episode,
            total_episodes=episodes,
            lr_max=lr,
        )

        state, _ = env.reset()
        done = False

        while not done:
            action = trainer.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            state = next_state

            if episode >= warmup_episodes and len(buffer) >= batch_size:
                sample = buffer.sample(batch_size)
                if sample is not None:
                    s, a, r, ns, d, idxs, is_w = sample
                    _, td_errors = trainer.train_step(s, a, r, ns, d, is_w)
                    buffer.update_priorities(idxs, td_errors)
                    train_updates += 1

                    if (
                        demo_states is not None
                        and train_updates % bc_interval == 0
                    ):
                        demo_batch_size = min(batch_size, demo_states.shape[0])
                        demo_idx = torch.randint(
                            low=0,
                            high=demo_states.shape[0],
                            size=(demo_batch_size,),
                            device=trainer.device,
                        )
                        trainer.behavior_clone_step(
                            demo_states[demo_idx],
                            demo_labels[demo_idx],
                            bc_weight=bc_weight,
                        )

            trainer.soft_sync_target_network()

        if (episode + 1) % val_eval_interval == 0:
            val_metrics = drl_train._evaluate_on_validation(
                trainer=trainer,
                data_3d=val_data_3d,
                tickers=tickers,
                columns=columns,
                initial_balance=10_000_000.0,
                commission=0.002,
            )
            reported_sharpe = float(val_metrics['sharpe'])
            if val_metrics['trades'] == 0:
                reported_sharpe -= NO_TRADE_SHARPE_PENALTY
                no_trade_eval_count += 1
            else:
                no_trade_eval_count = 0
                if reported_sharpe > best_objective:
                    best_objective = reported_sharpe
                    best_episode = episode + 1
                    best_val_metrics = dict(val_metrics)

            logger.info(
                f"Trial {trial.number} Ep {episode + 1}/{episodes} | "
                f"Val Return={val_metrics['return']:.2f}% | "
                f"Sharpe={val_metrics['sharpe']:.4f} | "
                f"Objective={reported_sharpe:.4f} | "
                f"MaxDD={val_metrics['max_drawdown']:.2f}% | "
                f"Trades={val_metrics['trades']} | "
                f"AvgPos={val_metrics['avg_positions']:.1f} | "
                f"Actions H/B/S="
                f"{val_metrics['action_hold_pct']:.1%}/"
                f"{val_metrics['action_buy_pct']:.1%}/"
                f"{val_metrics['action_sell_pct']:.1%}"
            )

            trial.report(reported_sharpe, episode + 1)
            if no_trade_eval_count >= 1 and best_val_metrics is not None:
                logger.info(
                    f"Trial {trial.number} stopping after first post-trade "
                    f"collapse; returning best checkpoint from episode "
                    f"{best_episode} with objective={best_objective:.4f}"
                )
                break
            if no_trade_eval_count >= 2:
                raise optuna.TrialPruned(
                    f"Pruned after {no_trade_eval_count} consecutive no-trade "
                    f"validation checks at episode {episode + 1}."
                )
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned at episode {episode + 1}: objective={reported_sharpe:.4f}"
                )

    final_val_metrics = drl_train._evaluate_on_validation(
        trainer=trainer,
        data_3d=val_data_3d,
        tickers=tickers,
        columns=columns,
        initial_balance=10_000_000.0,
        commission=0.002,
    )
    final_sharpe = float(final_val_metrics['sharpe'])
    objective_value = final_sharpe
    if final_val_metrics['trades'] == 0:
        objective_value -= NO_TRADE_SHARPE_PENALTY

    if final_val_metrics['trades'] > 0 and objective_value > best_objective:
        best_objective = objective_value
        best_episode = episodes
        best_val_metrics = dict(final_val_metrics)

    if best_val_metrics is not None:
        objective_value = best_objective
    else:
        best_episode = 0

    trial.set_user_attr("final_val_return", float(final_val_metrics['return']))
    trial.set_user_attr("final_val_max_drawdown", float(final_val_metrics['max_drawdown']))
    trial.set_user_attr("final_val_trades", int(final_val_metrics['trades']))
    trial.set_user_attr("final_val_avg_positions", float(final_val_metrics['avg_positions']))
    trial.set_user_attr("final_val_action_hold_pct", float(final_val_metrics['action_hold_pct']))
    trial.set_user_attr("final_val_action_buy_pct", float(final_val_metrics['action_buy_pct']))
    trial.set_user_attr("final_val_action_sell_pct", float(final_val_metrics['action_sell_pct']))
    trial.set_user_attr("best_val_episode", int(best_episode))
    trial.set_user_attr("best_val_objective", float(objective_value))
    if best_val_metrics is not None:
        trial.set_user_attr("best_val_return", float(best_val_metrics['return']))
        trial.set_user_attr("best_val_sharpe", float(best_val_metrics['sharpe']))
        trial.set_user_attr("best_val_max_drawdown", float(best_val_metrics['max_drawdown']))
        trial.set_user_attr("best_val_trades", int(best_val_metrics['trades']))
        trial.set_user_attr("best_val_avg_positions", float(best_val_metrics['avg_positions']))
    logger.info(
        f"Trial {trial.number} final | "
        f"Val Return={final_val_metrics['return']:.2f}% | "
        f"Sharpe={final_sharpe:.4f} | "
        f"Objective={objective_value:.4f} | "
        f"MaxDD={final_val_metrics['max_drawdown']:.2f}% | "
        f"Trades={final_val_metrics['trades']} | "
        f"AvgPos={final_val_metrics['avg_positions']:.1f} | "
        f"Actions H/B/S="
        f"{final_val_metrics['action_hold_pct']:.1%}/"
        f"{final_val_metrics['action_buy_pct']:.1%}/"
        f"{final_val_metrics['action_sell_pct']:.1%}"
    )
    if best_val_metrics is not None:
        logger.info(
            f"Trial {trial.number} best | "
            f"Ep={best_episode} | "
            f"Return={best_val_metrics['return']:.2f}% | "
            f"Sharpe={best_val_metrics['sharpe']:.4f} | "
            f"Objective={objective_value:.4f} | "
            f"MaxDD={best_val_metrics['max_drawdown']:.2f}% | "
            f"Trades={best_val_metrics['trades']} | "
            f"AvgPos={best_val_metrics['avg_positions']:.1f}"
        )
    return objective_value


if __name__ == "__main__":
    os.makedirs("reports/experiments", exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name="drl_agent_hpo",
    )
    study.optimize(objective, n_trials=50)

    best_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "study_name": study.study_name,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_trial_number": int(study.best_trial.number),
        "best_trial_user_attrs": dict(study.best_trial.user_attrs),
    }

    best_params_path = "reports/experiments/best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_payload, f, indent=2)

    logger.info(f"Best parameters saved to {best_params_path}")
