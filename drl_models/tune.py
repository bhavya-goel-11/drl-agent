import json
import os
from datetime import datetime, timezone

import numpy as np
import optuna
import pandas as pd

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
    buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

    episodes = 1500
    warmup_episodes = 50

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

            trainer.soft_sync_target_network()

        if (episode + 1) % 100 == 0:
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

    trial.set_user_attr("final_val_return", float(final_val_metrics['return']))
    trial.set_user_attr("final_val_max_drawdown", float(final_val_metrics['max_drawdown']))
    trial.set_user_attr("final_val_trades", int(final_val_metrics['trades']))
    trial.set_user_attr("final_val_avg_positions", float(final_val_metrics['avg_positions']))
    trial.set_user_attr("final_val_action_hold_pct", float(final_val_metrics['action_hold_pct']))
    trial.set_user_attr("final_val_action_buy_pct", float(final_val_metrics['action_buy_pct']))
    trial.set_user_attr("final_val_action_sell_pct", float(final_val_metrics['action_sell_pct']))
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
