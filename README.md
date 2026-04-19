# Hybrid Machine Learning & DRL Portfolio Optimizer

This repository contains a state-of-the-art Quantitative Trading architecture designed for academic research. It implements a vectorised, multi-asset portfolio optimization system that rigorously compares Deep Reinforcement Learning (DRL) against traditional Machine Learning algorithms. 

The capstone of this architecture is a mathematically robust **Capital-Split Ensemble** that successfully neutralizes model-specific noise to generate superior risk-adjusted returns across out-of-sample data.

## 📊 Core Architecture & Models

The evaluation pipeline runs a direct, Out-Of-Sample (OOS) comparative backtest across 4 distinct systems:

1. **D3QN Agent (Deep Dueling Double Q-Network)**:
   - A highly advanced Reinforcement Learning model with a custom multi-asset environment.
   - Designed to understand complex, non-linear market regimes by evaluating the entire portfolio state (including drawdowns and active capital allocation).
2. **XGBoost Classifier (Gradient Boosting)**:
   - A deterministic, high-conviction decision tree model that trades actively on momentum signals.
   - Fine-tuned using randomized hyperparameter search to maximize alpha generation.
3. **Random Forest Classifier (Bagging)**:
   - A robust baseline model that acts as a stabilizing benchmark for traditional ML performance.
4. **The Capital-Split Ensemble (Meta-Policy)**:
   - **The ultimate strategy:** This ensemble proves that heterogeneous models can be combined to achieve superior risk-adjusted returns.
   - It divides total capital equally across the DRL, XGBoost, and Random Forest models. Each model independently manages its sub-portfolio, and the final equity curve is the weighted sum of their un-correlated returns.
   - This prevents the catastrophic whipsawing associated with action-level voting across models of varying trading frequencies.

## 🚀 Quickstart

The entire backtesting pipeline, data fetching, and reporting infrastructure are encapsulated within a Dockerized environment.

### 1. Build and Run the Environment
Make sure you have Docker and Docker Compose installed, then spin up the TimescaleDB database and the main algorithm application:

```bash
docker-compose up -d --build
```

### 2. Execute the Evaluation Pipeline
Run the vectorised evaluation script to automatically fetch historical data, align technical indicators across all 46 tickers, compute OOS performance, and generate the final ensemble portfolio.

```bash
docker exec -it algo-app python -m backtesting.evaluate
```

## 📈 Outputs & Reports
Running the pipeline will automatically generate a highly detailed `reports/` folder. This folder contains all the necessary data for academic publication:
- `comparative_equity_curves.png`: A stunning visual comparison of all model performances across the out-of-sample period.
- `model_comparison_metrics.csv` & `.json`: Comprehensive quantitative metrics (Sharpe, Max Drawdown, Calmar, Alpha, Return) for every model.
- `*_detailed_report.png`: Individual tear sheets for every model containing sub-plots of their equity curves and max drawdowns.
- `*_detailed_trade_ledger.csv`: A transparent, row-by-row log of every execution made by the models.

## ⚙️ Repository Structure
- `backtesting/`: Contains the core vectorised backtesting engines.
  - `engine.py`: Base engine for the D3QN model.
  - `ml_baselines.py`: Trains and tests XGBoost and Random Forest.
  - `ensemble_engine.py`: Combines the models via Capital-Split logic.
  - `evaluate.py`: The master orchestrator.
- `drl_models/`: Contains the architecture and pre-trained weights (`.pth`) for the D3QN Agent.
- `data_pipeline/`: Responsible for fetching market data and calculating technical indicators.
- `reports/`: Automatically generated outputs. 

## 📝 Research Conclusion
This architecture proves that while highly parameterized models like DRL might struggle with the combinatorial explosion of action spaces in individual stock selection, they provide incredibly powerful diversification benefits when mathematically ensembled with traditional, high-frequency decision trees.
