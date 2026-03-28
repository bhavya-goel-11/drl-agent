# Institutional-Grade DRL Agent for Algorithmic Trading

## Project Description

This repository hosts a robust, end-to-end algorithmic trading system powered by Deep Reinforcement Learning (DRL). Utilizing a PyTorch-based Deep Q-Network (DQN) architecture, the agent learns optimal trading strategies (Buy, Sell, Hold) based on historical financial data and 31 engineered technical indicators across the full Nifty-50 universe.

The pipeline is **100% Git-Native** — all code execution happens on Google Colab with free GPU, while all state (processed data, model weights, backtest reports) is automatically versioned and pushed back to this GitHub repository via Git LFS.

## 🏗️ Architecture & Project Structure

```
drl-agent/
├── colab_pipeline.ipynb     ← Master Colab notebook (run this)
├── data/                    ← Processed .parquet data (Git LFS)
├── models/                  ← Trained .pth weights (Git LFS)
├── reports/                 ← Backtest PNGs & CSVs (Git LFS)
├── data_pipeline/
│   ├── loader.py            ← yfinance fetch → feature eng → parquet
│   └── features.py          ← 31-feature engineering (pandas-ta)
├── drl_models/
│   ├── agent.py             ← DQN + Target Network + Replay Buffer
│   ├── env.py               ← OpenAI Gym trading environment
│   └── train.py             ← Training loop with periodic git-sync
├── backtesting/
│   ├── engine.py            ← Event-driven multi-asset backtest
│   ├── evaluate.py          ← OOS evaluation entry point
│   └── metrics.py           ← Sharpe, Max Drawdown calculators
├── execution_engine/        ← (Scaffolded) Live trading OMS
├── utils/
│   └── git_sync.py          ← Git commit/push automation
└── requirements.txt
```

---

## 🛠️ Prerequisites

1. A **Google account** with access to [Google Colab](https://colab.research.google.com)
2. A **GitHub Personal Access Token (PAT)** with `repo` scope
   - Create at: [github.com/settings/tokens](https://github.com/settings/tokens) → Classic → Scope: `repo`
3. **Git LFS** enabled on this repository (one-time setup):
   ```bash
   git lfs install
   git lfs track "data/*.parquet" "models/*.pth" "reports/*.png"
   git add .gitattributes && git commit -m "Enable Git LFS"
   ```

---

## 🚀 Running the Full Pipeline (Google Colab)

### Option A: One-Click (Recommended)

1. Open [`colab_pipeline.ipynb`](colab_pipeline.ipynb) in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially
4. Enter your GitHub PAT when prompted

The notebook will automatically:
- Clone this repo
- Install all dependencies in a venv
- Execute the 3-stage pipeline
- Push all results back to GitHub

### Option B: Manual (Local or Any Environment)

```bash
# Clone and setup
git clone https://github.com/bhavya-goel-11/drl-agent.git
cd drl-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Stage 1: Fetch data & engineer features
python3 -m data_pipeline.loader

# Stage 2: Train the DRL agent
python3 -m drl_models.train

# Stage 3: Out-of-sample backtest
python3 -m backtesting.evaluate
```

---

## 📊 Pipeline Stages

### Stage 1: Data Pipeline
Downloads OHLCV data for 46 Nifty-50 stocks + `^NSEI` benchmark (2008-2026) via `yfinance`, applies 31 technical indicators (SMA, RSI, MACD, Bollinger Bands, ATR, ADX, ROC, CMF, VWAP, RS Momentum), and saves the result as a compressed Parquet file.

### Stage 2: DRL Training
Trains a Double DQN with Experience Replay on pre-2023 data across all tickers simultaneously. The agent learns to Buy/Sell/Hold based on the 31-feature state space. Model weights are checkpointed every 50 episodes and pushed to GitHub.

### Stage 3: Out-of-Sample Backtesting
Evaluates the frozen model on unseen post-2023 data with independent ₹1L accounts per stock. Generates equity curves, alpha distribution charts, Sharpe Ratios, and Max Drawdown metrics.

---

## 📁 Output Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Processed Data | `data/processed_data.parquet` | 31-feature engineered dataset |
| Best Model | `models/best_model.pth` | Highest-reward model weights |
| Final Model | `models/final_model.pth` | End-of-training weights |
| Equity Curve | `reports/equity_curve.png` | AI vs Benchmark portfolio |
| Alpha Chart | `reports/alpha_distribution.png` | Per-stock alpha bars |
| Metrics CSV | `reports/ticker_comparative_metrics.csv` | Full performance table |
| Trade Ledger | `reports/detailed_trade_ledger.csv` | Every trade executed |
