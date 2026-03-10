# Institutional-Grade DRL Agent for Algorithmic Trading

## Project Description

This repository hosts a robust, end-to-end algorithmic trading system powered by Deep Reinforcement Learning (DRL). Utilizing a PyTorch-based Deep Q-Network (DQN) architecture, the agent learns optimal trading strategies (Buy, Sell, Hold) based on historical financial data and engineered technical indicators. 

The pipeline includes an automated chronologically-ordered data fetcher connected to a high-performance **PostgreSQL + TimescaleDB** backend, a fully functional reinforcement learning training loop with an **Experience Replay Buffer** and **Target Network Syncing**, and an event-driven out-of-sample backtesting engine to evaluate the learned strategies with realistic portfolio metrics (Sharpe Ratio, Max Drawdown).

## 🏗️ Architecture & Project Structure

The repository is modularly structured, mirroring professional algorithmic-trading platforms:

- **`data_pipeline/`**: Handles the ETL process. Fetches historical market data via `yfinance` (`loader.py`), cleans it, and persists it into TimescaleDB. Adds actionable technical indicators (`features.py`) using `pandas-ta`.
- **`database/`**: Contains database connections and SQLAlchemy ORM models (`connection.py`, `models.py`) connecting to the locally hosted TimescaleDB container.
- **`drl_models/`**: The core AI logic.
  - `env.py`: An OpenAI Gym styled environment representing the stock market, taking in states and yielding rewards.
  - `agent.py`: Defines the PyTorch Neural Network, DQN Target Network, and Replay Buffer.
  - `train.py`: Initializes the environment and executes the reinforcement learning training suite with an epsilon-greedy algorithm.
- **`backtesting/`**: 
  - `engine.py`: A chronological, event-driven backtesting engine to simulate an out-of-sample live market block.
  - `evaluate.py`: The entry point script that connects the trained high-water mark model (`.pth`) to the testing block.
- **`execution_engine/`**: (Scaffolded) Manages signal generation, OMS (Order Management System), and Mock Broker interactions.

---

## 🛠️ Prerequisites & Installation

### 1. Requirements

Before running the pipeline, ensure you have the following installed:
- **Python 3.12+** (or compatible 3.x version)
- **Docker & Docker Compose** (for hosting TimescaleDB locally)

### 2. Set Up Database Services

The project uses TimescaleDB via Docker for high-performance time-series data storage.

```bash
# Start TimescaleDB in detached mode
docker-compose up -d timescaledb
```

Wait a few seconds for the database to fully initialize and pass its health check.

### 3. Install Python Dependencies

It is highly recommended to use a virtual environment (`venv`).

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Full Pipeline

Follow these steps in chronological order to fetch data, train the agent, and evaluate robust out-of-sample results.

### Step 1: Ingest Data into TimescaleDB
Run the data loader to fetch SPY ticker data and populate your local database. By default, it downloads data from 2020 to 2026.

```bash
python -m data_pipeline.loader
```
*Expected Output: "Saved X records for SPY to DB."*

### Step 2: Train the DRL Agent
Launch the Deep Reinforcement Learning pipeline. This script pulls the saved data, engineers technical indicators securely, splits the dataset into a strictly pre-2025 chunk, and trains the Deep Q-Network.

```bash
python -m drl_models.train
```
*Expected Output: Training logs showing episode progress, epsilon decay, and total reward/profit. High-water mark models will automatically save to the `drl_models/` directory.*

### Step 3: Out-of-Sample Backtesting
Evaluate the locked, trained neural network weights (`drl_models/best_dqn_trader.pth`) against unseen (out-of-sample) data. 

```bash
python -m backtesting.evaluate
```
*Expected Output: Simulation logs of trades culminating in a performance summary featuring Total Profit, Benchmark Comparisons, Sharpe Ratio, and Max Drawdown.*

### Step 4: Live Execution Engine (Scaffold)
Starts the scaffolding for the real-time Order Management System and mock broker signals listening for DRL activations.

```bash
python -m execution_engine.main
```

---

## ⚙️ Docker Deployment (Optional)

We also provide a full `Dockerfile` capable of building a containerized instance of the DRL app.

```bash
# Build the application image
docker-compose build algo-app

# Run the complete stack (DB + Application)
docker-compose up algo-app
```
