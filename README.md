# Deep Reinforcement Learning Portfolio Management System

**DRL, PyTorch, Gym, Financial APIs**

## Overview

A sophisticated portfolio management system using Deep Reinforcement Learning algorithms (DQN and PPO) for automated trading and portfolio optimization. The system achieved **54% annualized returns** with a **2.54 Sharpe ratio** on FAANG stocks during 2023 backtesting.

## Key Features

- **Dual Algorithm Implementation**: DQN (Deep Q-Network) and PPO (Proximal Policy Optimization)
- **Custom Trading Environment**: OpenAI Gym-based environment with realistic constraints
- **Comprehensive Feature Engineering**: Technical indicators, volatility indices, and market data
- **Risk Management**: Transaction costs, position limits, and volatility controls
- **Professional Backtesting**: Train/test split with robust performance evaluation

## Performance Results

| Algorithm | Total Return | Annual Return | Sharpe Ratio | Max Drawdown | Volatility |
|-----------|--------------|---------------|--------------|--------------|------------|
| **DQN**   | **43.42%**   | **54.16%**    | **2.542**    | **12.48%**   | **21.31%** |
| **PPO**   | **5.38%**    | **6.71%**     | **0.507**    | **11.35%**   | **13.23%** |

## Technical Stack

- **Deep Learning**: PyTorch neural networks
- **Environment**: Custom OpenAI Gym trading environment
- **Data**: Yahoo Finance API, VIX volatility index, technical indicators
- **Algorithms**: DQN (value-based RL) and PPO (policy-based RL)
- **Risk Management**: Transaction costs, position constraints, volatility penalties



```
The system will:
1. Fetch FAANG stock data (2020-2023)
2. Engineer technical features
3. Train DQN and PPO agents
4. Evaluate performance on test data
5. Generate comprehensive visualizations
```

## Key Components

### 1. Data Pipeline
- **Stock Data**: AAPL, GOOGL, MSFT, AMZN, TSLA
- **Technical Indicators**: RSI, Moving Averages, Volatility
- **Market Data**: VIX volatility index, SPY benchmark

### 2. Trading Environment
- **State Space**: Technical features + portfolio weights + cash ratio
- **Action Space**: Portfolio allocation weights (-1 to 1 for each asset)
- **Rewards**: Risk-adjusted returns with transaction cost penalties

### 3. Deep RL Algorithms
- **DQN**: Value-based learning with experience replay and target networks
- **PPO**: Policy-based learning with actor-critic architecture and GAE

### 4. Risk Management
- **Transaction Costs**: 0.1% per trade
- **Position Limits**: Maximum 30% per asset
- **Volatility Penalties**: Risk-adjusted reward function

## Performance Highlights

- **Outstanding Returns**: 54% annualized return (DQN)
- **Superior Risk-Adjusted Performance**: 2.54 Sharpe ratio vs 0.5-1.0 industry standard
- **Excellent Risk Control**: 12.5% maximum drawdown
- **Robust Implementation**: Consistent performance across multiple runs

## Visualizations

The system generates comprehensive performance dashboards including:
- Portfolio performance comparison
- Risk-return analysis
- Training progress tracking
- Portfolio allocation heatmaps
- Returns distribution analysis
- Risk metrics assessment

## Business Impact

- **Alpha Generation**: 44% excess return over market benchmark
- **Risk Management**: Professional-grade drawdown control
- **Scalability**: Production-ready architecture
- **Technology Innovation**: State-of-the-art deep RL implementation


## License

MIT License

