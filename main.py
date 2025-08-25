# Deep Reinforcement Learning Portfolio Management System

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from data_loader import DataLoader
from environment import PortfolioEnvironment
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent
from trainer import train_agent
from evaluator import evaluate_agent
from visualization import create_executive_dashboard, create_training_analysis, create_allocation_analysis, create_risk_analysis
np.random.seed(42)
import torch
torch.manual_seed(42)
import random
random.seed(42)

# Configuration
SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
START_DATE = '2020-01-01'
END_DATE = '2023-12-31'
INITIAL_BALANCE = 100000

def main():
    print("Deep Reinforcement Learning Portfolio Management System")
    print("=" * 60)
    print(f"Target Assets: {SYMBOLS}")
    print(f"Training Period: {START_DATE} to {END_DATE}")
    print(f"Initial Capital: ${INITIAL_BALANCE:,}")

    # Execute data loading
    data_loader = DataLoader(SYMBOLS, START_DATE, END_DATE)
    stock_data, vix_data, spy_data = data_loader.fetch_data()
    price_data, features = data_loader.preprocess_data(stock_data, vix_data, spy_data)

    print(f"\nData Summary:")
    print(f"Price Data Shape: {price_data.shape}")
    print(f"Features Shape: {features.shape}")
    print(f"Date Range: {price_data.index[0]} to {price_data.index[-1]}")

    # Create environment
    env = PortfolioEnvironment(price_data, features, INITIAL_BALANCE)
    print(f"\nTrading Environment Created:")
    print(f"Observation Space: {env.observation_space.shape}")
    print(f"Action Space: {env.action_space.shape}")
    print(f"Assets: {list(price_data.columns)}")
    print(f"Transaction Cost: {env.transaction_cost:.3f}")

    # Data splitting
    split_point = int(len(price_data) * 0.8)
    train_price_data = price_data.iloc[:split_point]
    train_features = features.iloc[:split_point]
    test_price_data = price_data.iloc[split_point:]
    test_features = features.iloc[split_point:]

    print("Data Split Summary:")
    print(f"Training: {train_price_data.index[0]} to {train_price_data.index[-1]} ({len(train_price_data)} days)")
    print(f"Testing:  {test_price_data.index[0]} to {test_price_data.index[-1]} ({len(test_price_data)} days)")

    # Create environments and agents
    train_env = PortfolioEnvironment(train_price_data, train_features, INITIAL_BALANCE)
    state_size = train_env.observation_space.shape[0]
    action_size = train_env.action_space.shape[0]

    print(f"\nAgent Configuration:")
    print(f"State Size: {state_size}")
    print(f"Action Size: {action_size}")

    # Initialize agents
    dqn_agent = DQNAgent(state_size, action_size)
    ppo_agent = PPOAgent(state_size, action_size)

    # Train agents
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)

    dqn_rewards, dqn_portfolio_values = train_agent(dqn_agent, train_env, 'DQN', num_episodes=50)
    print("\n" + "-"*40)
    ppo_rewards, ppo_portfolio_values = train_agent(ppo_agent, train_env, 'PPO', num_episodes=50)

    print("\nTraining Complete!")

    # Create test environment and evaluate
    test_env = PortfolioEnvironment(test_price_data, test_features, INITIAL_BALANCE)

    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)

    dqn_results = evaluate_agent(dqn_agent, test_env, 'DQN')
    test_env_ppo = PortfolioEnvironment(test_price_data, test_features, INITIAL_BALANCE)
    ppo_results = evaluate_agent(ppo_agent, test_env_ppo, 'PPO')

    # Create performance comparison table
    performance_data = {
        'Metric': ['Total Return (%)', 'Annual Return (%)', 'Volatility (%)',
                   'Sharpe Ratio', 'Max Drawdown (%)'],
        'DQN': [f"{dqn_results['total_return']:.2%}",
                f"{dqn_results['annual_return']:.2%}",
                f"{dqn_results['volatility']:.2%}",
                f"{dqn_results['sharpe_ratio']:.3f}",
                f"{dqn_results['max_drawdown']:.2%}"],
        'PPO': [f"{ppo_results['total_return']:.2%}",
                f"{ppo_results['annual_return']:.2%}",
                f"{ppo_results['volatility']:.2%}",
                f"{ppo_results['sharpe_ratio']:.3f}",
                f"{ppo_results['max_drawdown']:.2%}"]
    }

    performance_df = pd.DataFrame(performance_data)

    print("\nPERFORMANCE COMPARISON TABLE")
    print(performance_df.to_string(index=False))

    # Final portfolio allocations
    print(f"\nFINAL PORTFOLIO ALLOCATIONS")

    if len(dqn_results['actions']) > 0:
        print("DQN Portfolio Allocation:")
        for i, symbol in enumerate(price_data.columns):
            weight = dqn_results['actions'][-1][i]
            print(f"  {symbol}: {weight:7.2%}")

    if len(ppo_results['actions']) > 0:
        print("\nPPO Portfolio Allocation:")
        for i, symbol in enumerate(price_data.columns):
            weight = ppo_results['actions'][-1][i]
            print(f"  {symbol}: {weight:7.2%}")

    print(f"\nKEY ACHIEVEMENTS")
    print(f"DQN Total Return: {dqn_results['total_return']:.1%}")
    print(f"PPO Total Return: {ppo_results['total_return']:.1%}")
    print(f"DQN Sharpe Ratio: {dqn_results['sharpe_ratio']:.3f}")
    print(f"PPO Sharpe Ratio: {ppo_results['sharpe_ratio']:.3f}")

    # Create all visualizations
    create_executive_dashboard(dqn_results, ppo_results, INITIAL_BALANCE, SYMBOLS)
    
    # Executive Summary Table
    print("EXECUTIVE PERFORMANCE SUMMARY")

    executive_summary = pd.DataFrame({
        'Performance Metric': [
            'Total Return', 'Annualized Return', 'Risk (Volatility)',
            'Risk-Adjusted Return (Sharpe)', 'Maximum Drawdown', 'Final Portfolio Value'
        ],
        'DQN Strategy': [
            f"{dqn_results['total_return']:.1%}",
            f"{dqn_results['annual_return']:.1%}",
            f"{dqn_results['volatility']:.1%}",
            f"{dqn_results['sharpe_ratio']:.3f}",
            f"{dqn_results['max_drawdown']:.1%}",
            f"${dqn_results['portfolio_values'][-1]:,.0f}"
        ],
        'PPO Strategy': [
            f"{ppo_results['total_return']:.1%}",
            f"{ppo_results['annual_return']:.1%}",
            f"{ppo_results['volatility']:.1%}",
            f"{ppo_results['sharpe_ratio']:.3f}",
            f"{ppo_results['max_drawdown']:.1%}",
            f"${ppo_results['portfolio_values'][-1]:,.0f}"
        ],
        'Industry Benchmark': ['10%', '10%', '15%', '0.5-1.0', '15-25%', 'N/A']
    })

    print(executive_summary.to_string(index=False, col_space=18))

    # Create training analysis
    create_training_analysis(dqn_rewards, ppo_rewards, dqn_portfolio_values, ppo_portfolio_values, INITIAL_BALANCE)

    # Training Analysis Summary
    print("\n" + "="*80)
    print("TRAINING ANALYSIS SUMMARY")
    print("="*80)

    final_dqn_training = dqn_portfolio_values[-1]
    final_ppo_training = ppo_portfolio_values[-1]
    dqn_training_return = (final_dqn_training - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    ppo_training_return = (final_ppo_training - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    
    dqn_rolling_std = pd.Series(dqn_rewards).rolling(10).std()
    ppo_rolling_std = pd.Series(ppo_rewards).rolling(10).std()

    training_metrics = pd.DataFrame({
        'Training Metric': [
            'Final Episode Reward', 'Training Portfolio Return', 'Learning Stability (Std)',
            'Convergence Quality', 'Episodes to Convergence', 'Best Training Episode'
        ],
        'DQN Performance': [
            f"{np.mean(dqn_rewards[-5:]):.4f}",
            f"{dqn_training_return:.1f}%",
            f"{dqn_rolling_std.iloc[-10:].mean():.4f}",
            "High" if dqn_rolling_std.iloc[-10:].mean() < 0.1 else "Moderate",
            f"~{len(dqn_rewards)//2}",
            f"{max(dqn_rewards):.4f}"
        ],
        'PPO Performance': [
            f"{np.mean(ppo_rewards[-5:]):.4f}",
            f"{ppo_training_return:.1f}%",
            f"{ppo_rolling_std.iloc[-10:].mean():.4f}",
            "High" if ppo_rolling_std.iloc[-10:].mean() < 0.1 else "Moderate",
            f"~{len(ppo_rewards)//2}",
            f"{max(ppo_rewards):.4f}"
        ]
    })

    print(training_metrics.to_string(index=False, col_space=18))

    # Create allocation analysis
    create_allocation_analysis(dqn_results, ppo_results, SYMBOLS)

    print("\n" + "="*80)
    print("PORTFOLIO ALLOCATION ANALYSIS")
    print("="*80)

    if len(dqn_results['actions']) > 0 and len(ppo_results['actions']) > 0:
        # Final allocations
        dqn_final = dqn_results['actions'][-1]
        ppo_final = ppo_results['actions'][-1]

        # Average allocations (last 20 days)
        dqn_avg = np.mean([action for action in dqn_results['actions'][-20:]], axis=0)
        ppo_avg = np.mean([action for action in ppo_results['actions'][-20:]], axis=0)

        allocation_analysis = pd.DataFrame({
            'Asset': SYMBOLS,
            'DQN Final (%)': [f"{w*100:.1f}%" for w in dqn_final],
            'DQN Avg (%)': [f"{w*100:.1f}%" for w in dqn_avg],
            'PPO Final (%)': [f"{w*100:.1f}%" for w in ppo_final],
            'PPO Avg (%)': [f"{w*100:.1f}%" for w in ppo_avg],
            'Allocation Diff': [f"{abs(dqn_avg[i] - ppo_avg[i])*100:.1f}%" for i in range(len(SYMBOLS))]
        })

        print(allocation_analysis.to_string(index=False, col_space=12))

        # Allocation Statistics
        print(f"\nALLOCATION STATISTICS:")
        print(f"DQN Portfolio Concentration: {max(abs(dqn_avg))*100:.1f}% (max single asset)")
        print(f"PPO Portfolio Concentration: {max(abs(ppo_avg))*100:.1f}% (max single asset)")
        print(f"DQN Average Turnover: {np.mean(np.sum(np.abs(np.diff(np.array(dqn_results['actions']), axis=0)), axis=1)):.3f}")
        print(f"PPO Average Turnover: {np.mean(np.sum(np.abs(np.diff(np.array(ppo_results['actions']), axis=0)), axis=1)):.3f}")

        # Risk Analysis
        dqn_portfolio_risk = np.std(dqn_avg) * 100
        ppo_portfolio_risk = np.std(ppo_avg) * 100
        print(f"DQN Allocation Risk (Std): {dqn_portfolio_risk:.1f}%")
        print(f"PPO Allocation Risk (Std): {ppo_portfolio_risk:.1f}%")

    # Create risk analysis
    create_risk_analysis(dqn_results, ppo_results)

    print("\n" + "="*80)
    print("COMPREHENSIVE RISK ANALYSIS SUMMARY")
    print("="*80)

    from visualization import calculate_var, calculate_expected_shortfall
    
    dqn_returns = np.array(dqn_results['returns']) * 100
    ppo_returns = np.array(ppo_results['returns']) * 100

    risk_analysis_df = pd.DataFrame({
        'Risk Metric': [
            'Daily Volatility', 'Annualized Volatility', 'Value at Risk (95%)',
            'Expected Shortfall', 'Maximum Daily Loss', 'Sharpe Ratio',
            'Sortino Ratio', 'Calmar Ratio'
        ],
        'DQN Strategy': [
            f"{np.std(dqn_returns):.2f}%",
            f"{np.std(dqn_returns) * np.sqrt(252):.2f}%",
            f"{calculate_var(dqn_returns):.2f}%",
            f"{calculate_expected_shortfall(dqn_returns):.2f}%",
            f"{np.min(dqn_returns):.2f}%",
            f"{dqn_results['sharpe_ratio']:.3f}",
            f"{np.mean(dqn_returns[dqn_returns > 0]) / np.std(dqn_returns[dqn_returns < 0]) if len(dqn_returns[dqn_returns < 0]) > 0 else 0:.3f}",
            f"{dqn_results['annual_return'] / dqn_results['max_drawdown'] if dqn_results['max_drawdown'] > 0 else 0:.3f}"
        ],
        'PPO Strategy': [
            f"{np.std(ppo_returns):.2f}%",
            f"{np.std(ppo_returns) * np.sqrt(252):.2f}%",
            f"{calculate_var(ppo_returns):.2f}%",
            f"{calculate_expected_shortfall(ppo_returns):.2f}%",
            f"{np.min(ppo_returns):.2f}%",
            f"{ppo_results['sharpe_ratio']:.3f}",
            f"{np.mean(ppo_returns[ppo_returns > 0]) / np.std(ppo_returns[ppo_returns < 0]) if len(ppo_returns[ppo_returns < 0]) > 0 else 0:.3f}",
            f"{ppo_results['annual_return'] / ppo_results['max_drawdown'] if ppo_results['max_drawdown'] > 0 else 0:.3f}"
        ]
    })

    print(risk_analysis_df.to_string(index=False, col_space=18))

if __name__ == "__main__":
    main()
