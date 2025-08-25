import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def create_executive_dashboard(dqn_results, ppo_results, INITIAL_BALANCE, SYMBOLS):
    # Executive Performance Dashboard
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 1. Portfolio Performance Comparison
    ax1 = axes[0, 0]
    dqn_test_values = dqn_results['portfolio_values']
    ppo_test_values = ppo_results['portfolio_values']
    benchmark_line = [INITIAL_BALANCE] * len(dqn_test_values)

    ax1.plot(dqn_test_values, label='DQN Strategy', linewidth=4, color='#2E8B57')
    ax1.plot(ppo_test_values, label='PPO Strategy', linewidth=4, color='#4169E1')
    ax1.plot(benchmark_line, label='Initial Capital', linestyle='--', color='black', alpha=0.7, linewidth=2)
    ax1.set_title('Portfolio Performance Comparison\n2023 Test Period', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Trading Days', fontsize=14)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=14)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add performance annotations
    final_dqn = dqn_test_values[-1]
    final_ppo = ppo_test_values[-1]
    ax1.annotate(f'DQN: ${final_dqn:,.0f}\n+{((final_dqn/INITIAL_BALANCE)-1)*100:.1f}%',
                 xy=(len(dqn_test_values)-1, final_dqn), xytext=(10, 10),
                 textcoords='offset points', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

    # 2. Risk-Return Analysis
    ax2 = axes[0, 1]
    agents = ['DQN', 'PPO']
    returns = [dqn_results['annual_return'] * 100, ppo_results['annual_return'] * 100]
    risks = [dqn_results['volatility'] * 100, ppo_results['volatility'] * 100]
    colors = ['#2E8B57', '#4169E1']

    scatter = ax2.scatter(risks, returns, s=500, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    for i, agent in enumerate(agents):
        ax2.annotate(f'{agent}\nSharpe: {[dqn_results["sharpe_ratio"], ppo_results["sharpe_ratio"]][i]:.3f}',
                    (risks[i], returns[i]), xytext=(15, 15), textcoords='offset points',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    ax2.set_title('Risk-Return Profile Analysis', fontsize=18, fontweight='bold', pad=20)
    ax2.set_xlabel('Volatility (% Annual)', fontsize=14)
    ax2.set_ylabel('Annual Return (%)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Add efficient frontier reference
    ax2.axhline(y=10, color='red', linestyle=':', alpha=0.5, linewidth=2, label='Market Average (10%)')
    ax2.legend(fontsize=12)

    # 3. Sharpe Ratio Comparison
    ax3 = axes[1, 0]
    sharpe_ratios = [dqn_results['sharpe_ratio'], ppo_results['sharpe_ratio']]
    bars = ax3.bar(agents, sharpe_ratios, color=colors, alpha=0.8, width=0.6, edgecolor='black', linewidth=2)
    ax3.set_title('Risk-Adjusted Performance\n(Sharpe Ratio)', fontsize=18, fontweight='bold', pad=20)
    ax3.set_ylabel('Sharpe Ratio', fontsize=14)
    ax3.grid(True, axis='y', alpha=0.3)

    # Add benchmark lines
    ax3.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Excellent (1.0)')
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Market Average (0.5)')
    ax3.legend(fontsize=12)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # 4. Maximum Drawdown Risk Analysis
    ax4 = axes[1, 1]
    max_drawdowns = [dqn_results['max_drawdown'] * 100, ppo_results['max_drawdown'] * 100]
    bars = ax4.bar(agents, max_drawdowns, color=['#DC143C', '#FF6347'], alpha=0.8, width=0.6,
                   edgecolor='black', linewidth=2)
    ax4.set_title('Maximum Risk Exposure\n(Drawdown Analysis)', fontsize=18, fontweight='bold', pad=20)
    ax4.set_ylabel('Maximum Drawdown (%)', fontsize=14)
    ax4.grid(True, axis='y', alpha=0.3)

    # Add risk tolerance benchmarks
    ax4.axhline(y=20, color='red', linestyle='--', alpha=0.7, linewidth=2, label='High Risk (20%)')
    ax4.axhline(y=10, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Moderate Risk (10%)')
    ax4.legend(fontsize=12)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=16, fontweight='bold')

    plt.suptitle('DEEP RL PORTFOLIO MANAGEMENT - EXECUTIVE DASHBOARD\nFAANG Portfolio Performance 2023',
                 fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.2)
    plt.show()

def create_training_analysis(dqn_rewards, ppo_rewards, dqn_portfolio_values, ppo_portfolio_values, INITIAL_BALANCE):
    # Training Progress and Learning Analysis
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 1. Training Rewards Evolution
    ax1 = axes[0, 0]
    ax1.plot(dqn_rewards, label='DQN Episode Rewards', alpha=0.8, color='#2E8B57', linewidth=2)
    ax1.plot(ppo_rewards, label='PPO Episode Rewards', alpha=0.8, color='#4169E1', linewidth=2)

    # Add moving averages
    window = 5
    dqn_ma = pd.Series(dqn_rewards).rolling(window).mean()
    ppo_ma = pd.Series(ppo_rewards).rolling(window).mean()
    ax1.plot(dqn_ma, label=f'DQN {window}-Episode MA', color='#228B22', linewidth=3)
    ax1.plot(ppo_ma, label=f'PPO {window}-Episode MA', color='#0000CD', linewidth=3)

    ax1.set_title('Training Rewards Evolution\n(Learning Convergence)', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Training Episodes', fontsize=14)
    ax1.set_ylabel('Episode Reward', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Add performance annotations
    final_dqn_reward = dqn_ma.iloc[-1] if not pd.isna(dqn_ma.iloc[-1]) else dqn_rewards[-1]
    final_ppo_reward = ppo_ma.iloc[-1] if not pd.isna(ppo_ma.iloc[-1]) else ppo_rewards[-1]
    ax1.annotate(f'Final DQN: {final_dqn_reward:.3f}', xy=(len(dqn_rewards)-1, final_dqn_reward),
                 xytext=(10, 10), textcoords='offset points', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

    # 2. Training Portfolio Values
    ax2 = axes[0, 1]
    ax2.plot(dqn_portfolio_values, label='DQN Portfolio Growth', linewidth=3, color='#2E8B57')
    ax2.plot(ppo_portfolio_values, label='PPO Portfolio Growth', linewidth=3, color='#4169E1')
    ax2.axhline(y=INITIAL_BALANCE, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Break-even')

    ax2.set_title('Training Portfolio Value Evolution\n(Learning Effectiveness)', fontsize=18, fontweight='bold', pad=20)
    ax2.set_xlabel('Training Episodes', fontsize=14)
    ax2.set_ylabel('Portfolio Value ($)', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Calculate and display final training performance
    final_dqn_training = dqn_portfolio_values[-1]
    final_ppo_training = ppo_portfolio_values[-1]
    dqn_training_return = (final_dqn_training - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    ppo_training_return = (final_ppo_training - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    ax2.annotate(f'DQN Training Return: {dqn_training_return:.1f}%',
                 xy=(len(dqn_portfolio_values)-1, final_dqn_training), xytext=(10, 10),
                 textcoords='offset points', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

    # 3. Learning Stability Analysis (Reward Variance)
    ax3 = axes[1, 0]
    window_size = 10
    dqn_rolling_std = pd.Series(dqn_rewards).rolling(window_size).std()
    ppo_rolling_std = pd.Series(ppo_rewards).rolling(window_size).std()

    ax3.plot(dqn_rolling_std, label='DQN Reward Volatility', linewidth=3, color='#2E8B57', alpha=0.8)
    ax3.plot(ppo_rolling_std, label='PPO Reward Volatility', linewidth=3, color='#4169E1', alpha=0.8)

    ax3.set_title(f'Learning Stability Analysis\n({window_size}-Episode Rolling Std)', fontsize=18, fontweight='bold', pad=20)
    ax3.set_xlabel('Training Episodes', fontsize=14)
    ax3.set_ylabel('Reward Standard Deviation', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Add stability interpretation
    stable_threshold = 0.1
    ax3.axhline(y=stable_threshold, color='red', linestyle=':', alpha=0.7, linewidth=2,
                label=f'Stability Threshold ({stable_threshold})')
    ax3.legend(fontsize=12)

    # 4. Algorithm Comparison Metrics
    ax4 = axes[1, 1]
    metrics = ['Avg Training\nReward', 'Final Training\nPortfolio', 'Test Period\nReturn', 'Risk-Adjusted\n(Sharpe)']

    dqn_metrics = [
        np.mean(dqn_rewards[-10:]),
        (final_dqn_training - INITIAL_BALANCE) / 1000,
        dqn_results['total_return'] * 100,
        dqn_results['sharpe_ratio']
    ]

    ppo_metrics = [
        np.mean(ppo_rewards[-10:]),
        (final_ppo_training - INITIAL_BALANCE) / 1000,
        ppo_results['total_return'] * 100,
        ppo_results['sharpe_ratio']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax4.bar(x - width/2, dqn_metrics, width, label='DQN', color='#2E8B57', alpha=0.8)
    bars2 = ax4.bar(x + width/2, ppo_metrics, width, label='PPO', color='#4169E1', alpha=0.8)

    ax4.set_title('Comprehensive Algorithm Comparison', fontsize=18, fontweight='bold', pad=20)
    ax4.set_ylabel('Performance Score', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, fontsize=11)
    ax4.legend(fontsize=12)
    ax4.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1, height2 = bar1.get_height(), bar2.get_height()

        if i < 2:
            ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + max(dqn_metrics)*0.02,
                     f'{height1:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + max(ppo_metrics)*0.02,
                     f'{height2:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + max(dqn_metrics)*0.02,
                     f'{height1:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + max(ppo_metrics)*0.02,
                     f'{height2:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('DEEP REINFORCEMENT LEARNING - TRAINING ANALYSIS DASHBOARD\nAlgorithm Learning and Convergence Metrics',
                 fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.2)
    plt.show()

def create_allocation_analysis(dqn_results, ppo_results, SYMBOLS):
    # Portfolio Allocation and Asset Analysis
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))

    # 1. Portfolio Allocation Heatmap - DQN
    ax1 = axes[0, 0]
    if len(dqn_results['actions']) > 0:
        allocation_days = min(50, len(dqn_results['actions']))
        allocation_matrix = np.array([action for action in dqn_results['actions'][-allocation_days:]])

        im1 = ax1.imshow(allocation_matrix.T, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Portfolio Weight', fontsize=12, fontweight='bold')

        ax1.set_yticks(range(len(SYMBOLS)))
        ax1.set_yticklabels(SYMBOLS, fontsize=12, fontweight='bold')
        ax1.set_title(f'DQN Portfolio Allocation Strategy\n(Last {allocation_days} Trading Days)',
                      fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Trading Days (Most Recent)', fontsize=12)
        ax1.set_ylabel('FAANG Assets', fontsize=12)

        ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)

    # 2. Portfolio Allocation Heatmap - PPO
    ax2 = axes[0, 1]
    if len(ppo_results['actions']) > 0:
        allocation_days = min(50, len(ppo_results['actions']))
        allocation_matrix = np.array([action for action in ppo_results['actions'][-allocation_days:]])

        im2 = ax2.imshow(allocation_matrix.T, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Portfolio Weight', fontsize=12, fontweight='bold')

        ax2.set_yticks(range(len(SYMBOLS)))
        ax2.set_yticklabels(SYMBOLS, fontsize=12, fontweight='bold')
        ax2.set_title(f'PPO Portfolio Allocation Strategy\n(Last {allocation_days} Trading Days)',
                      fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Trading Days (Most Recent)', fontsize=12)
        ax2.set_ylabel('FAANG Assets', fontsize=12)

        ax2.grid(True, alpha=0.3, color='white', linewidth=0.5)

    # 3. Average Asset Allocation Comparison
    ax3 = axes[1, 0]
    if len(dqn_results['actions']) > 0 and len(ppo_results['actions']) > 0:
        dqn_avg_allocation = np.mean([action for action in dqn_results['actions'][-20:]], axis=0)
        ppo_avg_allocation = np.mean([action for action in ppo_results['actions'][-20:]], axis=0)

        x = np.arange(len(SYMBOLS))
        width = 0.35

        bars1 = ax3.bar(x - width/2, dqn_avg_allocation, width, label='DQN Average',
                        color='#2E8B57', alpha=0.8, edgecolor='black')
        bars2 = ax3.bar(x + width/2, ppo_avg_allocation, width, label='PPO Average',
                        color='#4169E1', alpha=0.8, edgecolor='black')

        ax3.set_title('Average Portfolio Allocation\n(Last 20 Trading Days)',
                      fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('FAANG Assets', fontsize=12)
        ax3.set_ylabel('Average Portfolio Weight', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(SYMBOLS, fontsize=12, fontweight='bold')
        ax3.legend(fontsize=12)
        ax3.grid(True, axis='y', alpha=0.3)
        ax3.axhline(y=0, color='black', linewidth=1)

        for bar in bars1:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height > 0 else height - 0.03,
                     f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                     fontsize=10, fontweight='bold')

        for bar in bars2:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height > 0 else height - 0.03,
                     f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                     fontsize=10, fontweight='bold')

    # 4. Daily Returns Distribution
    ax4 = axes[1, 1]
    dqn_returns = np.array(dqn_results['returns']) * 100
    ppo_returns = np.array(ppo_results['returns']) * 100

    bins = 25
    ax4.hist(dqn_returns, bins=bins, alpha=0.7, label='DQN Daily Returns',
             density=True, color='#2E8B57', edgecolor='black')
    ax4.hist(ppo_returns, bins=bins, alpha=0.7, label='PPO Daily Returns',
             density=True, color='#4169E1', edgecolor='black')

    dqn_mean, dqn_std = np.mean(dqn_returns), np.std(dqn_returns)
    ppo_mean, ppo_std = np.mean(ppo_returns), np.std(ppo_returns)

    ax4.axvline(dqn_mean, color='#2E8B57', linestyle='-', alpha=0.8, linewidth=3)
    ax4.axvline(ppo_mean, color='#4169E1', linestyle='-', alpha=0.8, linewidth=3)

    ax4.set_title('Daily Returns Distribution\nStatistical Comparison', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xlabel('Daily Return (%)', fontsize=12)
    ax4.set_ylabel('Probability Density', fontsize=12)
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('PORTFOLIO ALLOCATION ANALYSIS DASHBOARD\nAsset Distribution and Statistical Analysis',
                 fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.25)
    plt.show()
