import numpy as np

def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values[0]
    max_drawdown = 0

    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown

def evaluate_agent(agent, env, agent_type='DQN'):
    print(f"Evaluating {agent_type} Agent...")

    if agent_type == 'DQN':
        original_epsilon = agent.epsilon
        agent.epsilon = 0

    state = env.reset()
    step = 0

    while step < len(env.price_data) - 1:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        step += 1

        if done:
            break

    # Calculate performance metrics
    portfolio_values = env.portfolio_values
    returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
               for i in range(1, len(portfolio_values))]

    total_return = (portfolio_values[-1] - env.initial_balance) / env.initial_balance
    annual_return = total_return / (len(portfolio_values) / 252)
    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    max_drawdown = calculate_max_drawdown(portfolio_values)

    if agent_type == 'DQN':
        agent.epsilon = original_epsilon

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'portfolio_values': portfolio_values,
        'returns': returns,
        'actions': env.actions_taken
    }
