import numpy as np

def train_agent(agent, env, agent_type='DQN', num_episodes=100):
    print(f"Training {agent_type} Agent")

    episode_rewards = []
    episode_portfolio_values = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        step = 0

        while step < len(env.price_data) - 1:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            if agent_type == 'DQN':
                agent.remember(state, action, reward, next_state, done)
            elif agent_type == 'PPO':
                agent.store_reward(reward)
                agent.store_done(done)

            episode_reward += reward
            state = next_state
            step += 1

            if done:
                break

        # Update agent
        if agent_type == 'DQN':
            if len(agent.memory) > agent.batch_size:
                agent.replay()
                if episode % 10 == 0:
                    agent.update_target_network()
        elif agent_type == 'PPO':
            agent.update()

        episode_rewards.append(episode_reward)
        episode_portfolio_values.append(env.portfolio_values[-1])

        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_portfolio_value = np.mean(episode_portfolio_values[-10:])
            total_return = (avg_portfolio_value - env.initial_balance) / env.initial_balance * 100

            print(f"Episode {episode:3d} | Avg Reward: {avg_reward:8.4f} | "
                  f"Portfolio: ${avg_portfolio_value:10,.0f} | Return: {total_return:6.2f}%")

    return episode_rewards, episode_portfolio_values
