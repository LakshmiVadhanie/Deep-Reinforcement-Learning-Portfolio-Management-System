import numpy as np
import gym
from gym import spaces

class PortfolioEnvironment(gym.Env):
    def __init__(self, price_data, features, initial_balance=100000,
                 transaction_cost=0.001, max_position=0.3):
        super().__init__()

        self.price_data = price_data
        self.features = features
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position

        # Align data
        start_idx = max(features.first_valid_index(), price_data.first_valid_index())
        end_idx = min(features.last_valid_index(), price_data.last_valid_index())
        self.price_data = price_data.loc[start_idx:end_idx]
        self.features = features.loc[start_idx:end_idx]

        self.n_assets = len(price_data.columns)
        self.n_features = len(features.columns)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)
        obs_dim = self.n_features + self.n_assets + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(obs_dim,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.positions = np.zeros(self.n_assets)
        self.weights = np.zeros(self.n_assets)
        self.cash_ratio = 1.0

        self.portfolio_values = [self.initial_balance]
        self.returns = []
        self.actions_taken = []

        return self._get_observation()

    def _get_observation(self):
        if self.current_step >= len(self.features):
            return np.zeros(self.observation_space.shape[0])

        current_features = self.features.iloc[self.current_step].values
        portfolio_state = np.concatenate([current_features, self.weights, [self.cash_ratio]])
        return portfolio_state.astype(np.float32)

    def _calculate_portfolio_value(self):
        if self.current_step >= len(self.price_data):
            return self.portfolio_value

        current_prices = self.price_data.iloc[self.current_step].values
        stock_value = np.sum(self.positions * current_prices)
        return stock_value + self.balance

    def step(self, action):
        if self.current_step >= len(self.price_data) - 1:
            return self._get_observation(), 0, True, {}

        # Normalize and clip actions
        action = np.clip(action, -1, 1)
        action_sum = np.sum(np.abs(action))
        if action_sum > 1:
            action = action / action_sum

        # Execute trades
        current_value = self._calculate_portfolio_value()
        target_values = action * current_value
        current_prices = self.price_data.iloc[self.current_step].values
        current_stock_values = self.positions * current_prices
        trades_needed = target_values - current_stock_values

        total_transaction_cost = 0
        for i, trade_value in enumerate(trades_needed):
            if abs(trade_value) > current_value * 0.01:
                shares_to_trade = trade_value / current_prices[i]
                cost = abs(trade_value) * self.transaction_cost

                if trade_value > 0 and self.balance >= trade_value + cost:
                    self.positions[i] += shares_to_trade
                    self.balance -= (trade_value + cost)
                    total_transaction_cost += cost
                elif trade_value < 0:
                    shares_to_sell = min(abs(shares_to_trade), self.positions[i])
                    actual_sell_value = shares_to_sell * current_prices[i]
                    self.positions[i] -= shares_to_sell
                    self.balance += actual_sell_value - cost
                    total_transaction_cost += cost

        # Move to next step
        self.current_step += 1
        new_portfolio_value = self._calculate_portfolio_value()

        # Calculate return and reward
        if len(self.portfolio_values) > 0:
            portfolio_return = (new_portfolio_value - self.portfolio_values[-1]) / self.portfolio_values[-1]
        else:
            portfolio_return = 0

        self.portfolio_value = new_portfolio_value
        self.portfolio_values.append(new_portfolio_value)
        self.returns.append(portfolio_return)
        self.actions_taken.append(action.copy())

        # Update weights
        if new_portfolio_value > 0:
            stock_values = self.positions * self.price_data.iloc[self.current_step].values
            self.weights = stock_values / new_portfolio_value
            self.cash_ratio = self.balance / new_portfolio_value

        # Calculate reward with risk adjustment
        reward = portfolio_return
        if len(self.returns) >= 10:
            recent_volatility = np.std(self.returns[-10:])
            reward -= recent_volatility * 0.1
        reward -= total_transaction_cost / current_value

        done = self.current_step >= len(self.price_data) - 1

        info = {
            'portfolio_value': new_portfolio_value,
            'portfolio_return': portfolio_return,
            'transaction_cost': total_transaction_cost,
            'positions': self.positions.copy(),
            'weights': self.weights.copy(),
            'cash_ratio': self.cash_ratio
        }

        return self._get_observation(), reward, done, info
