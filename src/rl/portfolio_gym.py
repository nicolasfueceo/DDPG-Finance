import gym
from gym import spaces
import numpy as np
import random

class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, price_tensor, transaction_cost=0.0, initial_balance=10000, reward_function='log_return'):
        super(PortfolioEnv, self).__init__()
        self.price_tensor = price_tensor
        self.transaction_cost = transaction_cost
        self.initial_balance = initial_balance
        self.reward_function = reward_function
        self.num_assets = price_tensor.shape[3]
        self.window_size = price_tensor.shape[2]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "price_tensor": spaces.Box(low=0, high=np.inf,
                                       shape=(price_tensor.shape[1], price_tensor.shape[2], price_tensor.shape[3]),
                                       dtype=np.float32),
            "portfolio_weights": spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        })
        self.reset()

    def reset(self):
        self.current_step = random.randint(0, len(self.price_tensor) - self.window_size - 1)

        #equal proportions
        self.portfolio_weights = np.random.dirichlet(np.ones(self.num_assets), size=1)[0]
        self.portfolio_value = self.initial_balance
        return self._get_obs()

    def _get_obs(self):
        return {
            "price_tensor": self.price_tensor[self.current_step],
            "portfolio_weights": self.portfolio_weights,
        }

    def _calculate_reward(self, portfolio_return):
        # Simplified log return
        return np.log(portfolio_return) if portfolio_return > 0 else -1.0

    def step(self, action):
        #action = np.clip(action, 0, 1)
        #action = action / np.sum(action) if np.sum(action) > 0 else np.ones(self.num_assets) / self.num_assets
        transaction_costs = self.transaction_cost * np.sum(np.abs(action - self.portfolio_weights))
        self.portfolio_weights = action.flatten()
        close_current = self.price_tensor[self.current_step, 2, -1, :]
        close_next = self.price_tensor[self.current_step + 1, 2, -1, :]
        price_rel = close_next / close_current
        portfolio_return = float(np.dot(price_rel, self.portfolio_weights))
        self.portfolio_value *= portfolio_return * (1 - transaction_costs)
        reward = np.log(portfolio_return) if portfolio_return > 0 else -1.0
        self.current_step += 1
        done = self.current_step >= (len(self.price_tensor) - 1)
        return self._get_obs(), reward, done, {"portfolio_value": self.portfolio_value}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Value: {self.portfolio_value}, Weights: {self.portfolio_weights}")

    def seed(self, seed=None):
        np.random.seed(seed)
