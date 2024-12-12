import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = []
        self.capacity = capacity

    def add(self, price_tensor, portfolio_weights, action, reward, next_price_tensor, next_portfolio_weights, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((price_tensor, portfolio_weights, action, reward, next_price_tensor, next_portfolio_weights, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def size(self) -> int:
        return len(self.buffer)
