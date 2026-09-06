import random
from collections import deque


class ReplayMemory:
    """Fixed-size ring buffer of (state, action, next_state, reward, terminated) transitions."""

    def __init__(self, capacity, seed=None):
        self.memory = deque(maxlen=capacity)
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
