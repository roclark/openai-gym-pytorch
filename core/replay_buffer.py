import numpy as np
from random import sample
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self._buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self._buffer),
                                   batch_size,
                                   replace=False)
        batch = zip(*[self._buffer[i] for i in indices])
        state, action, reward, next_state, done = batch
        return (np.array(state),
                np.array(action),
                np.array(reward, dtype=np.float32),
                np.array(next_state),
                np.array(done, dtype=np.uint8))

    def __len__(self):
        return len(self._buffer)
