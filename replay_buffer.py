# replay_buffer.py
import random
import collections
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor([ns if ns is not None else np.zeros_like(state[0]) for ns in next_state]),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buffer)
