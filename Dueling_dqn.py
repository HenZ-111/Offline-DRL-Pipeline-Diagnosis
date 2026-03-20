import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Value stream
        self.value = nn.Linear(64, 1)

        # Advantage stream
        self.advantage = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: [B, input_dim]
        x = self.feature(x)

        value = self.value(x)                 # [B, 1]
        advantage = self.advantage(x)         # [B, A]

        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
