import torch.nn as nn

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
        x = self.feature(x)

        value = self.value(x)
        advantage = self.advantage(x)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
