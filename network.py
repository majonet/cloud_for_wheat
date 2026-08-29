import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """
    Input  -> shared MLP trunk -> {Value stream (1), Advantage stream (n_actions)}
    Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
    """

    def __init__(self, state_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)

        self.value_stream = nn.Sequential(nn.Linear(prev, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(prev, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x):
        h = self.trunk(x)
        v = self.value_stream(h)                       # (batch, 1)
        a = self.advantage_stream(h)                    # (batch, n_actions)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
