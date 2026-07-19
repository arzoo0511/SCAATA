"""Behavior-cloning model, ported unchanged from the v1 notebook.

This is the piece that was trained and saved in v1 but never loaded back
into the RL policy — `scaata/rl/policy_init.py` is where that gap actually
gets closed.
"""
import torch
import torch.nn as nn


class ImitationModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        return self.net(x)
