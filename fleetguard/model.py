import torch
from torch import nn


class WindowLSTM(nn.Module):
    """Small sequence-to-one LSTM regressor."""
    def __init__(self, n_features: int, hidden_size: int = 48, dropout: float = 0.15):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)
