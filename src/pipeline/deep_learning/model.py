import torch
import torch.nn as nn
import params as P


class MultiTaskLSTM(nn.Module):

    def __init__(self, n_features, targets):
        super().__init__()
        self.lstm = nn.LSTM(n_features, P.HIDDEN, P.LSTM_LAYERS, batch_first=True,dropout=P.DROPOUT if P.LSTM_LAYERS > 1 else 0.0)
        self.shared = nn.Sequential(nn.Dropout(P.DROPOUT), nn.Linear(P.HIDDEN, P.DENSE), nn.ReLU())
        self.heads = nn.ModuleDict({t: nn.Linear(P.DENSE, 1) for t in targets})

    def forward(self, x):
        h, _ = self.lstm(x)
        z = self.shared(h[:, -1])
        return torch.cat([head(z) for head in self.heads.values()], dim=1)