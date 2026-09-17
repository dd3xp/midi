"""Standard recurrent amplitude baseline; no custom note-aware computation."""
import torch
from torch import nn
from torch.nn import functional as F

class IndependentBiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(20, 256, num_layers=2, dropout=.3,
                          bidirectional=True, batch_first=True)
        self.f0_head = nn.Linear(512, 81)
        self.amp_head = nn.Linear(512, 1)

    def forward(self, features):
        states, _ = self.gru(features)
        return self.f0_head(states), F.softplus(self.amp_head(states).squeeze(-1))
