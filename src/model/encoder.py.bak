"""
MIDI Encoder: converts frame-level note features to contextual embeddings.
Linear(7, 128) -> BiGRU(128, hidden=128) -> output (T, 256)
"""

import torch
import torch.nn as nn


class MIDIEncoder(nn.Module):
    def __init__(self, input_dim=7, linear_dim=128, gru_hidden=128, dropout=0.3):
        super().__init__()
        self.linear = nn.Linear(input_dim, linear_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(
            input_size=linear_dim,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
            num_layers=2,
        )
        self.output_dim = gru_hidden * 2  # 256

    def forward(self, frame_features):
        """
        Args:
            frame_features: (B, T, 7)
        Returns:
            context: (B, T, 256)
        """
        x = self.relu(self.linear(frame_features))
        x, _ = self.gru(x)
        return x
