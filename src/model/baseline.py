"""
Baseline BiGRU model for deterministic f0 + amp prediction.
Takes encoder output C(t) (T, 256) -> BiGRU -> f0 classification + amp regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MIDIEncoder


class BaselineModel(nn.Module):
    def __init__(self, encoder_dim=256, gru_hidden=256, n_f0_bins=81, dropout=0.3):
        super().__init__()
        self.encoder = MIDIEncoder()
        self.gru = nn.GRU(
            input_size=encoder_dim,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
            num_layers=2,
        )
        gru_out_dim = gru_hidden * 2  # 512
        self.f0_head = nn.Linear(gru_out_dim, n_f0_bins)
        # amp RMS values empirically in [0, ~0.06] (99th pct);
        # Softplus ensures positive output without wasting range like Sigmoid [0,1]
        self.amp_head = nn.Sequential(
            nn.Linear(gru_out_dim, 1),
            nn.Softplus(),
        )
        self.n_f0_bins = n_f0_bins

    def forward(self, frame_features):
        """
        Args:
            frame_features: (B, T, 12)
        Returns:
            f0_logits: (B, T, 81)
            amp_pred: (B, T)
        """
        context = self.encoder(frame_features)
        h, _ = self.gru(context)
        f0_logits = self.f0_head(h)
        amp_pred = self.amp_head(h).squeeze(-1)
        return f0_logits, amp_pred

    def get_encoder(self):
        return self.encoder


def compute_baseline_loss(f0_logits, amp_pred, f0_bins, f0_gt, amp_gt, lam=1.0):
    """Compute baseline loss: CE(f0, voiced only) + lambda * MSE(amp, log space).

    Args:
        f0_logits: (B, T, 81)
        amp_pred: (B, T) positive (Softplus output)
        f0_bins: (B, T) target bin indices
        f0_gt: (B, T) ground truth f0 in Hz (used only for voicing mask)
        amp_gt: (B, T) ground truth amplitude
        lam: weight for amp loss
    """
    B, T, C = f0_logits.shape

    # f0 CE loss: only voiced frames (f0 > 0)
    voiced_mask = f0_gt > 0  # (B, T)
    f0_logits_flat = f0_logits.reshape(-1, C)
    f0_bins_flat = f0_bins.reshape(-1)
    voiced_flat = voiced_mask.reshape(-1)

    if voiced_flat.any():
        f0_loss = F.cross_entropy(
            f0_logits_flat[voiced_flat],
            f0_bins_flat[voiced_flat],
        )
    else:
        f0_loss = torch.tensor(0.0, device=f0_logits.device)

    # amp MSE loss in log space
    eps = 1e-7
    amp_pred_log = torch.log(amp_pred + eps)
    amp_gt_log = torch.log(amp_gt + eps)
    amp_loss = F.mse_loss(amp_pred_log, amp_gt_log)

    total_loss = f0_loss + lam * amp_loss
    return total_loss, f0_loss, amp_loss


def logits_to_f0(f0_logits, notes, hop_time):
    """Convert f0 logits to Hz using soft argmax over cent bins.

    Args:
        f0_logits: (T, 81) logits for a single sample
        notes: (N, 4) note array
        hop_time: frame hop time in seconds
    Returns:
        f0_hz: (T,) predicted f0 in Hz
    """
    T = f0_logits.shape[0]
    probs = torch.softmax(f0_logits, dim=-1)

    # cent bin centers: -200, -195, ..., +195, then unvoiced
    cent_bins = torch.arange(80, device=f0_logits.device).float() * 5.0 - 200.0
    voiced_probs = probs[:, :80]
    unvoiced_prob = probs[:, 80]

    cent_offset = (voiced_probs * cent_bins.unsqueeze(0)).sum(dim=-1)

    # Get MIDI pitch per frame from notes
    note_midi = torch.zeros(T, device=f0_logits.device)
    for onset, offset, midi_pitch, _ in notes:
        start = max(0, int(onset / hop_time))
        end = min(T, int(offset / hop_time))
        note_midi[start:end] = float(midi_pitch)

    f0_midi = note_midi + cent_offset / 100.0
    f0_hz = 440.0 * 2.0 ** ((f0_midi - 69.0) / 12.0)

    # Zero out unvoiced frames
    f0_hz = torch.where(unvoiced_prob > 0.5, torch.zeros_like(f0_hz), f0_hz)
    f0_hz = torch.where(note_midi > 0, f0_hz, torch.zeros_like(f0_hz))

    return f0_hz
