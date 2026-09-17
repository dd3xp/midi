"""
Baseline BiGRU model for deterministic f0 + amp prediction.
Takes encoder output C(t) (T, 256) -> BiGRU -> f0 classification + amp regression.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MIDIEncoder, TransformerMIDIEncoder, HybridEncoder, ConformerEncoder, S4DEncoder, MambaEncoder


# ============ Differentiable Filter Layers ============

class DifferentiableGaussianLayer(nn.Module):
    """Learnable 1D Gaussian smoothing with trainable sigma.

    Applied to amp predictions (B, T) after the amp head.
    Sigma is parameterized via softplus to ensure positivity.
    """
    def __init__(self, init_sigma=5.0, max_kernel_size=51):
        super().__init__()
        # Raw parameter; actual sigma = softplus(raw)
        # Initialize so that softplus(raw) ≈ init_sigma
        raw_init = math.log(math.exp(init_sigma) - 1.0)  # inverse softplus
        self.sigma_raw = nn.Parameter(torch.tensor(raw_init))
        self.max_kernel_size = max_kernel_size

    def forward(self, x):
        """x: (B, T) -> smoothed (B, T)"""
        sigma = F.softplus(self.sigma_raw).clamp(min=0.1, max=20.0)
        # Kernel size = 6*sigma + 1 (rounded to nearest odd)
        ks = int(min(6 * sigma.item() + 1, self.max_kernel_size))
        if ks % 2 == 0:
            ks += 1
        ks = max(3, ks)
        # Create Gaussian kernel
        half = ks // 2
        t = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-0.5 * (t / sigma) ** 2)
        kernel = kernel / kernel.sum()
        # Apply as 1D conv with reflect padding
        x_3d = x.unsqueeze(1)  # (B, 1, T)
        x_padded = F.pad(x_3d, (half, half), mode='reflect')
        kernel_3d = kernel.view(1, 1, -1)
        out = F.conv1d(x_padded, kernel_3d)
        return out.squeeze(1)  # (B, T)


class DifferentiableSavGolLayer(nn.Module):
    """Differentiable Savitzky-Golay-like smoothing layer.

    Uses a fixed window size with learnable polynomial-fitting weights.
    Initialized to approximate a SavGol filter (window=31, order=3).
    """
    def __init__(self, window_size=31, poly_order=3):
        super().__init__()
        assert window_size % 2 == 1, "window_size must be odd"
        self.window_size = window_size
        self.half = window_size // 2

        # Initialize weights to SavGol coefficients
        from scipy.signal import savgol_coeffs
        import numpy as np
        coeffs = savgol_coeffs(window_size, poly_order)
        self.weights = nn.Parameter(torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, x):
        """x: (B, T) -> smoothed (B, T)"""
        # Normalize weights to sum to 1 (ensures no DC offset change)
        w = self.weights / self.weights.sum()
        x_3d = x.unsqueeze(1)  # (B, 1, T)
        x_padded = F.pad(x_3d, (self.half, self.half), mode='reflect')
        kernel = w.view(1, 1, -1)
        out = F.conv1d(x_padded, kernel)
        return out.squeeze(1)  # (B, T)


# ============ DSP Aux Loss Utilities (R228, exp530) ============

def gaussian_smooth_1d(x, sigma):
    """Non-learnable 1D Gaussian smoothing for band-limited aux loss.

    Args:
        x: (B, T) tensor in log-amp space
        sigma: smoothing sigma in frames (e.g. 5)
    Returns:
        smoothed: (B, T) low-frequency component
    """
    ks = int(6 * sigma + 1)
    if ks % 2 == 0:
        ks += 1
    ks = max(3, ks)
    half = ks // 2
    t = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    x_3d = x.unsqueeze(1)  # (B, 1, T)
    x_padded = F.pad(x_3d, (half, half), mode='reflect')
    kernel_3d = kernel.view(1, 1, -1)
    out = F.conv1d(x_padded, kernel_3d)
    return out.squeeze(1)  # (B, T)


def compute_band_aux_loss(amp_pred_log, amp_gt_log, sigma=5):
    """Highpass residual MSE: encourage correct micro-dynamics.

    Decomposes amp into low (Gaussian smoothed) + high (residual),
    then computes MSE on the highpass residual only.
    """
    pred_low = gaussian_smooth_1d(amp_pred_log, sigma)
    gt_low = gaussian_smooth_1d(amp_gt_log, sigma)
    pred_high = amp_pred_log - pred_low
    gt_high = amp_gt_log - gt_low
    return F.mse_loss(pred_high, gt_high)


def compute_band_weighted_loss(amp_pred_log, amp_gt_log,
                                w_low=1.0, w_mid=1.0, w_vib=0.5, w_noise=0.1,
                                sigma_low=33.0, sigma_mid=5.0, sigma_vib=1.6,
                                huber=False, huber_delta=1.0):
    """4-band weighted MSE/Huber loss in log-amp space (R249, exp960).

    Decomposes amp(t) into 4 bands via cascaded Gaussian smoothing
    (frame rate = 100 Hz, hop = 10 ms):
      - low (<0.5 Hz, phrase + DC): smooth(σ_low=33)
      - mid (0.5-3 Hz, note envelope): smooth(σ_mid=5) - smooth(σ_low)
      - vibrato (3-10 Hz, vibrato AM): smooth(σ_vib=1.6) - smooth(σ_mid)
      - noise (>10 Hz, bow/breath/quantization): residual after smooth(σ_vib)
    Each band's MSE/Huber error is weighted independently.

    Default weights downweight vibrato (w_vib=0.5, since the AM phase is
    structurally unpredictable, see paper §7.5) and noise (w_noise=0.1)
    while keeping note-envelope and phrase bands at full weight.
    Setting w_low=w_mid=w_vib=w_noise=1.0 recovers the standard MSE loss
    (up to filter-transition leakage).
    """
    pred_low = gaussian_smooth_1d(amp_pred_log, sigma_low)
    pred_mid_smooth = gaussian_smooth_1d(amp_pred_log, sigma_mid)
    pred_vib_smooth = gaussian_smooth_1d(amp_pred_log, sigma_vib)
    gt_low = gaussian_smooth_1d(amp_gt_log, sigma_low)
    gt_mid_smooth = gaussian_smooth_1d(amp_gt_log, sigma_mid)
    gt_vib_smooth = gaussian_smooth_1d(amp_gt_log, sigma_vib)

    pred_mid = pred_mid_smooth - pred_low
    pred_vib = pred_vib_smooth - pred_mid_smooth
    pred_noise = amp_pred_log - pred_vib_smooth
    gt_mid = gt_mid_smooth - gt_low
    gt_vib = gt_vib_smooth - gt_mid_smooth
    gt_noise = amp_gt_log - gt_vib_smooth

    if huber:
        loss_fn = lambda a, b: F.huber_loss(a, b, delta=huber_delta)
    else:
        loss_fn = F.mse_loss

    loss = (w_low * loss_fn(pred_low, gt_low)
            + w_mid * loss_fn(pred_mid, gt_mid)
            + w_vib * loss_fn(pred_vib, gt_vib)
            + w_noise * loss_fn(pred_noise, gt_noise))
    return loss


def compute_multi_res_stft_loss(amp_pred_log, amp_gt_log, fft_sizes=(128, 256, 512)):
    """Multi-resolution spectral loss on 1D amp signal.

    Computes STFT of pred and gt log-amp, then MSE on magnitude spectra
    at multiple FFT sizes. Encourages matching frequency content.
    """
    total_loss = 0.0
    for n_fft in fft_sizes:
        if amp_pred_log.shape[-1] < n_fft:
            continue
        hop = n_fft // 4
        # (B, T) -> (B, freq, time) complex STFT
        pred_stft = torch.stft(amp_pred_log, n_fft=n_fft, hop_length=hop,
                               win_length=n_fft, return_complex=True,
                               window=torch.hann_window(n_fft, device=amp_pred_log.device))
        gt_stft = torch.stft(amp_gt_log, n_fft=n_fft, hop_length=hop,
                             win_length=n_fft, return_complex=True,
                             window=torch.hann_window(n_fft, device=amp_gt_log.device))
        # Magnitude + log compression
        pred_mag = torch.log1p(pred_stft.abs())
        gt_mag = torch.log1p(gt_stft.abs())
        total_loss = total_loss + F.mse_loss(pred_mag, gt_mag)
    return total_loss / max(len(fft_sizes), 1)


# ============ Adversarial Amp Discriminator (exp235) ============

class AmpDiscriminator(nn.Module):
    """1D convolutional discriminator for adversarial amp training (LSGAN).

    Distinguishes real vs predicted amp sequences conditioned on MIDI features.
    Input: amp (B, T) + MIDI features (B, T, midi_dim) → real/fake score (B,).
    """

    def __init__(self, midi_dim=20, hidden=64, n_layers=3, kernel_size=5):
        super().__init__()
        in_ch = 1 + midi_dim  # amp (1) + MIDI features
        layers = []
        for i in range(n_layers):
            out_ch = hidden * (2 ** min(i, 2))
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, stride=2,
                                    padding=kernel_size // 2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, amp, midi_features):
        """
        amp: (B, T) amplitude sequence
        midi_features: (B, T, D) MIDI frame features
        Returns: score (B,) — higher = more real
        """
        # Concat amp + MIDI features along channel dim
        x = torch.cat([amp.unsqueeze(-1), midi_features], dim=-1)  # (B, T, D+1)
        x = x.permute(0, 2, 1)  # (B, D+1, T)
        h = self.conv_stack(x)  # (B, C, T')
        h = self.pool(h).squeeze(-1)  # (B, C)
        return self.fc(h).squeeze(-1)  # (B,)


# ============ Amp Diffusion Head (exp116) ============

def cosine_beta_schedule_small(timesteps, s=0.008):
    """Cosine schedule for small diffusion head."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class AmpDiffusionHead(nn.Module):
    """Small conditional DDPM for amplitude prediction.

    Instead of deterministic regression, models amp as a diffusion process
    conditioned on encoder output. Uses a lightweight 1D conv denoiser.
    """

    def __init__(self, cond_dim=512, hidden_channels=64, n_steps=100, dropout=0.1):
        super().__init__()
        self.n_steps = n_steps
        self.hidden_channels = hidden_channels

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(64, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Condition projection: cond_dim -> hidden_channels
        self.cond_proj = nn.Conv1d(cond_dim, hidden_channels, 1)

        # Denoiser: 4-layer dilated conv stack
        # Input: 1 (noisy amp) + hidden_channels (condition) = hidden_channels+1
        self.input_conv = nn.Conv1d(1 + hidden_channels, hidden_channels, 3, padding=1)
        self.res_blocks = nn.ModuleList()
        for i in range(4):
            dilation = 2 ** i
            self.res_blocks.append(nn.ModuleDict({
                'norm': nn.GroupNorm(8, hidden_channels),
                'conv1': nn.Conv1d(hidden_channels, hidden_channels, 3,
                                   padding=dilation, dilation=dilation),
                'conv2': nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
                'time_proj': nn.Linear(hidden_channels, hidden_channels * 2),
                'dropout': nn.Dropout(dropout),
            }))
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, 1, 1),
        )

        # Register noise schedule buffers
        betas = cosine_beta_schedule_small(n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _time_embedding(self, t):
        """Sinusoidal time embedding. t: (B,) -> (B, 64)"""
        half_dim = 32
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_emb(emb)  # (B, hidden_channels)

    def _denoise(self, x_t, t, condition):
        """
        x_t: (B, 1, T) noisy amp
        t: (B,) timesteps
        condition: (B, cond_dim, T) encoder features
        Returns: noise prediction (B, 1, T)
        """
        t_emb = self._time_embedding(t)  # (B, hidden_channels)
        cond = self.cond_proj(condition)   # (B, hidden_channels, T)

        h = torch.cat([x_t, cond], dim=1)  # (B, 1+hidden_channels, T)
        h = self.input_conv(h)              # (B, hidden_channels, T)

        for block in self.res_blocks:
            residual = h
            h = block['norm'](h)
            h = F.silu(h)
            h = block['conv1'](h)
            # FiLM conditioning from time
            scale, shift = block['time_proj'](t_emb).chunk(2, dim=-1)
            h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
            h = block['dropout'](F.silu(h))
            h = block['conv2'](h)
            h = h + residual

        return self.output_conv(h)  # (B, 1, T)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

    def training_loss(self, x_0, condition):
        """
        x_0: (B, 1, T) clean log-amp (normalized)
        condition: (B, cond_dim, T)
        Returns: scalar loss
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)
        noise_pred = self._denoise(x_t, t, condition)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def ddim_sample(self, condition, n_steps=10, eta=0.0):
        """DDIM sampling for amp generation.

        Args:
            condition: (B, cond_dim, T)
            n_steps: sampling steps (default 10)
        Returns:
            x_0: (B, 1, T) generated amp
        """
        B, _, T = condition.shape
        device = condition.device

        step_size = max(1, self.n_steps // n_steps)
        timesteps = list(range(self.n_steps - 1, -1, -step_size))[:n_steps]

        x = torch.randn(B, 1, T, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self._denoise(x, t_batch, condition)

            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)

            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -5.0, 5.0)

            sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            dir_xt = torch.sqrt(torch.clamp(1 - alpha_prev - sigma_t ** 2, min=0.0)) * noise_pred

            noise = torch.randn_like(x) if (eta > 0 and i + 1 < len(timesteps)) else torch.zeros_like(x)
            x = torch.sqrt(alpha_prev) * x0_pred + dir_xt + sigma_t * noise

        return x


class NoteAwareAmpHead(nn.Module):
    """Two-stage amp prediction: note-level dynamics + frame-level envelope.

    Stage 1: Pool frame features per note -> note-level GRU -> per-note amp
    Stage 2: Broadcast note amp to frames + position_in_note -> envelope head
    """

    def __init__(self, input_dim=512, hidden=128, dropout=0.3):
        super().__init__()
        # Stage 1: Note-level amplitude (pool frame features per note)
        self.note_proj = nn.Sequential(
            nn.Linear(input_dim + 2, hidden),  # +2 for note duration, pitch
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.note_gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.note_amp = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Stage 2: Frame-level envelope refinement
        # Input: frame features + note_amp (broadcast) + position_in_note
        self.envelope_head = nn.Sequential(
            nn.Linear(input_dim + 2, 128),  # +2 for note_amp, position_in_note
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

    def _segment_notes(self, frame_features):
        """Segment frames into notes using is_onset (feature index 4).

        Returns:
            note_ids: (B, T) int tensor, -1 for unvoiced frames, 0..N-1 for note index
            max_notes: int, maximum number of notes in any sample
        """
        B, T, _ = frame_features.shape
        is_onset = frame_features[:, :, 4]  # (B, T)
        is_voiced = frame_features[:, :, 0]  # (B, T)

        note_ids = torch.full((B, T), -1, dtype=torch.long, device=frame_features.device)
        max_notes = 0
        for b in range(B):
            note_idx = -1
            in_note = False
            for t in range(T):
                if is_voiced[b, t] > 0.5:
                    if is_onset[b, t] > 0.5 or not in_note:
                        note_idx += 1
                        in_note = True
                    note_ids[b, t] = note_idx
                else:
                    in_note = False
            max_notes = max(max_notes, note_idx + 1)
        max_notes = max(max_notes, 1)  # at least 1
        return note_ids, max_notes

    def forward(self, h, frame_features):
        """
        h: (B, T, D) GRU output
        frame_features: (B, T, F) original MIDI features
            feat[:, :, 0] = is_voiced
            feat[:, :, 1] = normalized_pitch (midi/127)
            feat[:, :, 2] = position_in_note (0-1)
            feat[:, :, 4] = is_onset

        Returns: amp_pred (B, T), note_amp (B, max_notes), note_ids (B, T)
        """
        B, T, D = h.shape
        device = h.device

        # Segment frames into notes
        note_ids, max_notes = self._segment_notes(frame_features)

        # Vectorized pooling: scatter_mean over note_ids
        # Map unvoiced frames (note_ids == -1) to a dummy bin (max_notes), then discard
        safe_ids = note_ids.clone()
        safe_ids[safe_ids < 0] = max_notes  # dummy bin at index max_notes

        # One-hot counting for scatter
        # h: (B, T, D), safe_ids: (B, T)
        ids_expanded = safe_ids.unsqueeze(-1)  # (B, T, 1)

        # Sum frame features per note (including dummy bin)
        note_feats_sum = torch.zeros(B, max_notes + 1, D, device=device)
        note_feats_sum.scatter_add_(1, ids_expanded.expand(-1, -1, D), h)
        note_feats_sum = note_feats_sum[:, :max_notes, :]  # drop dummy bin

        # Count frames per note
        ones = torch.ones(B, T, 1, device=device)
        note_counts = torch.zeros(B, max_notes + 1, 1, device=device)
        note_counts.scatter_add_(1, ids_expanded, ones)
        note_counts = note_counts[:, :max_notes, :]  # (B, max_notes, 1)
        note_counts_clamped = note_counts.clamp(min=1)

        # Mean pooling
        note_feats_mean = note_feats_sum / note_counts_clamped  # (B, max_notes, D)

        # Note duration (fraction of crop) and average pitch
        pitch_vals = frame_features[:, :, 1:2]  # (B, T, 1)
        pitch_sum = torch.zeros(B, max_notes + 1, 1, device=device)
        pitch_sum.scatter_add_(1, ids_expanded, pitch_vals)
        pitch_sum = pitch_sum[:, :max_notes, :]
        avg_pitch = pitch_sum / note_counts_clamped  # (B, max_notes, 1)
        note_duration = note_counts[:, :, :] / T  # (B, max_notes, 1)

        note_feats = torch.cat([note_feats_mean, note_duration, avg_pitch], dim=-1)  # (B, max_notes, D+2)
        note_mask = (note_counts.squeeze(-1) > 0)  # (B, max_notes)

        # Note-level GRU
        note_proj = self.note_proj(note_feats)  # (B, max_notes, hidden)
        note_h, _ = self.note_gru(note_proj)  # (B, max_notes, hidden*2)
        note_amp_vals = self.note_amp(note_h).squeeze(-1)  # (B, max_notes)

        # Vectorized broadcast: gather note amp back to frames
        # For unvoiced frames, clamp to 0 to avoid index errors, then zero out
        gather_ids = note_ids.clamp(min=0)  # (B, T), unvoiced mapped to note 0 (will be zeroed)
        note_amp_per_frame = note_amp_vals.gather(1, gather_ids)  # (B, T)
        note_amp_per_frame = note_amp_per_frame * (note_ids >= 0).float()  # zero out unvoiced

        # Envelope head: concat frame features + note_amp + position_in_note
        pos_in_note = frame_features[:, :, 2:3]  # (B, T, 1)
        note_amp_expanded = note_amp_per_frame.unsqueeze(-1)  # (B, T, 1)
        envelope_input = torch.cat([h, note_amp_expanded, pos_in_note], dim=-1)  # (B, T, D+2)
        amp_pred = self.envelope_head(envelope_input).squeeze(-1)  # (B, T)

        return amp_pred, note_amp_vals, note_ids


class NoteAwareAmpHeadV2(nn.Module):
    """Simplified note-level amp: predict per-note dynamics, add frame-level residual.

    Modes:
        note_only=False (default): note amp + frame-level residual (original behavior)
        note_only=True (exp139): pure note-level bottleneck, no frame features for amp.
            Only note_amp broadcast + learned position_in_note envelope.
        return_note_info=True (exp140): also return note_amp/note_ids for coarse-to-fine loss.
    """

    def __init__(self, input_dim=512, hidden=256, dropout=0.3,
                 note_only=False, return_note_info=False):
        super().__init__()
        self._note_only = note_only
        self._return_note_info = return_note_info
        # Note-level: pool frames per note -> predict note dynamics
        self.note_gru = nn.GRU(input_dim, hidden, batch_first=True, bidirectional=True)
        self.note_amp_proj = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        if note_only:
            # Pure note bottleneck: position_in_note -> learned envelope shape
            # Input: note_amp (broadcast) + position_in_note
            self.envelope_head = nn.Sequential(
                nn.Linear(2, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
        else:
            # Frame-level residual: small correction for within-note envelope
            self.frame_residual = nn.Sequential(
                nn.Linear(input_dim + 2, 128),  # +2: note_amp (broadcast), position_in_note
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

    def forward(self, h, frame_features):
        """
        h: (B, T, D) from GRU/attention
        frame_features: (B, T, F) — uses is_voiced[0], position_in_note[2], is_onset[4]
        Returns: amp_pred (B, T)
        """
        B, T, D = h.shape
        device = h.device

        # 1. Build note masks using is_onset + is_voiced
        is_onset = frame_features[:, :, 4]   # (B, T)
        is_voiced = frame_features[:, :, 0]  # (B, T)

        # Assign each voiced frame to a note via cumulative onset count
        onset_in_voiced = is_onset * (is_voiced > 0.5).float()
        note_ids = torch.cumsum(onset_in_voiced > 0.5, dim=-1)  # (B, T), 1-indexed
        note_ids = note_ids * (is_voiced > 0.5).long()  # 0 = unvoiced
        note_ids = note_ids - 1  # (B, T), -1 = unvoiced, 0..N-1 = note
        max_notes = note_ids.max().item() + 1
        max_notes = max(max_notes, 1)

        # 2. Pool frame features per note (vectorized with scatter_mean)
        safe_ids = note_ids.clone()
        safe_ids[safe_ids < 0] = max_notes  # dummy bin for unvoiced
        ids_exp = safe_ids.unsqueeze(-1).expand(-1, -1, D)  # (B, T, D)

        note_sum = torch.zeros(B, max_notes + 1, D, device=device)
        note_sum.scatter_add_(1, ids_exp, h)
        note_count = torch.zeros(B, max_notes + 1, 1, device=device)
        note_count.scatter_add_(1, safe_ids.unsqueeze(-1), torch.ones(B, T, 1, device=device))
        note_count = note_count.clamp(min=1)

        note_feats = note_sum[:, :max_notes] / note_count[:, :max_notes]  # (B, max_notes, D)

        # 3. Note-level GRU -> per-note amp
        note_h, _ = self.note_gru(note_feats)
        note_amp = self.note_amp_proj(note_h).squeeze(-1)  # (B, max_notes)

        # 4. Broadcast note amp to frames (vectorized gather)
        gather_ids = note_ids.clamp(min=0)  # (B, T)
        frame_note_amp = note_amp.gather(1, gather_ids)  # (B, T)
        frame_note_amp = frame_note_amp * (note_ids >= 0).float()  # zero unvoiced

        # 5. Frame-level prediction
        pos_in_note = frame_features[:, :, 2]  # (B, T)

        if self._note_only:
            # exp139: Pure note bottleneck — no frame-level features, only note_amp + position
            env_input = torch.stack([frame_note_amp, pos_in_note], dim=-1)  # (B, T, 2)
            amp_pred = F.softplus(self.envelope_head(env_input).squeeze(-1))  # (B, T)
        else:
            # Original: frame-level residual
            residual_input = torch.cat([h, frame_note_amp.unsqueeze(-1), pos_in_note.unsqueeze(-1)], dim=-1)
            residual = self.frame_residual(residual_input).squeeze(-1)  # (B, T)
            amp_pred = F.softplus(frame_note_amp + residual)

        if self._return_note_info:
            # exp140: return note-level info for coarse-to-fine loss
            return amp_pred, {"note_amp": note_amp, "note_ids": note_ids}
        return amp_pred


class SlowFastAmpHeadV2(nn.Module):
    """Slow+Fast amp decomposition head (Round 159, exp225).

    Predicts two per-frame log-amplitude components:
      slow_log(t) — captures phrase-level envelope (learned under Gaussian-smoothed GT)
      fast_log(t) — captures bow-change / vibrato AM / articulation transients

    Output amp_pred = exp(slow_log + fast_log) (nonnegative, log-space additive).

    When `return_info=True`, returns (amp_pred, {"slow_log": ..., "fast_log": ...})
    so that `compute_baseline_loss` can compute the auxiliary slow/fast losses.
    """

    def __init__(self, input_dim=512, hidden=256, dropout=0.3, return_info=True):
        super().__init__()
        self._return_info = return_info
        self.slow_head = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.fast_head = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, h, frame_features):
        # h: (B, T, D)
        slow_log = self.slow_head(h).squeeze(-1)  # (B, T), unconstrained log-amp
        fast_log = self.fast_head(h).squeeze(-1)  # (B, T), unconstrained residual
        # Clamp sum to avoid overflow in exp
        log_amp = (slow_log + fast_log).clamp(min=-20.0, max=10.0)
        amp_pred = torch.exp(log_amp)
        if self._return_info:
            return amp_pred, {"slow_log": slow_log, "fast_log": fast_log}
        return amp_pred


class FiLMAmpHead(nn.Module):
    """FiLM-conditioned amp head.

    Uses local context (note density, pitch range, position in piece)
    to modulate frame-level features via γ, β affine transforms.
    """
    def __init__(self, input_dim=512, context_dim=4, hidden=256, dropout=0.3):
        super().__init__()
        # Context encoder: note_density, pitch_range, position_in_piece, mean_pitch
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden * 2),  # γ and β
        )
        self.pre = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.post = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, h, frame_features):
        """
        h: (B, T, D) from BiGRU
        frame_features: (B, T, F) MIDI features
        """
        B, T, _ = h.shape

        is_onset = frame_features[:, :, 4:5]  # (B, T, 1)
        pitch = frame_features[:, :, 1:2]  # (B, T, 1)

        # Rolling window stats via avg_pool1d
        kernel = min(64, T)  # handle short sequences
        pad_left = (kernel - 1) // 2
        pad_right = kernel - 1 - pad_left
        onset_padded = F.pad(is_onset.permute(0, 2, 1), (pad_left, pad_right))
        density = F.avg_pool1d(onset_padded, kernel, stride=1).permute(0, 2, 1) * kernel  # count

        pitch_padded = F.pad(pitch.permute(0, 2, 1), (pad_left, pad_right))
        mean_pitch = F.avg_pool1d(pitch_padded, kernel, stride=1).permute(0, 2, 1)

        # Position in sequence
        pos = torch.linspace(0, 1, T, device=h.device).view(1, T, 1).expand(B, -1, -1)

        # Pitch range approximated by variance
        pitch_sq_padded = F.pad((pitch ** 2).permute(0, 2, 1), (pad_left, pad_right))
        mean_pitch_sq = F.avg_pool1d(pitch_sq_padded, kernel, stride=1).permute(0, 2, 1)
        pitch_var = (mean_pitch_sq - mean_pitch ** 2).clamp(min=0)

        context = torch.cat([density, mean_pitch, pos, pitch_var], dim=-1)  # (B, T, 4)

        # FiLM: compute γ, β from context
        film_params = self.context_proj(context)  # (B, T, hidden*2)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, T, hidden)

        # Apply FiLM
        h_pre = self.pre(h)  # (B, T, hidden)
        h_film = gamma * h_pre + beta  # affine modulation

        return self.post(h_film).squeeze(-1)  # (B, T)


class AmpPosteriorEncoder(nn.Module):
    """CVAE posterior encoder: infer latent z from GT amp + MIDI context during training."""
    def __init__(self, input_dim, hidden_dim=64, latent_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, gru_out, amp_gt):
        """
        gru_out: (B, T, D) encoder/GRU output
        amp_gt: (B, T) ground truth amplitude
        Returns: mu (B, T, latent_dim), logvar (B, T, latent_dim)
        """
        x = torch.cat([gru_out, amp_gt.unsqueeze(-1)], dim=-1)  # (B, T, D+1)
        h = self.net(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class AmpClassificationHead(nn.Module):
    """Classify log-amplitude into discrete bins using cross-entropy (exp191).

    Discretizes log-amp into n_bins uniform bins over [log_amp_min, log_amp_max].
    At inference, returns expected value: softmax(logits) @ bin_centers.
    """

    def __init__(self, input_dim, n_bins=64, dropout=0.3):
        super().__init__()
        self.n_bins = n_bins
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_bins),
        )

    def forward(self, h):
        """h: (B, T, D) -> logits: (B, T, n_bins)"""
        return self.head(h)


class MultiLayerAmpHead(nn.Module):
    """Amp prediction head that takes concatenated multi-layer encoder representations.

    Designed to tap early layers (where linear probes show peak amp info) alongside
    the final layer. Uses LayerNorm + residual MLP.

    Input: concat of encoder layers at specified indices → (B, T, n_layers * d_model)
    Output: amp_pred (B, T)
    """

    def __init__(self, input_dim, hidden=512, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Residual projection: project input to scalar for skip connection
        self.residual_proj = nn.Linear(input_dim, 1)

    def forward(self, h_multi):
        """
        h_multi: (B, T, n_layers * d_model) concatenated multi-layer features
        Returns: amp_pred (B, T)
        """
        h = self.norm(h_multi)
        out = self.mlp(h).squeeze(-1)  # (B, T)
        residual = self.residual_proj(h_multi).squeeze(-1)  # (B, T)
        return F.softplus(out + residual)


class BaselineModel(nn.Module):
    def __init__(self, encoder_dim=256, gru_hidden=256, n_f0_bins=81, dropout=0.3,
                 encoder_input_dim=6, encoder_type="gru", encoder_config=None,
                 deep_amp_head=False, amp_attention=False,
                 amp_attn_heads=8, amp_attn_layers=2, amp_attn_ff=1024,
                 instrument_embedding=False, n_instruments=10, inst_emb_dim=32,
                 note_aware_amp=False, note_aware_hidden=128,
                 film_amp=False, note_aware_amp_v2=False,
                 dual_encoder=False,
                 amp_diffusion_head=False, amp_diff_steps=100,
                 note_level_amp=False, coarse_to_fine_amp=False,
                 instrument_residual_amp=False,
                 use_cvae=False, cvae_latent_dim=8,
                 amp_classification=False, n_amp_bins=64,
                 slow_fast_amp=False, slow_fast_hidden=256,
                 f0_expressive_feats=False,
                 # Round 163: new input signal experiments
                 performer_embedding=False, num_performers=200, performer_emb_dim=32,
                 vibrato_head=False,
                 use_predicted_f0_for_amp=False,
                 articulation_head=False, n_articulation_classes=4,
                 # Round 165: autoregressive amp head
                 autoregressive_amp=False, ar_amp_hidden=64,
                 # Round 166: dual head amp decomposition
                 dual_head_amp=False,
                 # Round 170 (exp263c): family-conditioned amp heads
                 family_amp_heads=False,
                 # Round 172 (exp281): multi-layer amp head with skip connections
                 multi_layer_amp=False, amp_layers=None, amp_head_hidden=512,
                 # Round 172 (exp283): auxiliary amp loss at intermediate layer
                 aux_amp_layer=None,
                 # Round 188 (exp284): detached layer amp — concat detached encoder layers to GRU output
                 detached_layer_amp=False, detached_amp_layers=None,
                 # R244 (exp600): Heteroscedastic amp head — output (mean, log_sigma)
                 heteroscedastic_amp=False):
        super().__init__()
        # R244 (exp600): Heteroscedastic amp head
        self._heteroscedastic_amp = heteroscedastic_amp
        # Round 172 (exp281): multi-layer amp head
        self._multi_layer_amp = multi_layer_amp
        self._amp_layers = amp_layers or []
        self._aux_amp_layer = aux_amp_layer
        # Round 188 (exp284): detached layer amp
        self._detached_layer_amp = detached_layer_amp
        self._detached_amp_layers = detached_amp_layers or []
        self._slow_fast_amp = slow_fast_amp
        self._f0_expressive_feats = f0_expressive_feats
        self._dual_encoder = dual_encoder
        # Round 166: dual head amp decomposition (exp250d)
        self._dual_head_amp = dual_head_amp
        # Round 170 (exp263c): family-conditioned amp heads
        self._family_amp_heads = family_amp_heads
        # Round 165: autoregressive amp head
        self._autoregressive_amp = autoregressive_amp
        self._ar_amp_hidden = ar_amp_hidden
        # Round 163: new input signal experiments
        self._performer_embedding = performer_embedding
        self._performer_emb_dim = performer_emb_dim if performer_embedding else 0
        self._vibrato_head = vibrato_head
        self._use_predicted_f0_for_amp = use_predicted_f0_for_amp
        self._articulation_head = articulation_head
        self._amp_diffusion_head = amp_diffusion_head
        self._amp_classification = amp_classification
        self._coarse_to_fine_amp = coarse_to_fine_amp
        self._use_cvae = use_cvae
        self._cvae_latent_dim = cvae_latent_dim
        if encoder_type == "transformer":
            cfg = encoder_config or {}
            self.encoder = TransformerMIDIEncoder(
                input_dim=encoder_input_dim,
                d_model=cfg.get("d_model", 256),
                nhead=cfg.get("nhead", 8),
                num_layers=cfg.get("num_layers", 6),
                dim_feedforward=cfg.get("dim_feedforward", 1024),
                dropout=cfg.get("dropout", 0.2),
            )
            encoder_dim = self.encoder.output_dim
        elif encoder_type == "hybrid":
            cfg = encoder_config or {}
            self.encoder = HybridEncoder(
                input_dim=encoder_input_dim,
                linear_dim=cfg.get("linear_dim", 128),
                gru_hidden=cfg.get("gru_hidden", 128),
                gru_layers=cfg.get("gru_layers", 2),
                gru_dropout=cfg.get("gru_dropout", 0.3),
                n_transformer_layers=cfg.get("n_transformer_layers", 1),
                nhead=cfg.get("nhead", 4),
                dim_feedforward=cfg.get("dim_feedforward", 512),
                transformer_dropout=cfg.get("transformer_dropout", 0.3),
                output_dim=cfg.get("output_dim", None),
            )
            encoder_dim = self.encoder.output_dim
        elif encoder_type == "conformer":
            cfg = encoder_config or {}
            self.encoder = ConformerEncoder(
                input_dim=encoder_input_dim,
                d_model=cfg.get("d_model", 256),
                nhead=cfg.get("nhead", 8),
                num_layers=cfg.get("num_layers", 4),
                conv_kernel_size=cfg.get("conv_kernel_size", 31),
                dim_ff=cfg.get("dim_ff", 1024),
                dropout=cfg.get("dropout", 0.2),
            )
            encoder_dim = self.encoder.output_dim
        elif encoder_type == "s4d":
            cfg = encoder_config or {}
            self.encoder = S4DEncoder(
                input_dim=encoder_input_dim,
                d_model=cfg.get("d_model", 256),
                n_layers=cfg.get("n_layers", 4),
                d_state=cfg.get("d_state", 64),
                dropout=cfg.get("dropout", 0.2),
                bidirectional=cfg.get("bidirectional", True),
            )
            encoder_dim = self.encoder.output_dim
        elif encoder_type == "mamba":
            cfg = encoder_config or {}
            self.encoder = MambaEncoder(
                input_dim=encoder_input_dim,
                d_model=cfg.get("d_model", 256),
                n_layers=cfg.get("n_layers", 4),
                d_state=cfg.get("d_state", 64),
                d_conv=cfg.get("d_conv", 4),
                expand=cfg.get("expand", 2),
                dropout=cfg.get("dropout", 0.2),
                bidirectional=cfg.get("bidirectional", True),
            )
            encoder_dim = self.encoder.output_dim
        else:
            cfg_e = encoder_config or {}
            self.encoder = MIDIEncoder(
                input_dim=encoder_input_dim,
                linear_dim=cfg_e.get("linear_dim", 128),
                gru_hidden=cfg_e.get("gru_hidden", 128),
                dropout=cfg_e.get("dropout", 0.3),
                num_layers=cfg_e.get("num_layers", 2),
                output_dim=cfg_e.get("output_dim", None),
            )
            encoder_dim = self.encoder.output_dim

        # Dual encoder: separate encoder+GRU for amp (exp115+)
        if dual_encoder:
            if encoder_type == "transformer":
                cfg = encoder_config or {}
                self.amp_encoder = TransformerMIDIEncoder(
                    input_dim=encoder_input_dim,
                    d_model=cfg.get("d_model", 256),
                    nhead=cfg.get("nhead", 8),
                    num_layers=cfg.get("num_layers", 6),
                    dim_feedforward=cfg.get("dim_feedforward", 1024),
                    dropout=cfg.get("dropout", 0.2),
                )
            elif encoder_type == "hybrid":
                cfg = encoder_config or {}
                self.amp_encoder = HybridEncoder(
                    input_dim=encoder_input_dim,
                    linear_dim=cfg.get("linear_dim", 128),
                    gru_hidden=cfg.get("gru_hidden", 128),
                    gru_layers=cfg.get("gru_layers", 2),
                    gru_dropout=cfg.get("gru_dropout", 0.3),
                    n_transformer_layers=cfg.get("n_transformer_layers", 1),
                    nhead=cfg.get("nhead", 4),
                    dim_feedforward=cfg.get("dim_feedforward", 512),
                    transformer_dropout=cfg.get("transformer_dropout", 0.3),
                    output_dim=cfg.get("output_dim", None),
                )
            else:
                cfg_e = encoder_config or {}
                self.amp_encoder = MIDIEncoder(
                    input_dim=encoder_input_dim,
                    linear_dim=cfg_e.get("linear_dim", 128),
                    gru_hidden=cfg_e.get("gru_hidden", 128),
                    dropout=cfg_e.get("dropout", 0.3),
                    num_layers=cfg_e.get("num_layers", 2),
                    output_dim=cfg_e.get("output_dim", None),
                )
            self.amp_gru = nn.GRU(
                input_size=encoder_dim,
                hidden_size=gru_hidden,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
                num_layers=2,
            )

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

        # Instrument embedding (exp097+)
        self._instrument_embedding = instrument_embedding
        if instrument_embedding:
            self.instrument_emb = nn.Embedding(n_instruments, inst_emb_dim)
            amp_input_dim = gru_out_dim + inst_emb_dim
        else:
            self.instrument_emb = None
            amp_input_dim = gru_out_dim

        # amp_head: configurable depth (exp092+)
        self._deep_amp_head = deep_amp_head
        self._amp_attention = amp_attention
        # R244: Heteroscedastic amp head outputs (mean, log_sigma)
        _amp_out_dim = 2 if heteroscedastic_amp else 1
        if deep_amp_head:
            self.amp_head = nn.Sequential(
                nn.Linear(amp_input_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, _amp_out_dim),
                *([nn.Softplus()] if not heteroscedastic_amp else []),
            )
        else:
            # Original simple head (backward compatible)
            if heteroscedastic_amp:
                self.amp_head = nn.Linear(amp_input_dim, 2)
            else:
                self.amp_head = nn.Sequential(
                    nn.Linear(amp_input_dim, 1),
                    nn.Softplus(),
                )

        # Round 172 (exp281): Multi-layer amp head with skip connections
        if multi_layer_amp and self._amp_layers:
            n_concat_layers = len(self._amp_layers)
            multi_input_dim = encoder_dim * n_concat_layers
            self.multi_layer_amp_head = MultiLayerAmpHead(
                input_dim=multi_input_dim, hidden=amp_head_hidden, dropout=dropout,
            )

        # Round 188 (exp284): Detached layer amp — wider amp head for concat(GRU, detached layers)
        if detached_layer_amp and self._detached_amp_layers:
            n_detached = len(self._detached_amp_layers)
            detached_input_dim = gru_out_dim + encoder_dim * n_detached
            self.detached_amp_head = nn.Sequential(
                nn.Linear(detached_input_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
                nn.Softplus(),
            )

        # Round 172 (exp283): Auxiliary amp loss at intermediate encoder layer
        if aux_amp_layer is not None:
            self.aux_amp_probe = nn.Sequential(
                nn.Linear(encoder_dim, 1),
                nn.Softplus(),
            )

        # Round 166 (exp250d): Dual-head amp decomposition
        # Two separate heads: one for slow envelope, one for fast fluctuation
        if dual_head_amp:
            self.amp_head_slow = nn.Sequential(
                nn.Linear(amp_input_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )
            self.amp_head_fast = nn.Sequential(
                nn.Linear(amp_input_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )

        # Round 165 (exp244f): Autoregressive amp head
        # Replaces per-frame linear with GRU that conditions on previous predicted amp
        if autoregressive_amp:
            self.ar_amp_gru = nn.GRUCell(amp_input_dim + 1, ar_amp_hidden)
            self.ar_amp_out = nn.Sequential(
                nn.Linear(ar_amp_hidden, 1),
                nn.Softplus(),
            )

        # Optional note-aware amp head (exp103+)
        self._note_aware_amp = note_aware_amp
        if note_aware_amp:
            self.note_amp_head = NoteAwareAmpHead(
                input_dim=gru_out_dim, hidden=note_aware_hidden, dropout=dropout,
            )

        # Optional NoteAwareAmpHeadV2 (exp112+)
        # Round 159 (exp226): f0_expressive_feats adds +3 derived dims to head input.
        # Round 163: performer embedding (+performer_emb_dim) and predicted f0 (+1)
        # Round 188 (exp284): add detached layer dims to v2_input_dim
        detached_extra_dim = encoder_dim * len(self._detached_amp_layers) if detached_layer_amp else 0
        v2_input_dim = gru_out_dim + (3 if f0_expressive_feats else 0) \
            + (performer_emb_dim if performer_embedding else 0) \
            + (1 if use_predicted_f0_for_amp else 0) \
            + detached_extra_dim
        self._note_aware_amp_v2 = note_aware_amp_v2
        if note_aware_amp_v2:
            self.note_amp_head_v2 = NoteAwareAmpHeadV2(
                input_dim=v2_input_dim, hidden=256, dropout=dropout,
                note_only=note_level_amp,
                return_note_info=coarse_to_fine_amp,
            )

        # R244 (exp600): Separate log_sigma head for heteroscedastic amp
        # Works alongside any amp head (NoteAwareAmpHeadV2, standard, etc.)
        if heteroscedastic_amp:
            _sigma_input_dim = v2_input_dim if note_aware_amp_v2 else amp_input_dim
            self.amp_log_sigma_head = nn.Sequential(
                nn.Linear(_sigma_input_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        # Round 170 (exp263c): 3 independent NoteAwareAmpHeadV2 for each family
        if family_amp_heads and note_aware_amp_v2:
            self.family_heads = nn.ModuleDict({
                'strings': NoteAwareAmpHeadV2(
                    input_dim=v2_input_dim, hidden=256, dropout=dropout,
                    note_only=note_level_amp, return_note_info=coarse_to_fine_amp),
                'winds': NoteAwareAmpHeadV2(
                    input_dim=v2_input_dim, hidden=256, dropout=dropout,
                    note_only=note_level_amp, return_note_info=coarse_to_fine_amp),
                'brass': NoteAwareAmpHeadV2(
                    input_dim=v2_input_dim, hidden=256, dropout=dropout,
                    note_only=note_level_amp, return_note_info=coarse_to_fine_amp),
            })

        # Round 159 (exp225): Slow+Fast amp head
        if slow_fast_amp:
            self.slow_fast_amp_head = SlowFastAmpHeadV2(
                input_dim=v2_input_dim, hidden=slow_fast_hidden,
                dropout=dropout, return_info=True,
            )

        # Optional FiLM-conditioned amp head (exp110+)
        self._film_amp = film_amp
        if film_amp:
            self.film_amp_head = FiLMAmpHead(
                input_dim=gru_out_dim, context_dim=4, hidden=256, dropout=dropout,
            )

        # Optional amp-specific self-attention (exp093+)
        if amp_attention:
            self.amp_attn = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=gru_out_dim, nhead=amp_attn_heads,
                    dim_feedforward=amp_attn_ff, dropout=dropout,
                    batch_first=True, activation='gelu',
                ),
                num_layers=amp_attn_layers,
            )
        else:
            self.amp_attn = None

        self.n_f0_bins = n_f0_bins

        # Instrument-family residual amp (exp168+)
        self._instrument_residual_amp = instrument_residual_amp
        if instrument_residual_amp:
            # 3 instrument families: string (0), woodwind (1), brass (2)
            # Each gets a lightweight residual MLP applied to GRU output
            self.inst_amp_residual = nn.ModuleDict({
                'string': nn.Sequential(nn.Linear(gru_out_dim, 64), nn.GELU(), nn.Linear(64, 1)),
                'woodwind': nn.Sequential(nn.Linear(gru_out_dim, 64), nn.GELU(), nn.Linear(64, 1)),
                'brass': nn.Sequential(nn.Linear(gru_out_dim, 64), nn.GELU(), nn.Linear(64, 1)),
            })
            self._inst_family_names = ['string', 'woodwind', 'brass']

        # Amp classification head (exp191+): discretize log-amp into bins
        if amp_classification:
            self.amp_cls_head = AmpClassificationHead(
                input_dim=gru_out_dim, n_bins=n_amp_bins, dropout=dropout,
            )

        # Amp diffusion head (exp116+): replaces deterministic amp prediction
        if amp_diffusion_head:
            self.amp_diff_head = AmpDiffusionHead(
                cond_dim=gru_out_dim, hidden_channels=64,
                n_steps=amp_diff_steps, dropout=dropout,
            )

        # Round 163 (exp242a): Performer embedding — per-track learned style vector
        if performer_embedding:
            self.performer_emb = nn.Embedding(num_performers, performer_emb_dim)

        # Round 163 (exp242b): Vibrato auxiliary head — predict per-frame vibrato depth
        if vibrato_head:
            self.vibrato_pred_head = nn.Sequential(
                nn.Linear(gru_out_dim, 64), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        # Round 163 (exp242f): Articulation auxiliary head — classify note attack shape
        if articulation_head:
            self.articulation_cls_head = nn.Sequential(
                nn.Linear(gru_out_dim, 64), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(64, n_articulation_classes),
            )

        # CVAE for amp (exp174+): latent z captures "performance intention"
        if use_cvae:
            self.posterior_enc = AmpPosteriorEncoder(
                gru_out_dim + 1, hidden_dim=64, latent_dim=cvae_latent_dim,
            )
            # Project z to match gru_out_dim so we can add (not concat) to h_amp
            # This avoids changing input_dim of downstream heads
            self.cvae_z_proj = nn.Sequential(
                nn.Linear(cvae_latent_dim, gru_out_dim),
                nn.GELU(),
            )

    def _apply_instrument_residual(self, h_amp, frame_features, amp_pred):
        """Apply instrument-family-specific residual correction to amp prediction.

        Uses frame_features[:, :, 12:15] (instrument family one-hot: string/woodwind/brass)
        to select the appropriate residual MLP per frame.

        Args:
            h_amp: (B, T, D) GRU output after attention
            frame_features: (B, T, F) with family one-hot at indices 12-14
            amp_pred: (B, T) pre-Softplus amp prediction (log-space)
        Returns:
            amp_pred: (B, T) corrected amp prediction (post-Softplus)
        """
        B, T, D = h_amp.shape
        # Compute all residuals: (B, T, 1) each
        residuals = torch.stack([
            self.inst_amp_residual[name](h_amp) for name in self._inst_family_names
        ], dim=-1)  # (B, T, 1, 3)
        residuals = residuals.squeeze(2)  # (B, T, 3)
        # Get family weights from frame features (one-hot at dims 12-14)
        family_weights = frame_features[:, :, 12:15]  # (B, T, 3)
        # Weighted sum of residuals
        inst_residual = (residuals * family_weights).sum(dim=-1)  # (B, T)
        # amp_pred is post-Softplus, so we need to go back to log space,
        # add residual, and re-apply Softplus
        eps = 1e-7
        amp_log = torch.log(amp_pred + eps)
        amp_corrected = F.softplus(amp_log + inst_residual)
        return amp_corrected

    def forward(self, frame_features, instrument_id=None, amp_gt=None, performer_id=None,
                amp_encoder_grad_scale=0.0):
        """
        Args:
            frame_features: (B, T, D)
            instrument_id: (B,) int tensor, optional — used when instrument_embedding=True
            amp_gt: (B, T) ground truth amp — only used when amp_diffusion_head=True (training)
            performer_id: (B,) int tensor, optional — used when performer_embedding=True (exp242a)
            amp_encoder_grad_scale: float — if >0, scale amp gradients flowing back to encoder
        Returns:
            f0_logits: (B, T, 81)
            amp_pred: (B, T)
            note_info: dict or None — only when note_aware_amp=True
              or amp_diff_loss when amp_diffusion_head=True and training
        """
        # Round 172/188: get intermediate layer outputs if needed
        _need_intermediates = self._multi_layer_amp or (self._aux_amp_layer is not None) \
            or self._detached_layer_amp
        if _need_intermediates and hasattr(self.encoder, 'forward') and \
                isinstance(self.encoder, MambaEncoder):
            context, intermediates = self.encoder(frame_features, return_intermediates=True)
        else:
            context = self.encoder(frame_features)
            intermediates = {}

        h, _ = self.gru(context)
        f0_logits = self.f0_head(h)

        # Dual encoder: use separate encoder+GRU for amp features (exp115+)
        if self._dual_encoder:
            amp_context = self.amp_encoder(frame_features)
            h_for_amp, _ = self.amp_gru(amp_context)
        else:
            h_for_amp = h

        # Round 188 (exp286): Scale amp gradients flowing back through shared encoder
        if amp_encoder_grad_scale > 0 and self.training and not self._dual_encoder:
            scale = amp_encoder_grad_scale
            # h_for_amp goes to amp head; hook scales its gradient before it reaches GRU/encoder
            h_for_amp = h_for_amp * 1.0  # create a new node in the computation graph
            h_for_amp.register_hook(lambda grad, s=scale: grad * s)

        # Round 159 (exp226): F0-derived expressive features for amp head.
        # Uses f0 distribution's expected cent offset, temporal derivative, and
        # sliding stddev (vibrato depth proxy, ~0.3s window) as extra inputs
        # to the amp head. Detached so amp gradients do not affect f0 head.
        # Computed here as a (B, T, 3) tensor and concatenated to h_amp AFTER
        # amp_attn (which expects fixed gru_out_dim) in each pathway below.
        extra_f0_feats = None
        if self._f0_expressive_feats:
            B_f = h_for_amp.shape[0]
            f0_probs = F.softmax(f0_logits.detach(), dim=-1)[..., :80]  # (B, T, 80)
            cent_centers = torch.arange(
                80, device=h_for_amp.device, dtype=h_for_amp.dtype
            ) * 5.0 - 200.0  # (80,)
            f0_cents = (f0_probs * cent_centers).sum(-1)  # (B, T)
            f0_deriv = torch.cat(
                [torch.zeros(B_f, 1, device=h_for_amp.device, dtype=h_for_amp.dtype),
                 f0_cents[:, 1:] - f0_cents[:, :-1]],
                dim=1,
            )  # (B, T)
            win = 15
            pad = win // 2
            f0_padded = F.pad(f0_cents.unsqueeze(1), (pad, pad), mode="replicate")
            f0_win = f0_padded.unfold(-1, win, 1).squeeze(1)  # (B, T, win)
            f0_std_window = f0_win.std(dim=-1)  # (B, T)
            extra_f0_feats = torch.stack(
                [f0_cents / 100.0, f0_deriv / 50.0, f0_std_window / 30.0], dim=-1
            )  # (B, T, 3)

        # Amp diffusion head pathway (exp116+)
        if self._amp_diffusion_head:
            h_amp = self.amp_attn(h_for_amp) if self.amp_attn is not None else h_for_amp
            # h_amp: (B, T, D) -> transpose to (B, D, T) for conv-based diffusion
            cond = h_amp.permute(0, 2, 1)  # (B, D, T)
            if self.training and amp_gt is not None:
                # Training: compute diffusion loss on log-amp
                voiced_mask = frame_features[:, :, 0] > 0.5
                log_amp = torch.log1p(amp_gt).unsqueeze(1)  # (B, 1, T)
                # Normalize per-sample
                amp_mean = log_amp.mean(dim=-1, keepdim=True)
                amp_std = log_amp.std(dim=-1, keepdim=True).clamp(min=1e-6)
                log_amp_norm = (log_amp - amp_mean) / amp_std
                diff_loss = self.amp_diff_head.training_loss(log_amp_norm, cond)
                # Also compute a deterministic amp prediction for evaluation during training
                amp_pred = self.amp_head(h_amp).squeeze(-1) if not self._note_aware_amp_v2 else \
                    self.note_amp_head_v2(h_amp, frame_features)
                return f0_logits, amp_pred, {"amp_diff_loss": diff_loss,
                                              "amp_mean": amp_mean, "amp_std": amp_std}
            else:
                # Inference: DDIM sample
                sampled = self.amp_diff_head.ddim_sample(cond, n_steps=10)  # (B, 1, T)
                # Denormalize: we don't have stats at inference, use raw output
                amp_pred = torch.expm1(F.softplus(sampled.squeeze(1)))  # (B, T)
                return f0_logits, amp_pred

        # Amp classification pathway (exp191+): returns logits over amp bins
        if self._amp_classification:
            h_amp = self.amp_attn(h_for_amp) if self.amp_attn is not None else h_for_amp
            amp_logits = self.amp_cls_head(h_amp)  # (B, T, n_amp_bins)
            return f0_logits, amp_logits  # amp_logits used as classification target

        # Note-aware amp pathway (exp103+)
        if self._note_aware_amp:
            amp_pred, note_amp_vals, note_ids = self.note_amp_head(h_for_amp, frame_features)
            return f0_logits, amp_pred, {
                "note_amp": note_amp_vals,
                "note_ids": note_ids,
            }

        # Round 159 (exp225): Slow+Fast amp head pathway
        if self._slow_fast_amp:
            h_amp = self.amp_attn(h_for_amp) if self.amp_attn is not None else h_for_amp
            if extra_f0_feats is not None:
                h_amp = torch.cat([h_amp, extra_f0_feats], dim=-1)
            amp_pred, sf_info = self.slow_fast_amp_head(h_amp, frame_features)
            return f0_logits, amp_pred, sf_info

        # Round 163 (exp242b): Vibrato auxiliary head — predict per-frame vibrato depth
        aux_outputs = {}
        if self._vibrato_head:
            vibrato_pred = self.vibrato_pred_head(h).squeeze(-1)  # (B, T)
            aux_outputs['vibrato_pred'] = vibrato_pred

        # Round 163 (exp242f): Articulation auxiliary head
        if self._articulation_head:
            art_logits = self.articulation_cls_head(h)  # (B, T, n_classes)
            aux_outputs['articulation_logits'] = art_logits

        # NoteAwareAmpHeadV2 pathway (exp112+)
        if self._note_aware_amp_v2:
            h_amp = self.amp_attn(h_for_amp) if self.amp_attn is not None else h_for_amp
            # Round 159 (exp226): concat f0-derived expressive features after attn
            if extra_f0_feats is not None:
                h_amp = torch.cat([h_amp, extra_f0_feats], dim=-1)

            # Round 163 (exp242a): concat performer embedding after attn
            if self._performer_embedding and performer_id is not None:
                B_p, T_p, _ = h_amp.shape
                p_emb = self.performer_emb(performer_id)  # (B, emb_dim)
                # Embedding dropout: 50% chance of zeroing during training (Round 165 fix C2)
                if self.training and torch.rand(1).item() < 0.5:
                    p_emb = torch.zeros_like(p_emb)
                p_emb = p_emb.unsqueeze(1).expand(-1, T_p, -1)  # (B, T, emb_dim)
                h_amp = torch.cat([h_amp, p_emb], dim=-1)
            elif self._performer_embedding:
                # No performer_id (test time with unseen tracks): use zeros
                B_p, T_p, _ = h_amp.shape
                h_amp = torch.cat([h_amp, torch.zeros(B_p, T_p, self._performer_emb_dim,
                                                       device=h_amp.device)], dim=-1)

            # Round 163 (exp242d): concat predicted f0 as amp feature
            if self._use_predicted_f0_for_amp:
                f0_probs = F.softmax(f0_logits.detach(), dim=-1)[..., :80]  # (B, T, 80)
                cent_centers = torch.arange(80, device=h_amp.device, dtype=h_amp.dtype) * 5.0 - 200.0
                pred_f0_cents = (f0_probs * cent_centers).sum(-1, keepdim=True) / 100.0  # (B, T, 1)
                h_amp = torch.cat([h_amp, pred_f0_cents], dim=-1)

            # CVAE (exp174+): inject latent z into h_amp
            kl_loss = torch.tensor(0.0, device=h_amp.device)
            if self._use_cvae:
                B, T, D = h_amp.shape
                if self.training and amp_gt is not None:
                    mu, logvar = self.posterior_enc(h_amp, amp_gt)
                    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                    kl_loss = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(-1).mean()
                else:
                    z = torch.zeros(B, T, self._cvae_latent_dim, device=h_amp.device)
                h_amp = h_amp + self.cvae_z_proj(z)  # additive injection

            # Round 170 (exp263c): family-conditioned amp heads
            if self._family_amp_heads:
                # Route each sample to its family head based on frame_features dims 12-14
                B_fam, T_fam, _ = h_amp.shape
                amp_pred = torch.zeros(B_fam, T_fam, device=h_amp.device)
                # Determine family per sample from instrument one-hot (dims 12-14)
                # Sum across time to get dominant family per sample
                fam_scores = frame_features[:, :, 12:15].mean(dim=1)  # (B, 3)
                fam_idx = fam_scores.argmax(dim=1)  # (B,) 0=strings, 1=brass, 2=woodwind
                fam_names = ['strings', 'brass', 'winds']
                for fi, fname in enumerate(fam_names):
                    mask = (fam_idx == fi)
                    if not mask.any():
                        continue
                    h_sub = h_amp[mask]
                    ff_sub = frame_features[mask]
                    v2_out = self.family_heads[fname](h_sub, ff_sub)
                    sub_pred = v2_out if not isinstance(v2_out, tuple) else v2_out[0]
                    amp_pred[mask] = sub_pred
                return f0_logits, amp_pred

            # Round 188 (exp284): Concat detached encoder layers to h_amp
            if self._detached_layer_amp and self._detached_amp_layers and intermediates:
                detached_feats = []
                for layer_idx in self._detached_amp_layers:
                    if layer_idx in intermediates:
                        detached_feats.append(intermediates[layer_idx].detach())
                    else:
                        detached_feats.append(context.detach())
                h_amp = torch.cat([h_amp] + detached_feats, dim=-1)

            v2_out = self.note_amp_head_v2(h_amp, frame_features)
            # R244: compute log_sigma from h_amp if heteroscedastic
            if self._heteroscedastic_amp:
                _h_log_sigma = self.amp_log_sigma_head(h_amp).squeeze(-1).clamp(-5, 2)  # (B, T)
                aux_outputs["amp_log_sigma"] = _h_log_sigma
            # Round 172 (exp283): inject aux amp pred into outputs
            if self._aux_amp_layer is not None and self._aux_amp_layer in intermediates:
                aux_h = intermediates[self._aux_amp_layer]
                aux_outputs["aux_amp_pred"] = self.aux_amp_probe(aux_h).squeeze(-1)
            if self._coarse_to_fine_amp and isinstance(v2_out, tuple):
                amp_pred, note_info = v2_out
                if self._use_cvae:
                    note_info["kl_loss"] = kl_loss
                note_info.update(aux_outputs)
                return f0_logits, amp_pred, note_info
            else:
                amp_pred = v2_out if not isinstance(v2_out, tuple) else v2_out[0]
                # Instrument-family residual (exp168+): add family-specific correction
                if self._instrument_residual_amp:
                    amp_pred = self._apply_instrument_residual(h_amp, frame_features, amp_pred)
                if self._use_cvae:
                    aux_outputs["kl_loss"] = kl_loss
                if aux_outputs:
                    return f0_logits, amp_pred, aux_outputs
                return f0_logits, amp_pred

        # Round 172 (exp281): Multi-layer amp head — concat encoder layer outputs
        if self._multi_layer_amp and self._amp_layers and intermediates:
            layer_feats = []
            for layer_idx in self._amp_layers:
                if layer_idx in intermediates:
                    layer_feats.append(intermediates[layer_idx])
                else:
                    # Fallback: use final encoder output
                    layer_feats.append(context)
            h_multi = torch.cat(layer_feats, dim=-1)  # (B, T, n_layers * d_model)
            amp_pred = self.multi_layer_amp_head(h_multi)
            # Also compute aux amp loss if requested
            aux_info = {}
            if self._aux_amp_layer is not None and self._aux_amp_layer in intermediates:
                aux_h = intermediates[self._aux_amp_layer]
                aux_amp_pred = self.aux_amp_probe(aux_h).squeeze(-1)  # (B, T)
                aux_info["aux_amp_pred"] = aux_amp_pred
            if aux_info:
                return f0_logits, amp_pred, aux_info
            return f0_logits, amp_pred

        # Round 172 (exp283): Auxiliary amp loss only (no multi-layer head)
        if self._aux_amp_layer is not None and intermediates and not self._multi_layer_amp:
            h_amp = self.amp_attn(h_for_amp) if self.amp_attn is not None else h_for_amp
            amp_pred = self.amp_head(h_amp).squeeze(-1)
            aux_h = intermediates[self._aux_amp_layer]
            aux_amp_pred = self.aux_amp_probe(aux_h).squeeze(-1)
            return f0_logits, amp_pred, {"aux_amp_pred": aux_amp_pred}

        # FiLM amp pathway (exp110+)
        if self._film_amp:
            h_amp = self.amp_attn(h_for_amp) if self.amp_attn is not None else h_for_amp
            amp_pred = self.film_amp_head(h_amp, frame_features)
            return f0_logits, amp_pred

        # Standard amp pathway: optionally route through self-attention
        h_amp = self.amp_attn(h_for_amp) if self.amp_attn is not None else h_for_amp
        # optionally concat instrument embedding (exp097+)
        if self._instrument_embedding and self.instrument_emb is not None and instrument_id is not None:
            B, T, _ = h_amp.shape
            inst_emb = self.instrument_emb(instrument_id)  # (B, inst_emb_dim)
            inst_emb = inst_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, inst_emb_dim)
            h_amp = torch.cat([h_amp, inst_emb], dim=-1)  # (B, T, gru_out + inst_emb_dim)
        # Round 166 (exp250d): Dual-head amp decomposition
        if self._dual_head_amp:
            amp_slow_pred = F.softplus(self.amp_head_slow(h_amp).squeeze(-1))  # (B, T)
            amp_fast_pred = self.amp_head_fast(h_amp).squeeze(-1)  # (B, T), can be negative
            amp_pred = F.softplus(torch.log(amp_slow_pred + 1e-7) + amp_fast_pred)
            return f0_logits, amp_pred, {
                "amp_slow_pred": amp_slow_pred,
                "amp_fast_pred": amp_fast_pred,
            }
        # Round 165 (exp244f): Autoregressive amp prediction
        if self._autoregressive_amp:
            B, T, D = h_amp.shape
            amp_preds = []
            h_gru = torch.zeros(B, self._ar_amp_hidden, device=h_amp.device)
            prev_amp = torch.zeros(B, 1, device=h_amp.device)
            for t in range(T):
                inp = torch.cat([h_amp[:, t, :], prev_amp], dim=-1)  # (B, D+1)
                h_gru = self.ar_amp_gru(inp, h_gru)  # (B, ar_hidden)
                a_t = self.ar_amp_out(h_gru)  # (B, 1)
                amp_preds.append(a_t.squeeze(-1))
                # Teacher forcing during training, autoregressive during eval
                if self.training and amp_gt is not None:
                    prev_amp = amp_gt[:, t:t+1]  # (B, 1) ground truth
                else:
                    prev_amp = a_t.detach()  # (B, 1) own prediction
            amp_pred = torch.stack(amp_preds, dim=1)  # (B, T)
            return f0_logits, amp_pred

        # Round 188 (exp284): Detached layer amp — concat GRU output + detached encoder layers
        if self._detached_layer_amp and self._detached_amp_layers and intermediates:
            detached_feats = []
            for layer_idx in self._detached_amp_layers:
                if layer_idx in intermediates:
                    detached_feats.append(intermediates[layer_idx].detach())
                else:
                    detached_feats.append(context.detach())
            h_concat = torch.cat([h_amp] + detached_feats, dim=-1)  # (B, T, gru_out + n*d_model)
            amp_pred = self.detached_amp_head(h_concat).squeeze(-1)
            # Also compute aux amp loss if requested
            aux_info = {}
            if self._aux_amp_layer is not None and self._aux_amp_layer in intermediates:
                aux_h = intermediates[self._aux_amp_layer]
                aux_info["aux_amp_pred"] = self.aux_amp_probe(aux_h).squeeze(-1)
            if aux_info:
                return f0_logits, amp_pred, aux_info
            return f0_logits, amp_pred

        amp_raw = self.amp_head(h_amp)  # (B, T, 1) or (B, T, 2) if heteroscedastic
        if self._heteroscedastic_amp:
            # R244: Split into mean (via Softplus for positivity) and bounded log_sigma
            amp_pred = F.softplus(amp_raw[..., 0])  # (B, T) — mean prediction
            amp_log_sigma = amp_raw[..., 1].clamp(-5, 2)  # (B, T) — bounded log std
            return f0_logits, amp_pred, {"amp_log_sigma": amp_log_sigma}
        else:
            amp_pred = amp_raw.squeeze(-1)
            return f0_logits, amp_pred

    def get_encoder(self):
        return self.encoder


def compute_baseline_loss(f0_logits, amp_pred, f0_bins, f0_gt, amp_gt, lam=1.0,
                          corr_weight=0.0, grad_weight=0.0, per_crop_amp_norm=False,
                          huber_loss=False, huber_delta=1.0, multiscale_weight=0.0,
                          note_info=None, note_loss_weight=0.0,
                          contrastive_dynamics_weight=0.0,
                          note_ranking_weight=0.0,
                          contrastive_margin=0.1,
                          frame_features=None,
                          ccc_weight=0.0,
                          slow_fast_amp_weight=0.0,
                          slow_fast_sigma_frames=40,
                          slow_fast_fast_weight=0.5,
                          amp_norm_mode="none",
                          amp_norm_stats=None,
                          amp_frame_weights=None,
                          band_weighted_loss=None):
    """Compute baseline loss: CE(f0, voiced only) + lambda * amp_loss(log space).

    Args:
        f0_logits: (B, T, 81)
        amp_pred: (B, T) positive (Softplus output)
        f0_bins: (B, T) target bin indices
        f0_gt: (B, T) ground truth f0 in Hz (used only for voicing mask)
        amp_gt: (B, T) ground truth amplitude
        lam: weight for amp loss
        corr_weight: weight for correlation loss (1 - pearson_corr) on voiced frames
        grad_weight: weight for gradient loss (MSE of temporal diffs)
        amp_norm_mode: normalization mode for amp loss (Round 206).
            "none" = raw log-amp MSE
            "per_crop_mean" = subtract per-crop voiced mean (legacy per_crop_amp_norm)
            "per_crop_zscore" = per-crop z-score (mean + std)
            "per_piece_zscore" = per-piece z-score using precomputed stats
            "per_piece_minmax" = per-piece min-max normalization
            "global_zscore" = global z-score using training set stats
            "per_piece_robust" = per-piece robust scaling (median/IQR)
            "per_instrument_zscore" = per-instrument z-score
        amp_norm_stats: dict with precomputed stats for normalization modes that need them
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

    # Round 206: Normalization ablation — multiple modes
    # Legacy per_crop_amp_norm flag maps to "per_crop_mean" mode
    effective_norm = amp_norm_mode
    if effective_norm == "none" and per_crop_amp_norm:
        effective_norm = "per_crop_mean"

    if effective_norm == "per_crop_mean":
        # Original per-crop normalization: subtract per-sample voiced mean (exp094)
        voiced_mask_f = (f0_gt > 0).float()
        n_voiced = voiced_mask_f.sum(dim=-1, keepdim=True).clamp(min=1)
        gt_mean = (amp_gt_log * voiced_mask_f).sum(dim=-1, keepdim=True) / n_voiced
        amp_pred_log = amp_pred_log - gt_mean
        amp_gt_log = amp_gt_log - gt_mean
    elif effective_norm == "per_crop_zscore":
        # Per-crop z-score: subtract mean AND divide by std
        voiced_mask_f = (f0_gt > 0).float()
        n_voiced = voiced_mask_f.sum(dim=-1, keepdim=True).clamp(min=1)
        gt_mean = (amp_gt_log * voiced_mask_f).sum(dim=-1, keepdim=True) / n_voiced
        gt_var = ((amp_gt_log - gt_mean) ** 2 * voiced_mask_f).sum(dim=-1, keepdim=True) / n_voiced
        gt_std = gt_var.sqrt().clamp(min=1e-6)
        amp_pred_log = (amp_pred_log - gt_mean) / gt_std
        amp_gt_log = (amp_gt_log - gt_mean) / gt_std
    elif effective_norm == "per_piece_zscore" and amp_norm_stats is not None:
        # Per-piece z-score using precomputed piece-level stats
        p_mean = amp_norm_stats["piece_logamp_mean"].unsqueeze(-1)  # (B, 1)
        p_std = amp_norm_stats["piece_logamp_std"].unsqueeze(-1)    # (B, 1)
        amp_pred_log = (amp_pred_log - p_mean) / p_std
        amp_gt_log = (amp_gt_log - p_mean) / p_std
    elif effective_norm == "per_piece_minmax" and amp_norm_stats is not None:
        # Per-piece min-max normalization to [0, 1] in log space
        p_min = amp_norm_stats["piece_logamp_min"].unsqueeze(-1)    # (B, 1)
        p_max = amp_norm_stats["piece_logamp_max"].unsqueeze(-1)    # (B, 1)
        p_range = (p_max - p_min).clamp(min=1e-6)
        amp_pred_log = (amp_pred_log - p_min) / p_range
        amp_gt_log = (amp_gt_log - p_min) / p_range
    elif effective_norm == "global_zscore" and amp_norm_stats is not None:
        # Global z-score using training set mean/std
        g_mean = amp_norm_stats["global_mean"]  # scalar
        g_std = amp_norm_stats["global_std"]    # scalar
        amp_pred_log = (amp_pred_log - g_mean) / g_std
        amp_gt_log = (amp_gt_log - g_mean) / g_std
    elif effective_norm == "per_piece_robust" and amp_norm_stats is not None:
        # Per-piece robust scaling: (x - median) / IQR
        p_med = amp_norm_stats["piece_logamp_median"].unsqueeze(-1)  # (B, 1)
        p_iqr = amp_norm_stats["piece_logamp_iqr"].unsqueeze(-1)    # (B, 1)
        amp_pred_log = (amp_pred_log - p_med) / p_iqr
        amp_gt_log = (amp_gt_log - p_med) / p_iqr
    elif effective_norm == "per_instrument_zscore" and amp_norm_stats is not None:
        # Per-instrument z-score using precomputed per-instrument stats
        inst_ids = amp_norm_stats["instrument_id"]  # (B,)
        inst_means = amp_norm_stats["inst_means"]   # (n_instruments,)
        inst_stds = amp_norm_stats["inst_stds"]     # (n_instruments,)
        i_mean = inst_means[inst_ids].unsqueeze(-1)  # (B, 1)
        i_std = inst_stds[inst_ids].unsqueeze(-1)    # (B, 1)
        amp_pred_log = (amp_pred_log - i_mean) / i_std
        amp_gt_log = (amp_gt_log - i_mean) / i_std
    # else: "none" — no normalization, raw log-amp MSE

    # Frame-level amp loss: MSE or Huber (exp105+)
    # R223: per-frame weighting for onset/silence masking
    # R249: band-weighted loss replaces the standard frame loss when configured
    if band_weighted_loss is not None:
        # 4-band weighted loss in log-amp space (paper §3 — only direction
        # in our 200+ ablations that puts per-band attention in the loss)
        bw = band_weighted_loss
        frame_amp_loss = compute_band_weighted_loss(
            amp_pred_log, amp_gt_log,
            w_low=float(bw.get("w_low", 1.0)),
            w_mid=float(bw.get("w_mid", 1.0)),
            w_vib=float(bw.get("w_vib", 0.5)),
            w_noise=float(bw.get("w_noise", 0.1)),
            sigma_low=float(bw.get("sigma_low", 33.0)),
            sigma_mid=float(bw.get("sigma_mid", 5.0)),
            sigma_vib=float(bw.get("sigma_vib", 1.6)),
            huber=bool(huber_loss),
            huber_delta=float(huber_delta),
        )
    elif amp_frame_weights is not None:
        # Element-wise loss with per-frame weights
        if huber_loss:
            per_frame = F.huber_loss(amp_pred_log, amp_gt_log, delta=huber_delta, reduction='none')
        else:
            per_frame = (amp_pred_log - amp_gt_log) ** 2
        frame_amp_loss = (per_frame * amp_frame_weights).sum() / amp_frame_weights.sum().clamp(min=1e-8)
    else:
        if huber_loss:
            frame_amp_loss = F.huber_loss(amp_pred_log, amp_gt_log, delta=huber_delta)
        else:
            frame_amp_loss = F.mse_loss(amp_pred_log, amp_gt_log)

    amp_loss = frame_amp_loss

    # Multi-scale loss: beat-level (hop=8 mean pooling) (exp105+)
    if multiscale_weight > 0:
        kernel = 8
        if amp_pred_log.shape[-1] >= kernel:
            pred_beat = F.avg_pool1d(amp_pred_log.unsqueeze(1), kernel, kernel).squeeze(1)
            gt_beat = F.avg_pool1d(amp_gt_log.unsqueeze(1), kernel, kernel).squeeze(1)
            ms_loss = F.mse_loss(pred_beat, gt_beat)
            amp_loss = amp_loss + multiscale_weight * ms_loss

    # Note-level loss (exp103+): MSE between predicted note amp and mean GT amp per note
    if note_info is not None and note_loss_weight > 0:
        note_amp = note_info["note_amp"]  # (B, max_notes)
        note_ids = note_info["note_ids"]  # (B, T)
        B_n = note_amp.shape[0]
        max_notes = note_amp.shape[1]
        note_losses = []
        for b in range(B_n):
            for n in range(max_notes):
                mask = (note_ids[b] == n)
                if mask.any():
                    gt_note_amp = amp_gt_log[b, mask].mean()
                    note_losses.append((note_amp[b, n] - gt_note_amp) ** 2)
        if note_losses:
            note_loss = torch.stack(note_losses).mean()
            amp_loss = amp_loss + note_loss_weight * note_loss

    # Correlation loss on voiced frames — per-sample then average (exp095: fix W2)
    if corr_weight > 0:
        corr_losses = []
        for i in range(B):
            voiced_i = voiced_mask[i]  # (T,)
            if voiced_i.sum() > 2:
                p = amp_pred_log[i, voiced_i]
                g = amp_gt_log[i, voiced_i]
                p_c = p - p.mean()
                g_c = g - g.mean()
                numer = (p_c * g_c).sum()
                denom = (p_c.norm() * g_c.norm()).clamp(min=1e-8)
                corr_losses.append(1.0 - numer / denom)
        if corr_losses:
            corr_loss = torch.stack(corr_losses).mean()
        else:
            corr_loss = torch.tensor(0.0, device=f0_logits.device)
        amp_loss = amp_loss + corr_weight * corr_loss

    # Gradient (temporal diff) loss — only consecutive voiced frames (exp095: fix W3)
    if grad_weight > 0:
        voiced_both = voiced_mask[:, 1:] & voiced_mask[:, :-1]  # (B, T-1)
        pred_diff = amp_pred_log[:, 1:] - amp_pred_log[:, :-1]
        gt_diff = amp_gt_log[:, 1:] - amp_gt_log[:, :-1]
        if voiced_both.any():
            grad_loss = ((pred_diff - gt_diff) ** 2 * voiced_both.float()).sum() / voiced_both.float().sum().clamp(min=1)
        else:
            grad_loss = torch.tensor(0.0, device=f0_logits.device)
        amp_loss = amp_loss + grad_weight * grad_loss

    # CCC (Concordance Correlation Coefficient) loss (exp194+):
    # Directly optimizes the evaluation metric (Pearson corr with mean/var penalty)
    if ccc_weight > 0:
        ccc_losses = []
        for i in range(B):
            voiced_i = voiced_mask[i]
            if voiced_i.sum() > 2:
                p = amp_pred_log[i, voiced_i]
                g = amp_gt_log[i, voiced_i]
                mu_p, mu_g = p.mean(), g.mean()
                var_p, var_g = p.var(), g.var()
                cov = ((p - mu_p) * (g - mu_g)).mean()
                ccc = 2 * cov / (var_p + var_g + (mu_p - mu_g) ** 2 + 1e-8)
                ccc_losses.append(1.0 - ccc)
        if ccc_losses:
            ccc_loss = torch.stack(ccc_losses).mean()
        else:
            ccc_loss = torch.tensor(0.0, device=f0_logits.device)
        amp_loss = amp_loss + ccc_weight * ccc_loss

    # Contrastive dynamics loss (exp162+): note-pair relative amplitude consistency
    # Segments notes at unvoiced gaps AND f0 jumps >1 semitone, computes mean amp per note,
    # then penalizes when predicted relative dynamics disagree with GT
    if contrastive_dynamics_weight > 0 or note_ranking_weight > 0:
        # Compute per-note mean amp using MIDI annotation-driven segmentation
        ranking_losses = []
        contrastive_losses = []
        margin = contrastive_margin
        for b in range(B):
            pred_log_b = amp_pred_log[b]  # (T,)
            gt_log_b = amp_gt_log[b]  # (T,)

            if frame_features is not None:
                # MIDI-driven segmentation using is_onset + is_voiced
                is_voiced_b = frame_features[b, :, 0] > 0.5  # (T,)
                is_onset_b = frame_features[b, :, 4] > 0.5   # (T,)

                if is_voiced_b.sum() < 10:
                    continue

                # Segment notes: new note at onset or voiced-entry after silence
                note_boundaries = []
                in_note = False
                for t in range(T):
                    if is_voiced_b[t]:
                        if is_onset_b[t] or not in_note:
                            note_boundaries.append(t)
                            in_note = True
                    else:
                        in_note = False
                note_boundaries.append(T)  # sentinel

                # Compute per-note mean amp
                note_pred_amps = []
                note_gt_amps = []
                for i in range(len(note_boundaries) - 1):
                    start = note_boundaries[i]
                    end = note_boundaries[i + 1]
                    seg_voiced = is_voiced_b[start:end]
                    voiced_frames = torch.where(seg_voiced)[0] + start
                    if len(voiced_frames) < 3:
                        continue
                    note_pred_amps.append(pred_log_b[voiced_frames].mean())
                    note_gt_amps.append(gt_log_b[voiced_frames].mean())
            else:
                # Fallback: unvoiced-gap-only segmentation (exp162 behavior)
                voiced_b = f0_gt[b] > 0  # (T,)
                if voiced_b.sum() < 10:
                    continue
                voiced_idx = torch.where(voiced_b)[0]
                if len(voiced_idx) < 2:
                    continue
                gaps = torch.where(voiced_idx[1:] - voiced_idx[:-1] > 1)[0]
                boundaries = [0] + (gaps + 1).tolist() + [len(voiced_idx)]

                note_pred_amps = []
                note_gt_amps = []
                for i in range(len(boundaries) - 1):
                    seg_idx = voiced_idx[boundaries[i]:boundaries[i+1]]
                    if len(seg_idx) < 3:
                        continue
                    note_pred_amps.append(pred_log_b[seg_idx].mean())
                    note_gt_amps.append(gt_log_b[seg_idx].mean())

            if len(note_pred_amps) < 2:
                continue

            note_pred = torch.stack(note_pred_amps)
            note_gt = torch.stack(note_gt_amps)

            # Note ranking loss: consecutive note pairs should maintain relative order
            if note_ranking_weight > 0:
                for i in range(len(note_pred) - 1):
                    gt_diff = note_gt[i+1] - note_gt[i]
                    if gt_diff.abs() < 0.05:  # skip near-equal pairs
                        continue
                    pred_diff = note_pred[i+1] - note_pred[i]
                    sign = gt_diff.sign()
                    # Hinge: loss if predicted diff doesn't match GT direction
                    ranking_losses.append(torch.relu(margin - sign * pred_diff))

            # Contrastive dynamics loss: random note pairs with large GT difference
            if contrastive_dynamics_weight > 0 and len(note_pred) >= 3:
                n_notes = len(note_pred)
                # Compare all pairs with sufficient GT difference
                for i in range(min(n_notes, 8)):
                    for j in range(i+1, min(n_notes, 8)):
                        gt_diff = (note_gt[i] - note_gt[j]).abs()
                        if gt_diff < 0.2:  # threshold for "different" dynamics
                            continue
                        pred_diff = (note_pred[i] - note_pred[j]).abs()
                        contrastive_losses.append(torch.relu(margin - pred_diff))

        if ranking_losses and note_ranking_weight > 0:
            ranking_loss = torch.stack(ranking_losses).mean()
            amp_loss = amp_loss + note_ranking_weight * ranking_loss
        if contrastive_losses and contrastive_dynamics_weight > 0:
            contrastive_loss = torch.stack(contrastive_losses).mean()
            amp_loss = amp_loss + contrastive_dynamics_weight * contrastive_loss

    # Slow+Fast decomposition auxiliary loss (Round 159, exp225)
    # When note_info contains slow_log/fast_log, force the model's slow and fast
    # components to separately match Gaussian-smoothed (slow) and residual (fast)
    # decomposition of ground-truth log-amp, on voiced frames only.
    if (slow_fast_amp_weight > 0 and note_info is not None
            and "slow_log" in note_info and "fast_log" in note_info):
        slow_pred = note_info["slow_log"]  # (B, T)
        fast_pred = note_info["fast_log"]  # (B, T)
        eps_sf = 1e-7
        log_amp_gt_raw = torch.log(amp_gt + eps_sf)  # (B, T)
        voiced_mask_f = (f0_gt > 0).float()
        # Gaussian smoothing along time axis to derive slow GT
        sigma = float(slow_fast_sigma_frames)
        radius = int(3 * sigma)
        radius = max(radius, 1)
        k = torch.arange(
            -radius, radius + 1, device=log_amp_gt_raw.device, dtype=log_amp_gt_raw.dtype
        )
        w = torch.exp(-0.5 * (k / sigma) ** 2)
        w = w / w.sum()
        kernel = w.view(1, 1, -1)  # (1, 1, 2r+1)
        # Smooth log-amp, replicating boundaries to avoid edge artifacts.
        gt_padded = F.pad(log_amp_gt_raw.unsqueeze(1), (radius, radius), mode="replicate")
        slow_gt = F.conv1d(gt_padded, kernel).squeeze(1)  # (B, T)
        fast_gt = log_amp_gt_raw - slow_gt
        denom_sf = voiced_mask_f.sum().clamp(min=1.0)
        slow_l = ((slow_pred - slow_gt) ** 2 * voiced_mask_f).sum() / denom_sf
        fast_l = ((fast_pred - fast_gt) ** 2 * voiced_mask_f).sum() / denom_sf
        amp_loss = amp_loss + slow_fast_amp_weight * (slow_l + slow_fast_fast_weight * fast_l)

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
