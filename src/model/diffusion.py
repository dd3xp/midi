"""
Conditional DDPM with 1D U-Net denoiser for f0 + amp generation.
Cosine noise schedule, FiLM time embedding, DDIM sampling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ Noise Schedule ============

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule from Nichol & Dhariwal 2021."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


# ============ Time Embedding ============

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: (B,) integer timesteps -> (B, dim)"""
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TimeMLPEmbedding(nn.Module):
    def __init__(self, time_dim=256):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t):
        return self.mlp(self.sinusoidal(t))


# ============ ResBlock with FiLM ============

class ResBlock1D(nn.Module):
    def __init__(self, channels, time_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

        # FiLM: time embedding -> scale and shift
        self.time_proj = nn.Linear(time_dim, channels * 2)

    def forward(self, x, t_emb):
        """
        x: (B, C, L)
        t_emb: (B, time_dim)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # FiLM conditioning
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)
        h = self.norm2(h)
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]

        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return x + h


# ============ Self-Attention 1D ============

class SelfAttention1D(nn.Module):
    def __init__(self, channels, n_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x):
        """x: (B, C, L) -> (B, C, L)"""
        B, C, L = x.shape
        h = self.norm(x)
        h = h.permute(0, 2, 1)  # (B, L, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1)  # (B, C, L)
        return x + h


# ============ U-Net Blocks ============

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=256, n_res=2, has_attn=True):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for _ in range(n_res):
            self.res_blocks.append(ResBlock1D(out_ch, time_dim))
            if has_attn:
                self.attn_blocks.append(SelfAttention1D(out_ch))
            else:
                self.attn_blocks.append(nn.Identity())
        self.downsample = nn.Conv1d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        h = self.proj(x)
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            h = res(h, t_emb)
            if not isinstance(attn, nn.Identity):
                h = attn(h)
        skip = h
        h = self.downsample(h)
        return h, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=256, n_res=2, has_attn=True):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1)
        self.proj = nn.Conv1d(in_ch + out_ch, out_ch, 1)
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for _ in range(n_res):
            self.res_blocks.append(ResBlock1D(out_ch, time_dim))
            if has_attn:
                self.attn_blocks.append(SelfAttention1D(out_ch))
            else:
                self.attn_blocks.append(nn.Identity())

    def forward(self, x, skip, t_emb):
        h = self.upsample(x)
        if h.shape[-1] != skip.shape[-1]:
            h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
        h = torch.cat([h, skip], dim=1)
        h = self.proj(h)
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            h = res(h, t_emb)
            if not isinstance(attn, nn.Identity):
                h = attn(h)
        return h


class MidBlock(nn.Module):
    def __init__(self, channels, time_dim=256, n_res=2):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock1D(channels, time_dim) for _ in range(n_res)
        ])

    def forward(self, x, t_emb):
        for res in self.res_blocks:
            x = res(x, t_emb)
        return x


# ============ 1D U-Net Denoiser ============

class UNet1D(nn.Module):
    """Conditional 1D U-Net for noise prediction.

    Input: concat of x_t (2 channels) and C(t) (256 channels) = 258 channels.
    Output: predicted noise (2 channels).
    Channels: [128, 256, 512], 3 levels.
    """
    def __init__(self, input_ch=258, output_ch=2, channels=(128, 256, 512),
                 time_dim=256, n_res=2):
        super().__init__()
        self.time_embed = TimeMLPEmbedding(time_dim)
        self.input_conv = nn.Conv1d(input_ch, channels[0], 3, padding=1)

        # Attention at first two levels, not the deepest
        attn_flags = [True, False, False]

        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_blocks.append(
                DownBlock(channels[i], channels[i + 1], time_dim, n_res, attn_flags[i])
            )

        self.mid_block = MidBlock(channels[-1], time_dim, n_res)

        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up_blocks.append(
                UpBlock(channels[i], channels[i - 1], time_dim, n_res, attn_flags[i - 1])
            )

        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], output_ch, 1),
        )

    def forward(self, x_t, t, condition):
        """
        Args:
            x_t: (B, 2, T) noisy signal
            t: (B,) integer timesteps
            condition: (B, 256, T) encoder output
        Returns:
            noise_pred: (B, 2, T)
        """
        t_emb = self.time_embed(t)
        h = torch.cat([x_t, condition], dim=1)  # (B, 258, T)
        h = self.input_conv(h)

        skips = []
        for down in self.down_blocks:
            h, skip = down(h, t_emb)
            skips.append(skip)

        h = self.mid_block(h, t_emb)

        for up, skip in zip(self.up_blocks, reversed(skips)):
            h = up(h, skip, t_emb)

        return self.output_conv(h)


# ============ Diffusion Model ============

class ConditionalDDPM(nn.Module):
    def __init__(self, n_steps=1000, channels=(128, 256, 512)):
        super().__init__()
        self.n_steps = n_steps
        self.unet = UNet1D(channels=channels)

        betas = cosine_beta_schedule(n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise

    def training_loss(self, x_0, condition):
        """Compute noise prediction MSE loss.

        Args:
            x_0: (B, 2, T) clean signal [normalized f0, normalized amp]
            condition: (B, 256, T) encoder output
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)
        noise_pred = self.unet(x_t, t, condition)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def ddim_sample(self, condition, n_steps=50):
        """DDIM sampling with configurable steps.

        Args:
            condition: (B, 256, T) encoder output
            n_steps: number of sampling steps
        Returns:
            x_0: (B, 2, T) generated signal
        """
        B, _, T = condition.shape
        device = condition.device

        # Uniform subsequence of timesteps
        step_size = self.n_steps // n_steps
        timesteps = list(range(self.n_steps - 1, -1, -step_size))[:n_steps]

        x = torch.randn(B, 2, T, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.unet(x, t_batch, condition)

            alpha_t = self.alphas_cumprod[t]
            if i + 1 < len(timesteps):
                alpha_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=device)

            # DDIM deterministic update
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
            x = torch.sqrt(alpha_prev) * x0_pred + \
                torch.sqrt(1 - alpha_prev) * noise_pred

        return x


# ============ Normalization Utilities ============

def normalize_f0_cents(f0_hz, notes, hop_time):
    """Normalize f0 to cent offset / 200, relative to active note pitch.

    Args:
        f0_hz: (T,) f0 in Hz
        notes: (N, 4) note array
        hop_time: frame interval
    Returns:
        f0_norm: (T,) normalized cent offset
        voiced_mask: (T,) boolean mask for voiced frames
    """
    T = len(f0_hz)
    note_midi = torch.zeros(T, device=f0_hz.device)
    for onset, offset, midi_pitch, _ in notes:
        start = max(0, int(float(onset) / hop_time))
        end = min(T, int(float(offset) / hop_time))
        note_midi[start:end] = float(midi_pitch)

    voiced_mask = (f0_hz > 0) & (note_midi > 0)
    f0_midi = torch.zeros_like(f0_hz)
    f0_midi[voiced_mask] = 69.0 + 12.0 * torch.log2(f0_hz[voiced_mask] / 440.0)
    cent_offset = (f0_midi - note_midi) * 100.0
    f0_norm = cent_offset / 200.0
    f0_norm[~voiced_mask] = 0.0
    return f0_norm, voiced_mask


def normalize_amp_log_zscore(amp, eps=1e-7, mean=None, std=None):
    """Normalize amp: log -> z-score.

    Returns:
        amp_norm: normalized amplitude
        mean: log-space mean (for denormalization)
        std: log-space std (for denormalization)
    """
    amp_log = torch.log(amp + eps)
    if mean is None:
        mean = amp_log.mean()
    if std is None:
        std = amp_log.std() + eps
    amp_norm = (amp_log - mean) / std
    return amp_norm, mean, std


def denormalize_f0(f0_norm, notes, hop_time):
    """Denormalize f0 from cent offset / 200 back to Hz."""
    T = len(f0_norm)
    note_midi = torch.zeros(T, device=f0_norm.device)
    for onset, offset, midi_pitch, _ in notes:
        start = max(0, int(float(onset) / hop_time))
        end = min(T, int(float(offset) / hop_time))
        note_midi[start:end] = float(midi_pitch)

    cent_offset = f0_norm * 200.0
    f0_midi = note_midi + cent_offset / 100.0
    f0_hz = 440.0 * 2.0 ** ((f0_midi - 69.0) / 12.0)
    f0_hz[note_midi == 0] = 0.0
    return f0_hz


def denormalize_amp(amp_norm, mean, std, eps=1e-7):
    """Denormalize amp from z-score back to linear."""
    amp_log = amp_norm * std + mean
    return torch.exp(amp_log) - eps
