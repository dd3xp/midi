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
        # Skip connection from DownBlock has in_ch channels (not out_ch),
        # so after concat: in_ch (upsampled) + in_ch (skip) = 2 * in_ch
        self.proj = nn.Conv1d(in_ch * 2, out_ch, 1)
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

    Input: concat of x_t (n_channels) and C(t) (256 channels).
    Output: predicted noise (n_channels).
    Channels: [128, 256, 512], 3 levels.
    """
    def __init__(self, input_ch=258, output_ch=2, channels=(128, 256, 512),
                 time_dim=256, n_res=2):
        super().__init__()
        self.time_embed = TimeMLPEmbedding(time_dim)
        self.input_conv = nn.Conv1d(input_ch, channels[0], 3, padding=1)

        # Attention at first two levels, not the deepest (model.md §5.5.3)
        attn_flags = [True, True, False]

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
            x_t: (B, n_ch, T) noisy signal
            t: (B,) integer timesteps
            condition: (B, 256, T) encoder output
        Returns:
            noise_pred: (B, n_ch, T)
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
    def __init__(self, n_steps=1000, channels=(128, 256, 512), n_channels=2):
        super().__init__()
        self.n_steps = n_steps
        self.n_channels = n_channels
        self.unet = UNet1D(
            input_ch=n_channels + 256,  # x_t channels + condition channels
            output_ch=n_channels,
            channels=channels,
        )

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
            x_0: (B, n_channels, T) clean signal
            condition: (B, 256, T) encoder output
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)
        noise_pred = self.unet(x_t, t, condition)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def ddim_sample(self, condition, n_steps=50, eta=0.0):
        """DDIM sampling with configurable steps and stochasticity.

        Args:
            condition: (B, 256, T) encoder output
            n_steps: number of sampling steps
            eta: stochasticity parameter (0=deterministic DDIM, 1=DDPM-like)
        Returns:
            x_0: (B, n_channels, T) generated signal
        """
        B, _, T = condition.shape
        device = condition.device

        # Uniform subsequence of timesteps
        step_size = self.n_steps // n_steps
        timesteps = list(range(self.n_steps - 1, -1, -step_size))[:n_steps]

        x = torch.randn(B, self.n_channels, T, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.unet(x, t_batch, condition)

            alpha_t = self.alphas_cumprod[t]
            if i + 1 < len(timesteps):
                alpha_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=device)

            # Predict x_0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

            # Stochastic DDIM (Song et al., 2020 Eq. 12)
            # sigma_t controls noise injection; eta=0 is deterministic
            sigma_t = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            )
            # Direction pointing to x_t
            dir_xt = torch.sqrt(
                torch.clamp(1 - alpha_prev - sigma_t ** 2, min=0.0)
            ) * noise_pred
            # Noise injection
            if eta > 0 and i + 1 < len(timesteps):
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = torch.sqrt(alpha_prev) * x0_pred + dir_xt + sigma_t * noise

        return x


# ============ EncoderAdapter (exp051) ============

class EncoderAdapter(nn.Module):
    """Lightweight bottleneck adapter to transform encoder features for amp prediction.

    Learns amp-specific nonlinear feature transformations without modifying the encoder.
    Residual connection ensures at worst it passes through original features.

    Args:
        dim: encoder output dimension (256)
        bottleneck_dim: bottleneck size (controls capacity)
        n_layers: number of bottleneck layers (1 or 2)
        dropout: dropout rate
    """
    def __init__(self, dim=256, bottleneck_dim=64, n_layers=1, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_dim = dim if i == 0 else bottleneck_dim
            out_dim = bottleneck_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(bottleneck_dim, dim))
        self.adapter = nn.Sequential(*layers)
        # Initialize final linear near zero so adapter starts as identity
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x):
        """x: (B, T, 256) -> (B, T, 256) — adapted features with residual"""
        return x + self.adapter(x)


# ============ AmpPredictor ============

class AmpPredictor(nn.Module):
    """Predicts log-amplitude from MIDI encoder condition.
    Separate from diffusion to allow independent optimization.
    Output is in log-space; apply exp() at inference to get linear amp.
    Larger capacity version (exp017): 2-layer BiGRU with hidden=128.

    f0_conditioned (exp022+): optionally concatenate normalized f0 (1 dim)
    to the condition input, so amp prediction can leverage f0 patterns
    (vibrato -> dynamics correlation).

    use_attention (exp023+): add self-attention (TransformerEncoder) after GRU
    to capture long-range amplitude patterns (phrase-level dynamics).

    note_position_conditioned (exp044+): concatenate note_position (0=onset, 1=offset)
    directly to AmpPredictor input. Provides strong inductive bias for ADSR envelope.
    """
    def __init__(self, cond_dim=256, hidden=256, gru_hidden=128,
                 n_gru_layers=2, dropout=0.3, f0_conditioned=False,
                 use_attention=False, n_attn_heads=4, n_attn_layers=2,
                 instrument_conditioned=False, n_instruments=10, inst_embed_dim=32,
                 note_position_conditioned=False,
                 velocity_conditioned=False, condition_dropout=0.0):
        super().__init__()
        self.f0_conditioned = f0_conditioned
        self.instrument_conditioned = instrument_conditioned
        self.note_position_conditioned = note_position_conditioned
        self.velocity_conditioned = velocity_conditioned
        self.condition_dropout = condition_dropout
        self.use_attention = use_attention

        if instrument_conditioned:
            self.inst_embed = nn.Embedding(n_instruments, inst_embed_dim)

        input_dim = cond_dim + (1 if f0_conditioned else 0) + (inst_embed_dim if instrument_conditioned else 0) + (1 if note_position_conditioned else 0) + (1 if velocity_conditioned else 0)
        self.proj = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(
            hidden, gru_hidden,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )
        gru_out_dim = gru_hidden * 2  # bidirectional

        if use_attention:
            attn_layer = nn.TransformerEncoderLayer(
                d_model=gru_out_dim,
                nhead=n_attn_heads,
                dim_feedforward=gru_out_dim * 2,  # 512 for gru_hidden=128
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,  # Pre-LN for better training stability
            )
            self.attn = nn.TransformerEncoder(attn_layer, num_layers=n_attn_layers)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(gru_out_dim, 1)  # no activation, log-space

    def forward(self, condition, f0=None, instrument_id=None, note_position=None, velocity=None):
        """
        Args:
            condition: (B, T, 256) encoder output
            f0: (B, T) normalized f0 (cent_offset / 200), optional.
                Required when f0_conditioned=True.
            instrument_id: (B,) integer instrument IDs, optional.
                Required when instrument_conditioned=True.
            note_position: (B, T) position within note (0=onset, 1=offset), optional.
                Required when note_position_conditioned=True.
            velocity: (B, T) normalized velocity (0-1), optional.
                Required when velocity_conditioned=True.
        Returns:
            log_amp: (B, T) in log-space
        """
        # Condition dropout: randomly zero out encoder features during training
        if self.condition_dropout > 0 and self.training:
            mask = torch.bernoulli(
                torch.full_like(condition, 1.0 - self.condition_dropout)
            )
            condition = condition * mask / (1.0 - self.condition_dropout)  # scale to preserve magnitude

        if self.f0_conditioned:
            if f0 is not None:
                h = torch.cat([condition, f0.unsqueeze(-1)], dim=-1)  # (B, T, 257)
            else:
                zeros = torch.zeros(*condition.shape[:2], 1, device=condition.device)
                h = torch.cat([condition, zeros], dim=-1)
        else:
            h = condition

        if self.instrument_conditioned and instrument_id is not None:
            inst_emb = self.inst_embed(instrument_id)  # (B, inst_embed_dim)
            inst_emb = inst_emb.unsqueeze(1).expand(-1, h.shape[1], -1)  # (B, T, inst_embed_dim)
            h = torch.cat([h, inst_emb], dim=-1)  # (B, T, 256+32)

        if self.note_position_conditioned:
            if note_position is not None:
                h = torch.cat([h, note_position.unsqueeze(-1)], dim=-1)  # (B, T, +1)
            else:
                zeros = torch.zeros(*h.shape[:2], 1, device=h.device)
                h = torch.cat([h, zeros], dim=-1)

        if self.velocity_conditioned:
            if velocity is not None:
                h = torch.cat([h, velocity.unsqueeze(-1)], dim=-1)  # (B, T, +1)
            else:
                zeros = torch.zeros(*h.shape[:2], 1, device=h.device)
                h = torch.cat([h, zeros], dim=-1)

        h = F.relu(self.proj(h))
        h, _ = self.gru(h)
        if self.use_attention:
            h = self.attn(h)  # (B, T, 256) self-attention over time
        h = self.dropout(h)
        return self.out(h).squeeze(-1)


# ============ TwoStepAmpPredictor (exp047) ============

class TwoStepAmpPredictor(nn.Module):
    """Two-step amp prediction: note-level mean + frame-level residual.

    Step 1: NoteLevelMLP predicts per-note mean log-amp from velocity + pitch
            + smoothed encoder features (avg_pool1d over smooth_kernel frames).
    Step 2: ResidualPredictor (BiGRU + optional attention) predicts frame-level
            residual from encoder condition + note_level.
    Final output: note_level + residual.

    This structural decomposition forces the model to separate "how loud is this
    note" (simple, from velocity/pitch/encoder context) from "what shape is the
    envelope" (complex, from temporal context).
    """
    def __init__(self, cond_dim=256, hidden=256, gru_hidden=128,
                 n_gru_layers=2, dropout=0.3,
                 use_attention=True, n_attn_heads=4, n_attn_layers=2,
                 smooth_kernel=31):
        super().__init__()
        self.smooth_kernel = smooth_kernel

        # Step 1: Note-level predictor with encoder features
        # Input: velocity(1) + pitch(1) + smooth_encoder(cond_dim) = cond_dim + 2
        note_input_dim = cond_dim + 2
        self.note_mlp = nn.Sequential(
            nn.Linear(note_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Step 2: Frame-level residual predictor
        # Input: encoder condition (cond_dim) + note_level (1) = cond_dim + 1
        input_dim = cond_dim + 1
        self.proj = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(
            hidden, gru_hidden,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )
        gru_out_dim = gru_hidden * 2  # bidirectional

        self.use_attention = use_attention
        if use_attention:
            attn_layer = nn.TransformerEncoderLayer(
                d_model=gru_out_dim, nhead=n_attn_heads,
                dim_feedforward=gru_out_dim * 2, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True,
            )
            self.attn = nn.TransformerEncoder(attn_layer, num_layers=n_attn_layers)

        self.dropout_layer = nn.Dropout(dropout)
        self.out = nn.Linear(gru_out_dim, 1)

    def forward(self, condition, velocity, pitch, **kwargs):
        """
        Args:
            condition: (B, T, 256) encoder output
            velocity: (B, T) normalized velocity (0-1)
            pitch: (B, T) normalized pitch (midi/127)
        Returns:
            log_amp: (B, T) = note_level + residual, in log-space
            note_level: (B, T) note-level prediction (for auxiliary loss)
        """
        # Step 1: Smooth pool encoder features for note-level context
        # condition: (B, T, cond_dim) → transpose → pool → transpose
        cond_t = condition.permute(0, 2, 1)  # (B, cond_dim, T)
        pad = self.smooth_kernel // 2
        smoothed = F.avg_pool1d(cond_t, self.smooth_kernel, stride=1, padding=pad)
        smoothed = smoothed.permute(0, 2, 1)  # (B, T, cond_dim)
        # Handle edge case: pool might produce T+1 length
        smoothed = smoothed[:, :condition.shape[1], :]

        # Note-level prediction with rich features
        note_features = torch.cat([
            velocity.unsqueeze(-1),   # (B, T, 1)
            pitch.unsqueeze(-1),      # (B, T, 1)
            smoothed                   # (B, T, cond_dim)
        ], dim=-1)  # (B, T, cond_dim + 2)
        note_level = self.note_mlp(note_features).squeeze(-1)  # (B, T)

        # Step 2: Frame-level residual (NO detach — allow cooperative training)
        cond_aug = torch.cat([condition, note_level.unsqueeze(-1)], dim=-1)
        h = F.relu(self.proj(cond_aug))
        h, _ = self.gru(h)
        if self.use_attention:
            h = self.attn(h)
        h = self.dropout_layer(h)
        residual = self.out(h).squeeze(-1)  # (B, T)

        return note_level + residual, note_level


# ============ TCN AmpPredictor (exp027) ============

class TCNBlock(nn.Module):
    """Residual dilated causal conv block for TCN AmpPredictor."""
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        # 'same' padding for dilated conv
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation, padding=padding)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, C, T)"""
        return x + self.dropout(self.act(self.norm(self.conv(x))))


class TCNAmpPredictor(nn.Module):
    """WaveNet-style dilated CNN for amplitude prediction.
    Replaces GRU+Attention with stacked dilated convolutions.
    Fewer parameters (~400K vs ~2.3M), multi-scale temporal modeling.
    """
    def __init__(self, cond_dim=256, n_channels=128, kernel_size=3,
                 n_layers=8, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(cond_dim, n_channels)
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(n_channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_layers)
        ])
        self.final_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(n_channels, 1)

    def forward(self, condition, f0=None, **kwargs):
        """
        Args:
            condition: (B, T, 256) encoder output
            f0: ignored (API compatibility)
            **kwargs: absorbs instrument_id, note_position, velocity (not used by TCN)
        Returns:
            log_amp: (B, T) in log-space
        """
        h = F.gelu(self.input_proj(condition))  # (B, T, C)
        h = h.permute(0, 2, 1)  # (B, C, T) for Conv1d
        for block in self.tcn_blocks:
            h = block(h)
        h = h.permute(0, 2, 1)  # (B, T, C)
        h = self.final_dropout(h)
        return self.output_proj(h).squeeze(-1)  # (B, T)


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
