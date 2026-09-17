"""
MIDI Encoder: converts frame-level note features to contextual embeddings.

GRU variant: Linear(6, 128) -> BiGRU(128, hidden=128) -> output (T, 256)
Transformer variant: Linear(input_dim, d_model) -> Sinusoidal PE -> TransformerEncoder -> output (T, d_model)

Input features (6-dim, velocity removed):
  0: is_voiced, 1: normalized_pitch, 2: position_in_note,
  3: time_since_onset, 4: is_onset, 5: is_offset
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MIDIEncoder(nn.Module):
    def __init__(self, input_dim=6, linear_dim=128, gru_hidden=128, dropout=0.3,
                 num_layers=2, output_dim=None):
        super().__init__()
        self.linear = nn.Linear(input_dim, linear_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(
            input_size=linear_dim,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            num_layers=num_layers,
        )
        gru_out = gru_hidden * 2
        if output_dim is not None and output_dim != gru_out:
            self.output_proj = nn.Linear(gru_out, output_dim)
            self.output_dim = output_dim
        else:
            self.output_proj = None
            self.output_dim = gru_out

    def forward(self, frame_features):
        """
        Args:
            frame_features: (B, T, 6)
        Returns:
            context: (B, T, output_dim)
        """
        x = self.relu(self.linear(frame_features))
        x, _ = self.gru(x)
        if self.output_proj is not None:
            x = self.output_proj(x)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer.

    Dynamically expands PE buffer if input exceeds max_len (handles variable-length eval).
    """
    def __init__(self, d_model, max_len=16384):
        super().__init__()
        self.d_model = d_model
        pe = self._build_pe(max_len, d_model)
        self.register_buffer("pe", pe)  # (1, max_len, d_model)

    @staticmethod
    def _build_pe(length, d_model):
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, length, d_model)

    def forward(self, x):
        """x: (B, T, d_model)"""
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Dynamically expand PE for longer sequences (e.g., full-length eval)
            self.pe = self._build_pe(seq_len, self.d_model).to(x.device)
        return x + self.pe[:, :seq_len]


class TransformerMIDIEncoder(nn.Module):
    """Full Transformer encoder replacing BiGRU.

    Input: (B, T, input_dim) frame features
    -> Linear projection to d_model
    -> Sinusoidal positional encoding
    -> N layers of TransformerEncoderLayer (Pre-LN, GELU)
    -> Output: (B, T, d_model=256)
    """
    def __init__(self, input_dim=6, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, dropout=0.2, max_len=16384):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.input_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = d_model

    def forward(self, frame_features):
        """
        Args:
            frame_features: (B, T, input_dim)
        Returns:
            context: (B, T, d_model)
        """
        x = self.input_proj(frame_features)  # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.input_dropout(x)
        x = self.transformer(x)  # (B, T, d_model)
        return x


class HybridEncoder(nn.Module):
    """BiGRU (local) + Transformer (global) hybrid encoder.

    BiGRU captures local sequential patterns with strong inductive bias.
    Transformer layers on top enable global context aggregation.
    """
    def __init__(self, input_dim=6, linear_dim=128, gru_hidden=128,
                 gru_layers=2, gru_dropout=0.3,
                 n_transformer_layers=1, nhead=4,
                 dim_feedforward=512, transformer_dropout=0.3,
                 output_dim=None):
        super().__init__()
        self.gru_enc = MIDIEncoder(
            input_dim=input_dim, linear_dim=linear_dim,
            gru_hidden=gru_hidden, dropout=gru_dropout,
            num_layers=gru_layers, output_dim=None,
        )
        gru_out = gru_hidden * 2  # BiGRU output dim
        d_model = gru_out

        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.tf_dropout = nn.Dropout(transformer_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        if output_dim is not None and output_dim != d_model:
            self.output_proj = nn.Linear(d_model, output_dim)
            self.output_dim = output_dim
        else:
            self.output_proj = None
            self.output_dim = d_model

    def forward(self, frame_features):
        x = self.gru_enc(frame_features)  # (B, T, gru_out)
        x = self.pos_enc(x)
        x = self.tf_dropout(x)
        x = self.transformer(x)  # (B, T, d_model)
        if self.output_proj is not None:
            x = self.output_proj(x)
        return x


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN -> MHSA -> Conv -> FFN (Macaron-style).

    Combines local (depthwise conv) and global (self-attention) context.
    Half-step feed-forward modules sandwich the attention and conv modules.
    """

    def __init__(self, d_model=256, nhead=8, conv_kernel_size=31,
                 dim_ff=1024, dropout=0.2):
        super().__init__()
        # First half-step FFN
        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

        # Multi-head self-attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)

        # Convolution module
        self.conv_norm = nn.LayerNorm(d_model)
        # Pointwise -> GLU -> Depthwise -> BatchNorm -> SiLU -> Pointwise
        self.conv_pw1 = nn.Linear(d_model, d_model * 2)
        padding = (conv_kernel_size - 1) // 2
        self.conv_dw = nn.Conv1d(
            d_model, d_model, conv_kernel_size,
            padding=padding, groups=d_model
        )
        self.conv_bn = nn.BatchNorm1d(d_model)
        self.conv_pw2 = nn.Linear(d_model, d_model)
        self.conv_dropout = nn.Dropout(dropout)

        # Second half-step FFN
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, T, d_model)"""
        # Half-step FFN 1
        residual = x
        x = self.ffn1_norm(x)
        x = residual + 0.5 * self.ffn1(x)

        # Multi-head self-attention
        residual = x
        x = self.attn_norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + self.attn_dropout(attn_out)

        # Convolution module
        residual = x
        x = self.conv_norm(x)
        x = self.conv_pw1(x)  # (B, T, 2*d_model)
        x = x[..., :x.shape[-1] // 2] * torch.sigmoid(x[..., x.shape[-1] // 2:])  # GLU
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.conv_dw(x)
        x = self.conv_bn(x)
        x = torch.nn.functional.silu(x)
        x = x.transpose(1, 2)  # (B, T, d_model)
        x = self.conv_pw2(x)
        x = residual + self.conv_dropout(x)

        # Half-step FFN 2
        residual = x
        x = self.ffn2_norm(x)
        x = residual + 0.5 * self.ffn2(x)

        x = self.final_norm(x)
        return x


class ConformerEncoder(nn.Module):
    """Conformer encoder for MIDI-to-expression (exp157).

    Combines CNN (local patterns) + self-attention (global context).
    Input: (B, T, input_dim) -> Linear -> PE -> N ConformerBlocks -> (B, T, d_model)
    """

    def __init__(self, input_dim=15, d_model=256, nhead=8,
                 num_layers=4, conv_kernel_size=31,
                 dim_ff=1024, dropout=0.2, max_len=16384):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, nhead, conv_kernel_size, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_dim = d_model

    def forward(self, frame_features):
        """
        Args:
            frame_features: (B, T, input_dim)
        Returns:
            context: (B, T, d_model)
        """
        x = self.input_proj(frame_features)
        x = self.pos_enc(x)
        x = self.input_dropout(x)
        for block in self.blocks:
            x = block(x)
        return x


class S4DLayer(nn.Module):
    """Simplified S4D (Diagonal State Space) layer, pure PyTorch.

    Implements a diagonal state-space model:
        x'(t) = A x(t) + B u(t)
        y(t)  = Re[C x(t)] + D u(t)

    where A is diagonal (complex), allowing parallel computation via convolution.
    Reference: "On the Parameterization and Initialization of Diagonal State Space Models"
    (Gu et al., 2022).
    """

    def __init__(self, d_model, d_state=64, dropout=0.1, bidirectional=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional

        # S4D-Lin initialization: A = -1/2 + i * pi * n
        A_real = torch.full((d_model, d_state), -0.5)
        A_imag = math.pi * torch.arange(d_state).float().unsqueeze(0).expand(d_model, -1)
        self.A_log_real = nn.Parameter(torch.log(-A_real))  # log(-real part) for stability
        self.A_imag = nn.Parameter(A_imag)

        # [C-S4D-1] Learnable dt parameter (was hardcoded dt=1)
        # Initialize log_dt ~ Uniform(log(0.001), log(0.1)) => dt in [0.001, 0.1]
        self.log_dt = nn.Parameter(torch.rand(d_model) * math.log(100) + math.log(0.001))

        # [C-S4D-2] B, C: complex, scaled by 1/sqrt(2*d_state) (was 0.5, too large)
        scale = 1.0 / math.sqrt(2 * d_state)
        self.B_re = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.B_im = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) * scale)

        # [C-S4D-3] Separate C parameters for backward direction
        if bidirectional:
            self.C_bwd_re = nn.Parameter(torch.randn(d_model, d_state) * scale)
            self.C_bwd_im = nn.Parameter(torch.randn(d_model, d_state) * scale)

        # [W-S4D-1] D initialized to zeros (was ones, dominated SSM output)
        self.D = nn.Parameter(torch.zeros(d_model))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if bidirectional:
            self.out_proj = nn.Linear(d_model * 2, d_model)
        else:
            self.out_proj = nn.Linear(d_model, d_model)

    def _s4d_kernel(self, L, direction="fwd"):
        """Compute S4D convolution kernel of length L.

        Uses ZOH (zero-order hold) discretization with learnable dt:
            A_bar = exp(A * dt)
            K[k] = Re(C * A_bar^k * (A_bar - 1) / A * B)  for k = 0..L-1

        Args:
            L: sequence length
            direction: "fwd" or "bwd" (uses separate C params for backward)

        Returns: (d_model, L) real kernel
        """
        # A = -exp(A_log_real) + i * A_imag
        A = -torch.exp(self.A_log_real) + 1j * self.A_imag  # (d_model, d_state)
        B = self.B_re + 1j * self.B_im  # (d_model, d_state)

        # [C-S4D-3] Use separate C for backward direction
        if direction == "bwd" and self.bidirectional:
            C = self.C_bwd_re + 1j * self.C_bwd_im
        else:
            C = self.C_re + 1j * self.C_im  # (d_model, d_state)

        # [C-S4D-1] Learnable dt
        dt = torch.exp(self.log_dt)  # (d_model,)
        # ZOH discretization: A_bar = exp(A * dt)
        A_dt = A * dt.unsqueeze(-1)  # (d_model, d_state)
        A_bar = torch.exp(A_dt)  # (d_model, d_state)

        # K[k] = sum_n C_n * A_bar_n^k * B_n * (A_bar_n - 1) / A_n
        coeff = C * B * (A_bar - 1) / A  # (d_model, d_state)

        # Powers: A_bar^k = exp(k * A_dt)
        k = torch.arange(L, device=A.device).float()
        vandermonde = torch.exp(A_dt.unsqueeze(-1) * k.unsqueeze(0).unsqueeze(0))  # (d_model, d_state, L)
        kernel = torch.einsum('dn,dnl->dl', coeff, vandermonde).real  # (d_model, L)

        return kernel

    def forward(self, x):
        """x: (B, T, d_model) -> (B, T, d_model)"""
        residual = x
        x = self.norm(x)
        B, T, D = x.shape

        # Compute forward kernel
        kernel_fwd = self._s4d_kernel(T, direction="fwd")  # (d_model, T)

        # Apply via FFT convolution (efficient for long sequences)
        u = x.permute(0, 2, 1)  # (B, D, T)

        # FFT conv
        L_fft = 2 * T  # avoid circular convolution artifacts
        U_f = torch.fft.rfft(u, n=L_fft, dim=-1)  # (B, D, L_fft//2+1)
        K_fwd_f = torch.fft.rfft(kernel_fwd, n=L_fft, dim=-1)  # (D, L_fft//2+1)
        Y_f = U_f * K_fwd_f.unsqueeze(0)  # (B, D, L_fft//2+1)
        y = torch.fft.irfft(Y_f, n=L_fft, dim=-1)[..., :T]  # (B, D, T)

        # Add D * u (skip connection in SSM)
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * u  # (B, D, T)

        if self.bidirectional:
            # [C-S4D-3] Backward direction uses independent C parameters
            kernel_bwd = self._s4d_kernel(T, direction="bwd")  # (d_model, T)
            K_bwd_f = torch.fft.rfft(kernel_bwd, n=L_fft, dim=-1)
            u_rev = u.flip(-1)
            U_rev_f = torch.fft.rfft(u_rev, n=L_fft, dim=-1)
            Y_rev_f = U_rev_f * K_bwd_f.unsqueeze(0)
            y_rev = torch.fft.irfft(Y_rev_f, n=L_fft, dim=-1)[..., :T]
            y_rev = y_rev + self.D.unsqueeze(0).unsqueeze(-1) * u_rev
            y_rev = y_rev.flip(-1)
            y = torch.cat([y, y_rev], dim=1)  # (B, 2*D, T)

        y = y.permute(0, 2, 1)  # (B, T, D or 2*D)
        y = self.out_proj(y)  # (B, T, D)
        y = self.dropout(y)

        return residual + y


class S4DEncoder(nn.Module):
    """S4D-based encoder for MIDI-to-expression.

    Replaces BiGRU/Transformer with stacked S4D layers.
    Input: (B, T, input_dim) -> Linear -> N S4D layers -> (B, T, d_model)
    """

    def __init__(self, input_dim=20, d_model=256, n_layers=4, d_state=64,
                 dropout=0.2, bidirectional=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            S4DLayer(d_model, d_state, dropout, bidirectional)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_dim = d_model

    def forward(self, frame_features):
        """
        Args:
            frame_features: (B, T, input_dim)
        Returns:
            context: (B, T, d_model)

        For long sequences (T > chunk_size), uses chunked inference with
        overlapping windows and Hann-window blending to avoid OOM from
        the Vandermonde matrix in S4D kernel computation.
        """
        B, T, D = frame_features.shape
        chunk_size = 1024
        overlap = 128

        if T <= chunk_size or self.training:
            # Short sequence or training: standard forward pass
            x = self.input_proj(frame_features)
            x = self.input_dropout(x)
            for layer in self.layers:
                x = layer(x)
            x = self.final_norm(x)
            return x

        # Chunked inference for long sequences (avoids Vandermonde OOM)
        stride = chunk_size - overlap
        device = frame_features.device

        # Pre-compute Hann window weights for overlap blending
        hann = torch.hann_window(overlap * 2, device=device)  # (2*overlap,)
        fade_out = hann[:overlap]   # decreasing from 1 to 0
        fade_in = hann[overlap:]    # increasing from 0 to 1

        # Accumulate weighted outputs
        out_sum = torch.zeros(B, T, self.output_dim, device=device)
        weight_sum = torch.zeros(T, device=device)

        starts = list(range(0, T - chunk_size + 1, stride))
        if starts[-1] + chunk_size < T:
            starts.append(T - chunk_size)

        for start in starts:
            end = start + chunk_size
            chunk_in = frame_features[:, start:end, :]

            # Forward through S4D layers
            x = self.input_proj(chunk_in)
            x = self.input_dropout(x)
            for layer in self.layers:
                x = layer(x)
            x = self.final_norm(x)  # (B, chunk_size, d_model)

            # Build per-frame weights: 1.0 in center, fade in/out at edges
            w = torch.ones(chunk_size, device=device)
            if start > 0:
                # Fade in at the beginning (overlap region with previous chunk)
                w[:overlap] = fade_in
            if end < T:
                # Fade out at the end (overlap region with next chunk)
                w[-overlap:] = fade_out

            out_sum[:, start:end, :] += x * w.unsqueeze(0).unsqueeze(-1)
            weight_sum[start:end] += w

        # Normalize by total weight
        weight_sum = weight_sum.clamp(min=1e-8)
        out = out_sum / weight_sum.unsqueeze(0).unsqueeze(-1)
        return out


# ============ Mamba (Selective State Space) Encoder ============
# Pure PyTorch implementation based on mamba-minimal
# (https://github.com/johnma2006/mamba-minimal)
# No dependency on mamba-ssm CUDA package.


class SelectiveSSM(nn.Module):
    """Selective State Space Model (Mamba core).

    Unlike S4D where A/B/C are fixed, Mamba makes B/C/dt input-dependent
    (selective), allowing the model to dynamically decide what to remember.

    Input: (B, T, D) -> (B, T, D)
    """

    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        # Input projection: x -> (z, x_proj) for gating
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D depthwise conv (local context before SSM)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner,
        )

        # Input-dependent projections for SSM parameters
        # x -> dt, B, C (selective mechanism)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)

        # dt projection: low-rank (d_inner -> d_inner via bottleneck)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        # Initialize dt bias for dt in [0.001, 0.1]
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            # Inverse of softplus to set bias
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_dt)

        # A parameter (diagonal, fixed structure, learned magnitude)
        # S4D-Lin initialization: A_n = -1/2 + i*pi*n (we use real part only)
        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))  # log for numerical stability

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _ssm_scan(self, x, dt, B, C):
        """Parallel selective scan using cumulative sum approximation.

        Args:
            x: (B, T, D_inner) input after conv
            dt: (B, T, D_inner) discretization step
            B: (B, T, N) input-dependent B
            C: (B, T, N) input-dependent C

        Returns: (B, T, D_inner) SSM output
        """
        batch, T, D = x.shape
        N = B.shape[-1]

        # Discretize A: A_bar = exp(-A * dt)
        A = -torch.exp(self.A_log)  # (D, N)
        # Memory-efficient: compute dtA, A_bar, dtB, x_dtB per-timestep in the loop
        # to avoid materializing full (B, T, D, N) tensors.
        # For small d_state (N<=16), the loop overhead is negligible.

        # Sequential scan — memory-efficient version that avoids (B,T,D,N) intermediates
        h = torch.zeros(batch, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :]  # (B, D)
            B_t = B[:, t, :]    # (B, N)
            C_t = C[:, t, :]    # (B, N)
            x_t = x[:, t, :]    # (B, D)

            # dtA_t: (B, D, N) — per-timestep, no full tensor
            dtA_t = dt_t.unsqueeze(-1) * A.unsqueeze(0)  # (B, D, N)
            A_bar_t = torch.exp(dtA_t)  # (B, D, N)

            # dtB_t: (B, D, N)
            dtB_t = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, D, N)
            x_dtB_t = x_t.unsqueeze(-1) * dtB_t  # (B, D, N)

            h = A_bar_t * h + x_dtB_t  # (B, D, N)
            y_t = torch.einsum('bdn,bn->bd', h, C_t)  # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, T, D)
        return y

    def forward(self, x):
        """x: (B, T, d_model) -> (B, T, d_model)"""
        residual = x
        x = self.norm(x)
        B_sz, T, D = x.shape

        # Project and split into x_proj and gate z
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # 1D depthwise conv (causal: trim right padding)
        x_conv = x_proj.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T]  # causal trim
        x_conv = F.silu(x_conv).transpose(1, 2)  # (B, T, d_inner)

        # Input-dependent SSM parameters
        x_ssm_params = self.x_proj(x_conv)  # (B, T, N*2 + d_inner)
        N = self.d_state
        B_param = x_ssm_params[:, :, :N]  # (B, T, N)
        C_param = x_ssm_params[:, :, N:2*N]  # (B, T, N)
        dt_param = x_ssm_params[:, :, 2*N:]  # (B, T, d_inner)

        # dt through projection + softplus
        dt = F.softplus(self.dt_proj(dt_param))  # (B, T, d_inner)

        # SSM scan
        y = self._ssm_scan(x_conv, dt, B_param, C_param)

        # D skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)

        return residual + y


class MambaEncoder(nn.Module):
    """Mamba-based encoder using selective state spaces.

    Bidirectional: runs forward + backward Mamba independently, concatenates,
    then projects back to d_model. This captures both past and future context.

    Input: (B, T, input_dim) -> Linear -> N Mamba layers -> (B, T, d_model)
    """

    def __init__(self, input_dim=20, d_model=256, n_layers=4, d_state=64,
                 d_conv=4, expand=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional

        if bidirectional:
            half_d = d_model // 2
            self.fwd_layers = nn.ModuleList([
                SelectiveSSM(half_d, d_state, d_conv, expand, dropout)
                for _ in range(n_layers)
            ])
            self.bwd_layers = nn.ModuleList([
                SelectiveSSM(half_d, d_state, d_conv, expand, dropout)
                for _ in range(n_layers)
            ])
            self.input_proj = nn.Linear(input_dim, d_model)
            self.fwd_proj = nn.Linear(d_model, half_d)
            self.bwd_proj = nn.Linear(d_model, half_d)
            self.combine_proj = nn.Linear(d_model, d_model)
        else:
            self.layers = nn.ModuleList([
                SelectiveSSM(d_model, d_state, d_conv, expand, dropout)
                for _ in range(n_layers)
            ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_dim = d_model

    def forward(self, frame_features, return_intermediates=False):
        """
        Args:
            frame_features: (B, T, input_dim)
            return_intermediates: if True, also return per-layer concatenated outputs
        Returns:
            context: (B, T, d_model)
            intermediates: dict mapping layer index to (B, T, d_model) — only if return_intermediates=True
        """
        x = self.input_proj(frame_features)  # (B, T, d_model)
        x = self.input_dropout(x)

        intermediates = {}

        if self.bidirectional:
            # Split into forward and backward streams
            x_fwd = self.fwd_proj(x)  # (B, T, half_d)
            x_bwd = self.bwd_proj(x)  # (B, T, half_d)

            # Backward direction: flip once before processing
            x_bwd = x_bwd.flip(1)

            # Process layers, capturing intermediates
            n_layers = len(self.fwd_layers)
            for i in range(n_layers):
                x_fwd = self.fwd_layers[i](x_fwd)
                x_bwd = self.bwd_layers[i](x_bwd)
                if return_intermediates:
                    # Combine fwd + bwd at this layer (flip bwd back to original order)
                    layer_out = torch.cat([x_fwd, x_bwd.flip(1)], dim=-1)  # (B, T, d_model)
                    intermediates[i] = self.combine_proj(layer_out)

            # Flip backward back to original order
            x_bwd = x_bwd.flip(1)

            # Concatenate and project
            x = torch.cat([x_fwd, x_bwd], dim=-1)  # (B, T, d_model)
            x = self.combine_proj(x)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if return_intermediates:
                    intermediates[i] = x.clone()

        x = self.final_norm(x)

        if return_intermediates:
            return x, intermediates
        return x
