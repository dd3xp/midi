"""
Training entry point for MIDI-to-expression model.
Stage 1: Encoder + Baseline (joint), Stage 2: Diffusion (encoder frozen).
"""

import os
import sys
import json
import copy
import random
import hashlib
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Limit GPU memory to 75% to leave room for OS/desktop
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.model.dataset import ExpressionDataset, collate_fn
from src.model.baseline import BaselineModel, compute_baseline_loss
from src.model.diffusion import (
    ConditionalDDPM, AmpPredictor, TwoStepAmpPredictor, EncoderAdapter,
    normalize_f0_cents, normalize_amp_log_zscore,
)


DEFAULT_CONFIG = {
    "data_dir": os.path.join(PROJECT_ROOT, "datagen", "solo", "URMP"),
    "bach10_dir": os.path.join(PROJECT_ROOT, "datagen", "solo", "Bach10"),
    "phenicx_dir": os.path.join(PROJECT_ROOT, "datagen", "solo", "PHENICX"),
    "trios_dir": os.path.join(PROJECT_ROOT, "datagen", "solo", "TRIOS"),
    "output_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
    "instruments": ["vn", "va", "vc", "fl", "ob", "cl", "sax", "tpt", "tbn", "bn"],
    "crop_len": 512,
    "seed": 42,
    "stage1": {
        "batch_size": 32,
        "lr": 1e-3,
        "epochs": 100,
        "lam": 1.0,
        "patience": 15,
        "lr_patience": 5,
        "lr_factor": 0.5,
        "dropout": 0.3,
    },
    "stage2": {
        "batch_size": 16,
        "lr": 2e-4,
        "epochs": 200,
        "ema_decay": 0.995,
        "n_diffusion_steps": 1000,
        "samples_per_epoch": None,
    },
    "stage3": {
        "batch_size": 16,
        "lr": 2e-4,
        "epochs": 200,
        "ema_decay": 0.995,
        "n_diffusion_steps": 1000,
        "samples_per_epoch": None,
    },
    "save_every": 10,
}


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_stage1(config, device):
    """Stage 1: Train MIDI Encoder + Baseline jointly."""
    cfg = config["stage1"]
    print("=" * 60)
    print("Stage 1: Training Encoder + Baseline")
    print("=" * 60)

    train_ds = ExpressionDataset(
        data_dir=config["data_dir"],
        bach10_dir=config.get("bach10_dir"),
        phenicx_dir=config.get("phenicx_dir"),
        trios_dir=config.get("trios_dir"),
        instruments=config["instruments"],
        split="train",
        crop_len=config["crop_len"],
        seed=config["seed"],
    )
    test_ds = ExpressionDataset(
        data_dir=config["data_dir"],
        bach10_dir=config.get("bach10_dir"),
        phenicx_dir=config.get("phenicx_dir"),
        trios_dir=config.get("trios_dir"),
        instruments=config["instruments"],
        split="test",
        crop_len=config["crop_len"],
        seed=config["seed"],
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"Train: {len(train_ds)} tracks, Test: {len(test_ds)} tracks")

    model = BaselineModel(dropout=cfg["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg["lr_patience"], factor=cfg["lr_factor"],
    )

    os.makedirs(config["output_dir"], exist_ok=True)
    history = {"train_loss": [], "test_loss": [], "train_f0_loss": [], "train_amp_loss": []}
    best_test_loss = float("inf")
    no_improve = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss, epoch_f0, epoch_amp, n_batches = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            ff = batch["frame_features"].to(device)
            f0 = batch["f0"].to(device)
            amp = batch["amp"].to(device)
            f0_bins = batch["f0_bins"].to(device)

            f0_logits, amp_pred = model(ff)
            loss, f0_loss, amp_loss = compute_baseline_loss(
                f0_logits, amp_pred, f0_bins, f0, amp, lam=cfg["lam"],
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_f0 += f0_loss.item()
            epoch_amp += amp_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_f0 = epoch_f0 / max(n_batches, 1)
        avg_amp = epoch_amp / max(n_batches, 1)

        # Test
        model.eval()
        test_loss_sum, test_n = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                ff = batch["frame_features"].to(device)
                f0 = batch["f0"].to(device)
                amp = batch["amp"].to(device)
                f0_bins = batch["f0_bins"].to(device)
                f0_logits, amp_pred = model(ff)
                loss, _, _ = compute_baseline_loss(
                    f0_logits, amp_pred, f0_bins, f0, amp, lam=cfg["lam"],
                )
                test_loss_sum += loss.item()
                test_n += 1

        test_loss = test_loss_sum / max(test_n, 1)
        scheduler.step(test_loss)

        history["train_loss"].append(avg_loss)
        history["test_loss"].append(test_loss)
        history["train_f0_loss"].append(avg_f0)
        history["train_amp_loss"].append(avg_amp)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{cfg['epochs']}  "
              f"train={avg_loss:.4f} (f0={avg_f0:.4f} amp={avg_amp:.4f})  "
              f"test={test_loss:.4f}  lr={lr:.2e}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(config["output_dir"], "baseline_best.pt"))
        else:
            no_improve += 1

        if epoch % config["save_every"] == 0:
            torch.save(model.state_dict(),
                       os.path.join(config["output_dir"], f"baseline_ep{epoch}.pt"))

        if no_improve >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    with open(os.path.join(config["output_dir"], "stage1_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Load best checkpoint
    model.load_state_dict(
        torch.load(os.path.join(config["output_dir"], "baseline_best.pt"),
                    map_location=device, weights_only=True)
    )
    return model


def compute_amp_stats(dataset, device):
    """Compute global amp log-space mean and std for normalization."""
    all_amp_log = []
    eps = 1e-7
    # Iterate over actual tracks, not virtual samples_per_epoch length
    for i in range(len(dataset.tracks)):
        item = dataset._load(i)
        amp = torch.from_numpy(item["amp"])
        amp_log = torch.log(amp + eps)
        all_amp_log.append(amp_log)
    all_amp_log = torch.cat(all_amp_log)
    return all_amp_log.mean().to(device), all_amp_log.std().to(device)


def compute_pitch_amp_anchor(dataset, device):
    """Compute average log_amp per MIDI pitch from training data.

    Returns a 128-element tensor where anchor[pitch] = mean(log(amp)) for all
    frames with that MIDI pitch. For pitches with no data, interpolate from neighbors.
    Similar to f0 cent offset: model predicts amp_offset = log_amp - anchor[pitch].
    """
    eps = 1e-7
    pitch_amp_sums = torch.zeros(128)
    pitch_amp_counts = torch.zeros(128)

    for i in range(len(dataset.tracks)):
        item = dataset._load(i)
        amp = item["amp"]
        notes = item["notes"]
        hop_time = item["hop_time"]
        n_frames = len(amp)

        # Build per-frame pitch from notes
        frame_pitch = np.full(n_frames, -1, dtype=np.int32)
        for onset, offset, midi_pitch, velocity in notes:
            start = max(0, int(onset / hop_time))
            end = min(n_frames, int(offset / hop_time))
            for f in range(start, end):
                frame_pitch[f] = int(midi_pitch)

        amp_log = np.log(amp + eps)
        for f in range(n_frames):
            p = frame_pitch[f]
            if 0 <= p <= 127 and amp[f] > eps:
                pitch_amp_sums[p] += amp_log[f]
                pitch_amp_counts[p] += 1

    # Compute mean; for pitches with no data, use global mean
    anchor = torch.zeros(128)
    valid = pitch_amp_counts > 0
    anchor[valid] = pitch_amp_sums[valid] / pitch_amp_counts[valid]
    global_mean = pitch_amp_sums[valid].sum() / pitch_amp_counts[valid].sum() if valid.any() else 0.0

    # Fill missing pitches with nearest neighbor interpolation
    for p in range(128):
        if not valid[p]:
            # Find nearest valid pitch
            left, right = p - 1, p + 1
            while left >= 0 and not valid[left]:
                left -= 1
            while right < 128 and not valid[right]:
                right += 1
            if left >= 0 and right < 128:
                # Linear interpolation
                anchor[p] = anchor[left] + (anchor[right] - anchor[left]) * (p - left) / (right - left)
            elif left >= 0:
                anchor[p] = anchor[left]
            elif right < 128:
                anchor[p] = anchor[right]
            else:
                anchor[p] = global_mean

    print(f"  Pitch anchor: {int(valid.sum())}/128 pitches have data")
    print(f"  Range: [{anchor[valid].min():.3f}, {anchor[valid].max():.3f}], "
          f"global_mean={global_mean:.3f}")
    return anchor.to(device)


def compute_amp_stats_per_inst(dataset, device):
    """Compute per-instrument amp log-space mean and std for normalization.

    Returns dict: {inst_id: {"mean": tensor, "std": tensor}}
    """
    from collections import defaultdict
    eps = 1e-7
    inst_amp_log = defaultdict(list)
    for i in range(len(dataset.tracks)):
        item = dataset._load(i)
        inst_id = item["instrument_id"]
        amp = torch.from_numpy(item["amp"])
        amp_log = torch.log(amp + eps)
        inst_amp_log[inst_id].append(amp_log)

    stats = {}
    for inst_id, logs in inst_amp_log.items():
        cat = torch.cat(logs)
        stats[inst_id] = {
            "mean": cat.mean().to(device),
            "std": (cat.std() + eps).to(device),
        }
        print(f"  Instrument {inst_id}: mean={stats[inst_id]['mean']:.4f}, "
              f"std={stats[inst_id]['std']:.4f}, n_frames={len(cat)}")
    return stats


def compute_note_mean(log_amp_gt, pitch, voiced):
    """Compute per-note mean of log_amp_gt.

    Args:
        log_amp_gt: (B, T) ground truth log amplitude
        pitch: (B, T) normalized pitch (midi/127)
        voiced: (B, T) bool, True where a note is active
    Returns:
        note_mean: (B, T) per-note mean, same value for all frames in a note
    """
    B, T = log_amp_gt.shape
    note_mean = torch.zeros_like(log_amp_gt)
    for b in range(B):
        p = pitch[b]  # (T,)
        v = voiced[b]  # (T,)
        # Detect note boundaries: pitch changes or voiced→unvoiced
        changes = torch.zeros(T, dtype=torch.bool, device=p.device)
        changes[0] = True
        changes[1:] = (p[1:] != p[:-1]) | (v[1:] != v[:-1])
        # Assign note IDs
        note_ids = changes.long().cumsum(0) - 1  # 0, 0, ..., 1, 1, ..., 2, ...
        n_notes = note_ids.max().item() + 1
        for nid in range(n_notes):
            mask = (note_ids == nid) & v
            if mask.any():
                note_mean[b, note_ids == nid] = log_amp_gt[b, mask].mean()
    return note_mean


def train_stage2(config, encoder, device):
    """Stage 2: Train f0-only diffusion + AmpPredictor with frozen encoder."""
    cfg = config["stage2"]
    eps = 1e-7
    print("\n" + "=" * 60)
    print("Stage 2: Training 1ch Diffusion (f0) + AmpPredictor")
    print("=" * 60)

    samples_per_epoch = cfg.get("samples_per_epoch", None)
    train_ds = ExpressionDataset(
        data_dir=config["data_dir"],
        bach10_dir=config.get("bach10_dir"),
        phenicx_dir=config.get("phenicx_dir"),
        trios_dir=config.get("trios_dir"),
        instruments=config["instruments"],
        split="train",
        crop_len=config["crop_len"],
        seed=config["seed"],
        samples_per_epoch=samples_per_epoch,
    )
    test_ds = ExpressionDataset(
        data_dir=config["data_dir"],
        bach10_dir=config.get("bach10_dir"),
        phenicx_dir=config.get("phenicx_dir"),
        trios_dir=config.get("trios_dir"),
        instruments=config["instruments"],
        split="test",
        crop_len=config["crop_len"],
        seed=config["seed"],
    )

    print(f"Train: {len(train_ds.tracks)} tracks, samples_per_epoch={len(train_ds)}")

    # Compute amp normalization stats from training set (for compatibility)
    amp_mean, amp_std = compute_amp_stats(train_ds, device)
    print(f"Amp log stats: mean={amp_mean:.4f}, std={amp_std:.4f}")
    torch.save({"amp_mean": amp_mean, "amp_std": amp_std},
               os.path.join(config["output_dir"], "norm_stats.pt"))

    # Per-instrument amp normalization (exp029+)
    amp_per_inst_norm = cfg.get("amp_per_inst_norm", False)
    amp_per_inst_stats = None
    if amp_per_inst_norm:
        print("Computing per-instrument amp stats...")
        amp_per_inst_stats = compute_amp_stats_per_inst(train_ds, device)
        # Save for evaluation
        save_stats = {k: {"mean": v["mean"].cpu(), "std": v["std"].cpu()}
                      for k, v in amp_per_inst_stats.items()}
        torch.save(save_stats,
                   os.path.join(config["output_dir"], "amp_per_inst_stats.pt"))
        print(f"Per-instrument amp stats saved ({len(amp_per_inst_stats)} instruments)")

    # Pitch-anchor amp offset (exp030+)
    amp_pitch_anchor_enabled = cfg.get("amp_pitch_anchor", False)
    pitch_amp_anchor = None
    if amp_pitch_anchor_enabled:
        print("Computing pitch-based amp anchor table...")
        pitch_amp_anchor = compute_pitch_amp_anchor(train_ds, device)
        torch.save(pitch_amp_anchor.cpu(),
                   os.path.join(config["output_dir"], "pitch_amp_anchor.pt"))
        print(f"Pitch amp anchor saved to {config['output_dir']}/pitch_amp_anchor.pt")

    # Scheduled sampling: online DDIM or pre-cached diffusion f0 (exp032+/exp036+)
    freeze_diffusion = cfg.get("freeze_diffusion", False)  # read early for ss_online check
    amp_scheduled_sampling = cfg.get("amp_scheduled_sampling", False)
    cached_f0_dir = cfg.get("cached_f0_dir", None)
    cached_f0_data = {}  # track_idx -> (n_samples, T) tensor
    ss_online = False  # True = online DDIM sampling (no cache needed)
    # Support both old (warmup/decay) and new (start/end epoch) naming
    ss_start_epoch = cfg.get("amp_ss_start_epoch", None)
    ss_end_epoch = cfg.get("amp_ss_end_epoch", None)
    if ss_start_epoch is not None and ss_end_epoch is not None:
        ss_warmup = ss_start_epoch
        ss_decay = ss_end_epoch - ss_start_epoch
    else:
        ss_warmup = cfg.get("amp_ss_warmup_epochs", 50)
        ss_decay = cfg.get("amp_ss_decay_epochs", 100)
    # Online DDIM config
    ss_ddim_steps = cfg.get("amp_ss_ddim_steps", 20)
    ss_eta = cfg.get("amp_ss_eta", 0.3)
    if amp_scheduled_sampling and cached_f0_dir and os.path.isdir(cached_f0_dir):
        index_path = os.path.join(cached_f0_dir, "index.pt")
        if os.path.isfile(index_path):
            index = torch.load(index_path, map_location="cpu", weights_only=True)
            n_loaded = 0
            for tidx in range(len(train_ds.tracks)):
                track_path = train_ds.tracks[tidx]["path"]
                t_hash = index.get(track_path)
                if t_hash:
                    cache_path = os.path.join(cached_f0_dir, f"{t_hash}.pt")
                    if os.path.isfile(cache_path):
                        data = torch.load(cache_path, map_location="cpu", weights_only=True)
                        cached_f0_data[tidx] = data["f0_samples"]  # (n_samples, T)
                        n_loaded += 1
            print(f"Scheduled sampling (cached): loaded f0 for {n_loaded}/{len(train_ds.tracks)} tracks")
            print(f"  ss_warmup={ss_warmup}, ss_decay={ss_decay}")
        else:
            print(f"WARNING: cached f0 index not found at {index_path}, disabling scheduled sampling")
            amp_scheduled_sampling = False
    elif amp_scheduled_sampling and freeze_diffusion:
        # Online DDIM sampling with frozen diffusion (exp036+)
        ss_online = True
        print(f"Scheduled sampling (online DDIM): steps={ss_ddim_steps}, eta={ss_eta}")
        print(f"  ss_warmup={ss_warmup}, ss_decay={ss_decay}")
    elif amp_scheduled_sampling:
        print(f"WARNING: scheduled sampling requires freeze_diffusion=true or cached_f0_dir, disabling")
        amp_scheduled_sampling = False

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Encoder: freeze or unfreeze (exp055+)
    unfreeze_encoder = cfg.get("unfreeze_encoder", False)
    encoder_lr = float(cfg.get("encoder_lr", 2e-6))
    encoder = encoder.to(device)
    if unfreeze_encoder:
        encoder.train()
        print(f"Encoder UNFROZEN for amp training (encoder_lr={encoder_lr})")
    else:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    # n_channels from config (default 2 for backward compat)
    n_channels = cfg.get("n_channels", 2)
    print(f"Diffusion n_channels={n_channels}")

    diffusion = ConditionalDDPM(
        n_steps=cfg["n_diffusion_steps"], n_channels=n_channels,
    ).to(device)

    # Optionally load pre-trained diffusion and freeze it (exp022+)
    freeze_diffusion = cfg.get("freeze_diffusion", False)
    diff_ckpt_path = cfg.get("diffusion_checkpoint", None)
    if diff_ckpt_path and os.path.isfile(diff_ckpt_path):
        diffusion.load_state_dict(
            torch.load(diff_ckpt_path, map_location=device, weights_only=True)
        )
        print(f"Loaded pre-trained diffusion from {diff_ckpt_path}")
    if freeze_diffusion:
        diffusion.eval()
        for param in diffusion.parameters():
            param.requires_grad = False
        print("Diffusion FROZEN — only training AmpPredictor")

    # AmpPredictor (only when n_channels=1, i.e., f0-only diffusion)
    amp_predictor = None
    amp_f0_conditioned = cfg.get("amp_f0_conditioned", False)
    amp_inst_conditioned = cfg.get("amp_instrument_conditioned", False)
    amp_note_pos_conditioned = cfg.get("amp_note_position_conditioned", False)
    amp_vel_conditioned = cfg.get("amp_velocity_conditioned", False)
    amp_condition_dropout = float(cfg.get("amp_condition_dropout", 0.0))
    amp_two_step = cfg.get("amp_two_step", False)
    amp_note_aux_weight = float(cfg.get("amp_note_aux_weight", 0.5))
    amp_residual_penalty_weight = float(cfg.get("amp_residual_penalty_weight", 0.0))
    amp_smooth_kernel = int(cfg.get("amp_smooth_kernel", 31))
    if n_channels == 1:
        amp_dropout = cfg.get("amp_dropout", 0.2)
        amp_type = cfg.get("amp_type", "gru")  # "gru" (default) or "tcn"
        if amp_two_step:
            # Two-step amp prediction (exp047+)
            amp_hidden = cfg.get("amp_hidden", 256)
            amp_gru_hidden = cfg.get("amp_gru_hidden", 128)
            amp_gru_layers = cfg.get("amp_gru_layers", 2)
            amp_use_attention = cfg.get("amp_use_attention", True)
            amp_n_attn_heads = cfg.get("amp_n_attn_heads", 4)
            amp_n_attn_layers = cfg.get("amp_n_attn_layers", 2)
            amp_predictor = TwoStepAmpPredictor(
                cond_dim=256, hidden=amp_hidden, gru_hidden=amp_gru_hidden,
                n_gru_layers=amp_gru_layers, dropout=amp_dropout,
                use_attention=amp_use_attention,
                n_attn_heads=amp_n_attn_heads,
                n_attn_layers=amp_n_attn_layers,
                smooth_kernel=amp_smooth_kernel,
            ).to(device)
            n_params_ts = sum(p.numel() for p in amp_predictor.parameters())
            print(f"TwoStepAmpPredictor enabled (hidden={amp_hidden}, gru_hidden={amp_gru_hidden}, "
                  f"layers={amp_gru_layers}, dropout={amp_dropout}, "
                  f"use_attention={amp_use_attention}, "
                  f"n_attn_heads={amp_n_attn_heads}, n_attn_layers={amp_n_attn_layers}, "
                  f"smooth_kernel={amp_smooth_kernel}, "
                  f"note_aux_weight={amp_note_aux_weight}, "
                  f"residual_penalty_weight={amp_residual_penalty_weight}, "
                  f"params={n_params_ts:,})")
        elif amp_type == "tcn":
            from src.model.diffusion import TCNAmpPredictor
            tcn_channels = cfg.get("tcn_channels", 128)
            tcn_layers = cfg.get("tcn_layers", 8)
            tcn_kernel = cfg.get("tcn_kernel_size", 3)
            amp_predictor = TCNAmpPredictor(
                cond_dim=256, n_channels=tcn_channels, kernel_size=tcn_kernel,
                n_layers=tcn_layers, dropout=amp_dropout,
            ).to(device)
            print(f"TCNAmpPredictor (channels={tcn_channels}, layers={tcn_layers}, "
                  f"kernel={tcn_kernel}, dropout={amp_dropout})")
        else:
            amp_hidden = cfg.get("amp_hidden", 256)
            amp_gru_hidden = cfg.get("amp_gru_hidden", 128)
            amp_gru_layers = cfg.get("amp_gru_layers", 2)
            amp_use_attention = cfg.get("amp_use_attention", False)
            amp_n_attn_heads = cfg.get("amp_n_attn_heads", 4)
            amp_n_attn_layers = cfg.get("amp_n_attn_layers", 2)
            amp_predictor = AmpPredictor(
                cond_dim=256, hidden=amp_hidden, gru_hidden=amp_gru_hidden,
                n_gru_layers=amp_gru_layers, dropout=amp_dropout,
                f0_conditioned=amp_f0_conditioned,
                use_attention=amp_use_attention,
                n_attn_heads=amp_n_attn_heads,
                n_attn_layers=amp_n_attn_layers,
                instrument_conditioned=amp_inst_conditioned,
                note_position_conditioned=amp_note_pos_conditioned,
                velocity_conditioned=amp_vel_conditioned,
                condition_dropout=amp_condition_dropout,
            ).to(device)
            print(f"AmpPredictor enabled (hidden={amp_hidden}, gru_hidden={amp_gru_hidden}, "
                  f"layers={amp_gru_layers}, dropout={amp_dropout}, "
                  f"f0_conditioned={amp_f0_conditioned}, "
                  f"use_attention={amp_use_attention}, "
                  f"n_attn_heads={amp_n_attn_heads}, n_attn_layers={amp_n_attn_layers}, "
                  f"instrument_conditioned={amp_inst_conditioned}, "
                  f"note_position_conditioned={amp_note_pos_conditioned}, "
                  f"velocity_conditioned={amp_vel_conditioned}, "
                  f"condition_dropout={amp_condition_dropout})")

    # EncoderAdapter (exp051+): lightweight adapter between encoder and AmpPredictor
    adapter = None
    if cfg.get("amp_encoder_adapter", False):
        adapter_bottleneck = cfg.get("adapter_bottleneck", 64)
        adapter_layers = cfg.get("adapter_layers", 1)
        adapter_dropout = cfg.get("adapter_dropout", 0.1)
        adapter = EncoderAdapter(
            dim=256, bottleneck_dim=adapter_bottleneck,
            n_layers=adapter_layers, dropout=adapter_dropout,
        ).to(device)
        n_adapter_params = sum(p.numel() for p in adapter.parameters())
        print(f"EncoderAdapter enabled (bottleneck={adapter_bottleneck}, "
              f"layers={adapter_layers}, dropout={adapter_dropout}, "
              f"params={n_adapter_params:,})")

    # Optimizer: separate optimizers for diffusion and AmpPredictor
    stage2_lr = float(cfg["lr"])
    diff_optimizer = None
    diff_scheduler = None
    if not freeze_diffusion:
        diff_optimizer = torch.optim.Adam(
            diffusion.parameters(), lr=stage2_lr, betas=(0.9, 0.999),
        )
    amp_optimizer = None
    if amp_predictor is not None:
        amp_lr = float(cfg.get("amp_lr", stage2_lr))
        amp_wd = float(cfg.get("amp_weight_decay", 0.0))
        if unfreeze_encoder:
            # Separate param groups: amp_predictor at amp_lr, encoder at encoder_lr (exp055+)
            param_groups = [
                {"params": list(amp_predictor.parameters()), "lr": amp_lr},
                {"params": list(encoder.parameters()), "lr": encoder_lr},
            ]
            if adapter is not None:
                param_groups.append({"params": list(adapter.parameters()), "lr": amp_lr})
            amp_optimizer = torch.optim.AdamW(
                param_groups, weight_decay=amp_wd,
            )
            print(f"AmpPredictor optimizer: amp_lr={amp_lr}, encoder_lr={encoder_lr}, "
                  f"weight_decay={amp_wd}"
                  f"{', +adapter params' if adapter is not None else ''}")
        else:
            # Include adapter parameters in amp optimizer (exp051+)
            amp_params = list(amp_predictor.parameters())
            if adapter is not None:
                amp_params += list(adapter.parameters())
            amp_optimizer = torch.optim.Adam(
                amp_params, lr=amp_lr, betas=(0.9, 0.999),
                weight_decay=amp_wd,
            )
            print(f"AmpPredictor optimizer: lr={amp_lr}, weight_decay={amp_wd}"
                  f"{', +adapter params' if adapter is not None else ''}")

    # Cosine LR scheduling (if lr_min specified)
    lr_min = cfg.get("lr_min", None)
    if diff_optimizer is not None and lr_min is not None:
        lr_min = float(lr_min)
        diff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            diff_optimizer, T_max=cfg["epochs"], eta_min=lr_min,
        )
        print(f"Using CosineAnnealingLR (diff): lr {stage2_lr} -> {lr_min} over {cfg['epochs']} epochs")

    amp_scheduler = None
    if amp_optimizer is not None:
        amp_lr_min = float(cfg.get("amp_lr_min", lr_min if lr_min is not None else 1e-6))
        # When encoder is unfrozen (exp055+), eta_min must not exceed encoder_lr
        # otherwise CosineAnnealingLR would INCREASE encoder lr during training
        if unfreeze_encoder and amp_lr_min > encoder_lr:
            amp_lr_min = encoder_lr * 0.1  # decay encoder lr to 10% of initial
            print(f"Adjusted amp_lr_min to {amp_lr_min:.2e} (encoder_lr constraint)")
        amp_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            amp_optimizer, T_max=cfg["epochs"], eta_min=amp_lr_min,
        )
        amp_lr_val = float(cfg.get("amp_lr", stage2_lr))
        print(f"Using CosineAnnealingLR (amp): lr {amp_lr_val} -> {amp_lr_min} over {cfg['epochs']} epochs")

    # Print amplitude augmentation config
    if cfg.get("amp_augment", False):
        print(f"Amplitude augmentation ENABLED: scale={cfg.get('amp_augment_scale', 0.2)}, jitter={cfg.get('amp_augment_jitter', 0.05)}")
    else:
        print("Amplitude augmentation: disabled")

    # Knowledge Distillation (exp054+): load pre-computed teacher predictions
    kd_teacher_preds = None
    kd_weight = float(cfg.get("kd_weight", 0.0))
    kd_teacher_path = cfg.get("kd_teacher_preds", None)
    if kd_teacher_path and os.path.isfile(kd_teacher_path):
        kd_teacher_preds = torch.load(kd_teacher_path, map_location="cpu", weights_only=True)
        print(f"Knowledge Distillation ENABLED: loaded teacher preds for "
              f"{len(kd_teacher_preds)} tracks, kd_weight={kd_weight}")
    elif kd_teacher_path:
        print(f"WARNING: KD teacher preds not found at {kd_teacher_path}, disabling KD")
    else:
        if kd_weight > 0:
            print(f"WARNING: kd_weight={kd_weight} but no kd_teacher_preds path, disabling KD")
            kd_weight = 0.0

    ema = None
    if not freeze_diffusion:
        ema = EMA(diffusion, decay=cfg["ema_decay"])

    os.makedirs(config["output_dir"], exist_ok=True)
    history = {
        "train_loss": [], "test_loss": [],
        "train_diff_loss": [], "train_amp_loss": [],
        "test_diff_loss": [], "test_amp_loss": [],
    }
    best_test_loss = float("inf")
    best_test_amp_loss = float("inf")  # independent amp early stopping

    for epoch in range(1, cfg["epochs"] + 1):
        if not freeze_diffusion:
            diffusion.train()
        if amp_predictor is not None:
            amp_predictor.train()
        if adapter is not None:
            adapter.train()
        if unfreeze_encoder:
            encoder.train()
        epoch_loss, epoch_diff_loss, epoch_amp_loss, n_batches = 0.0, 0.0, 0.0, 0

        for batch in train_loader:
            ff = batch["frame_features"].to(device)
            f0 = batch["f0"].to(device)
            amp = batch["amp"].to(device)
            instrument_id = batch["instrument_id"].to(device) if amp_inst_conditioned else None
            note_position = ff[:, :, 3] if amp_note_pos_conditioned else None  # position_in_note (0-1)
            velocity = ff[:, :, 2] if (amp_vel_conditioned or amp_two_step) else None  # normalized velocity (0-1)

            if unfreeze_encoder:
                # Encoder unfrozen (exp055+): compute condition WITH gradients for amp path
                condition = encoder(ff)  # (B, T, 256)
                condition_perm = condition.permute(0, 2, 1).detach()  # (B, 256, T) — detach for diffusion path
            else:
                with torch.no_grad():
                    condition = encoder(ff)  # (B, T, 256)
                    condition_perm = condition.permute(0, 2, 1)  # (B, 256, T)

            # Apply adapter for amp path (exp051+): adapter is trainable, condition is frozen
            condition_amp = adapter(condition) if adapter is not None else condition

            # Normalize f0: cent offset / 200
            note_pitch = ff[:, :, 1] * 127.0  # denormalize pitch
            voiced = ff[:, :, 0] > 0.5
            safe_f0 = torch.clamp(f0, min=1.0)
            f0_midi = 69.0 + 12.0 * torch.log2(safe_f0 / 440.0)
            cent_offset = (f0_midi - note_pitch) * 100.0
            f0_norm = cent_offset / 200.0
            f0_norm[~voiced] = 0.0
            f0_norm[f0 <= 0] = 0.0

            if n_channels == 1:
                # f0-only diffusion loss
                x_0 = f0_norm.unsqueeze(1)  # (B, 1, T)
                if freeze_diffusion:
                    with torch.no_grad():
                        diff_loss = diffusion.training_loss(x_0, condition_perm)
                else:
                    diff_loss = diffusion.training_loss(x_0, condition_perm)

                # AmpPredictor loss (pred is already in log-space)
                note_level_pred = None  # for two-step auxiliary loss
                if amp_two_step:
                    # Two-step amp prediction (exp047+)
                    pitch_norm = ff[:, :, 1]  # normalized pitch (midi/127)
                    log_amp_pred, note_level_pred = amp_predictor(condition_amp, velocity=velocity, pitch=pitch_norm)
                elif amp_f0_conditioned:
                    # f0-conditioned: pass GT or generated diffusion f0 (scheduled sampling)
                    # Scheduled sampling (exp032+/exp036+): mix GT and diffusion f0
                    if amp_scheduled_sampling:
                        if epoch < ss_warmup:
                            p_gt = 1.0
                        elif epoch < ss_warmup + ss_decay:
                            p_gt = 1.0 - (epoch - ss_warmup) / ss_decay
                        else:
                            p_gt = 0.0
                        use_gt = random.random() < p_gt
                        if use_gt:
                            f0_for_amp = f0_norm
                        elif ss_online:
                            # Online DDIM sampling (exp036+): generate f0 on the fly
                            with torch.no_grad():
                                diffusion.eval()
                                x_gen = diffusion.ddim_sample(
                                    condition_perm, n_steps=ss_ddim_steps, eta=ss_eta,
                                )
                                f0_for_amp = x_gen[:, 0, :]  # (B, T) normalized f0
                        elif cached_f0_data:
                            # Use cached diffusion f0 for each batch item
                            track_idxs = batch["track_idx"]
                            crop_starts = batch["crop_start"]
                            B, T = f0_norm.shape
                            f0_for_amp = f0_norm.clone()  # fallback to GT
                            for b in range(B):
                                tidx = track_idxs[b].item()
                                cstart = crop_starts[b].item()
                                if tidx in cached_f0_data:
                                    samples = cached_f0_data[tidx]  # (n_samples, T_full)
                                    si = random.randint(0, samples.shape[0] - 1)
                                    T_full = samples.shape[1]
                                    cend = min(cstart + T, T_full)
                                    seg = samples[si, cstart:cend]
                                    if len(seg) < T:
                                        seg = torch.cat([seg, torch.zeros(T - len(seg))])
                                    f0_for_amp[b] = seg[:T].to(device)
                        else:
                            f0_for_amp = f0_norm  # fallback to GT
                    else:
                        f0_for_amp = f0_norm
                    log_amp_pred = amp_predictor(condition_amp, f0=f0_for_amp, instrument_id=instrument_id, note_position=note_position, velocity=velocity)
                else:
                    log_amp_pred = amp_predictor(condition_amp, instrument_id=instrument_id, note_position=note_position, velocity=velocity)
                log_amp_gt = torch.log(amp + eps)
                # Amplitude augmentation (exp039+): training only
                if cfg.get("amp_augment", False):
                    aug_scale = cfg.get("amp_augment_scale", 0.2)
                    aug_jitter = cfg.get("amp_augment_jitter", 0.05)
                    # Random global scale: shift entire sample's amp in log space
                    if aug_scale > 0:
                        offset = torch.empty(log_amp_gt.shape[0], 1, device=log_amp_gt.device).uniform_(-aug_scale, aug_scale)
                        log_amp_gt = log_amp_gt + offset
                    # Random per-frame jitter
                    if aug_jitter > 0:
                        noise = torch.randn_like(log_amp_gt) * aug_jitter
                        log_amp_gt = log_amp_gt + noise
                # Pitch-anchor amp offset (exp030+): target = log_amp - anchor[pitch]
                if amp_pitch_anchor_enabled and pitch_amp_anchor is not None:
                    pitch_idx = note_pitch.long().clamp(0, 127)  # (B, T)
                    anchor_per_frame = pitch_amp_anchor[pitch_idx]  # (B, T)
                    log_amp_gt = log_amp_gt - anchor_per_frame
                    # Zero out unvoiced frames (no note = no anchor)
                    log_amp_gt[~voiced] = 0.0
                # Per-instrument amp normalization: z-score per instrument
                elif amp_per_inst_norm and amp_per_inst_stats is not None and instrument_id is not None:
                    for b in range(log_amp_gt.shape[0]):
                        iid = instrument_id[b].item()
                        if iid in amp_per_inst_stats:
                            log_amp_gt[b] = (log_amp_gt[b] - amp_per_inst_stats[iid]["mean"]) / amp_per_inst_stats[iid]["std"]
                mse = nn.functional.mse_loss(log_amp_pred, log_amp_gt)

                # Correlation-aligned loss components (exp024+)
                amp_corr_weight = cfg.get("amp_corr_weight", 0.0)
                amp_grad_weight = cfg.get("amp_grad_weight", 0.0)
                if amp_corr_weight > 0:
                    pred_c = log_amp_pred - log_amp_pred.mean(dim=-1, keepdim=True)
                    gt_c = log_amp_gt - log_amp_gt.mean(dim=-1, keepdim=True)
                    corr = (pred_c * gt_c).sum(dim=-1) / (
                        pred_c.norm(dim=-1) * gt_c.norm(dim=-1) + 1e-8
                    )
                    corr_loss_val = (1.0 - corr).mean()
                else:
                    corr_loss_val = 0.0
                if amp_grad_weight > 0:
                    d_pred = log_amp_pred[:, 1:] - log_amp_pred[:, :-1]
                    d_gt = log_amp_gt[:, 1:] - log_amp_gt[:, :-1]
                    grad_loss_val = nn.functional.mse_loss(d_pred, d_gt)
                else:
                    grad_loss_val = 0.0
                amp_loss = mse + amp_corr_weight * corr_loss_val + amp_grad_weight * grad_loss_val
                # Multi-scale temporal amp loss (exp031+)
                amp_multiscale = cfg.get("amp_multiscale", False)
                if amp_multiscale:
                    ms_scales = cfg.get("amp_multiscale_scales", [8, 32])
                    ms_weight = cfg.get("amp_multiscale_weight", 0.5)
                    ms_loss = 0.0
                    pred_1d = log_amp_pred.unsqueeze(1)  # (B, 1, T)
                    gt_1d = log_amp_gt.unsqueeze(1)      # (B, 1, T)
                    for scale in ms_scales:
                        if pred_1d.shape[-1] >= scale:
                            pred_ds = nn.functional.avg_pool1d(pred_1d, scale, stride=scale).squeeze(1)
                            gt_ds = nn.functional.avg_pool1d(gt_1d, scale, stride=scale).squeeze(1)
                            ms_loss += nn.functional.mse_loss(pred_ds, gt_ds)
                    amp_loss = amp_loss + ms_weight * ms_loss
                # Two-step auxiliary loss (exp047+): note_level → per-note mean
                if amp_two_step and note_level_pred is not None:
                    pitch_norm = ff[:, :, 1]  # normalized pitch (midi/127)
                    note_mean_gt = compute_note_mean(log_amp_gt, pitch_norm, voiced)
                    note_aux_loss = nn.functional.mse_loss(note_level_pred, note_mean_gt)
                    amp_loss = amp_loss + amp_note_aux_weight * note_aux_loss
                    # Residual L1 penalty (exp048+): force small residual
                    if amp_residual_penalty_weight > 0:
                        residual = log_amp_pred - note_level_pred
                        residual_penalty = torch.abs(residual).mean()
                        amp_loss = amp_loss + amp_residual_penalty_weight * residual_penalty
                # Knowledge Distillation loss (exp054+)
                if kd_teacher_preds is not None and kd_weight > 0:
                    track_idxs = batch["track_idx"]
                    crop_starts = batch["crop_start"]
                    B_kd, T_kd = log_amp_pred.shape
                    teacher_batch = torch.zeros(B_kd, T_kd, device=log_amp_pred.device)
                    for b in range(B_kd):
                        tidx = track_idxs[b].item()
                        cstart = crop_starts[b].item()
                        if tidx in kd_teacher_preds:
                            t_pred = kd_teacher_preds[tidx]  # (T_full,) on CPU
                            cend = min(cstart + T_kd, t_pred.shape[0])
                            seg = t_pred[cstart:cend]
                            teacher_batch[b, :len(seg)] = seg.to(log_amp_pred.device)
                    kd_loss = nn.functional.mse_loss(log_amp_pred, teacher_batch.detach())
                    amp_loss = amp_loss + kd_weight * kd_loss
                loss = (0.0 if freeze_diffusion else diff_loss) + amp_loss
            else:
                # Legacy 2-channel mode
                amp_log = torch.log(amp + eps)
                amp_norm = (amp_log - amp_mean) / amp_std
                x_0 = torch.stack([f0_norm, amp_norm], dim=1)
                diff_loss = diffusion.training_loss(x_0, condition_perm)
                amp_loss = torch.tensor(0.0)
                loss = diff_loss

            if diff_optimizer is not None:
                diff_optimizer.zero_grad()
            if amp_optimizer is not None:
                amp_optimizer.zero_grad()
            loss.backward()
            if not freeze_diffusion:
                nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            if amp_predictor is not None:
                nn.utils.clip_grad_norm_(amp_predictor.parameters(), 1.0)
            if adapter is not None:
                nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            if unfreeze_encoder:
                nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            if diff_optimizer is not None:
                diff_optimizer.step()
            if amp_optimizer is not None:
                amp_optimizer.step()
            if ema is not None:
                ema.update(diffusion)

            epoch_loss += loss.item()
            epoch_diff_loss += diff_loss.item()
            epoch_amp_loss += amp_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_diff = epoch_diff_loss / max(n_batches, 1)
        avg_amp = epoch_amp_loss / max(n_batches, 1)

        # Test loss
        diffusion.eval()
        if amp_predictor is not None:
            amp_predictor.eval()
        if unfreeze_encoder:
            encoder.eval()
        if adapter is not None:
            adapter.eval()
        test_loss_sum, test_diff_sum, test_amp_sum, test_n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                ff = batch["frame_features"].to(device)
                f0 = batch["f0"].to(device)
                amp = batch["amp"].to(device)
                test_instrument_id = batch["instrument_id"].to(device) if amp_inst_conditioned else None
                test_note_position = ff[:, :, 3] if amp_note_pos_conditioned else None
                test_velocity = ff[:, :, 2] if (amp_vel_conditioned or amp_two_step) else None

                condition = encoder(ff)  # (B, T, 256)
                condition_perm = condition.permute(0, 2, 1)
                # Apply adapter for amp path (exp051+)
                condition_amp = adapter(condition) if adapter is not None else condition

                note_pitch = ff[:, :, 1] * 127.0
                voiced = ff[:, :, 0] > 0.5
                safe_f0 = torch.clamp(f0, min=1.0)
                f0_midi = 69.0 + 12.0 * torch.log2(safe_f0 / 440.0)
                cent_offset = (f0_midi - note_pitch) * 100.0
                f0_norm = cent_offset / 200.0
                f0_norm[~voiced] = 0.0
                f0_norm[f0 <= 0] = 0.0

                if n_channels == 1:
                    x_0 = f0_norm.unsqueeze(1)
                    diff_loss = diffusion.training_loss(x_0, condition_perm)
                    test_note_level_pred = None
                    if amp_two_step:
                        pitch_norm = ff[:, :, 1]
                        log_amp_pred, test_note_level_pred = amp_predictor(condition_amp, velocity=test_velocity, pitch=pitch_norm)
                    elif amp_f0_conditioned:
                        log_amp_pred = amp_predictor(condition_amp, f0=f0_norm, instrument_id=test_instrument_id, note_position=test_note_position, velocity=test_velocity)
                    else:
                        log_amp_pred = amp_predictor(condition_amp, instrument_id=test_instrument_id, note_position=test_note_position, velocity=test_velocity)
                    log_amp_gt = torch.log(amp + eps)
                    # Pitch-anchor amp offset for test (matching train)
                    if amp_pitch_anchor_enabled and pitch_amp_anchor is not None:
                        pitch_idx = note_pitch.long().clamp(0, 127)
                        anchor_per_frame = pitch_amp_anchor[pitch_idx]
                        log_amp_gt = log_amp_gt - anchor_per_frame
                        log_amp_gt[~voiced] = 0.0
                    # Per-instrument amp normalization for test (matching train)
                    elif amp_per_inst_norm and amp_per_inst_stats is not None and test_instrument_id is not None:
                        for b in range(log_amp_gt.shape[0]):
                            iid = test_instrument_id[b].item()
                            if iid in amp_per_inst_stats:
                                log_amp_gt[b] = (log_amp_gt[b] - amp_per_inst_stats[iid]["mean"]) / amp_per_inst_stats[iid]["std"]
                    mse = nn.functional.mse_loss(log_amp_pred, log_amp_gt)
                    # Same correlation-aligned loss for test (matching train)
                    amp_corr_weight = cfg.get("amp_corr_weight", 0.0)
                    amp_grad_weight = cfg.get("amp_grad_weight", 0.0)
                    if amp_corr_weight > 0:
                        pred_c = log_amp_pred - log_amp_pred.mean(dim=-1, keepdim=True)
                        gt_c = log_amp_gt - log_amp_gt.mean(dim=-1, keepdim=True)
                        corr = (pred_c * gt_c).sum(dim=-1) / (
                            pred_c.norm(dim=-1) * gt_c.norm(dim=-1) + 1e-8
                        )
                        corr_loss_val = (1.0 - corr).mean()
                    else:
                        corr_loss_val = 0.0
                    if amp_grad_weight > 0:
                        d_pred = log_amp_pred[:, 1:] - log_amp_pred[:, :-1]
                        d_gt = log_amp_gt[:, 1:] - log_amp_gt[:, :-1]
                        grad_loss_val = nn.functional.mse_loss(d_pred, d_gt)
                    else:
                        grad_loss_val = 0.0
                    amp_loss = mse + amp_corr_weight * corr_loss_val + amp_grad_weight * grad_loss_val
                    # Multi-scale temporal amp loss (exp031+)
                    amp_multiscale = cfg.get("amp_multiscale", False)
                    if amp_multiscale:
                        ms_scales = cfg.get("amp_multiscale_scales", [8, 32])
                        ms_weight = cfg.get("amp_multiscale_weight", 0.5)
                        ms_loss = 0.0
                        pred_1d = log_amp_pred.unsqueeze(1)  # (B, 1, T)
                        gt_1d = log_amp_gt.unsqueeze(1)      # (B, 1, T)
                        for scale in ms_scales:
                            if pred_1d.shape[-1] >= scale:
                                pred_ds = nn.functional.avg_pool1d(pred_1d, scale, stride=scale).squeeze(1)
                                gt_ds = nn.functional.avg_pool1d(gt_1d, scale, stride=scale).squeeze(1)
                                ms_loss += nn.functional.mse_loss(pred_ds, gt_ds)
                        amp_loss = amp_loss + ms_weight * ms_loss
                    # Two-step auxiliary loss for test (exp047+)
                    if amp_two_step and test_note_level_pred is not None:
                        pitch_norm = ff[:, :, 1]
                        note_mean_gt = compute_note_mean(log_amp_gt, pitch_norm, voiced)
                        note_aux_loss = nn.functional.mse_loss(test_note_level_pred, note_mean_gt)
                        amp_loss = amp_loss + amp_note_aux_weight * note_aux_loss
                        # Residual L1 penalty for test (exp048+)
                        if amp_residual_penalty_weight > 0:
                            residual = log_amp_pred - test_note_level_pred
                            residual_penalty = torch.abs(residual).mean()
                            amp_loss = amp_loss + amp_residual_penalty_weight * residual_penalty
                    # KD loss for test (exp054+) — no KD on test since teacher
                    # was trained on train set only, test KD would be unfair
                    loss = diff_loss + amp_loss
                else:
                    amp_log = torch.log(amp + eps)
                    amp_norm = (amp_log - amp_mean) / amp_std
                    x_0 = torch.stack([f0_norm, amp_norm], dim=1)
                    diff_loss = diffusion.training_loss(x_0, condition_perm)
                    amp_loss = torch.tensor(0.0)
                    loss = diff_loss

                test_loss_sum += loss.item()
                test_diff_sum += diff_loss.item()
                test_amp_sum += amp_loss.item()
                test_n += 1

        test_loss = test_loss_sum / max(test_n, 1)
        test_diff = test_diff_sum / max(test_n, 1)
        test_amp = test_amp_sum / max(test_n, 1)

        history["train_loss"].append(avg_loss)
        history["test_loss"].append(test_loss)
        history["train_diff_loss"].append(avg_diff)
        history["train_amp_loss"].append(avg_amp)
        history["test_diff_loss"].append(test_diff)
        history["test_amp_loss"].append(test_amp)

        diff_lr = diff_optimizer.param_groups[0]["lr"] if diff_optimizer else 0
        amp_lr_now = amp_optimizer.param_groups[0]["lr"] if amp_optimizer else 0
        # Compute scheduled sampling p_gt for logging
        ss_p_gt_str = ""
        if amp_scheduled_sampling:
            if epoch < ss_warmup:
                _p_gt = 1.0
            elif epoch < ss_warmup + ss_decay:
                _p_gt = 1.0 - (epoch - ss_warmup) / ss_decay
            else:
                _p_gt = 0.0
            ss_p_gt_str = f"  ss_p_gt={_p_gt:.2f}"
        print(f"Epoch {epoch}/{cfg['epochs']}  "
              f"train={avg_loss:.6f} (diff={avg_diff:.6f} amp={avg_amp:.6f})  "
              f"test={test_loss:.6f} (diff={test_diff:.6f} amp={test_amp:.6f})  "
              f"diff_lr={diff_lr:.2e} amp_lr={amp_lr_now:.2e}{ss_p_gt_str}")

        # Two-step diagnostics: log note_level stats every 10 epochs
        if amp_two_step and epoch % 10 == 0 and amp_predictor is not None:
            amp_predictor.eval()
            with torch.no_grad():
                # Use one test batch to get diagnostics
                for diag_batch in test_loader:
                    diag_ff = diag_batch["frame_features"].to(device)
                    diag_amp = diag_batch["amp"].to(device)
                    diag_cond = encoder(diag_ff)
                    diag_cond_amp = adapter(diag_cond) if adapter is not None else diag_cond
                    diag_vel = diag_ff[:, :, 2]
                    diag_pitch = diag_ff[:, :, 1]
                    diag_total, diag_note_level = amp_predictor(diag_cond_amp, velocity=diag_vel, pitch=diag_pitch)
                    diag_residual = diag_total - diag_note_level
                    diag_gt = torch.log(diag_amp + eps)
                    diag_voiced = diag_ff[:, :, 0] > 0.5
                    diag_note_mean = compute_note_mean(diag_gt, diag_pitch, diag_voiced)
                    diag_note_aux = nn.functional.mse_loss(diag_note_level, diag_note_mean)
                    diag_residual_l1 = torch.abs(diag_residual).mean()
                    print(f"  [TwoStep] note_level: mean={diag_note_level.mean():.4f}, std={diag_note_level.std():.4f} | "
                          f"residual: std={diag_residual.std():.4f}, L1={diag_residual_l1:.4f} | "
                          f"note_aux_loss={diag_note_aux:.4f} | gt_amp: std={diag_gt.std():.4f}")
                    break

        # Step LR schedulers
        if diff_scheduler is not None:
            diff_scheduler.step()
        if amp_scheduler is not None:
            amp_scheduler.step()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            if not freeze_diffusion:
                # Save raw (non-EMA) diffusion
                torch.save(diffusion.state_dict(),
                           os.path.join(config["output_dir"], "diffusion_best_raw.pt"))
                # Save EMA diffusion
                ema.apply(diffusion)
                torch.save(diffusion.state_dict(),
                           os.path.join(config["output_dir"], "diffusion_best_ema.pt"))
                ema.restore(diffusion)
            print(f"  -> New best test_loss={test_loss:.6f}")

        # Independent amp early stopping: save amp_predictor by best test_amp_loss
        if amp_predictor is not None and test_amp < best_test_amp_loss:
            best_test_amp_loss = test_amp
            torch.save(amp_predictor.state_dict(),
                       os.path.join(config["output_dir"], "amp_predictor_best.pt"))
            # Save adapter alongside amp_predictor (exp051+)
            if adapter is not None:
                torch.save(adapter.state_dict(),
                           os.path.join(config["output_dir"], "encoder_adapter_best.pt"))
            # Save fine-tuned encoder (exp055+)
            if unfreeze_encoder:
                torch.save(encoder.state_dict(),
                           os.path.join(config["output_dir"], "encoder_finetuned.pt"))
            print(f"  -> New best test_amp_loss={test_amp:.6f}, saved amp_predictor"
                  f"{' + adapter' if adapter is not None else ''}"
                  f"{' + encoder_finetuned' if unfreeze_encoder else ''}")

        if epoch % config["save_every"] == 0:
            if not freeze_diffusion:
                torch.save(diffusion.state_dict(),
                           os.path.join(config["output_dir"], f"diffusion_raw_ep{epoch}.pt"))
                ema.apply(diffusion)
                torch.save(diffusion.state_dict(),
                           os.path.join(config["output_dir"], f"diffusion_ema_ep{epoch}.pt"))
                ema.restore(diffusion)
            if amp_predictor is not None:
                torch.save(amp_predictor.state_dict(),
                           os.path.join(config["output_dir"], f"amp_predictor_ep{epoch}.pt"))
            if adapter is not None:
                torch.save(adapter.state_dict(),
                           os.path.join(config["output_dir"], f"encoder_adapter_ep{epoch}.pt"))

    with open(os.path.join(config["output_dir"], "stage2_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Load best EMA checkpoint (only if diffusion was trained)
    if not freeze_diffusion:
        diffusion.load_state_dict(
            torch.load(os.path.join(config["output_dir"], "diffusion_best_ema.pt"),
                        map_location=device, weights_only=True)
        )
    return diffusion


def train_stage3(config, encoder, device):
    """Stage 3: Train amp-only diffusion (1ch DDPM) with frozen encoder.
    Supports residual mode: train diffusion on (gt_amp - AmpPredictor_mean)."""
    cfg = config["stage3"]
    eps = 1e-7
    residual_mode = cfg.get("residual", False)
    print("\n" + "=" * 60)
    if residual_mode:
        print("Stage 3: Training RESIDUAL Amp Diffusion")
    else:
        print("Stage 3: Training 1ch Amp Diffusion")
    print("=" * 60)

    samples_per_epoch = cfg.get("samples_per_epoch", None)
    train_ds = ExpressionDataset(
        data_dir=config["data_dir"],
        bach10_dir=config.get("bach10_dir"),
        phenicx_dir=config.get("phenicx_dir"),
        trios_dir=config.get("trios_dir"),
        instruments=config["instruments"],
        split="train",
        crop_len=config["crop_len"],
        seed=config["seed"],
        samples_per_epoch=samples_per_epoch,
    )
    test_ds = ExpressionDataset(
        data_dir=config["data_dir"],
        bach10_dir=config.get("bach10_dir"),
        phenicx_dir=config.get("phenicx_dir"),
        trios_dir=config.get("trios_dir"),
        instruments=config["instruments"],
        split="test",
        crop_len=config["crop_len"],
        seed=config["seed"],
    )

    print(f"Train: {len(train_ds.tracks)} tracks, samples_per_epoch={len(train_ds)}")

    # Compute amp normalization stats (log z-score)
    amp_mean, amp_std = compute_amp_stats(train_ds, device)
    print(f"Amp log stats: mean={amp_mean:.4f}, std={amp_std:.4f}")
    torch.save({"amp_mean": amp_mean, "amp_std": amp_std},
               os.path.join(config["output_dir"], "amp_norm_stats.pt"))

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Freeze encoder
    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Load frozen AmpPredictor for residual mode
    frozen_amp_predictor = None
    if residual_mode:
        amp_pred_ckpt = cfg["amp_predictor_checkpoint"]
        amp_inst_cond = cfg.get("amp_instrument_conditioned", False)
        frozen_amp_predictor = AmpPredictor(
            cond_dim=256,
            hidden=cfg.get("amp_hidden", 256),
            gru_hidden=cfg.get("amp_gru_hidden", 128),
            n_gru_layers=cfg.get("amp_gru_layers", 2),
            dropout=cfg.get("amp_dropout", 0.3),
            use_attention=cfg.get("amp_use_attention", False),
            n_attn_heads=cfg.get("amp_n_attn_heads", 4),
            n_attn_layers=cfg.get("amp_n_attn_layers", 2),
            instrument_conditioned=amp_inst_cond,
        ).to(device)
        frozen_amp_predictor.load_state_dict(
            torch.load(amp_pred_ckpt, map_location=device, weights_only=True)
        )
        frozen_amp_predictor.eval()
        for param in frozen_amp_predictor.parameters():
            param.requires_grad = False
        n_ap_params = sum(p.numel() for p in frozen_amp_predictor.parameters())
        print(f"Loaded frozen AmpPredictor from {amp_pred_ckpt} ({n_ap_params:,} params)")
        print(f"  instrument_conditioned={amp_inst_cond}")

    # Amp diffusion: 1ch DDPM for normalized amp (or residual)
    amp_diffusion = ConditionalDDPM(
        n_steps=cfg["n_diffusion_steps"], n_channels=1,
    ).to(device)
    n_params = sum(p.numel() for p in amp_diffusion.parameters())
    print(f"Amp diffusion: {n_params:,} params")

    optimizer = torch.optim.Adam(
        amp_diffusion.parameters(), lr=float(cfg["lr"]), betas=(0.9, 0.999),
    )

    lr_min = cfg.get("lr_min", None)
    if lr_min is not None:
        lr_min = float(lr_min)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=lr_min,
        )
        print(f"CosineAnnealingLR: {cfg['lr']} -> {lr_min} over {cfg['epochs']} epochs")
    else:
        scheduler = None

    ema = EMA(amp_diffusion, decay=cfg["ema_decay"])

    os.makedirs(config["output_dir"], exist_ok=True)
    history = {"train_loss": [], "test_loss": []}
    best_test_loss = float("inf")

    # Amp augmentation config (read outside epoch loop)
    amp_augment = cfg.get("amp_augment", False)
    amp_augment_scale = cfg.get("amp_augment_scale", 0.2)
    amp_augment_jitter = cfg.get("amp_augment_jitter", 0.05)
    if amp_augment:
        print(f"Amp diffusion augmentation ENABLED: scale={amp_augment_scale}, jitter={amp_augment_jitter}")
    else:
        print("Amp diffusion augmentation DISABLED")

    # Residual normalization factor (default 1.0 = no rescaling)
    residual_std_value = 1.0

    # Diagnostic: compute residual statistics on first few batches
    if residual_mode and frozen_amp_predictor is not None:
        print("\n--- Residual Diagnostics ---")
        res_all = []
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= 10:  # first 10 batches
                    break
                ff = batch["frame_features"].to(device)
                amp = batch["amp"].to(device)
                instrument_id = batch["instrument_id"].to(device)
                condition = encoder(ff)
                amp_log = torch.log(amp + eps)
                amp_norm = (amp_log - amp_mean) / amp_std
                mu_raw_log = frozen_amp_predictor(condition, instrument_id=instrument_id)
                mu_norm = (mu_raw_log - amp_mean) / amp_std
                residual = amp_norm - mu_norm
                res_all.append(residual.cpu())
        res_cat = torch.cat(res_all, dim=0).flatten()
        print(f"Residual stats (z-score space): mean={res_cat.mean():.4f}, std={res_cat.std():.4f}, "
              f"min={res_cat.min():.4f}, max={res_cat.max():.4f}")
        if res_cat.std() > 0.5:
            print("WARNING: Residual std > 0.5 — AmpPredictor and GT normalization may be mismatched!")
        # [C1] Save residual_std for normalization (exp050+)
        residual_std_value = res_cat.std().item()
        print(f"Saving residual_std={residual_std_value:.4f} to amp_norm_stats.pt")
        torch.save({"amp_mean": amp_mean, "amp_std": amp_std, "residual_std": residual_std_value},
                   os.path.join(config["output_dir"], "amp_norm_stats.pt"))
        print("--- End Residual Diagnostics ---\n")

    for epoch in range(1, cfg["epochs"] + 1):
        amp_diffusion.train()
        epoch_loss, n_batches = 0.0, 0

        for batch in train_loader:
            ff = batch["frame_features"].to(device)
            amp = batch["amp"].to(device)

            with torch.no_grad():
                condition = encoder(ff)  # (B, T, 256)
                condition_perm = condition.permute(0, 2, 1)  # (B, 256, T)

            # Normalize amp: log -> z-score
            amp_log = torch.log(amp + eps)
            amp_norm = (amp_log - amp_mean) / amp_std

            # Residual mode: compute residual = gt_norm - predictor_norm, then rescale
            if residual_mode and frozen_amp_predictor is not None:
                with torch.no_grad():
                    instrument_id = batch["instrument_id"].to(device)
                    mu_raw_log = frozen_amp_predictor(condition, instrument_id=instrument_id)  # (B, T)
                    mu_norm = (mu_raw_log - amp_mean) / amp_std
                    residual = (amp_norm - mu_norm) / residual_std_value  # [C1] normalized residual
                x_0 = residual.unsqueeze(1)  # (B, 1, T)
                # [C3] Residual augmentation: light jitter only
                if amp_augment:
                    x_0 = x_0 + torch.randn_like(x_0) * amp_augment_jitter
            else:
                # Amp augmentation (training only)
                if amp_augment:
                    # Global scale shift in z-score space
                    scale_offset = (torch.rand(amp_norm.size(0), 1, device=device) * 2 - 1) * amp_augment_scale / amp_std
                    amp_norm = amp_norm + scale_offset
                    # Per-frame jitter
                    jitter = torch.randn_like(amp_norm) * amp_augment_jitter / amp_std
                    amp_norm = amp_norm + jitter
                x_0 = amp_norm.unsqueeze(1)  # (B, 1, T)

            loss = amp_diffusion.training_loss(x_0, condition_perm)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(amp_diffusion.parameters(), 1.0)
            optimizer.step()
            ema.update(amp_diffusion)

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Test loss
        amp_diffusion.eval()
        test_loss_sum, test_n = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                ff = batch["frame_features"].to(device)
                amp = batch["amp"].to(device)

                condition = encoder(ff)
                condition_perm = condition.permute(0, 2, 1)

                amp_log = torch.log(amp + eps)
                amp_norm = (amp_log - amp_mean) / amp_std

                if residual_mode and frozen_amp_predictor is not None:
                    instrument_id = batch["instrument_id"].to(device)
                    mu_raw_log = frozen_amp_predictor(condition, instrument_id=instrument_id)
                    mu_norm = (mu_raw_log - amp_mean) / amp_std
                    residual = (amp_norm - mu_norm) / residual_std_value  # [C1] normalized
                    x_0 = residual.unsqueeze(1)
                else:
                    x_0 = amp_norm.unsqueeze(1)

                loss = amp_diffusion.training_loss(x_0, condition_perm)
                test_loss_sum += loss.item()
                test_n += 1

        test_loss = test_loss_sum / max(test_n, 1)
        history["train_loss"].append(avg_loss)
        history["test_loss"].append(test_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{cfg['epochs']}  "
              f"train={avg_loss:.6f}  test={test_loss:.6f}  lr={lr_now:.2e}")

        if scheduler is not None:
            scheduler.step()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(amp_diffusion.state_dict(),
                       os.path.join(config["output_dir"], "amp_diffusion_best_raw.pt"))
            ema.apply(amp_diffusion)
            torch.save(amp_diffusion.state_dict(),
                       os.path.join(config["output_dir"], "amp_diffusion_best_ema.pt"))
            ema.restore(amp_diffusion)
            print(f"  -> New best test_loss={test_loss:.6f}")

        if epoch % config["save_every"] == 0:
            torch.save(amp_diffusion.state_dict(),
                       os.path.join(config["output_dir"], f"amp_diffusion_raw_ep{epoch}.pt"))
            ema.apply(amp_diffusion)
            torch.save(amp_diffusion.state_dict(),
                       os.path.join(config["output_dir"], f"amp_diffusion_ema_ep{epoch}.pt"))
            ema.restore(amp_diffusion)

    with open(os.path.join(config["output_dir"], "stage3_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Load best EMA checkpoint
    amp_diffusion.load_state_dict(
        torch.load(os.path.join(config["output_dir"], "amp_diffusion_best_ema.pt"),
                    map_location=device, weights_only=True)
    )
    return amp_diffusion


def main():
    parser = argparse.ArgumentParser(description="Train MIDI-to-expression model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--stage", type=int, default=0,
                        help="0=both, 1=baseline only, 2=diffusion only, 3=amp diffusion")
    args = parser.parse_args()

    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
        if user_config:
            for key, val in user_config.items():
                if isinstance(val, dict) and key in config:
                    config[key].update(val)
                else:
                    config[key] = val

    device = get_device()
    print(f"Device: {device}")
    print(f"Config: {json.dumps(config, indent=2, default=str)}")

    if args.stage in (0, 1):
        baseline = train_stage1(config, device)
        encoder = baseline.get_encoder()
    else:
        baseline = BaselineModel().to(device)
        # Allow specifying baseline checkpoint location in config
        ckpt_path = config.get("baseline_checkpoint",
                               os.path.join(config["output_dir"], "baseline_best.pt"))
        if os.path.isfile(ckpt_path):
            baseline.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True)
            )
            print(f"Loaded baseline from {ckpt_path}")
        else:
            print(f"WARNING: No baseline checkpoint at {ckpt_path}, using random init")
        encoder = baseline.get_encoder()

    if args.stage in (0, 2):
        train_stage2(config, encoder, device)

    if args.stage == 3:
        train_stage3(config, encoder, device)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
