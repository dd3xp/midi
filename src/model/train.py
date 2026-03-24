"""
Training entry point for MIDI-to-expression model.
Stage 1: Encoder + Baseline (joint), Stage 2: Diffusion (encoder frozen).
"""

import os
import sys
import json
import copy
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.model.dataset import ExpressionDataset, collate_fn
from src.model.baseline import BaselineModel, compute_baseline_loss
from src.model.diffusion import (
    ConditionalDDPM, normalize_f0_cents, normalize_amp_log_zscore,
)


DEFAULT_CONFIG = {
    "data_dir": os.path.join(PROJECT_ROOT, "datagen"),
    "output_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
    "instruments": ["vn", "tpt", "fl"],
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
        "ema_decay": 0.9999,
        "n_diffusion_steps": 1000,
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
        instruments=config["instruments"],
        split="train",
        crop_len=config["crop_len"],
        seed=config["seed"],
    )
    test_ds = ExpressionDataset(
        data_dir=config["data_dir"],
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
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
    for i in range(len(dataset)):
        item = dataset._load(i)
        amp = torch.from_numpy(item["amp"])
        amp_log = torch.log(amp + eps)
        all_amp_log.append(amp_log)
    all_amp_log = torch.cat(all_amp_log)
    return all_amp_log.mean().to(device), all_amp_log.std().to(device)


def train_stage2(config, encoder, device):
    """Stage 2: Train conditional diffusion model with frozen encoder."""
    cfg = config["stage2"]
    eps = 1e-7
    print("\n" + "=" * 60)
    print("Stage 2: Training Conditional Diffusion Model")
    print("=" * 60)

    train_ds = ExpressionDataset(
        data_dir=config["data_dir"],
        instruments=config["instruments"],
        split="train",
        crop_len=config["crop_len"],
        seed=config["seed"],
    )
    test_ds = ExpressionDataset(
        data_dir=config["data_dir"],
        instruments=config["instruments"],
        split="test",
        crop_len=config["crop_len"],
        seed=config["seed"],
    )

    # Compute amp normalization stats from training set
    amp_mean, amp_std = compute_amp_stats(train_ds, device)
    print(f"Amp log stats: mean={amp_mean:.4f}, std={amp_std:.4f}")
    torch.save({"amp_mean": amp_mean, "amp_std": amp_std},
               os.path.join(config["output_dir"], "norm_stats.pt"))

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

    diffusion = ConditionalDDPM(n_steps=cfg["n_diffusion_steps"]).to(device)
    optimizer = torch.optim.Adam(
        diffusion.parameters(), lr=cfg["lr"], betas=(0.9, 0.999),
    )
    ema = EMA(diffusion, decay=cfg["ema_decay"])

    os.makedirs(config["output_dir"], exist_ok=True)
    history = {"train_loss": [], "test_loss": []}
    best_train_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        diffusion.train()
        epoch_loss, n_batches = 0.0, 0

        for batch in train_loader:
            ff = batch["frame_features"].to(device)
            f0 = batch["f0"].to(device)
            amp = batch["amp"].to(device)

            with torch.no_grad():
                condition = encoder(ff)  # (B, T, 256)
                condition = condition.permute(0, 2, 1)  # (B, 256, T)

            # Normalize f0: use is_voiced from frame_features and pitch info
            # Simple batch normalization: cent offset / 200
            note_pitch = ff[:, :, 1] * 127.0  # denormalize pitch
            voiced = ff[:, :, 0] > 0.5
            safe_f0 = torch.clamp(f0, min=1.0)
            f0_midi = 69.0 + 12.0 * torch.log2(safe_f0 / 440.0)
            cent_offset = (f0_midi - note_pitch) * 100.0
            f0_norm = cent_offset / 200.0
            f0_norm[~voiced] = 0.0
            f0_norm[f0 <= 0] = 0.0

            # Normalize amp
            amp_log = torch.log(amp + eps)
            amp_norm = (amp_log - amp_mean) / amp_std

            # Stack as (B, 2, T)
            x_0 = torch.stack([f0_norm, amp_norm], dim=1)

            loss = diffusion.training_loss(x_0, condition)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            ema.update(diffusion)

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Test loss
        diffusion.eval()
        test_loss_sum, test_n = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                ff = batch["frame_features"].to(device)
                f0 = batch["f0"].to(device)
                amp = batch["amp"].to(device)

                condition = encoder(ff).permute(0, 2, 1)

                note_pitch = ff[:, :, 1] * 127.0
                voiced = ff[:, :, 0] > 0.5
                safe_f0 = torch.clamp(f0, min=1.0)
                f0_midi = 69.0 + 12.0 * torch.log2(safe_f0 / 440.0)
                cent_offset = (f0_midi - note_pitch) * 100.0
                f0_norm = cent_offset / 200.0
                f0_norm[~voiced] = 0.0
                f0_norm[f0 <= 0] = 0.0

                amp_log = torch.log(amp + eps)
                amp_norm = (amp_log - amp_mean) / amp_std

                x_0 = torch.stack([f0_norm, amp_norm], dim=1)
                loss = diffusion.training_loss(x_0, condition)
                test_loss_sum += loss.item()
                test_n += 1

        test_loss = test_loss_sum / max(test_n, 1)
        history["train_loss"].append(avg_loss)
        history["test_loss"].append(test_loss)

        print(f"Epoch {epoch}/{cfg['epochs']}  "
              f"train={avg_loss:.6f}  test={test_loss:.6f}")

        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            ema.apply(diffusion)
            torch.save(diffusion.state_dict(),
                       os.path.join(config["output_dir"], "diffusion_best_ema.pt"))
            ema.restore(diffusion)

        if epoch % config["save_every"] == 0:
            ema.apply(diffusion)
            torch.save(diffusion.state_dict(),
                       os.path.join(config["output_dir"], f"diffusion_ema_ep{epoch}.pt"))
            ema.restore(diffusion)

    with open(os.path.join(config["output_dir"], "stage2_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Load best EMA checkpoint
    diffusion.load_state_dict(
        torch.load(os.path.join(config["output_dir"], "diffusion_best_ema.pt"),
                    map_location=device, weights_only=True)
    )
    return diffusion


def main():
    parser = argparse.ArgumentParser(description="Train MIDI-to-expression model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--stage", type=int, default=0,
                        help="0=both, 1=baseline only, 2=diffusion only")
    args = parser.parse_args()

    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
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
        ckpt_path = os.path.join(config["output_dir"], "baseline_best.pt")
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

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
