"""
Pre-cache diffusion f0 samples for scheduled sampling training (exp032+).

For each training track, generates N diffusion f0 samples and saves them.
These are used during amp predictor training to gradually replace GT f0
with realistic diffusion f0, implementing scheduled sampling to reduce
exposure bias.

Usage:
    python src/model/cache_diffusion_f0.py \
        --checkpoint_dir experiments/checkpoints/exp017 \
        --baseline_checkpoint experiments/checkpoints/exp014/baseline_best.pt \
        --output_dir experiments/checkpoints/exp032/cached_f0 \
        --ddim_steps 50 --eta 0.3 --n_samples 3
"""

import os
import sys
import hashlib
import argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.model.dataset import ExpressionDataset, notes_to_frame_features
from src.model.baseline import BaselineModel
from src.model.encoder import MIDIEncoder
from src.model.diffusion import ConditionalDDPM


def track_hash(track_path):
    """Generate a short hash for a track path to use as filename."""
    return hashlib.md5(track_path.encode()).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description="Cache diffusion f0 for scheduled sampling")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory with diffusion checkpoint (e.g., exp017)")
    parser.add_argument("--baseline_checkpoint", type=str, required=True,
                        help="Path to baseline_best.pt (for encoder)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for cached f0 files")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "datagen", "solo", "URMP"))
    parser.add_argument("--bach10_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "datagen", "solo", "Bach10"))
    parser.add_argument("--instruments", nargs="+",
                        default=["vn", "va", "vc", "fl", "ob", "cl", "sax", "tpt", "tbn", "bn"])
    parser.add_argument("--crop_len", type=int, default=512)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of f0 samples to cache per track")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load encoder from baseline
    baseline = BaselineModel().to(device)
    baseline.load_state_dict(
        torch.load(args.baseline_checkpoint, map_location=device, weights_only=True)
    )
    encoder = baseline.get_encoder()
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    print(f"Loaded encoder from {args.baseline_checkpoint}")

    # Load diffusion model
    diffusion = ConditionalDDPM(n_steps=1000, n_channels=1).to(device)
    diff_ckpt = os.path.join(args.checkpoint_dir, "diffusion_best_ema.pt")
    diffusion.load_state_dict(
        torch.load(diff_ckpt, map_location=device, weights_only=True)
    )
    diffusion.eval()
    print(f"Loaded diffusion from {diff_ckpt}")

    # Load training dataset (full sequences, no cropping for caching)
    train_ds = ExpressionDataset(
        data_dir=args.data_dir,
        bach10_dir=args.bach10_dir,
        instruments=args.instruments,
        split="train",
        crop_len=99999,  # no cropping for caching
        seed=args.seed,
    )
    print(f"Training tracks: {len(train_ds.tracks)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # For each training track, generate and cache N diffusion f0 samples
    torch.manual_seed(args.seed)
    n_total = len(train_ds.tracks)

    with torch.no_grad():
        for idx in range(n_total):
            if device.type == "cuda":
                torch.cuda.empty_cache()

            item = train_ds._load(idx)
            track_path = train_ds.tracks[idx]["path"]
            t_hash = track_hash(track_path)

            ff = torch.from_numpy(item["frame_features"]).unsqueeze(0).to(device)
            T = ff.shape[1]

            condition = encoder(ff)  # (1, T, 256)
            condition_perm = condition.permute(0, 2, 1)  # (1, 256, T)

            # Compute GT f0_norm for this track (for reference/validation)
            f0 = torch.from_numpy(item["f0"]).to(device)
            notes = item["notes"]
            hop_time = item["hop_time"]
            note_pitch_frame = torch.zeros(T, device=device)
            for onset, offset, midi_pitch, _ in notes:
                start = max(0, int(onset / hop_time))
                end = min(T, int(offset / hop_time))
                note_pitch_frame[start:end] = float(midi_pitch)

            voiced = (f0 > 0) & (note_pitch_frame > 0)
            safe_f0 = torch.clamp(f0, min=1.0)
            f0_midi = 69.0 + 12.0 * torch.log2(safe_f0 / 440.0)
            gt_f0_norm = ((f0_midi - note_pitch_frame) * 100.0) / 200.0
            gt_f0_norm[~voiced] = 0.0
            gt_f0_norm[f0 <= 0] = 0.0

            # Generate N diffusion f0 samples
            f0_samples = []
            for s in range(args.n_samples):
                x_gen = diffusion.ddim_sample(
                    condition_perm, n_steps=args.ddim_steps, eta=args.eta
                )  # (1, 1, T)
                f0_norm_sample = x_gen[0, 0].cpu()  # (T,)
                f0_samples.append(f0_norm_sample)

            f0_samples = torch.stack(f0_samples)  # (n_samples, T)

            # Save: {track_hash}.pt containing f0_samples (n_samples, T)
            # Also save metadata for verification
            save_path = os.path.join(args.output_dir, f"{t_hash}.pt")
            torch.save({
                "f0_samples": f0_samples,        # (n_samples, T) normalized f0
                "gt_f0_norm": gt_f0_norm.cpu(),   # (T,) GT for comparison
                "track_path": track_path,
                "n_frames": T,
            }, save_path)

            if (idx + 1) % 10 == 0 or idx == 0 or idx == n_total - 1:
                print(f"[{idx+1}/{n_total}] Cached {args.n_samples} f0 samples "
                      f"for {os.path.basename(track_path)} (T={T}, hash={t_hash})")

    # Save an index mapping track_path -> hash for easy lookup
    index = {}
    for idx in range(n_total):
        track_path = train_ds.tracks[idx]["path"]
        index[track_path] = track_hash(track_path)
    torch.save(index, os.path.join(args.output_dir, "index.pt"))
    print(f"\nDone! Cached f0 for {n_total} tracks, {args.n_samples} samples each.")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
