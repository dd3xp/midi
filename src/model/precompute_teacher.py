"""
Pre-compute teacher AmpPredictor predictions for knowledge distillation (exp054).

Loads exp039 AmpPredictor (instrument_conditioned=True, use_attention=True)
with exp033 encoder, runs inference on all training tracks, and saves
full-length teacher predictions per track.

Output: {track_idx: tensor(T,)} dict saved to teacher_preds.pt
"""

import os
import sys
import argparse
import torch
import numpy as np

# Limit GPU memory
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.model.dataset import ExpressionDataset, INSTRUMENT_TO_ID
from src.model.baseline import BaselineModel
from src.model.diffusion import AmpPredictor


def main():
    parser = argparse.ArgumentParser(description="Pre-compute teacher amp predictions")
    parser.add_argument("--baseline_checkpoint", type=str,
                        default="experiments/checkpoints/exp033/baseline_best.pt",
                        help="Path to baseline checkpoint (for encoder)")
    parser.add_argument("--teacher_checkpoint", type=str,
                        default="experiments/checkpoints/exp039/amp_predictor_best.pt",
                        help="Path to teacher AmpPredictor checkpoint")
    parser.add_argument("--output", type=str,
                        default="experiments/checkpoints/exp054/teacher_preds.pt",
                        help="Output path for teacher predictions")
    parser.add_argument("--instruments", type=str, nargs="+",
                        default=["vn", "va", "vc", "fl", "ob", "cl", "sax", "tpt", "tbn", "bn"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load encoder from baseline checkpoint
    baseline = BaselineModel().to(device)
    ckpt_path = os.path.join(PROJECT_ROOT, args.baseline_checkpoint)
    baseline.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    encoder = baseline.get_encoder()
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    print(f"Loaded encoder from {ckpt_path}")

    # Load teacher AmpPredictor (exp039 config: instrument_conditioned=True, use_attention=True)
    teacher = AmpPredictor(
        cond_dim=256,
        hidden=256,
        gru_hidden=128,
        n_gru_layers=2,
        dropout=0.3,
        f0_conditioned=False,
        use_attention=True,
        n_attn_heads=4,
        n_attn_layers=2,
        instrument_conditioned=True,
        n_instruments=10,
        inst_embed_dim=32,
        note_position_conditioned=False,
        velocity_conditioned=False,
    ).to(device)
    teacher_ckpt = os.path.join(PROJECT_ROOT, args.teacher_checkpoint)
    teacher.load_state_dict(
        torch.load(teacher_ckpt, map_location=device, weights_only=True)
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    n_params = sum(p.numel() for p in teacher.parameters())
    print(f"Loaded teacher AmpPredictor from {teacher_ckpt} ({n_params:,} params)")

    # Load training dataset (full tracks, no cropping)
    from src.model.dataset import DATA_DIR, BACH10_DIR, PHENICX_DIR, TRIOS_DIR
    train_ds = ExpressionDataset(
        data_dir=DATA_DIR,
        bach10_dir=BACH10_DIR,
        phenicx_dir=PHENICX_DIR,
        trios_dir=TRIOS_DIR,
        instruments=args.instruments,
        split="train",
        crop_len=999999,  # effectively no cropping
        seed=42,
    )
    print(f"Training tracks: {len(train_ds.tracks)}")

    # Pre-compute teacher predictions for each training track
    teacher_preds = {}
    with torch.no_grad():
        for idx in range(len(train_ds.tracks)):
            data = train_ds._load(idx)
            ff = torch.from_numpy(data["frame_features"]).unsqueeze(0).to(device)  # (1, T, 7)
            inst_id = torch.tensor([data["instrument_id"]], dtype=torch.long, device=device)  # (1,)

            # Run encoder
            condition = encoder(ff)  # (1, T, 256)

            # Run teacher (instrument-conditioned)
            log_amp_pred = teacher(condition, instrument_id=inst_id)  # (1, T)
            teacher_preds[idx] = log_amp_pred.squeeze(0).cpu()  # (T,)

            if (idx + 1) % 20 == 0 or idx == 0:
                track_info = train_ds.tracks[idx]
                print(f"  [{idx+1}/{len(train_ds.tracks)}] {os.path.basename(track_info['path'])} "
                      f"inst={track_info['instrument']} T={ff.shape[1]} "
                      f"amp_range=[{teacher_preds[idx].min():.3f}, {teacher_preds[idx].max():.3f}]")

    # Save
    output_path = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(teacher_preds, output_path)

    # Summary
    total_frames = sum(t.shape[0] for t in teacher_preds.values())
    mem_mb = sum(t.numel() * 4 for t in teacher_preds.values()) / 1024 / 1024
    print(f"\nSaved teacher predictions for {len(teacher_preds)} tracks to {output_path}")
    print(f"Total frames: {total_frames:,}, Size: {mem_mb:.1f} MB")


if __name__ == "__main__":
    main()
