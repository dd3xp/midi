"""
Evaluation for MIDI-to-expression models.
Metrics: f0 RPA, f0 MAE, Amp Correlation, VDE, VRE.
Multi-sample evaluation for diffusion (oracle + mean).
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.model.dataset import ExpressionDataset
from src.model.baseline import BaselineModel, logits_to_f0
from src.model.encoder import MIDIEncoder
from src.model.diffusion import (
    ConditionalDDPM, denormalize_f0, denormalize_amp,
)


# ============ Metrics ============

def hz_to_cents(f0_hz, ref_hz):
    """Convert f0 difference to cents. Both must be > 0."""
    return 1200.0 * np.log2(f0_hz / ref_hz)


def rpa(pred_hz, gt_hz, threshold=50.0):
    """Raw Pitch Accuracy: fraction of voiced frames within threshold cents."""
    voiced = (gt_hz > 0) & (pred_hz > 0)
    if voiced.sum() == 0:
        return 0.0
    cent_err = np.abs(hz_to_cents(pred_hz[voiced], gt_hz[voiced]))
    return float((cent_err < threshold).mean())


def f0_mae_cents(pred_hz, gt_hz):
    """Mean absolute error in cents for voiced frames."""
    voiced = (gt_hz > 0) & (pred_hz > 0)
    if voiced.sum() == 0:
        return 0.0
    cent_err = np.abs(hz_to_cents(pred_hz[voiced], gt_hz[voiced]))
    return float(cent_err.mean())


def amp_correlation(pred_amp, gt_amp):
    """Pearson correlation between predicted and ground truth amplitude."""
    if np.std(pred_amp) < 1e-10 or np.std(gt_amp) < 1e-10:
        return 0.0
    return float(np.corrcoef(pred_amp, gt_amp)[0, 1])


def amp_rmse_log(pred_amp, gt_amp, eps=1e-7):
    """RMSE in log space."""
    pred_log = np.log(pred_amp + eps)
    gt_log = np.log(gt_amp + eps)
    return float(np.sqrt(np.mean((pred_log - gt_log) ** 2)))


def extract_vibrato(f0_hz, hop_time, min_duration=0.3, freq_range=(4.0, 8.0)):
    """Extract vibrato depth (cents) and rate (Hz) from f0 trajectory of a single note.

    Returns (depth, rate) or (None, None) if no vibrato detected.
    """
    if len(f0_hz) * hop_time < min_duration:
        return None, None
    voiced = f0_hz > 0
    if voiced.sum() < int(min_duration / hop_time):
        return None, None

    f0_voiced = f0_hz[voiced]
    f0_cents = 1200.0 * np.log2(f0_voiced / np.median(f0_voiced))

    # Remove linear trend
    x = np.arange(len(f0_cents))
    coeffs = np.polyfit(x, f0_cents, 1)
    f0_detrended = f0_cents - np.polyval(coeffs, x)

    # FFT to find vibrato frequency
    n = len(f0_detrended)
    if n < 8:
        return None, None
    spectrum = np.abs(np.fft.rfft(f0_detrended))
    freqs = np.fft.rfftfreq(n, d=hop_time)

    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not mask.any():
        return None, None

    masked_spectrum = spectrum.copy()
    masked_spectrum[~mask] = 0

    peak_idx = masked_spectrum.argmax()
    if masked_spectrum[peak_idx] < 1.0:  # threshold for vibrato presence
        return None, None

    rate = float(freqs[peak_idx])
    depth = float(spectrum[peak_idx] * 2.0 / n)  # approximate peak-to-peak / 2
    return depth, rate


def vibrato_errors(pred_hz, gt_hz, notes, hop_time):
    """Compute vibrato depth error (VDE) and vibrato rate error (VRE) per note."""
    vde_list, vre_list = [], []

    for onset, offset, _, _ in notes:
        start = max(0, int(onset / hop_time))
        end = int(offset / hop_time)
        if end <= start:
            continue

        gt_seg = gt_hz[start:min(end, len(gt_hz))]
        pred_seg = pred_hz[start:min(end, len(pred_hz))]
        if len(gt_seg) == 0 or len(pred_seg) == 0:
            continue

        gt_depth, gt_rate = extract_vibrato(gt_seg, hop_time)
        pred_depth, pred_rate = extract_vibrato(pred_seg, hop_time)

        if gt_depth is not None and pred_depth is not None:
            vde_list.append(abs(pred_depth - gt_depth))
            vre_list.append(abs(pred_rate - gt_rate))

    return {
        "vde_mean": float(np.mean(vde_list)) if vde_list else None,
        "vre_mean": float(np.mean(vre_list)) if vre_list else None,
        "n_vibrato_notes": len(vde_list),
    }


# ============ Evaluation Functions ============

def evaluate_baseline(model, dataset, device):
    """Evaluate baseline model on full test sequences."""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset._load(idx)
            ff = torch.from_numpy(item["frame_features"]).unsqueeze(0).to(device)
            f0_gt = item["f0"]
            amp_gt = item["amp"]
            notes = item["notes"]
            hop_time = item["hop_time"]

            f0_logits, amp_pred = model(ff)
            f0_logits = f0_logits.squeeze(0).cpu()
            amp_pred = amp_pred.squeeze(0).cpu().numpy()

            f0_pred_hz = logits_to_f0(f0_logits, notes, hop_time).numpy()

            metrics = {
                "track": dataset.tracks[idx]["path"],
                "rpa": rpa(f0_pred_hz, f0_gt),
                "f0_mae": f0_mae_cents(f0_pred_hz, f0_gt),
                "amp_corr": amp_correlation(amp_pred, amp_gt),
                "amp_rmse_log": amp_rmse_log(amp_pred, amp_gt),
            }
            vib = vibrato_errors(f0_pred_hz, f0_gt, notes, hop_time)
            metrics.update(vib)
            all_metrics.append(metrics)

    return all_metrics


def evaluate_diffusion(encoder, diffusion, dataset, device, n_samples=10,
                       amp_mean=0.0, amp_std=1.0, ddim_steps=50):
    """Evaluate diffusion model with multi-sample evaluation."""
    encoder.eval()
    diffusion.eval()

    all_metrics = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset._load(idx)
            ff = torch.from_numpy(item["frame_features"]).unsqueeze(0).to(device)
            f0_gt = item["f0"]
            amp_gt = item["amp"]
            notes = item["notes"]
            hop_time = item["hop_time"]

            condition = encoder(ff).permute(0, 2, 1)  # (1, 256, T)

            sample_metrics = []
            for s in range(n_samples):
                x_gen = diffusion.ddim_sample(condition, n_steps=ddim_steps)
                f0_norm = x_gen[0, 0].cpu()
                amp_norm = x_gen[0, 1].cpu()

                f0_pred_hz = denormalize_f0(
                    f0_norm, notes, hop_time,
                ).numpy()
                amp_pred = denormalize_amp(
                    amp_norm,
                    torch.tensor(amp_mean),
                    torch.tensor(amp_std),
                ).numpy()
                amp_pred = np.clip(amp_pred, 0.0, None)

                m = {
                    "rpa": rpa(f0_pred_hz, f0_gt),
                    "f0_mae": f0_mae_cents(f0_pred_hz, f0_gt),
                    "amp_corr": amp_correlation(amp_pred, amp_gt),
                    "amp_rmse_log": amp_rmse_log(amp_pred, amp_gt),
                }
                vib = vibrato_errors(f0_pred_hz, f0_gt, notes, hop_time)
                m.update(vib)
                sample_metrics.append(m)

            # Oracle: best f0_mae among samples
            oracle_idx = min(range(n_samples), key=lambda i: sample_metrics[i]["f0_mae"])

            numeric_keys = ["rpa", "f0_mae", "amp_corr", "amp_rmse_log"]
            mean_metrics = {}
            for k in numeric_keys:
                vals = [sm[k] for sm in sample_metrics]
                mean_metrics[f"{k}_mean"] = float(np.mean(vals))
                mean_metrics[f"{k}_oracle"] = sample_metrics[oracle_idx][k]

            for k in ["vde_mean", "vre_mean"]:
                vals = [sm[k] for sm in sample_metrics if sm[k] is not None]
                if vals:
                    mean_metrics[f"{k}_avg"] = float(np.mean(vals))
                    oracle_val = sample_metrics[oracle_idx].get(k)
                    mean_metrics[f"{k}_oracle"] = oracle_val
                else:
                    mean_metrics[f"{k}_avg"] = None
                    mean_metrics[f"{k}_oracle"] = None

            mean_metrics["track"] = dataset.tracks[idx]["path"]
            mean_metrics["n_vibrato_notes"] = sample_metrics[0].get("n_vibrato_notes", 0)
            all_metrics.append(mean_metrics)

    return all_metrics


def summarize_metrics(metrics_list, prefix=""):
    """Compute aggregate statistics from per-track metrics."""
    numeric_keys = [k for k in metrics_list[0] if isinstance(metrics_list[0][k], (int, float))
                    and metrics_list[0][k] is not None]
    summary = {}
    for k in numeric_keys:
        vals = [m[k] for m in metrics_list if m.get(k) is not None]
        if vals:
            summary[f"{prefix}{k}"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate MIDI-to-expression models")
    parser.add_argument("--checkpoint_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "checkpoints"))
    parser.add_argument("--mode", choices=["baseline", "diffusion", "both"], default="both")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_ds = ExpressionDataset(split="test")
    print(f"Test tracks: {len(test_ds)}")

    results = {}

    if args.mode in ("baseline", "both"):
        print("\n--- Evaluating Baseline ---")
        model = BaselineModel().to(device)
        ckpt = os.path.join(args.checkpoint_dir, "baseline_best.pt")
        if os.path.isfile(ckpt):
            model.load_state_dict(
                torch.load(ckpt, map_location=device, weights_only=True)
            )
        else:
            print(f"WARNING: {ckpt} not found, using random weights")

        baseline_metrics = evaluate_baseline(model, test_ds, device)
        baseline_summary = summarize_metrics(baseline_metrics, prefix="baseline_")
        results["baseline_per_track"] = baseline_metrics
        results["baseline_summary"] = baseline_summary
        print(json.dumps(baseline_summary, indent=2))

    if args.mode in ("diffusion", "both"):
        print("\n--- Evaluating Diffusion ---")
        baseline_model = BaselineModel().to(device)
        ckpt = os.path.join(args.checkpoint_dir, "baseline_best.pt")
        if os.path.isfile(ckpt):
            baseline_model.load_state_dict(
                torch.load(ckpt, map_location=device, weights_only=True)
            )
        encoder = baseline_model.get_encoder()
        encoder.eval()

        diffusion = ConditionalDDPM().to(device)
        diff_ckpt = os.path.join(args.checkpoint_dir, "diffusion_best_ema.pt")
        if os.path.isfile(diff_ckpt):
            diffusion.load_state_dict(
                torch.load(diff_ckpt, map_location=device, weights_only=True)
            )
        else:
            print(f"WARNING: {diff_ckpt} not found, using random weights")

        # Load normalization stats
        norm_path = os.path.join(args.checkpoint_dir, "norm_stats.pt")
        if os.path.isfile(norm_path):
            stats = torch.load(norm_path, map_location=device, weights_only=True)
            amp_mean = float(stats["amp_mean"])
            amp_std = float(stats["amp_std"])
        else:
            print("WARNING: norm_stats.pt not found, using defaults")
            amp_mean, amp_std = 0.0, 1.0

        diff_metrics = evaluate_diffusion(
            encoder, diffusion, test_ds, device,
            n_samples=args.n_samples, amp_mean=amp_mean, amp_std=amp_std,
            ddim_steps=args.ddim_steps,
        )
        diff_summary = summarize_metrics(diff_metrics, prefix="diffusion_")
        results["diffusion_per_track"] = diff_metrics
        results["diffusion_summary"] = diff_summary
        print(json.dumps(diff_summary, indent=2))

    # Save results
    output_path = args.output or os.path.join(args.checkpoint_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
