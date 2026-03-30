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

# Limit GPU memory to 75% to leave room for OS/desktop
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.model.dataset import ExpressionDataset, INSTRUMENT_TO_ID
from src.model.baseline import BaselineModel, logits_to_f0
from src.model.encoder import MIDIEncoder
from src.model.diffusion import (
    ConditionalDDPM, AmpPredictor, TwoStepAmpPredictor, EncoderAdapter,
    denormalize_f0, denormalize_amp,
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
                       amp_mean=0.0, amp_std=1.0, ddim_steps=50, eta=0.0,
                       amp_predictor=None, amp_diffusion=None,
                       amp_instrument_conditioned=False,
                       amp_per_inst_stats=None,
                       pitch_amp_anchor=None,
                       residual_scale=1.0, residual_std=1.0,
                       max_eval_len=0,
                       encoder_adapter=None):
    """Evaluate diffusion model with multi-sample evaluation.

    Amp source priority:
    0. amp_diffusion + amp_predictor: RESIDUAL mode (mean + sampled residual)
    1. amp_diffusion alone: separate 1ch DDPM for amp (stochastic, sampled per-sample)
    2. amp_predictor alone: deterministic BiGRU predictor (same amp for all samples)
    3. Legacy 2-channel: amp from diffusion channel 1
    """
    encoder.eval()
    diffusion.eval()
    if amp_predictor is not None:
        amp_predictor.eval()
    if amp_diffusion is not None:
        amp_diffusion.eval()
    if encoder_adapter is not None:
        encoder_adapter.eval()

    n_channels = diffusion.n_channels
    all_metrics = []

    # Use autocast for mixed-precision inference to reduce GPU memory usage
    autocast_ctx = torch.cuda.amp.autocast() if device.type == "cuda" else torch.inference_mode()

    with torch.no_grad(), autocast_ctx:
        for idx in range(len(dataset)):
            # Free GPU cache between tracks to avoid OOM on long sequences
            if device.type == "cuda":
                torch.cuda.empty_cache()
            item = dataset._load(idx)
            ff = torch.from_numpy(item["frame_features"]).unsqueeze(0).to(device)
            f0_gt = item["f0"]
            amp_gt = item["amp"]
            notes = item["notes"]
            hop_time = item["hop_time"]

            # Truncate long sequences to prevent OOM (Fix-1: exp050)
            if max_eval_len > 0 and ff.shape[1] > max_eval_len:
                orig_len = ff.shape[1]
                ff = ff[:, :max_eval_len, :]
                f0_gt = f0_gt[:max_eval_len]
                amp_gt = amp_gt[:max_eval_len]
                # Truncate notes that extend beyond max_eval_len
                max_time = max_eval_len * hop_time
                notes = [n for n in notes if n[0] < max_time]
                print(f"  Track {idx}: truncated {orig_len} -> {max_eval_len} frames ({orig_len - max_eval_len} dropped)")

            # Get instrument_id for instrument-conditioned AmpPredictor
            # (needed for both pure predictor and residual diffusion modes)
            instrument_id = None
            if amp_instrument_conditioned and amp_predictor is not None:
                inst_name = item.get("instrument", dataset.tracks[idx].get("instrument", "vn"))
                inst_id = INSTRUMENT_TO_ID.get(inst_name, 0)
                instrument_id = torch.tensor([inst_id], device=device, dtype=torch.long)

            # Extract note_position for note-position-conditioned AmpPredictor (exp044+)
            note_position = None
            if amp_predictor is not None and getattr(amp_predictor, 'note_position_conditioned', False):
                note_position = ff[:, :, 3]  # (1, T) position_in_note from frame_features

            # Extract velocity for velocity-conditioned AmpPredictor (exp046+) or TwoStep (exp047+)
            velocity = None
            amp_is_two_step = isinstance(amp_predictor, TwoStepAmpPredictor) if amp_predictor is not None else False
            if amp_predictor is not None and (getattr(amp_predictor, 'velocity_conditioned', False) or amp_is_two_step):
                velocity = ff[:, :, 2]  # (1, T) normalized velocity (0-1)

            condition = encoder(ff)  # (1, T, 256)
            condition_perm = condition.permute(0, 2, 1)  # (1, 256, T)
            # Apply adapter for amp path (exp051+)
            condition_amp = encoder_adapter(condition) if encoder_adapter is not None else condition

            # If using deterministic AmpPredictor (NOT f0-conditioned), predict amp once
            amp_is_f0_cond = getattr(amp_predictor, 'f0_conditioned', False) if amp_predictor else False
            if amp_predictor is not None and amp_diffusion is None and not amp_is_f0_cond:
                if amp_is_two_step:
                    pitch_norm = ff[:, :, 1]  # (1, T) normalized pitch
                    log_amp_pred, _ = amp_predictor(condition_amp, velocity=velocity, pitch=pitch_norm)
                else:
                    log_amp_pred = amp_predictor(condition_amp, instrument_id=instrument_id, note_position=note_position, velocity=velocity)  # (1, T) — normalized log-space
                # Pitch-anchor denormalization (exp030+): add back anchor[pitch]
                if pitch_amp_anchor is not None:
                    note_pitch = ff[:, :, 1] * 127.0  # (1, T) denormalized MIDI pitch
                    pitch_idx = note_pitch.long().clamp(0, 127)
                    anchor_per_frame = pitch_amp_anchor[pitch_idx]  # (1, T)
                    log_amp_pred = log_amp_pred + anchor_per_frame
                # Per-instrument denormalization (exp029+)
                elif amp_per_inst_stats is not None and instrument_id is not None:
                    iid = instrument_id[0].item()
                    if iid in amp_per_inst_stats:
                        inst_mean = amp_per_inst_stats[iid]["mean"].to(device)
                        inst_std = amp_per_inst_stats[iid]["std"].to(device)
                        log_amp_pred = log_amp_pred * inst_std + inst_mean
                amp_pred_det = torch.exp(log_amp_pred[0]).cpu().numpy()
                amp_pred_det = np.clip(amp_pred_det, 0.0, None)

            sample_metrics = []
            all_f0_preds = []
            all_amp_preds = []
            for s in range(n_samples):
                # Ensure f0 diffusion is on GPU for sampling
                if amp_diffusion is not None and device.type == "cuda":
                    diffusion.to(device)

                x_gen = diffusion.ddim_sample(condition_perm, n_steps=ddim_steps, eta=eta)
                f0_norm = x_gen[0, 0].cpu()

                # Offload f0 diffusion to CPU to free GPU memory for amp diffusion (Fix-2: exp050)
                if amp_diffusion is not None and device.type == "cuda":
                    diffusion.cpu()
                    torch.cuda.empty_cache()

                f0_pred_hz = denormalize_f0(
                    f0_norm, notes, hop_time,
                ).numpy()

                if amp_diffusion is not None and amp_predictor is not None:
                    # RESIDUAL mode: mean from AmpPredictor + sampled residual from diffusion
                    if amp_is_two_step:
                        pitch_norm = ff[:, :, 1]
                        mu_raw_log, _ = amp_predictor(condition_amp, velocity=velocity, pitch=pitch_norm)
                    else:
                        mu_raw_log = amp_predictor(condition_amp, instrument_id=instrument_id, note_position=note_position, velocity=velocity)  # (1, T)
                    mu_norm = (mu_raw_log - amp_mean) / amp_std  # (1, T)
                    # Sample residual
                    res_gen = amp_diffusion.ddim_sample(
                        condition_perm, n_steps=ddim_steps, eta=eta,
                    )
                    residual = res_gen[0, 0] * residual_std  # [C2] unscale normalized residual
                    # Combine: final amp in z-score = mean + residual
                    amp_norm_final = mu_norm[0] + residual_scale * residual  # (T,) on device
                    amp_pred = torch.exp(amp_norm_final * amp_std + amp_mean).cpu().numpy()
                    amp_pred = np.clip(amp_pred, 0.0, None)
                elif amp_diffusion is not None:
                    # Pure amp diffusion (stochastic, no mean predictor)
                    amp_gen = amp_diffusion.ddim_sample(
                        condition_perm, n_steps=ddim_steps, eta=eta,
                    )
                    amp_norm = amp_gen[0, 0].cpu()
                    amp_pred = denormalize_amp(
                        amp_norm,
                        torch.tensor(amp_mean),
                        torch.tensor(amp_std),
                    ).numpy()
                    amp_pred = np.clip(amp_pred, 0.0, None)
                elif n_channels == 1 and amp_predictor is not None and amp_is_f0_cond:
                    # f0-conditioned AmpPredictor: use sampled f0_norm
                    f0_for_amp = x_gen[:, 0, :]  # (1, T) — normalized f0 from diffusion
                    log_amp_pred_s = amp_predictor(condition_amp, f0=f0_for_amp, instrument_id=instrument_id, note_position=note_position, velocity=velocity)
                    # Pitch-anchor denormalization (exp030+)
                    if pitch_amp_anchor is not None:
                        note_pitch = ff[:, :, 1] * 127.0
                        pitch_idx = note_pitch.long().clamp(0, 127)
                        anchor_per_frame = pitch_amp_anchor[pitch_idx]
                        log_amp_pred_s = log_amp_pred_s + anchor_per_frame
                    # Per-instrument denormalization (exp029+)
                    elif amp_per_inst_stats is not None and instrument_id is not None:
                        iid = instrument_id[0].item()
                        if iid in amp_per_inst_stats:
                            inst_mean = amp_per_inst_stats[iid]["mean"].to(device)
                            inst_std = amp_per_inst_stats[iid]["std"].to(device)
                            log_amp_pred_s = log_amp_pred_s * inst_std + inst_mean
                    amp_pred = torch.exp(log_amp_pred_s[0]).cpu().numpy()
                    amp_pred = np.clip(amp_pred, 0.0, None)
                elif n_channels == 1 and amp_predictor is not None:
                    # Use deterministic AmpPredictor output
                    amp_pred = amp_pred_det
                else:
                    # Legacy 2-channel: amp from diffusion channel 1
                    amp_norm = x_gen[0, 1].cpu()
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
                all_f0_preds.append(f0_pred_hz.copy())
                all_amp_preds.append(amp_pred.copy())

            # Inter-sample diversity metrics
            f0_stack = np.stack(all_f0_preds)  # (N, T)
            amp_stack = np.stack(all_amp_preds)  # (N, T)

            # f0 diversity: std in cents across samples, averaged over voiced frames
            all_voiced = np.all(f0_stack > 0, axis=0)  # frame voiced in ALL samples
            if all_voiced.sum() > 0:
                f0_voiced = f0_stack[:, all_voiced]  # (N, T_voiced)
                f0_cents_all = 1200.0 * np.log2(f0_voiced / f0_voiced.mean(axis=0, keepdims=True))
                f0_diversity_cents = float(np.mean(np.std(f0_cents_all, axis=0)))
            else:
                f0_diversity_cents = 0.0

            # amp diversity: std across samples, averaged over all frames
            amp_diversity = float(np.mean(np.std(amp_stack, axis=0)))

            # Separate oracles: f0 oracle by f0_mae, amp oracle by amp_rmse_log
            f0_oracle_idx = min(range(n_samples), key=lambda i: sample_metrics[i]["f0_mae"])
            amp_oracle_idx = min(range(n_samples), key=lambda i: sample_metrics[i]["amp_rmse_log"])

            mean_metrics = {}
            for k in ["rpa", "f0_mae"]:
                vals = [sm[k] for sm in sample_metrics]
                mean_metrics[f"{k}_mean"] = float(np.mean(vals))
                mean_metrics[f"{k}_oracle"] = sample_metrics[f0_oracle_idx][k]

            for k in ["amp_corr", "amp_rmse_log"]:
                vals = [sm[k] for sm in sample_metrics]
                mean_metrics[f"{k}_mean"] = float(np.mean(vals))
                mean_metrics[f"{k}_oracle"] = sample_metrics[amp_oracle_idx][k]

            for k in ["vde_mean", "vre_mean"]:
                vals = [sm[k] for sm in sample_metrics if sm[k] is not None]
                if vals:
                    mean_metrics[f"{k}_avg"] = float(np.mean(vals))
                    oracle_val = sample_metrics[f0_oracle_idx].get(k)
                    mean_metrics[f"{k}_oracle"] = oracle_val
                else:
                    mean_metrics[f"{k}_avg"] = None
                    mean_metrics[f"{k}_oracle"] = None

            mean_metrics["f0_diversity_cents"] = f0_diversity_cents
            mean_metrics["amp_diversity"] = amp_diversity
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
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM stochasticity (0=deterministic, 1=DDPM-like)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--residual_scale", type=float, default=1.0,
                        help="Residual scaling factor alpha (0=pure AmpPredictor, 1=full residual)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML for dataset settings (instruments, bach10_dir)")
    parser.add_argument("--max_test_tracks", type=int, default=0,
                        help="Limit test tracks for fast iteration (0=use all)")
    parser.add_argument("--max_eval_len", type=int, default=0,
                        help="Max frames per track (0=no limit). Truncates long sequences to prevent OOM.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset config from YAML if provided, else use defaults
    from src.model.dataset import DATA_DIR, BACH10_DIR, PHENICX_DIR, TRIOS_DIR, ALL_INSTRUMENTS
    data_dir = DATA_DIR
    bach10_dir = BACH10_DIR  # default: load Bach10 if available
    phenicx_dir = PHENICX_DIR
    trios_dir = TRIOS_DIR
    instruments = ALL_INSTRUMENTS
    if args.config and os.path.isfile(args.config):
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        _cfg = cfg  # save for later use
        if "data_dir" in cfg:
            data_dir = cfg["data_dir"]
        if "bach10_dir" in cfg:
            bach10_dir = cfg["bach10_dir"]
        if "phenicx_dir" in cfg:
            phenicx_dir = cfg["phenicx_dir"]
        if "trios_dir" in cfg:
            trios_dir = cfg["trios_dir"]
        if "instruments" in cfg:
            instruments = cfg["instruments"]

    test_ds = ExpressionDataset(
        data_dir=data_dir, bach10_dir=bach10_dir,
        phenicx_dir=phenicx_dir, trios_dir=trios_dir,
        instruments=instruments, split="test",
    )
    if args.max_test_tracks > 0 and len(test_ds) > args.max_test_tracks:
        test_ds.tracks = test_ds.tracks[:args.max_test_tracks]
        print(f"Test tracks: {len(test_ds)} (limited from full set)")
    else:
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
            print(f"Loaded baseline from {ckpt}")
        else:
            # Fallback: check config for baseline_checkpoint (frozen encoder case)
            fallback_bl = None
            if args.config and os.path.isfile(args.config):
                import yaml as _yaml_bl
                with open(args.config, "r", encoding="utf-8") as _f:
                    _cfg_bl = _yaml_bl.safe_load(_f) or {}
                fallback_bl = _cfg_bl.get("baseline_checkpoint", None)
            if fallback_bl and os.path.isfile(fallback_bl):
                baseline_model.load_state_dict(
                    torch.load(fallback_bl, map_location=device, weights_only=True)
                )
                print(f"Loaded baseline from config fallback: {fallback_bl}")
            else:
                print(f"WARNING: {ckpt} not found and no fallback, using random weights")
        encoder = baseline_model.get_encoder()
        # Check for fine-tuned encoder (exp055+): overrides baseline encoder weights
        encoder_ft_path = os.path.join(args.checkpoint_dir, "encoder_finetuned.pt")
        if os.path.isfile(encoder_ft_path):
            encoder.load_state_dict(
                torch.load(encoder_ft_path, map_location=device, weights_only=True)
            )
            print(f"Using fine-tuned encoder from {encoder_ft_path}")
        encoder.eval()

        # Detect n_channels from config or amp_predictor checkpoint
        n_channels = 2  # default legacy
        if args.config and os.path.isfile(args.config):
            import yaml as _yaml
            with open(args.config, "r", encoding="utf-8") as f:
                _cfg = _yaml.safe_load(f) or {}
            n_channels = _cfg.get("stage2", {}).get("n_channels", 2)

        diffusion = ConditionalDDPM(n_channels=n_channels).to(device)
        diff_ckpt = os.path.join(args.checkpoint_dir, "diffusion_best_ema.pt")
        if os.path.isfile(diff_ckpt):
            diffusion.load_state_dict(
                torch.load(diff_ckpt, map_location=device, weights_only=True)
            )
        else:
            # Fallback: check config for diffusion_checkpoint (frozen diffusion case)
            fallback_ckpt = None
            if args.config and os.path.isfile(args.config):
                fallback_ckpt = _cfg.get("stage2", {}).get("diffusion_checkpoint", None)
            if fallback_ckpt and os.path.isfile(fallback_ckpt):
                diffusion.load_state_dict(
                    torch.load(fallback_ckpt, map_location=device, weights_only=True)
                )
                print(f"Loaded frozen diffusion from config fallback: {fallback_ckpt}")
            else:
                print(f"WARNING: {diff_ckpt} not found and no fallback, using random weights")

        # Check for amp diffusion (exp021+) or AmpPredictor
        amp_predictor = None
        amp_diffusion = None
        amp_diff_ckpt = os.path.join(args.checkpoint_dir, "amp_diffusion_best_ema.pt")

        if n_channels == 1 and os.path.isfile(amp_diff_ckpt):
            # Amp diffusion mode: load separate 1ch DDPM for amp
            amp_diffusion = ConditionalDDPM(n_steps=1000, n_channels=1).to(device)
            amp_diffusion.load_state_dict(
                torch.load(amp_diff_ckpt, map_location=device, weights_only=True)
            )
            print(f"Loaded Amp Diffusion from {amp_diff_ckpt}")

        # Also load AmpPredictor if present (for residual mode or standalone)
        if n_channels == 1:
            amp_dropout = _cfg.get("stage2", {}).get("amp_dropout", 0.2) if args.config else 0.2
            amp_type = _cfg.get("stage2", {}).get("amp_type", "gru") if args.config else "gru"
            _amp_two_step = _cfg.get("stage2", {}).get("amp_two_step", False) if args.config else False
            amp_pred_ckpt = os.path.join(args.checkpoint_dir, "amp_predictor_best.pt")
            if os.path.isfile(amp_pred_ckpt):
                if _amp_two_step:
                    # Two-step amp prediction (exp047+)
                    amp_hidden = _cfg.get("stage2", {}).get("amp_hidden", 256)
                    amp_gru_hidden = _cfg.get("stage2", {}).get("amp_gru_hidden", 128)
                    amp_gru_layers = _cfg.get("stage2", {}).get("amp_gru_layers", 2)
                    amp_use_attention = _cfg.get("stage2", {}).get("amp_use_attention", True)
                    amp_n_attn_heads = _cfg.get("stage2", {}).get("amp_n_attn_heads", 4)
                    amp_n_attn_layers = _cfg.get("stage2", {}).get("amp_n_attn_layers", 2)
                    amp_smooth_kernel = _cfg.get("stage2", {}).get("amp_smooth_kernel", 31)
                    amp_predictor = TwoStepAmpPredictor(
                        cond_dim=256, hidden=amp_hidden, gru_hidden=amp_gru_hidden,
                        n_gru_layers=amp_gru_layers, dropout=amp_dropout,
                        use_attention=amp_use_attention,
                        n_attn_heads=amp_n_attn_heads,
                        n_attn_layers=amp_n_attn_layers,
                        smooth_kernel=amp_smooth_kernel,
                    ).to(device)
                    amp_predictor.load_state_dict(
                        torch.load(amp_pred_ckpt, map_location=device, weights_only=True)
                    )
                    print(f"Loaded TwoStepAmpPredictor (hidden={amp_hidden}, gru_hidden={amp_gru_hidden}, "
                          f"layers={amp_gru_layers}, dropout={amp_dropout}, "
                          f"use_attention={amp_use_attention}) from {amp_pred_ckpt}")
                elif amp_type == "tcn":
                    from src.model.diffusion import TCNAmpPredictor
                    tcn_channels = _cfg.get("stage2", {}).get("tcn_channels", 128) if args.config else 128
                    tcn_layers = _cfg.get("stage2", {}).get("tcn_layers", 8) if args.config else 8
                    tcn_kernel = _cfg.get("stage2", {}).get("tcn_kernel_size", 3) if args.config else 3
                    amp_predictor = TCNAmpPredictor(
                        cond_dim=256, n_channels=tcn_channels, kernel_size=tcn_kernel,
                        n_layers=tcn_layers, dropout=amp_dropout,
                    ).to(device)
                    amp_predictor.load_state_dict(
                        torch.load(amp_pred_ckpt, map_location=device, weights_only=True)
                    )
                    print(f"Loaded TCNAmpPredictor (channels={tcn_channels}, layers={tcn_layers}, "
                          f"kernel={tcn_kernel}, dropout={amp_dropout}) from {amp_pred_ckpt}")
                else:
                    amp_hidden = _cfg.get("stage2", {}).get("amp_hidden", 256)
                    amp_gru_hidden = _cfg.get("stage2", {}).get("amp_gru_hidden", 128)
                    amp_gru_layers = _cfg.get("stage2", {}).get("amp_gru_layers", 2)
                    amp_f0_conditioned = _cfg.get("stage2", {}).get("amp_f0_conditioned", False) if args.config else False
                    amp_use_attention = _cfg.get("stage2", {}).get("amp_use_attention", False) if args.config else False
                    amp_n_attn_heads = _cfg.get("stage2", {}).get("amp_n_attn_heads", 4) if args.config else 4
                    amp_n_attn_layers = _cfg.get("stage2", {}).get("amp_n_attn_layers", 2) if args.config else 2
                    amp_inst_conditioned = _cfg.get("stage2", {}).get("amp_instrument_conditioned", False) if args.config else False
                    amp_note_pos_cond = _cfg.get("stage2", {}).get("amp_note_position_conditioned", False) if args.config else False
                    amp_vel_conditioned = _cfg.get("stage2", {}).get("amp_velocity_conditioned", False) if args.config else False
                    amp_cond_dropout = float(_cfg.get("stage2", {}).get("amp_condition_dropout", 0.0)) if args.config else 0.0
                    amp_predictor = AmpPredictor(
                        cond_dim=256, hidden=amp_hidden, gru_hidden=amp_gru_hidden,
                        n_gru_layers=amp_gru_layers, dropout=amp_dropout,
                        f0_conditioned=amp_f0_conditioned,
                        use_attention=amp_use_attention,
                        n_attn_heads=amp_n_attn_heads,
                        n_attn_layers=amp_n_attn_layers,
                        instrument_conditioned=amp_inst_conditioned,
                        note_position_conditioned=amp_note_pos_cond,
                        velocity_conditioned=amp_vel_conditioned,
                        condition_dropout=amp_cond_dropout,
                    ).to(device)
                    amp_predictor.load_state_dict(
                        torch.load(amp_pred_ckpt, map_location=device, weights_only=True)
                    )
                    print(f"Loaded AmpPredictor (hidden={amp_hidden}, gru_hidden={amp_gru_hidden}, "
                          f"layers={amp_gru_layers}, dropout={amp_dropout}, "
                          f"f0_conditioned={amp_f0_conditioned}, "
                          f"use_attention={amp_use_attention}, "
                          f"n_attn_heads={amp_n_attn_heads}, n_attn_layers={amp_n_attn_layers}, "
                          f"instrument_conditioned={amp_inst_conditioned}, "
                          f"note_position_conditioned={amp_note_pos_cond}, "
                          f"velocity_conditioned={amp_vel_conditioned}, "
                          f"condition_dropout={amp_cond_dropout}) "
                          f"from {amp_pred_ckpt}")
            else:
                print(f"WARNING: {amp_pred_ckpt} not found, amp will be incorrect")

        # Load EncoderAdapter if present (exp051+)
        adapter = None
        adapter_ckpt = os.path.join(args.checkpoint_dir, "encoder_adapter_best.pt")
        if os.path.isfile(adapter_ckpt):
            adapter_bottleneck = _cfg.get("stage2", {}).get("adapter_bottleneck", 64) if args.config else 64
            adapter_layers = _cfg.get("stage2", {}).get("adapter_layers", 1) if args.config else 1
            adapter_dropout = _cfg.get("stage2", {}).get("adapter_dropout", 0.1) if args.config else 0.1
            adapter = EncoderAdapter(
                dim=256, bottleneck_dim=adapter_bottleneck,
                n_layers=adapter_layers, dropout=adapter_dropout,
            ).to(device)
            adapter.load_state_dict(
                torch.load(adapter_ckpt, map_location=device, weights_only=True)
            )
            adapter.eval()
            print(f"Loaded EncoderAdapter (bottleneck={adapter_bottleneck}, layers={adapter_layers}) "
                  f"from {adapter_ckpt}")

        # Load normalization stats
        amp_mean, amp_std = 0.0, 1.0
        # Amp diffusion uses amp_norm_stats.pt; legacy uses norm_stats.pt
        amp_norm_path = os.path.join(args.checkpoint_dir, "amp_norm_stats.pt")
        norm_path = os.path.join(args.checkpoint_dir, "norm_stats.pt")
        residual_std = 1.0  # default: no rescaling
        if amp_diffusion is not None and os.path.isfile(amp_norm_path):
            stats = torch.load(amp_norm_path, map_location=device, weights_only=True)
            amp_mean = float(stats["amp_mean"])
            amp_std = float(stats["amp_std"])
            residual_std = float(stats.get("residual_std", 1.0))  # [C2] exp050+
            print(f"Amp norm stats (amp diffusion): mean={amp_mean:.4f}, std={amp_std:.4f}, residual_std={residual_std:.4f}")
        elif os.path.isfile(norm_path):
            stats = torch.load(norm_path, map_location=device, weights_only=True)
            amp_mean = float(stats["amp_mean"])
            amp_std = float(stats["amp_std"])
        elif n_channels == 2:
            print("WARNING: norm_stats.pt not found, using defaults")

        # Detect residual mode: both amp_diffusion and amp_predictor loaded
        if amp_diffusion is not None and amp_predictor is not None:
            print("*** RESIDUAL AMP DIFFUSION MODE: amp = AmpPredictor_mean + diffusion_residual ***")

        # Detect instrument conditioning from config
        _amp_inst_cond = _cfg.get("stage2", {}).get("amp_instrument_conditioned", False) if args.config else False

        # Load per-instrument amp stats (exp029+)
        _amp_per_inst_stats = None
        _amp_per_inst_norm = _cfg.get("stage2", {}).get("amp_per_inst_norm", False) if args.config else False
        if _amp_per_inst_norm:
            per_inst_path = os.path.join(args.checkpoint_dir, "amp_per_inst_stats.pt")
            if os.path.isfile(per_inst_path):
                _amp_per_inst_stats = torch.load(per_inst_path, map_location=device, weights_only=True)
                print(f"Loaded per-instrument amp stats from {per_inst_path} ({len(_amp_per_inst_stats)} instruments)")
            else:
                print(f"WARNING: amp_per_inst_norm=True but {per_inst_path} not found")

        # Load pitch-anchor amp table (exp030+)
        _pitch_amp_anchor = None
        _amp_pitch_anchor_enabled = _cfg.get("stage2", {}).get("amp_pitch_anchor", False) if args.config else False
        if _amp_pitch_anchor_enabled:
            anchor_path = os.path.join(args.checkpoint_dir, "pitch_amp_anchor.pt")
            if os.path.isfile(anchor_path):
                _pitch_amp_anchor = torch.load(anchor_path, map_location=device, weights_only=True)
                print(f"Loaded pitch amp anchor from {anchor_path} (shape={_pitch_amp_anchor.shape})")
            else:
                print(f"WARNING: amp_pitch_anchor=True but {anchor_path} not found")

        diff_metrics = evaluate_diffusion(
            encoder, diffusion, test_ds, device,
            n_samples=args.n_samples, amp_mean=amp_mean, amp_std=amp_std,
            ddim_steps=args.ddim_steps, eta=args.eta,
            amp_predictor=amp_predictor, amp_diffusion=amp_diffusion,
            amp_instrument_conditioned=_amp_inst_cond,
            amp_per_inst_stats=_amp_per_inst_stats,
            pitch_amp_anchor=_pitch_amp_anchor,
            residual_scale=args.residual_scale,
            residual_std=residual_std,
            max_eval_len=getattr(args, 'max_eval_len', 0),
            encoder_adapter=adapter,
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
