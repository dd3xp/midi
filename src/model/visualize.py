"""
Paper visualization: f0/amp contour comparisons, vibrato zoom, eta sweep, instrument comparison.
Uses exp003 baseline checkpoint + exp007 diffusion checkpoint.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.model.dataset import notes_to_frame_features
from src.model.baseline import BaselineModel, logits_to_f0
from src.model.encoder import MIDIEncoder
from src.model.diffusion import ConditionalDDPM, denormalize_f0, denormalize_amp


# ============ Matplotlib Config ============

plt.rcParams.update({
    "figure.figsize": (12, 4),
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "lines.linewidth": 1.2,
    "savefig.dpi": 200,
})


# ============ Model Loading ============

def load_models(baseline_ckpt_dir, diffusion_ckpt_dir, device):
    """Load baseline model, encoder, diffusion model, and normalization stats."""
    # Baseline
    baseline = BaselineModel().to(device)
    bl_path = os.path.join(baseline_ckpt_dir, "baseline_best.pt")
    baseline.load_state_dict(torch.load(bl_path, map_location=device, weights_only=True))
    baseline.eval()

    # Encoder (from baseline)
    encoder = baseline.get_encoder()
    encoder.eval()

    # Diffusion
    diffusion = ConditionalDDPM().to(device)
    diff_path = os.path.join(diffusion_ckpt_dir, "diffusion_best_ema.pt")
    diffusion.load_state_dict(torch.load(diff_path, map_location=device, weights_only=True))
    diffusion.eval()

    # Normalization stats
    norm_path = os.path.join(diffusion_ckpt_dir, "norm_stats.pt")
    stats = torch.load(norm_path, map_location=device, weights_only=True)
    amp_mean = float(stats["amp_mean"])
    amp_std = float(stats["amp_std"])

    return baseline, encoder, diffusion, amp_mean, amp_std


def load_track(npz_path):
    """Load a single track's data."""
    data = np.load(npz_path)
    notes = data["notes"].astype(np.float32)
    f0 = data["f0"].astype(np.float32)
    amp = data["amp"].astype(np.float32)
    hop_time = float(data["hop_time"])
    n_frames = len(f0)
    frame_features = notes_to_frame_features(notes, n_frames, hop_time)
    return {
        "frame_features": frame_features,
        "f0": f0,
        "amp": amp,
        "notes": notes,
        "hop_time": hop_time,
        "n_frames": n_frames,
    }


# ============ Prediction ============

def predict_baseline(baseline, track, device):
    """Get baseline f0 (Hz) and amp predictions for a full track."""
    ff = torch.from_numpy(track["frame_features"]).unsqueeze(0).to(device)
    with torch.no_grad():
        f0_logits, amp_pred = baseline(ff)
    f0_logits = f0_logits.squeeze(0).cpu()
    amp_pred = amp_pred.squeeze(0).cpu().numpy()
    f0_hz = logits_to_f0(f0_logits, track["notes"], track["hop_time"]).numpy()
    return f0_hz, amp_pred


def predict_diffusion(encoder, diffusion, track, device,
                       amp_mean, amp_std, n_samples=5, ddim_steps=50, eta=0.0):
    """Get N diffusion samples of f0 (Hz) and amp for a full track."""
    ff = torch.from_numpy(track["frame_features"]).unsqueeze(0).to(device)
    with torch.no_grad():
        condition = encoder(ff).permute(0, 2, 1)  # (1, 256, T)

    f0_samples = []
    amp_samples = []
    for _ in range(n_samples):
        with torch.no_grad():
            x_gen = diffusion.ddim_sample(condition, n_steps=ddim_steps, eta=eta)
        f0_norm = x_gen[0, 0].cpu()
        amp_norm = x_gen[0, 1].cpu()

        f0_hz = denormalize_f0(f0_norm, track["notes"], track["hop_time"]).numpy()
        amp_pred = denormalize_amp(
            amp_norm, torch.tensor(amp_mean), torch.tensor(amp_std)
        ).numpy()
        amp_pred = np.clip(amp_pred, 0.0, None)

        f0_samples.append(f0_hz)
        amp_samples.append(amp_pred)

    return f0_samples, amp_samples


# ============ Plotting Functions ============

def f0_to_midi_display(f0_hz):
    """Convert Hz to MIDI note number for display. 0 Hz -> NaN."""
    f0_midi = np.full_like(f0_hz, np.nan)
    voiced = f0_hz > 0
    f0_midi[voiced] = 69.0 + 12.0 * np.log2(f0_hz[voiced] / 440.0)
    return f0_midi


def plot_f0_contour(track, bl_f0, diff_f0_samples, instrument_name, piece_name,
                     output_dir, time_range=None):
    """Generate f0 contour overlay figure for one track.

    Args:
        track: loaded track dict
        bl_f0: (T,) baseline f0 in Hz
        diff_f0_samples: list of (T,) diffusion f0 arrays in Hz
        instrument_name: e.g. "vn", "tpt", "fl"
        piece_name: e.g. "Jupiter", "Nocturne"
        output_dir: path to save figures
        time_range: optional (start_sec, end_sec) to limit x-axis
    """
    hop_time = track["hop_time"]
    n_frames = track["n_frames"]
    times = np.arange(n_frames) * hop_time

    gt_midi = f0_to_midi_display(track["f0"])
    bl_midi = f0_to_midi_display(bl_f0)
    diff_midis = [f0_to_midi_display(s) for s in diff_f0_samples]

    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot diffusion samples first (background)
    for i, dm in enumerate(diff_midis):
        label = "Diffusion samples" if i == 0 else None
        ax.plot(times, dm, color="red", alpha=0.25, linewidth=0.8, label=label)

    # Baseline
    ax.plot(times, bl_midi, color="royalblue", alpha=0.8, linewidth=1.2, label="Baseline")

    # Ground truth
    ax.plot(times, gt_midi, color="black", alpha=0.9, linewidth=1.0, label="Ground Truth")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (MIDI note)")
    inst_display = {"vn": "Violin", "tpt": "Trumpet", "fl": "Flute"}
    ax.set_title(f"f0 Contour — {inst_display.get(instrument_name, instrument_name)}: {piece_name}")
    ax.legend(loc="upper right")

    if time_range:
        ax.set_xlim(time_range)

    # Set y limits to data range with padding
    all_midi = np.concatenate([gt_midi] + diff_midis + [bl_midi])
    valid = all_midi[~np.isnan(all_midi)]
    if len(valid) > 0:
        ymin, ymax = valid.min() - 2, valid.max() + 2
        ax.set_ylim(ymin, ymax)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = f"f0_contour_{instrument_name}_{piece_name}.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


def find_vibrato_region(f0_hz, notes, hop_time, target_duration_sec=3.0):
    """Find a region with clear INTRA-NOTE vibrato, not note transitions."""
    best_start = 0.0
    best_score = -1
    best_note_center = 0.0

    for onset, offset, midi_pitch, _ in notes:
        duration = offset - onset
        if duration < 0.6:  # only long notes can show vibrato
            continue

        # Use interior of note (skip attack/release: first 15% and last 10%)
        inner_start = onset + duration * 0.15
        inner_end = offset - duration * 0.10
        if inner_end - inner_start < 0.3:
            continue

        start_frame = max(0, int(inner_start / hop_time))
        end_frame = min(len(f0_hz), int(inner_end / hop_time))
        seg = f0_hz[start_frame:end_frame]
        voiced = seg[seg > 0]

        if len(voiced) < 40:
            continue

        # Score by PERIODIC f0 variation (vibrato is typically 4-8 Hz)
        f0_cents = 1200.0 * np.log2(voiced / np.median(voiced))
        # Remove trend (linear detrend)
        x = np.arange(len(f0_cents))
        if len(x) > 1:
            coeffs = np.polyfit(x, f0_cents, 1)
            f0_detrended = f0_cents - np.polyval(coeffs, x)
        else:
            f0_detrended = f0_cents

        variation = np.std(f0_detrended)
        # Penalize very large variation (likely pitch instability, not vibrato)
        if variation > 80:  # > 80 cents = not vibrato
            continue

        if variation > best_score:
            best_score = variation
            best_note_center = (onset + offset) / 2

    # Build window around the best vibrato note
    window_start = max(0, best_note_center - target_duration_sec / 2)
    return (window_start, window_start + target_duration_sec)


def plot_vibrato_zoom(track, bl_f0, diff_f0_samples, instrument_name, piece_name,
                       output_dir, zoom_range=None):
    """Zoomed view of a few notes showing vibrato detail."""
    hop_time = track["hop_time"]
    n_frames = track["n_frames"]
    times = np.arange(n_frames) * hop_time

    if zoom_range is None:
        zoom_range = find_vibrato_region(track["f0"], track["notes"], hop_time)

    gt_midi = f0_to_midi_display(track["f0"])
    bl_midi = f0_to_midi_display(bl_f0)
    diff_midis = [f0_to_midi_display(s) for s in diff_f0_samples]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Diffusion samples
    for i, dm in enumerate(diff_midis):
        label = "Diffusion samples" if i == 0 else None
        ax.plot(times, dm, color="red", alpha=0.35, linewidth=1.0, label=label)

    # Baseline
    ax.plot(times, bl_midi, color="royalblue", alpha=0.9, linewidth=1.5, label="Baseline")

    # Ground truth
    ax.plot(times, gt_midi, color="black", alpha=0.9, linewidth=1.5, label="Ground Truth")

    ax.set_xlim(zoom_range)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (MIDI note)")
    inst_display = {"vn": "Violin", "tpt": "Trumpet", "fl": "Flute"}
    ax.set_title(f"Vibrato Detail — {inst_display.get(instrument_name, instrument_name)}: {piece_name}")
    ax.legend(loc="upper right")

    # Y-axis: zoom to data range in the time window
    mask = (times >= zoom_range[0]) & (times <= zoom_range[1])
    all_midi = np.concatenate([gt_midi[mask]] + [dm[mask] for dm in diff_midis] + [bl_midi[mask]])
    valid = all_midi[~np.isnan(all_midi)]
    if len(valid) > 0:
        ymin, ymax = valid.min() - 1, valid.max() + 1
        ax.set_ylim(ymin, ymax)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = f"vibrato_zoom_{instrument_name}_{piece_name}.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


def plot_amp_contour(track, bl_amp, diff_amp_samples, instrument_name, piece_name,
                      output_dir):
    """Generate amplitude contour overlay figure."""
    hop_time = track["hop_time"]
    n_frames = track["n_frames"]
    times = np.arange(n_frames) * hop_time

    fig, ax = plt.subplots(figsize=(12, 3))

    for i, da in enumerate(diff_amp_samples):
        label = "Diffusion samples" if i == 0 else None
        ax.plot(times, da, color="red", alpha=0.25, linewidth=0.8, label=label)

    ax.plot(times, bl_amp, color="royalblue", alpha=0.8, linewidth=1.2, label="Baseline")
    ax.plot(times, track["amp"], color="black", alpha=0.9, linewidth=1.0, label="Ground Truth")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (RMS)")
    inst_display = {"vn": "Violin", "tpt": "Trumpet", "fl": "Flute"}
    ax.set_title(f"Amplitude Contour — {inst_display.get(instrument_name, instrument_name)}: {piece_name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = f"amp_contour_{instrument_name}_{piece_name}.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


def plot_eta_sweep_summary(results_dir, output_dir):
    """Bar chart / line plot of metrics across eta values.

    Reads from experiments/results/exp009_*.json and exp008_*.json.
    """
    # Collect all eta results
    eta_data = []
    files_and_etas = [
        ("exp009_eta00.json", 0.0),
        ("exp009_eta03.json", 0.3),
        ("exp011_eta05.json", 0.5),
        ("exp009_eta10.json", 1.0),
    ]

    for fname, eta_val in files_and_etas:
        fpath = os.path.join(results_dir, fname)
        if not os.path.isfile(fpath):
            print(f"  Warning: {fpath} not found, skipping eta={eta_val}")
            continue
        d = json.load(open(fpath))
        s = d["diffusion_summary"]
        eta_data.append({
            "eta": eta_val,
            "rpa_mean": s["diffusion_rpa_mean"]["mean"] * 100,
            "rpa_oracle": s["diffusion_rpa_oracle"]["mean"] * 100,
            "f0_mae_mean": s["diffusion_f0_mae_mean"]["mean"],
            "f0_mae_oracle": s["diffusion_f0_mae_oracle"]["mean"],
            "amp_corr_mean": s["diffusion_amp_corr_mean"]["mean"],
        })
        # Add diversity if available
        div_key = "diffusion_f0_diversity_cents"
        if div_key in s:
            eta_data[-1]["f0_diversity"] = s[div_key]["mean"]

    if not eta_data:
        print("  No eta sweep data found, skipping plot")
        return None

    etas = [d["eta"] for d in eta_data]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: RPA vs eta
    ax = axes[0]
    ax.plot(etas, [d["rpa_mean"] for d in eta_data], "o-", color="royalblue",
            label="Mean", markersize=6)
    ax.plot(etas, [d["rpa_oracle"] for d in eta_data], "s--", color="darkblue",
            label="Oracle", markersize=6)
    ax.axhline(y=97.14, color='gray', linestyle=':', linewidth=1.0, alpha=0.7, label='Baseline')
    ax.set_xlabel("η (DDIM stochasticity)")
    ax.set_ylabel("RPA (%)")
    ax.set_title("Pitch Accuracy vs η")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: f0 MAE vs eta
    ax = axes[1]
    ax.plot(etas, [d["f0_mae_mean"] for d in eta_data], "o-", color="orangered",
            label="Mean", markersize=6)
    ax.plot(etas, [d["f0_mae_oracle"] for d in eta_data], "s--", color="darkred",
            label="Oracle", markersize=6)
    ax.set_xlabel("η (DDIM stochasticity)")
    ax.set_ylabel("f0 MAE (cents)")
    ax.set_title("Pitch Error vs η")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: f0 diversity vs eta
    ax = axes[2]
    divs_raw = [d.get("f0_diversity") for d in eta_data]
    # Filter out None values to avoid false zero points
    valid = [(e, dv) for e, dv in zip(etas, divs_raw) if dv is not None]
    if valid:
        valid_etas, divs = zip(*valid)
        ax.plot(valid_etas, divs, "o-", color="green", markersize=6)
    ax.set_xlabel("η (DDIM stochasticity)")
    ax.set_ylabel("f0 Inter-sample Std (cents)")
    ax.set_title("Diversity vs η")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = "eta_sweep_summary.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


def plot_instrument_comparison(results_dir, output_dir):
    """Per-instrument bar chart: baseline vs diffusion on key metrics."""
    # Load baseline results (exp003)
    bl_path = os.path.join(results_dir, "exp003.json")
    diff_path = os.path.join(results_dir, "exp009_eta03.json")  # best mean eta

    if not os.path.isfile(bl_path) or not os.path.isfile(diff_path):
        print(f"  Missing result files for instrument comparison")
        return None

    bl_data = json.load(open(bl_path))
    diff_data = json.load(open(diff_path))

    # Group by instrument
    def extract_instrument(track_path):
        """Extract instrument from track path like ...processed_vn/..."""
        path_lower = track_path.replace("\\", "/").lower()
        if "processed_vn" in path_lower:
            return "Violin"
        elif "processed_tpt" in path_lower:
            return "Trumpet"
        elif "processed_fl" in path_lower:
            return "Flute"
        return "Unknown"

    instruments = ["Violin", "Trumpet", "Flute"]

    # Aggregate per-instrument metrics
    bl_by_inst = {inst: {"rpa": [], "vde": [], "amp_corr": []} for inst in instruments}
    for t in bl_data["baseline_per_track"]:
        inst = extract_instrument(t["track"])
        if inst in bl_by_inst:
            bl_by_inst[inst]["rpa"].append(t["rpa"])
            if t.get("vde_mean") is not None:
                bl_by_inst[inst]["vde"].append(t["vde_mean"])
            bl_by_inst[inst]["amp_corr"].append(t["amp_corr"])

    diff_by_inst = {inst: {"rpa": [], "vde": [], "amp_corr": []} for inst in instruments}
    for t in diff_data["diffusion_per_track"]:
        inst = extract_instrument(t["track"])
        if inst in diff_by_inst:
            diff_by_inst[inst]["rpa"].append(t["rpa_mean"])
            if t.get("vde_mean_avg") is not None:
                diff_by_inst[inst]["vde"].append(t["vde_mean_avg"])
            diff_by_inst[inst]["amp_corr"].append(t["amp_corr_mean"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(instruments))
    width = 0.35

    # Panel 1: RPA
    ax = axes[0]
    bl_rpa = [np.mean(bl_by_inst[inst]["rpa"]) * 100 if bl_by_inst[inst]["rpa"] else 0
              for inst in instruments]
    diff_rpa = [np.mean(diff_by_inst[inst]["rpa"]) * 100 if diff_by_inst[inst]["rpa"] else 0
                for inst in instruments]
    bars1 = ax.bar(x - width/2, bl_rpa, width, label="Baseline", color="royalblue", alpha=0.8)
    bars2 = ax.bar(x + width/2, diff_rpa, width, label="Diffusion (mean)", color="orangered", alpha=0.8)
    ax.set_ylabel("RPA (%)")
    ax.set_title("Pitch Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(instruments)
    ax.legend()
    ax.set_ylim(80, 100)
    ax.grid(True, alpha=0.3, axis="y")
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    # Panel 2: VDE (lower is better)
    ax = axes[1]
    bl_vde = [np.mean(bl_by_inst[inst]["vde"]) if bl_by_inst[inst]["vde"] else 0
              for inst in instruments]
    diff_vde = [np.mean(diff_by_inst[inst]["vde"]) if diff_by_inst[inst]["vde"] else 0
                for inst in instruments]
    bars1 = ax.bar(x - width/2, bl_vde, width, label="Baseline", color="royalblue", alpha=0.8)
    bars2 = ax.bar(x + width/2, diff_vde, width, label="Diffusion (mean)", color="orangered", alpha=0.8)
    ax.set_ylabel("VDE (cents)")
    ax.set_title("Vibrato Depth Error ↓")
    ax.set_xticks(x)
    ax.set_xticklabels(instruments)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    # Panel 3: Amp Correlation
    ax = axes[2]
    bl_amp = [np.mean(bl_by_inst[inst]["amp_corr"]) if bl_by_inst[inst]["amp_corr"] else 0
              for inst in instruments]
    diff_amp = [np.mean(diff_by_inst[inst]["amp_corr"]) if diff_by_inst[inst]["amp_corr"] else 0
                for inst in instruments]
    bars1 = ax.bar(x - width/2, bl_amp, width, label="Baseline", color="royalblue", alpha=0.8)
    bars2 = ax.bar(x + width/2, diff_amp, width, label="Diffusion (mean)", color="orangered", alpha=0.8)
    ax.set_ylabel("Envelope Correlation")
    ax.set_title("Amplitude Correlation")
    ax.set_xticks(x)
    ax.set_xticklabels(instruments)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fname = "instrument_comparison.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--baseline-ckpt", type=str, required=True,
                        help="Path to baseline checkpoint dir (exp003)")
    parser.add_argument("--diffusion-ckpt", type=str, required=True,
                        help="Path to diffusion checkpoint dir (exp007)")
    parser.add_argument("--results-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "experiments", "results"),
                        help="Path to results dir for eta sweep data")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "experiments", "figures"),
                        help="Output directory for figures")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Fixed seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load models
    print("Loading models...")
    baseline, encoder, diffusion, amp_mean, amp_std = load_models(
        args.baseline_ckpt, args.diffusion_ckpt, device
    )

    # Tracks to visualize
    tracks_info = [
        {
            "path": os.path.join(PROJECT_ROOT, "datagen", "processed_vn",
                                  "01_Jupiter_vn_vc_track1_vn", "data.npz"),
            "instrument": "vn",
            "piece": "Jupiter",
        },
        {
            "path": os.path.join(PROJECT_ROOT, "datagen", "processed_tpt",
                                  "18_Nocturne_vn_fl_tpt_track3_tpt", "data.npz"),
            "instrument": "tpt",
            "piece": "Nocturne",
        },
        {
            "path": os.path.join(PROJECT_ROOT, "datagen", "processed_fl",
                                  "17_Nocturne_vn_fl_cl_track2_fl", "data.npz"),
            "instrument": "fl",
            "piece": "Nocturne",
        },
    ]

    generated_files = []

    for ti in tracks_info:
        print(f"\nProcessing {ti['instrument']}: {ti['piece']}...")
        track = load_track(ti["path"])
        print(f"  Frames: {track['n_frames']}, Duration: {track['n_frames'] * track['hop_time']:.1f}s")

        # Baseline prediction
        print("  Predicting baseline...")
        bl_f0, bl_amp = predict_baseline(baseline, track, device)

        # Diffusion predictions
        print(f"  Generating {args.n_samples} diffusion samples (ddim_steps={args.ddim_steps}, eta={args.eta})...")
        diff_f0s, diff_amps = predict_diffusion(
            encoder, diffusion, track, device,
            amp_mean, amp_std,
            n_samples=args.n_samples, ddim_steps=args.ddim_steps, eta=args.eta,
        )

        # Full f0 contour
        print("  Plotting f0 contour...")
        f = plot_f0_contour(track, bl_f0, diff_f0s,
                            ti["instrument"], ti["piece"], args.output_dir)
        generated_files.append(f)

        # Vibrato zoom
        print("  Plotting vibrato zoom...")
        f = plot_vibrato_zoom(track, bl_f0, diff_f0s,
                              ti["instrument"], ti["piece"], args.output_dir)
        generated_files.append(f)

        # Amp contour
        print("  Plotting amp contour...")
        f = plot_amp_contour(track, bl_amp, diff_amps,
                             ti["instrument"], ti["piece"], args.output_dir)
        generated_files.append(f)

    # Eta sweep summary (from existing result files)
    print("\nPlotting eta sweep summary...")
    f = plot_eta_sweep_summary(args.results_dir, args.output_dir)
    if f:
        generated_files.append(f)

    # Instrument comparison
    print("Plotting instrument comparison...")
    f = plot_instrument_comparison(args.results_dir, args.output_dir)
    if f:
        generated_files.append(f)

    print(f"\n=== Done! Generated {len(generated_files)} figures in {args.output_dir} ===")
    for gf in generated_files:
        print(f"  {gf}")


if __name__ == "__main__":
    main()
