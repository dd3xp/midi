"""
exp020: Paper-quality figures and result tables for AIMC 2026.
Uses exp019 mixed checkpoint (1ch diffusion + AmpPredictor).
No training — pure visualization + table generation.
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
from matplotlib.ticker import MaxNLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.model.dataset import ExpressionDataset, notes_to_frame_features
from src.model.baseline import BaselineModel, logits_to_f0
from src.model.encoder import MIDIEncoder
from src.model.diffusion import (
    ConditionalDDPM, AmpPredictor, denormalize_f0, denormalize_amp,
)


# ============ Matplotlib Config (Paper Quality) ============

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.2,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 100,
    "font.family": "sans-serif",
})

# Consistent color scheme
COLOR_GT = "black"
COLOR_BL = "#4472C4"       # blue
COLOR_DIFF = "#E74C3C"     # red
COLOR_DIFF_ALPHA = "#E74C3C"
COLOR_DIFF2 = "#F39C12"    # orange (for additional bar)

INST_DISPLAY = {
    "vn": "Violin", "va": "Viola", "vc": "Cello",
    "fl": "Flute", "ob": "Oboe", "cl": "Clarinet",
    "sax": "Saxophone", "tpt": "Trumpet", "tbn": "Trombone",
    "bn": "Bassoon",
}


# ============ Model Loading ============

def load_models(ckpt_dir, device, config=None):
    """Load baseline, encoder, 1ch diffusion, AmpPredictor from exp019 mixed."""
    import yaml

    # Baseline
    baseline = BaselineModel().to(device)
    bl_path = os.path.join(ckpt_dir, "baseline_best.pt")
    baseline.load_state_dict(torch.load(bl_path, map_location=device, weights_only=True))
    baseline.eval()

    # Encoder (from baseline)
    encoder = baseline.get_encoder()
    encoder.eval()

    # Diffusion (1-channel)
    n_channels = 1
    if config:
        n_channels = config.get("stage2", {}).get("n_channels", 1)
    diffusion = ConditionalDDPM(n_channels=n_channels).to(device)
    diff_path = os.path.join(ckpt_dir, "diffusion_best_ema.pt")
    diffusion.load_state_dict(torch.load(diff_path, map_location=device, weights_only=True))
    diffusion.eval()

    # AmpPredictor
    amp_hidden = config.get("stage2", {}).get("amp_hidden", 256) if config else 256
    amp_gru_hidden = config.get("stage2", {}).get("amp_gru_hidden", 128) if config else 128
    amp_gru_layers = config.get("stage2", {}).get("amp_gru_layers", 2) if config else 2
    amp_dropout = config.get("stage2", {}).get("amp_dropout", 0.3) if config else 0.3

    amp_predictor = AmpPredictor(
        cond_dim=256, hidden=amp_hidden, gru_hidden=amp_gru_hidden,
        n_gru_layers=amp_gru_layers, dropout=amp_dropout,
    ).to(device)
    amp_path = os.path.join(ckpt_dir, "amp_predictor_best.pt")
    amp_predictor.load_state_dict(torch.load(amp_path, map_location=device, weights_only=True))
    amp_predictor.eval()
    print(f"Loaded AmpPredictor (hidden={amp_hidden}, gru={amp_gru_hidden}, "
          f"layers={amp_gru_layers}, dropout={amp_dropout})")

    # Norm stats (for legacy compat, but amp now from AmpPredictor)
    norm_path = os.path.join(ckpt_dir, "norm_stats.pt")
    stats = torch.load(norm_path, map_location=device, weights_only=True)

    return baseline, encoder, diffusion, amp_predictor, stats


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
        "f0": f0, "amp": amp, "notes": notes,
        "hop_time": hop_time, "n_frames": n_frames,
    }


# ============ Prediction ============

def predict_baseline(baseline, track, device):
    """Baseline f0 (Hz) and amp predictions."""
    ff = torch.from_numpy(track["frame_features"]).unsqueeze(0).to(device)
    with torch.no_grad():
        f0_logits, amp_pred = baseline(ff)
    f0_logits = f0_logits.squeeze(0).cpu()
    amp_pred = amp_pred.squeeze(0).cpu().numpy()
    f0_hz = logits_to_f0(f0_logits, track["notes"], track["hop_time"]).numpy()
    return f0_hz, amp_pred


def predict_diffusion(encoder, diffusion, amp_predictor, track, device,
                      n_samples=5, ddim_steps=50, eta=0.3):
    """Get N diffusion f0 samples + deterministic AmpPredictor amp."""
    ff = torch.from_numpy(track["frame_features"]).unsqueeze(0).to(device)
    with torch.no_grad():
        condition = encoder(ff)  # (1, T, 256)
        condition_perm = condition.permute(0, 2, 1)  # (1, 256, T)

        # Deterministic amp
        log_amp = amp_predictor(condition)  # (1, T)
        amp_pred = torch.exp(log_amp[0]).cpu().numpy()
        amp_pred = np.clip(amp_pred, 0.0, None)

    f0_samples = []
    for _ in range(n_samples):
        with torch.no_grad():
            x_gen = diffusion.ddim_sample(condition_perm, n_steps=ddim_steps, eta=eta)
        f0_norm = x_gen[0, 0].cpu()
        f0_hz = denormalize_f0(f0_norm, track["notes"], track["hop_time"]).numpy()
        f0_samples.append(f0_hz)

    # AmpPredictor is deterministic: same amp for all samples
    amp_samples = [amp_pred.copy() for _ in range(n_samples)]
    return f0_samples, amp_samples


# ============ Utility ============

def f0_to_midi_display(f0_hz):
    """Hz -> MIDI note number (0 Hz -> NaN)."""
    f0_midi = np.full_like(f0_hz, np.nan)
    voiced = f0_hz > 0
    f0_midi[voiced] = 69.0 + 12.0 * np.log2(f0_hz[voiced] / 440.0)
    return f0_midi


def find_vibrato_region(f0_hz, notes, hop_time, target_duration_sec=3.0):
    """Find a region with clear intra-note vibrato."""
    best_start = 0.0
    best_score = -1
    best_note_center = 0.0

    for onset, offset, midi_pitch, _ in notes:
        duration = offset - onset
        if duration < 0.6:
            continue
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

        f0_cents = 1200.0 * np.log2(voiced / np.median(voiced))
        x = np.arange(len(f0_cents))
        if len(x) > 1:
            coeffs = np.polyfit(x, f0_cents, 1)
            f0_detrended = f0_cents - np.polyval(coeffs, x)
        else:
            f0_detrended = f0_cents

        variation = np.std(f0_detrended)
        if variation > 80:
            continue
        if variation > best_score:
            best_score = variation
            best_note_center = (onset + offset) / 2

    window_start = max(0, best_note_center - target_duration_sec / 2)
    return (window_start, window_start + target_duration_sec)


def extract_instrument_from_path(path):
    """Extract instrument abbreviation from track path."""
    path_norm = path.replace("\\", "/").lower()
    for inst in INST_DISPLAY:
        if f"processed_{inst}" in path_norm or path_norm.endswith(f"_{inst}"):
            return inst
        # Bach10 pattern: .../{piece}_{inst}/data.npz
        if f"_{inst}/" in path_norm:
            return inst
    return "unknown"


# ============ Figure 1: f0 Contour ============

def plot_f0_contour(track, bl_f0, diff_f0_samples, inst, piece, output_dir,
                    time_range=None):
    """f0 contour overlay — single column width (3.5 inch)."""
    hop_time = track["hop_time"]
    times = np.arange(track["n_frames"]) * hop_time

    gt_midi = f0_to_midi_display(track["f0"])
    bl_midi = f0_to_midi_display(bl_f0)
    diff_midis = [f0_to_midi_display(s) for s in diff_f0_samples]

    fig, ax = plt.subplots(figsize=(7, 3))

    for i, dm in enumerate(diff_midis):
        label = "Diffusion" if i == 0 else None
        ax.plot(times, dm, color=COLOR_DIFF, alpha=0.25, linewidth=0.7, label=label)

    ax.plot(times, bl_midi, color=COLOR_BL, alpha=0.85, linewidth=1.0, label="Baseline")
    ax.plot(times, gt_midi, color=COLOR_GT, alpha=0.9, linewidth=0.9, label="Ground Truth")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (MIDI)")
    ax.set_title(f"f0 Contour — {INST_DISPLAY.get(inst, inst)}: {piece}")
    ax.legend(loc="upper right", framealpha=0.8)

    if time_range:
        ax.set_xlim(time_range)

    all_midi = np.concatenate([gt_midi] + diff_midis + [bl_midi])
    valid = all_midi[~np.isnan(all_midi)]
    if len(valid) > 0:
        ax.set_ylim(valid.min() - 2, valid.max() + 2)

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fname = f"f0_contour_{inst}.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


# ============ Figure 1b: Vibrato Zoom ============

def plot_vibrato_zoom(track, bl_f0, diff_f0_samples, inst, piece, output_dir):
    """Zoomed vibrato detail — single column."""
    hop_time = track["hop_time"]
    times = np.arange(track["n_frames"]) * hop_time
    zoom = find_vibrato_region(track["f0"], track["notes"], hop_time)

    gt_midi = f0_to_midi_display(track["f0"])
    bl_midi = f0_to_midi_display(bl_f0)
    diff_midis = [f0_to_midi_display(s) for s in diff_f0_samples]

    fig, ax = plt.subplots(figsize=(7, 3))

    for i, dm in enumerate(diff_midis):
        label = "Diffusion" if i == 0 else None
        ax.plot(times, dm, color=COLOR_DIFF, alpha=0.35, linewidth=1.0, label=label)

    ax.plot(times, bl_midi, color=COLOR_BL, alpha=0.9, linewidth=1.3, label="Baseline")
    ax.plot(times, gt_midi, color=COLOR_GT, alpha=0.9, linewidth=1.3, label="Ground Truth")

    ax.set_xlim(zoom)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (MIDI)")
    ax.set_title(f"Vibrato Detail — {INST_DISPLAY.get(inst, inst)}: {piece}")
    ax.legend(loc="upper right", framealpha=0.8)

    mask = (times >= zoom[0]) & (times <= zoom[1])
    all_midi = np.concatenate([gt_midi[mask]] + [dm[mask] for dm in diff_midis] + [bl_midi[mask]])
    valid = all_midi[~np.isnan(all_midi)]
    if len(valid) > 0:
        ax.set_ylim(valid.min() - 1, valid.max() + 1)

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fname = f"vibrato_zoom_{inst}.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


# ============ Figure 2: Amp Contour ============

def plot_amp_contour(track, bl_amp, diff_amp_samples, inst, piece, output_dir):
    """Amplitude contour overlay — single column."""
    hop_time = track["hop_time"]
    times = np.arange(track["n_frames"]) * hop_time

    fig, ax = plt.subplots(figsize=(7, 2.5))

    for i, da in enumerate(diff_amp_samples):
        label = "Diffusion" if i == 0 else None
        ax.plot(times, da, color=COLOR_DIFF, alpha=0.3, linewidth=0.7, label=label)

    ax.plot(times, bl_amp, color=COLOR_BL, alpha=0.85, linewidth=1.0, label="Baseline")
    ax.plot(times, track["amp"], color=COLOR_GT, alpha=0.9, linewidth=0.9, label="Ground Truth")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (RMS)")
    ax.set_title(f"Amplitude — {INST_DISPLAY.get(inst, inst)}: {piece}")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fname = f"amp_contour_{inst}.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


# ============ Figure 3: Eta Sweep ============

def plot_eta_sweep(track, encoder, diffusion, amp_predictor, device,
                   output_dir, n_samples=3, ddim_steps=50):
    """4-panel: same passage at eta=0.0, 0.3, 0.5, 1.0."""
    etas = [0.0, 0.3, 0.5, 1.0]
    fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True, sharey=True)

    hop_time = track["hop_time"]
    times = np.arange(track["n_frames"]) * hop_time
    gt_midi = f0_to_midi_display(track["f0"])

    # Find a good zoom window
    zoom = find_vibrato_region(track["f0"], track["notes"], hop_time, target_duration_sec=5.0)

    for ax, eta in zip(axes.flat, etas):
        f0_samples, _ = predict_diffusion(
            encoder, diffusion, amp_predictor, track, device,
            n_samples=n_samples, ddim_steps=ddim_steps, eta=eta,
        )
        diff_midis = [f0_to_midi_display(s) for s in f0_samples]

        ax.plot(times, gt_midi, color=COLOR_GT, alpha=0.8, linewidth=1.0, label="GT")
        for i, dm in enumerate(diff_midis):
            label = "Samples" if i == 0 else None
            ax.plot(times, dm, color=COLOR_DIFF, alpha=0.4, linewidth=0.8, label=label)

        ax.set_xlim(zoom)
        ax.set_title(f"$\\eta$ = {eta}", fontsize=11)
        ax.grid(True, alpha=0.2)
        if ax in axes[-1]:
            ax.set_xlabel("Time (s)")
        if ax in axes[:, 0]:
            ax.set_ylabel("Pitch (MIDI)")

    # Zoom y-axis
    mask = (times >= zoom[0]) & (times <= zoom[1])
    valid = gt_midi[mask]
    valid = valid[~np.isnan(valid)]
    if len(valid) > 0:
        for ax in axes.flat:
            ax.set_ylim(valid.min() - 2, valid.max() + 2)

    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Effect of DDIM Stochasticity ($\\eta$)", fontsize=13, y=1.02)
    fig.tight_layout()
    fname = "eta_sweep.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


# ============ Figure 4: Per-Instrument Bar Chart ============

def plot_instrument_comparison(results_dir, output_dir):
    """Per-instrument bar chart: Baseline vs Diffusion for RPA and Amp Corr."""
    bl_path = os.path.join(results_dir, "exp014.json")
    diff_path = os.path.join(results_dir, "exp019_eta03.json")

    if not os.path.isfile(bl_path) or not os.path.isfile(diff_path):
        print(f"  Missing result files for instrument comparison")
        return None

    bl_data = json.load(open(bl_path))
    diff_data = json.load(open(diff_path))

    # Group by instrument
    instruments_order = ["vn", "va", "vc", "fl", "ob", "cl", "sax", "tpt", "tbn", "bn"]

    bl_by_inst = {i: {"rpa": [], "amp_corr": []} for i in instruments_order}
    for t in bl_data["baseline_per_track"]:
        inst = extract_instrument_from_path(t["track"])
        if inst in bl_by_inst:
            bl_by_inst[inst]["rpa"].append(t["rpa"])
            bl_by_inst[inst]["amp_corr"].append(t["amp_corr"])

    diff_by_inst = {i: {"rpa": [], "amp_corr": []} for i in instruments_order}
    for t in diff_data["diffusion_per_track"]:
        inst = extract_instrument_from_path(t["track"])
        if inst in diff_by_inst:
            diff_by_inst[inst]["rpa"].append(t["rpa_mean"])
            diff_by_inst[inst]["amp_corr"].append(t["amp_corr_mean"])

    # Filter instruments with data
    active_insts = [i for i in instruments_order
                    if bl_by_inst[i]["rpa"] or diff_by_inst[i]["rpa"]]

    labels = [INST_DISPLAY.get(i, i) for i in active_insts]
    x = np.arange(len(active_insts))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    # (a) RPA
    bl_rpa = [np.mean(bl_by_inst[i]["rpa"]) * 100 if bl_by_inst[i]["rpa"] else 0
              for i in active_insts]
    diff_rpa = [np.mean(diff_by_inst[i]["rpa"]) * 100 if diff_by_inst[i]["rpa"] else 0
                for i in active_insts]

    ax1.bar(x - width/2, bl_rpa, width, label="Baseline", color=COLOR_BL, alpha=0.85)
    ax1.bar(x + width/2, diff_rpa, width, label="Diffusion", color=COLOR_DIFF, alpha=0.85)
    ax1.set_ylabel("RPA (%)")
    ax1.set_title("(a) Pitch Accuracy by Instrument")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.set_ylim(85, 100)
    ax1.grid(True, alpha=0.2, axis="y")

    # (b) Amp Corr
    bl_amp = [np.mean(bl_by_inst[i]["amp_corr"]) if bl_by_inst[i]["amp_corr"] else 0
              for i in active_insts]
    diff_amp = [np.mean(diff_by_inst[i]["amp_corr"]) if diff_by_inst[i]["amp_corr"] else 0
                for i in active_insts]

    ax2.bar(x - width/2, bl_amp, width, label="Baseline", color=COLOR_BL, alpha=0.85)
    ax2.bar(x + width/2, diff_amp, width, label="Diffusion", color=COLOR_DIFF, alpha=0.85)
    ax2.set_ylabel("Envelope Correlation")
    ax2.set_title("(b) Amplitude Correlation by Instrument")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fname = "instrument_comparison.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


# ============ Figure 5: VDE Comparison ============

def plot_vde_comparison(results_dir, output_dir):
    """Per-instrument VDE bar chart: Baseline vs Diffusion."""
    bl_path = os.path.join(results_dir, "exp014.json")
    diff_path = os.path.join(results_dir, "exp019_eta03.json")

    if not os.path.isfile(bl_path) or not os.path.isfile(diff_path):
        print(f"  Missing result files for VDE comparison")
        return None

    bl_data = json.load(open(bl_path))
    diff_data = json.load(open(diff_path))

    instruments_order = ["vn", "va", "vc", "fl", "ob", "cl", "sax", "tpt", "tbn", "bn"]

    bl_by_inst = {i: [] for i in instruments_order}
    for t in bl_data["baseline_per_track"]:
        inst = extract_instrument_from_path(t["track"])
        if inst in bl_by_inst and t.get("vde_mean") is not None:
            bl_by_inst[inst].append(t["vde_mean"])

    diff_by_inst = {i: [] for i in instruments_order}
    for t in diff_data["diffusion_per_track"]:
        inst = extract_instrument_from_path(t["track"])
        if inst in diff_by_inst and t.get("vde_mean_avg") is not None:
            diff_by_inst[inst].append(t["vde_mean_avg"])

    active_insts = [i for i in instruments_order
                    if bl_by_inst[i] or diff_by_inst[i]]
    labels = [INST_DISPLAY.get(i, i) for i in active_insts]
    x = np.arange(len(active_insts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3))

    bl_vde = [np.mean(bl_by_inst[i]) if bl_by_inst[i] else 0 for i in active_insts]
    diff_vde = [np.mean(diff_by_inst[i]) if diff_by_inst[i] else 0 for i in active_insts]

    bars1 = ax.bar(x - width/2, bl_vde, width, label="Baseline", color=COLOR_BL, alpha=0.85)
    bars2 = ax.bar(x + width/2, diff_vde, width, label="Diffusion", color=COLOR_DIFF, alpha=0.85)

    ax.set_ylabel("VDE (cents)")
    ax.set_title("Vibrato Depth Error by Instrument (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    fname = "vde_comparison.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


# ============ Figure 6: Eta Sweep (metrics) ============

def plot_eta_metrics(results_dir, output_dir):
    """Line charts: metrics vs eta from exp019 multi-eta results."""
    eta_data = []
    files = [
        ("exp019_eta00.json", 0.0),
        ("exp019_eta03.json", 0.3),
        ("exp019_eta05.json", 0.5),
        ("exp019_eta10.json", 1.0),
    ]

    for fname, eta_val in files:
        fpath = os.path.join(results_dir, fname)
        if not os.path.isfile(fpath):
            print(f"  Warning: {fpath} not found")
            continue
        d = json.load(open(fpath))
        s = d["diffusion_summary"]
        entry = {
            "eta": eta_val,
            "rpa_mean": s["diffusion_rpa_mean"]["mean"] * 100,
            "rpa_oracle": s["diffusion_rpa_oracle"]["mean"] * 100,
            "f0_mae": s["diffusion_f0_mae_mean"]["mean"],
            "vde": s["diffusion_vde_mean_avg"]["mean"],
            "vre": s["diffusion_vre_mean_avg"]["mean"],
        }
        if "diffusion_f0_diversity_cents" in s:
            entry["diversity"] = s["diffusion_f0_diversity_cents"]["mean"]
        eta_data.append(entry)

    if not eta_data:
        print("  No eta data found")
        return None

    etas = [d["eta"] for d in eta_data]

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))

    # (a) RPA
    ax = axes[0]
    ax.plot(etas, [d["rpa_mean"] for d in eta_data], "o-", color=COLOR_DIFF,
            markersize=5, label="Mean")
    ax.plot(etas, [d["rpa_oracle"] for d in eta_data], "s--", color=COLOR_DIFF,
            markersize=5, alpha=0.6, label="Oracle")
    ax.axhline(y=96.59, color=COLOR_BL, linestyle=":", linewidth=1.0, alpha=0.7, label="Baseline")
    ax.set_xlabel("$\\eta$")
    ax.set_ylabel("RPA (%)")
    ax.set_title("(a) Pitch Accuracy")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # (b) VDE
    ax = axes[1]
    ax.plot(etas, [d["vde"] for d in eta_data], "o-", color=COLOR_DIFF, markersize=5)
    ax.axhline(y=8.70, color=COLOR_BL, linestyle=":", linewidth=1.0, alpha=0.7, label="Baseline")
    ax.set_xlabel("$\\eta$")
    ax.set_ylabel("VDE (cents)")
    ax.set_title("(b) Vibrato Depth Error")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # (c) Diversity
    ax = axes[2]
    divs = [d.get("diversity") for d in eta_data]
    valid = [(e, dv) for e, dv in zip(etas, divs) if dv is not None]
    if valid:
        ve, vd = zip(*valid)
        ax.plot(ve, vd, "o-", color="#27AE60", markersize=5)
    ax.set_xlabel("$\\eta$")
    ax.set_ylabel("f0 Diversity (cents)")
    ax.set_title("(c) Inter-sample Diversity")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fname = "eta_metrics.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


# ============ Figure 7: Training Curves ============

def plot_training_curves(checkpoints_dir, output_dir):
    """Training loss curves from exp017 (diffusion) and exp018 (amp-focused)."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    for exp_name, label, ax in [("exp017", "exp017 (f0 diffusion)", axes[0]),
                                 ("exp018", "exp018 (amp-focused)", axes[1])]:
        hist_path = os.path.join(checkpoints_dir, exp_name, "stage2_history.json")
        if not os.path.isfile(hist_path):
            print(f"  Warning: {hist_path} not found")
            continue

        h = json.load(open(hist_path))
        epochs = np.arange(1, len(h["train_loss"]) + 1)

        ax.plot(epochs, h["train_diff_loss"], color=COLOR_DIFF, alpha=0.8,
                linewidth=1.0, label="Train diff loss")
        ax.plot(epochs, h["test_diff_loss"], color=COLOR_DIFF, alpha=0.5,
                linewidth=1.0, linestyle="--", label="Test diff loss")
        ax.plot(epochs, h["train_amp_loss"], color=COLOR_BL, alpha=0.8,
                linewidth=1.0, label="Train amp loss")
        ax.plot(epochs, h["test_amp_loss"], color=COLOR_BL, alpha=0.5,
                linewidth=1.0, linestyle="--", label="Test amp loss")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    fname = "training_curves.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


# ============ Paper Tables ============

def generate_paper_tables(results_dir, output_path):
    """Generate paper_tables.md from exp014 (baseline) + exp019 (diffusion) results."""
    # Load data
    bl_data = json.load(open(os.path.join(results_dir, "exp014.json")))
    bl_s = bl_data["baseline_summary"]

    eta_results = {}
    for eta_label, eta_file in [("0.0", "exp019_eta00.json"),
                                 ("0.3", "exp019_eta03.json"),
                                 ("0.5", "exp019_eta05.json"),
                                 ("1.0", "exp019_eta10.json")]:
        fpath = os.path.join(results_dir, eta_file)
        if os.path.isfile(fpath):
            d = json.load(open(fpath))
            eta_results[eta_label] = d["diffusion_summary"]

    lines = []
    lines.append("# Paper Results Tables (exp020)")
    lines.append("")
    lines.append("Generated from exp014 (baseline) and exp019 (diffusion) evaluations.")
    lines.append("Dataset: URMP 9 instruments + Bach10 4 instruments = 173 tracks.")
    lines.append("")

    # Table 1: Overall Comparison
    lines.append("## Table 1: Overall Comparison")
    lines.append("")

    def fmt(val, mult=1, prec=2):
        return f"{val * mult:.{prec}f}"

    def fmt_pm(summary, key, mult=1, prec=2):
        if key not in summary:
            return "—"
        m = summary[key]["mean"]
        s = summary[key]["std"]
        return f"{m * mult:.{prec}f} +/- {s * mult:.{prec}f}"

    s03 = eta_results.get("0.3", {})
    s10 = eta_results.get("1.0", {})

    lines.append("| Metric | Baseline | Diffusion eta=0.3 | Diffusion eta=1.0 |")
    lines.append("|--------|----------|-------------------|-------------------|")

    bl_rpa = bl_s["baseline_rpa"]["mean"] * 100
    bl_rpa_s = bl_s["baseline_rpa"]["std"] * 100
    d03_rpa = s03.get("diffusion_rpa_mean", {}).get("mean", 0) * 100
    d03_rpa_s = s03.get("diffusion_rpa_mean", {}).get("std", 0) * 100
    d10_rpa = s10.get("diffusion_rpa_mean", {}).get("mean", 0) * 100
    d10_rpa_s = s10.get("diffusion_rpa_mean", {}).get("std", 0) * 100
    lines.append(f"| f0 RPA (%) | {bl_rpa:.2f} +/- {bl_rpa_s:.2f} | {d03_rpa:.2f} +/- {d03_rpa_s:.2f} | {d10_rpa:.2f} +/- {d10_rpa_s:.2f} |")

    bl_mae = bl_s["baseline_f0_mae"]["mean"]
    bl_mae_s = bl_s["baseline_f0_mae"]["std"]
    d03_mae = s03.get("diffusion_f0_mae_mean", {}).get("mean", 0)
    d03_mae_s = s03.get("diffusion_f0_mae_mean", {}).get("std", 0)
    d10_mae = s10.get("diffusion_f0_mae_mean", {}).get("mean", 0)
    d10_mae_s = s10.get("diffusion_f0_mae_mean", {}).get("std", 0)
    lines.append(f"| f0 MAE (cents) | {bl_mae:.2f} +/- {bl_mae_s:.2f} | {d03_mae:.2f} +/- {d03_mae_s:.2f} | {d10_mae:.2f} +/- {d10_mae_s:.2f} |")

    bl_ac = bl_s["baseline_amp_corr"]["mean"]
    bl_ac_s = bl_s["baseline_amp_corr"]["std"]
    d03_ac = s03.get("diffusion_amp_corr_mean", {}).get("mean", 0)
    d03_ac_s = s03.get("diffusion_amp_corr_mean", {}).get("std", 0)
    d10_ac = s10.get("diffusion_amp_corr_mean", {}).get("mean", 0)
    d10_ac_s = s10.get("diffusion_amp_corr_mean", {}).get("std", 0)
    lines.append(f"| Amp Corr | {bl_ac:.3f} +/- {bl_ac_s:.3f} | {d03_ac:.3f} +/- {d03_ac_s:.3f} | {d10_ac:.3f} +/- {d10_ac_s:.3f} |")

    bl_ar = bl_s["baseline_amp_rmse_log"]["mean"]
    bl_ar_s = bl_s["baseline_amp_rmse_log"]["std"]
    d03_ar = s03.get("diffusion_amp_rmse_log_mean", {}).get("mean", 0)
    d03_ar_s = s03.get("diffusion_amp_rmse_log_mean", {}).get("std", 0)
    d10_ar = s10.get("diffusion_amp_rmse_log_mean", {}).get("mean", 0)
    d10_ar_s = s10.get("diffusion_amp_rmse_log_mean", {}).get("std", 0)
    lines.append(f"| Amp RMSE(log) | {bl_ar:.3f} +/- {bl_ar_s:.3f} | {d03_ar:.3f} +/- {d03_ar_s:.3f} | {d10_ar:.3f} +/- {d10_ar_s:.3f} |")

    bl_vde = bl_s["baseline_vde_mean"]["mean"]
    bl_vde_s = bl_s["baseline_vde_mean"]["std"]
    d03_vde = s03.get("diffusion_vde_mean_avg", {}).get("mean", 0)
    d03_vde_s = s03.get("diffusion_vde_mean_avg", {}).get("std", 0)
    d10_vde = s10.get("diffusion_vde_mean_avg", {}).get("mean", 0)
    d10_vde_s = s10.get("diffusion_vde_mean_avg", {}).get("std", 0)
    lines.append(f"| VDE (cents) | {bl_vde:.2f} +/- {bl_vde_s:.2f} | {d03_vde:.2f} +/- {d03_vde_s:.2f} | {d10_vde:.2f} +/- {d10_vde_s:.2f} |")

    bl_vre = bl_s["baseline_vre_mean"]["mean"]
    bl_vre_s = bl_s["baseline_vre_mean"]["std"]
    d03_vre = s03.get("diffusion_vre_mean_avg", {}).get("mean", 0)
    d03_vre_s = s03.get("diffusion_vre_mean_avg", {}).get("std", 0)
    d10_vre = s10.get("diffusion_vre_mean_avg", {}).get("mean", 0)
    d10_vre_s = s10.get("diffusion_vre_mean_avg", {}).get("std", 0)
    lines.append(f"| VRE (Hz) | {bl_vre:.3f} +/- {bl_vre_s:.3f} | {d03_vre:.3f} +/- {d03_vre_s:.3f} | {d10_vre:.3f} +/- {d10_vre_s:.3f} |")

    d03_div = s03.get("diffusion_f0_diversity_cents", {}).get("mean", 0)
    d03_div_s = s03.get("diffusion_f0_diversity_cents", {}).get("std", 0)
    d10_div = s10.get("diffusion_f0_diversity_cents", {}).get("mean", 0)
    d10_div_s = s10.get("diffusion_f0_diversity_cents", {}).get("std", 0)
    lines.append(f"| f0 Diversity (cents) | — | {d03_div:.2f} +/- {d03_div_s:.2f} | {d10_div:.2f} +/- {d10_div_s:.2f} |")

    lines.append("")

    # Table 2: Multi-eta
    lines.append("## Table 2: Multi-eta Evaluation")
    lines.append("")
    lines.append("| eta | RPA mean (%) | RPA oracle (%) | MAE mean | VDE | VRE | f0 Diversity (cents) |")
    lines.append("|-----|-------------|---------------|----------|-----|-----|---------------------|")

    for eta_label in ["0.0", "0.3", "0.5", "1.0"]:
        s = eta_results.get(eta_label, {})
        if not s:
            continue
        rpa_m = s.get("diffusion_rpa_mean", {}).get("mean", 0) * 100
        rpa_o = s.get("diffusion_rpa_oracle", {}).get("mean", 0) * 100
        mae_m = s.get("diffusion_f0_mae_mean", {}).get("mean", 0)
        vde_m = s.get("diffusion_vde_mean_avg", {}).get("mean", 0)
        vre_m = s.get("diffusion_vre_mean_avg", {}).get("mean", 0)
        div_m = s.get("diffusion_f0_diversity_cents", {}).get("mean", 0)
        lines.append(f"| {eta_label} | {rpa_m:.2f} | {rpa_o:.2f} | {mae_m:.2f} | {vde_m:.2f} | {vre_m:.3f} | {div_m:.2f} |")

    lines.append("")

    # Table 3: Per-instrument
    lines.append("## Table 3: Per-Instrument Performance (Selected)")
    lines.append("")
    lines.append("Baseline (exp014) vs Diffusion eta=0.3 (exp019)")
    lines.append("")

    instruments_sel = ["vn", "va", "fl", "ob", "tpt", "bn"]

    bl_by_inst = {}
    for t in bl_data["baseline_per_track"]:
        inst = extract_instrument_from_path(t["track"])
        if inst not in bl_by_inst:
            bl_by_inst[inst] = {"rpa": [], "amp_corr": [], "vde": []}
        bl_by_inst[inst]["rpa"].append(t["rpa"])
        bl_by_inst[inst]["amp_corr"].append(t["amp_corr"])
        if t.get("vde_mean") is not None:
            bl_by_inst[inst]["vde"].append(t["vde_mean"])

    diff03_data = json.load(open(os.path.join(results_dir, "exp019_eta03.json")))
    diff_by_inst = {}
    for t in diff03_data["diffusion_per_track"]:
        inst = extract_instrument_from_path(t["track"])
        if inst not in diff_by_inst:
            diff_by_inst[inst] = {"rpa": [], "amp_corr": [], "vde": []}
        diff_by_inst[inst]["rpa"].append(t["rpa_mean"])
        diff_by_inst[inst]["amp_corr"].append(t["amp_corr_mean"])
        if t.get("vde_mean_avg") is not None:
            diff_by_inst[inst]["vde"].append(t["vde_mean_avg"])

    lines.append("| Instrument | BL RPA (%) | Diff RPA (%) | BL Amp Corr | Diff Amp Corr | BL VDE | Diff VDE |")
    lines.append("|------------|-----------|-------------|------------|--------------|--------|---------|")

    for inst in instruments_sel:
        bl = bl_by_inst.get(inst, {"rpa": [], "amp_corr": [], "vde": []})
        di = diff_by_inst.get(inst, {"rpa": [], "amp_corr": [], "vde": []})
        bl_r = np.mean(bl["rpa"]) * 100 if bl["rpa"] else 0
        di_r = np.mean(di["rpa"]) * 100 if di["rpa"] else 0
        bl_a = np.mean(bl["amp_corr"]) if bl["amp_corr"] else 0
        di_a = np.mean(di["amp_corr"]) if di["amp_corr"] else 0
        bl_v = np.mean(bl["vde"]) if bl["vde"] else 0
        di_v = np.mean(di["vde"]) if di["vde"] else 0
        name = INST_DISPLAY.get(inst, inst)
        lines.append(f"| {name} | {bl_r:.1f} | {di_r:.1f} | {bl_a:.3f} | {di_a:.3f} | {bl_v:.1f} | {di_v:.1f} |")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by exp020 plot_final.py*")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved paper_tables.md to {output_path}")


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures + tables (exp020)")
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Path to exp019 mixed checkpoint dir")
    parser.add_argument("--results-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "experiments", "results"))
    parser.add_argument("--checkpoints-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "experiments", "checkpoints"),
                        help="Parent checkpoints dir (for training curves)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "experiments", "figures", "final"))
    parser.add_argument("--tables-output", type=str,
                        default=os.path.join(PROJECT_ROOT, "experiments", "results", "paper_tables.md"))
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML for model parameters")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip model-based figures (only tables + bar charts)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.tables_output), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    generated = []

    # Load config
    config = None
    if args.config and os.path.isfile(args.config):
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    if not args.skip_model:
        print("\n=== Loading models ===")
        baseline, encoder, diffusion, amp_predictor, stats = load_models(
            args.ckpt_dir, device, config
        )

        # Representative tracks (vn, fl, tpt)
        data_dir = os.path.join(PROJECT_ROOT, "datagen", "solo", "URMP")
        tracks_info = [
            {
                "path": os.path.join(data_dir, "processed_vn",
                                     "01_Jupiter_vn_vc_track1_vn", "data.npz"),
                "inst": "vn", "piece": "Jupiter",
            },
            {
                "path": os.path.join(data_dir, "processed_fl",
                                     "17_Nocturne_vn_fl_cl_track2_fl", "data.npz"),
                "inst": "fl", "piece": "Nocturne",
            },
            {
                "path": os.path.join(data_dir, "processed_tpt",
                                     "18_Nocturne_vn_fl_tpt_track3_tpt", "data.npz"),
                "inst": "tpt", "piece": "Nocturne",
            },
        ]

        # Figure 1-2: f0/vibrato/amp contours for 3 instruments
        for ti in tracks_info:
            if not os.path.isfile(ti["path"]):
                print(f"  Warning: {ti['path']} not found, skipping")
                continue
            print(f"\n--- {INST_DISPLAY[ti['inst']]}: {ti['piece']} ---")
            track = load_track(ti["path"])
            print(f"  Frames: {track['n_frames']}, "
                  f"Duration: {track['n_frames'] * track['hop_time']:.1f}s")

            bl_f0, bl_amp = predict_baseline(baseline, track, device)

            print(f"  Generating {args.n_samples} diffusion samples...")
            diff_f0s, diff_amps = predict_diffusion(
                encoder, diffusion, amp_predictor, track, device,
                n_samples=args.n_samples, ddim_steps=args.ddim_steps, eta=args.eta,
            )

            generated.append(plot_f0_contour(
                track, bl_f0, diff_f0s, ti["inst"], ti["piece"], args.output_dir))
            generated.append(plot_vibrato_zoom(
                track, bl_f0, diff_f0s, ti["inst"], ti["piece"], args.output_dir))
            generated.append(plot_amp_contour(
                track, bl_amp, diff_amps, ti["inst"], ti["piece"], args.output_dir))

        # Figure 3: Eta sweep (use violin track)
        print("\n--- Eta sweep (Violin) ---")
        vn_path = tracks_info[0]["path"]
        if os.path.isfile(vn_path):
            vn_track = load_track(vn_path)
            f = plot_eta_sweep(vn_track, encoder, diffusion, amp_predictor, device,
                               args.output_dir, n_samples=3, ddim_steps=args.ddim_steps)
            generated.append(f)

    # Figure 4: Instrument comparison (from JSON, no model needed)
    print("\n--- Instrument comparison ---")
    f = plot_instrument_comparison(args.results_dir, args.output_dir)
    if f:
        generated.append(f)

    # Figure 5: VDE comparison
    print("\n--- VDE comparison ---")
    f = plot_vde_comparison(args.results_dir, args.output_dir)
    if f:
        generated.append(f)

    # Figure 6: Eta metrics
    print("\n--- Eta metrics ---")
    f = plot_eta_metrics(args.results_dir, args.output_dir)
    if f:
        generated.append(f)

    # Figure 7: Training curves
    print("\n--- Training curves ---")
    f = plot_training_curves(args.checkpoints_dir, args.output_dir)
    if f:
        generated.append(f)

    # Paper tables
    print("\n--- Paper tables ---")
    generate_paper_tables(args.results_dir, args.tables_output)

    print(f"\n=== Done! Generated {len(generated)} figures ===")
    for g in generated:
        print(f"  {g}")
    print(f"Tables: {args.tables_output}")


if __name__ == "__main__":
    main()
