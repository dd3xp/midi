"""
Inference: Generate expressive f0 + amp from a MIDI file, then synthesize WAV.
Usage: python src/model/infer.py --midi path/to/file.mid --output output.wav
"""

import os
import sys
import argparse
import numpy as np
import torch
import pretty_midi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.model.baseline import BaselineModel, logits_to_f0
from src.model.encoder import MIDIEncoder
from src.model.diffusion import ConditionalDDPM, AmpPredictor, denormalize_f0, denormalize_amp
from src.model.dataset import ExpressionDataset


def midi_to_notes(midi_path, hop_time=0.01):
    """Convert MIDI file to notes array and frame features.
    Frame features match dataset.py format (7 dims):
        0: is_voiced, 1: pitch/127, 2: velocity/127,
        3: note_progress, 4: log1p(time_since_onset),
        5: is_onset, 6: is_offset
    """
    pm = pretty_midi.PrettyMIDI(midi_path)

    all_notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            all_notes.append([note.start, note.end, note.pitch, note.velocity])

    if not all_notes:
        raise ValueError("No notes found in MIDI file")

    all_notes.sort(key=lambda x: x[0])
    notes = np.array(all_notes, dtype=np.float32)

    total_time = notes[-1, 1] + 0.5
    n_frames = int(total_time / hop_time)
    frame_times = np.arange(n_frames) * hop_time

    ONSET_OFFSET_RADIUS = 2
    frame_features = np.zeros((n_frames, 7), dtype=np.float32)

    for onset, offset, pitch, vel in notes:
        start_frame = max(0, int(onset / hop_time))
        end_frame = min(n_frames, int(offset / hop_time))
        duration = offset - onset
        if duration <= 0:
            continue

        for f in range(start_frame, end_frame):
            frame_features[f, 0] = 1.0                    # is_voiced
            frame_features[f, 1] = pitch / 127.0           # normalized pitch
            frame_features[f, 2] = vel / 127.0             # normalized velocity
            t = frame_times[f] - onset
            frame_features[f, 3] = t / duration            # note progress
            frame_features[f, 4] = np.log1p(t)             # log time since onset

        onset_frame = int(onset / hop_time)
        for f in range(max(0, onset_frame - ONSET_OFFSET_RADIUS),
                       min(n_frames, onset_frame + ONSET_OFFSET_RADIUS + 1)):
            frame_features[f, 5] = 1.0                     # is_onset

        offset_frame = int(offset / hop_time)
        for f in range(max(0, offset_frame - ONSET_OFFSET_RADIUS),
                       min(n_frames, offset_frame + ONSET_OFFSET_RADIUS + 1)):
            frame_features[f, 6] = 1.0                     # is_offset

    return notes, frame_features, hop_time, n_frames


def synthesize_wav(f0_hz, amp, hop_time, sr=16000, output_path="output.wav"):
    """Synthesize WAV from f0 and amplitude using phase-continuous additive synthesis."""
    import scipy.io.wavfile as wavfile

    n_frames = len(f0_hz)
    samples_per_frame = int(hop_time * sr)
    total_samples = n_frames * samples_per_frame

    # Interpolate f0 and amp to sample rate for smooth synthesis
    frame_times = np.arange(n_frames) * hop_time
    sample_times = np.arange(total_samples) / sr

    f0_interp = np.interp(sample_times, frame_times, f0_hz)
    amp_interp = np.interp(sample_times, frame_times, amp)

    # Phase-continuous synthesis: integrate instantaneous frequency
    # phase[n] = sum of 2*pi*f[k]/sr for k=0..n-1
    instantaneous_phase = np.cumsum(2.0 * np.pi * f0_interp / sr)
    audio = amp_interp * np.sin(instantaneous_phase)

    # Zero out unvoiced regions
    audio[f0_interp <= 0] = 0.0

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.8

    wavfile.write(output_path, sr, (audio * 32767).astype(np.int16))
    print(f"WAV saved to {output_path} ({total_samples / sr:.1f}s, {sr}Hz)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate expressive performance from MIDI")
    parser.add_argument("--midi", type=str, required=True, help="Input MIDI file")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file")
    parser.add_argument("--checkpoint_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "experiments/checkpoints/exp039"))
    parser.add_argument("--baseline_checkpoint", type=str, default=None,
                        help="Baseline checkpoint (for encoder). Default: checkpoint_dir/baseline_best.pt")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--instrument", type=str, default="fl",
                        help="Instrument name for instrument-conditioned amp (default: fl)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Load MIDI
    print(f"Loading MIDI: {args.midi}")
    notes, frame_features, hop_time, n_frames = midi_to_notes(args.midi)
    print(f"  Notes: {len(notes)}, Frames: {n_frames}, Duration: {n_frames * hop_time:.1f}s")

    # Load models
    ckpt_dir = args.checkpoint_dir
    baseline_ckpt = args.baseline_checkpoint or os.path.join(ckpt_dir, "baseline_best.pt")

    # Encoder (from baseline)
    print("Loading encoder...")
    baseline = BaselineModel().to(device)
    baseline.load_state_dict(torch.load(baseline_ckpt, map_location=device, weights_only=True))
    encoder = baseline.get_encoder()
    encoder.eval()

    # Diffusion (f0)
    print("Loading diffusion model...")
    diffusion = ConditionalDDPM(n_channels=1).to(device)
    diff_ckpt = os.path.join(ckpt_dir, "diffusion_best_ema.pt")
    if os.path.isfile(diff_ckpt):
        diffusion.load_state_dict(torch.load(diff_ckpt, map_location=device, weights_only=True))
    diffusion.eval()

    # AmpPredictor
    print("Loading AmpPredictor...")
    # Detect config from checkpoint
    amp_ckpt = os.path.join(ckpt_dir, "amp_predictor_best.pt")
    amp_state = torch.load(amp_ckpt, map_location=device, weights_only=True)

    # Infer architecture from state dict
    has_inst = "inst_embed.weight" in amp_state
    has_attn = any("attn" in k for k in amp_state)
    n_instruments = amp_state["inst_embed.weight"].shape[0] if has_inst else 10
    inst_embed_dim = amp_state["inst_embed.weight"].shape[1] if has_inst else 32

    amp_predictor = AmpPredictor(
        instrument_conditioned=has_inst,
        n_instruments=n_instruments,
        inst_embed_dim=inst_embed_dim,
        use_attention=has_attn,
    ).to(device)
    amp_predictor.load_state_dict(amp_state)
    amp_predictor.eval()

    # Normalization stats
    norm_stats = torch.load(os.path.join(ckpt_dir, "norm_stats.pt"), map_location=device, weights_only=True)
    amp_mean = norm_stats.get("amp_mean", 0.0)
    amp_std = norm_stats.get("amp_std", 1.0)

    # Instrument ID
    ALL_INSTRUMENTS = ["vn", "va", "vc", "fl", "ob", "cl", "sax", "tpt", "tbn", "bn"]
    inst_name = args.instrument
    if inst_name in ALL_INSTRUMENTS:
        inst_id = ALL_INSTRUMENTS.index(inst_name)
    else:
        print(f"Warning: unknown instrument '{inst_name}', using fl (3)")
        inst_id = 3

    # Run inference
    print(f"Generating expression (instrument={inst_name}, eta={args.eta})...")
    ff_tensor = torch.from_numpy(frame_features).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode
        condition = encoder(ff_tensor)  # (1, T, 256)
        cond_t = condition.permute(0, 2, 1)  # (1, 256, T) for diffusion

        # Generate f0 via DDIM
        x_0 = diffusion.ddim_sample(cond_t, n_steps=args.ddim_steps, eta=args.eta)
        f0_norm = x_0[:, 0, :]  # (1, T)

        # Denormalize f0
        f0_norm_1d = f0_norm.squeeze(0)  # (T,)
        f0_hz = denormalize_f0(f0_norm_1d, notes, hop_time).cpu().numpy()

        # Generate amp
        inst_tensor = torch.tensor([inst_id], device=device)
        amp_kwargs = {}
        if has_inst:
            amp_kwargs["instrument_id"] = inst_tensor
        log_amp = amp_predictor(condition, **amp_kwargs).squeeze(-1)  # (1, T)
        # Denormalize: model outputs normalized log_amp, convert back to linear
        log_amp_raw = log_amp * amp_std + amp_mean
        amp_pred = torch.exp(log_amp_raw).squeeze(0).cpu().numpy()
        # Ensure positive
        amp_pred = np.maximum(amp_pred, 0.0)
        # For synthesis, normalize amp to reasonable range
        if amp_pred.max() > 0:
            amp_pred = amp_pred / amp_pred.max()

    # Zero out unvoiced frames
    voiced = frame_features[:, 1] > 0.5
    f0_hz[~voiced] = 0.0
    amp_pred[~voiced] = 0.0

    print(f"  f0 range: {f0_hz[voiced].min():.1f} - {f0_hz[voiced].max():.1f} Hz")
    print(f"  amp range: {amp_pred[voiced].min():.4f} - {amp_pred[voiced].max():.4f}")

    # Synthesize WAV
    print("Synthesizing WAV...")
    synthesize_wav(f0_hz, amp_pred, hop_time, output_path=args.output)
    print("Done!")


if __name__ == "__main__":
    main()
