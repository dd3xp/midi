"""
Dataset for MIDI-to-expression model training.
Loads .npz files, converts notes to frame-level features, supports random cropping.
"""

import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "datagen")

CROP_LEN = 512
ONSET_OFFSET_RADIUS = 3


def extract_piece_id(track_name):
    """Extract piece number from track folder name like '08_Spring_fl_vn_track1_fl'."""
    match = re.match(r"(\d+)_", track_name)
    if match:
        return match.group(1)
    return track_name


def notes_to_frame_features(notes, n_frames, hop_time):
    """Convert note array (N,4) to frame-level 7-dim features.

    Features per frame:
        0: is_voiced (0 or 1)
        1: normalized pitch (midi_pitch / 127)
        2: normalized velocity (velocity / 127)
        3: position_in_note (0 to 1)
        4: time_since_onset (log-transformed)
        5: is_onset (1 within ±RADIUS frames of note onset)
        6: is_offset (1 within ±RADIUS frames of note offset)
    """
    features = np.zeros((n_frames, 7), dtype=np.float32)
    frame_times = np.arange(n_frames) * hop_time

    for onset, offset, midi_pitch, velocity in notes:
        start_frame = int(onset / hop_time)
        end_frame = int(offset / hop_time)
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)
        if start_frame >= end_frame:
            continue

        duration = offset - onset
        if duration <= 0:
            continue

        for f in range(start_frame, end_frame):
            features[f, 0] = 1.0
            features[f, 1] = midi_pitch / 127.0
            features[f, 2] = velocity / 127.0
            t = frame_times[f] - onset
            features[f, 3] = t / duration
            features[f, 4] = np.log1p(t)
            features[f, 5] = 0.0
            features[f, 6] = 0.0

        onset_frame = int(onset / hop_time)
        for f in range(max(0, onset_frame - ONSET_OFFSET_RADIUS),
                       min(n_frames, onset_frame + ONSET_OFFSET_RADIUS + 1)):
            features[f, 5] = 1.0

        offset_frame = int(offset / hop_time)
        for f in range(max(0, offset_frame - ONSET_OFFSET_RADIUS),
                       min(n_frames, offset_frame + ONSET_OFFSET_RADIUS + 1)):
            features[f, 6] = 1.0

    return features


def f0_to_cent_bins(f0, notes, hop_time, n_bins=81):
    """Convert f0 (Hz) to classification bins relative to active note pitch.

    Returns array of bin indices (T,):
        bins 0-79: cent offsets from -200 to +195 in 5-cent steps
        bin 80: unvoiced
    """
    n_frames = len(f0)
    bins = np.full(n_frames, n_bins - 1, dtype=np.int64)  # default: unvoiced

    note_midi_per_frame = np.zeros(n_frames, dtype=np.float32)
    for onset, offset, midi_pitch, _ in notes:
        start_frame = max(0, int(onset / hop_time))
        end_frame = min(n_frames, int(offset / hop_time))
        for f in range(start_frame, end_frame):
            note_midi_per_frame[f] = midi_pitch

    for f in range(n_frames):
        if f0[f] <= 0 or note_midi_per_frame[f] <= 0:
            continue
        f0_midi = 69.0 + 12.0 * np.log2(f0[f] / 440.0)
        cent_offset = (f0_midi - note_midi_per_frame[f]) * 100.0
        cent_offset = np.clip(cent_offset, -200.0, 195.0)
        bin_idx = int((cent_offset + 200.0) / 5.0)
        bin_idx = np.clip(bin_idx, 0, n_bins - 2)
        bins[f] = bin_idx

    return bins


class ExpressionDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, instruments=("vn", "tpt", "fl"),
                 split="train", test_ratio=0.2, crop_len=CROP_LEN, seed=42):
        self.crop_len = crop_len
        self.split = split

        all_tracks = []
        for inst in instruments:
            inst_dir = os.path.join(data_dir, f"processed_{inst}")
            if not os.path.isdir(inst_dir):
                continue
            for track_dir in sorted(glob.glob(os.path.join(inst_dir, "*"))):
                npz_path = os.path.join(track_dir, "data.npz")
                if os.path.isfile(npz_path):
                    track_name = os.path.basename(track_dir)
                    piece_id = extract_piece_id(track_name)
                    all_tracks.append({
                        "path": npz_path,
                        "piece_id": piece_id,
                        "instrument": inst,
                    })

        piece_ids = sorted(set(t["piece_id"] for t in all_tracks))
        rng = np.random.RandomState(seed)
        rng.shuffle(piece_ids)
        n_test = max(1, int(len(piece_ids) * test_ratio))
        test_pieces = set(piece_ids[:n_test])
        train_pieces = set(piece_ids[n_test:])

        if split == "train":
            self.tracks = [t for t in all_tracks if t["piece_id"] in train_pieces]
        else:
            self.tracks = [t for t in all_tracks if t["piece_id"] in test_pieces]

        self._cache = {}

    def _load(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        track = self.tracks[idx]
        data = np.load(track["path"])
        notes = data["notes"].astype(np.float32)
        f0 = data["f0"].astype(np.float32)
        amp = data["amp"].astype(np.float32)
        hop_time = float(data["hop_time"])
        n_frames = len(f0)
        frame_features = notes_to_frame_features(notes, n_frames, hop_time)
        f0_bins = f0_to_cent_bins(f0, notes, hop_time)
        result = {
            "frame_features": frame_features,
            "f0": f0,
            "amp": amp,
            "f0_bins": f0_bins,
            "notes": notes,
            "hop_time": hop_time,
        }
        self._cache[idx] = result
        return result

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        data = self._load(idx)
        frame_features = data["frame_features"]
        f0 = data["f0"]
        amp = data["amp"]
        f0_bins = data["f0_bins"]
        notes = data["notes"]
        hop_time = data["hop_time"]
        n_frames = len(f0)

        if n_frames > self.crop_len:
            if self.split == "train":
                start = np.random.randint(0, n_frames - self.crop_len)
            else:
                start = 0
            end = start + self.crop_len
            frame_features = frame_features[start:end]
            f0 = f0[start:end]
            amp = amp[start:end]
            f0_bins = f0_bins[start:end]
        elif n_frames < self.crop_len:
            pad = self.crop_len - n_frames
            frame_features = np.pad(frame_features, ((0, pad), (0, 0)))
            f0 = np.pad(f0, (0, pad))
            amp = np.pad(amp, (0, pad))
            f0_bins = np.pad(f0_bins, (0, pad), constant_values=80)

        return {
            "frame_features": torch.from_numpy(frame_features),
            "f0": torch.from_numpy(f0),
            "amp": torch.from_numpy(amp),
            "f0_bins": torch.from_numpy(f0_bins),
            "notes": notes,
            "hop_time": hop_time,
        }


def collate_fn(batch):
    """Custom collate that handles variable-length notes."""
    return {
        "frame_features": torch.stack([b["frame_features"] for b in batch]),
        "f0": torch.stack([b["f0"] for b in batch]),
        "amp": torch.stack([b["amp"] for b in batch]),
        "f0_bins": torch.stack([b["f0_bins"] for b in batch]),
    }
