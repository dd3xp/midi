"""
Dataset for MIDI-to-expression model training.
Loads .npz files, converts notes to frame-level features, supports random cropping.
"""

import os
import re
import glob
import collections
import numpy as np
import torch
from torch.utils.data import Dataset

# Round 201 (V5.5): harmony / music-theory per-frame features (chord, key, phrase, cadence)
from src.data.solo.harmony_features import (
    HARMONY_FEATURE_DIMS,
    compute_harmony_features,
)

# Round 202 (V5.5): ADSR prior envelope for amplitude — REVERTED in Round 203
# from src.data.solo.instrument_adsr import compute_adsr_envelope


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "datagen", "solo", "URMP")
BACH10_DIR = os.path.join(PROJECT_ROOT, "datagen", "solo", "Bach10")
PHENICX_DIR = os.path.join(PROJECT_ROOT, "datagen", "solo", "PHENICX")
TRIOS_DIR = os.path.join(PROJECT_ROOT, "datagen", "solo", "TRIOS")
COCOCHORALES_DIR = os.path.join(PROJECT_ROOT, "datagen", "solo", "CocoChorales")

ALL_INSTRUMENTS = ("vn", "va", "vc", "fl", "ob", "cl", "sax", "tpt", "tbn", "bn")

INSTRUMENT_TO_ID = {
    "vn": 0, "va": 1, "vc": 2, "fl": 3, "ob": 4,
    "cl": 5, "sax": 6, "tpt": 7, "tbn": 8, "bn": 9,
    "db": 10, "hn": 11, "tba": 12, "erhu": 13,
}

CROP_LEN = 512
ONSET_OFFSET_RADIUS = 3


def _teacher_track_key(track_path):
    """Unique key for a track used by teacher_preds dict.

    Uses dataset + instrument folder + piece folder to avoid collisions when
    the same piece name appears under multiple instrument folders in Bach10,
    PHENICX, TRIOS (e.g. Bach10/processed_vn/07-HerrGott vs
    Bach10/processed_cl/07-HerrGott). Round 157 bug fix: previously used only
    `basename(dirname(track_path))`, which silently overwrote 15% of train
    tracks during precompute and injected wrong-instrument targets into
    distillation loss.

    Layout assumed:
        .../datagen/solo/<dataset>/processed_<inst>/<piece_folder>/data.npz
    Returns a string like "Bach10/processed_vn/07-HerrGott".
    """
    parts = os.path.normpath(track_path).split(os.sep)
    if len(parts) >= 4:
        return "/".join(parts[-4:-1])
    # Fallback for unexpected layouts: use full parent dir path
    return os.path.dirname(track_path)


def extract_piece_id(track_name, dataset_type="urmp"):
    """Extract piece NAME from track folder for split grouping.

    URMP:   '08_Spring_fl_vn_track1_fl' -> 'Spring'
    Bach10: '01-AchGottundHerr_vn'      -> 'AchGottundHerr'

    This ensures same-piece recordings are grouped together,
    preventing data leakage across train/test splits.
    """
    if dataset_type == "cocochorales":
        # Format: "random_track144006_1_violin" -> "coco_144006"
        m = re.match(r"random_track(\d+)_", track_name)
        return "coco_" + m.group(1) if m else "coco_" + track_name
    if dataset_type == "phenicx":
        # Format: "mozart_violin1" -> "mozart"
        return track_name.split("_")[0]
    if dataset_type == "trios":
        # Format: "mozart_clarinet" -> "mozart"
        return track_name.split("_")[0]
    if dataset_type == "bach10":
        # New format: piece dir is "01-AchGottundHerr" (no inst suffix)
        match = re.match(r"\d+-(.+)$", track_name)
        if match:
            return "b10_" + match.group(1)  # prefix to avoid URMP name collision
    else:
        match = re.match(r"\d+_([A-Za-z0-9]+)_", track_name)
        if match:
            return match.group(1)
    return track_name


def notes_to_frame_features(notes, n_frames, hop_time, n_features=6, instrument=None):
    """Convert note array (N,4) to frame-level features.

    Velocity has been removed: it was estimated from audio RMS which creates
    circular reasoning (model predicts amp from amp-derived velocity). Real MIDI
    files rarely have reliable velocity annotations anyway.

    Features per frame (6-dim base):
        0: is_voiced (0 or 1)
        1: normalized pitch (midi_pitch / 127)
        2: position_in_note (0 to 1)
        3: time_since_onset (log-transformed)
        4: is_onset (1 within ±RADIUS frames of note onset)
        5: is_offset (1 within ±RADIUS frames of note offset)

    Extended features (9-dim, when n_features=9):
        6: note_density (active notes in ±0.5s window, normalized)
        7: pitch_direction (pitch diff from previous note / 12, signed)
        8: interval_size (|pitch diff| / 24, normalized)

    Extended features (12-dim, when n_features=12):
        9: rest_before_note (log-transformed gap to previous note, phrase boundary signal)
        10: local_pitch_trend (±5 note window pitch regression slope / 12)
        11: rhythmic_regularity (IOI std / mean IOI in ±3 note window, rubato indicator)

    Velocity feature (13-dim, when n_features=13):
        12: velocity (note velocity / 127, from data file notes[:,3])
            NOTE: velocity comes from data file. URMP/Bach10/PHENICX have all-zero
            velocity; TRIOS has real MIDI velocity. May be RMS-estimated in some datasets.

    Instrument family features (15-dim, when n_features=15 and instrument is provided):
        12: is_string (1 if vn/va/vc, 0 otherwise)
        13: is_brass (1 if tpt/tbn, 0 otherwise)
        14: is_woodwind (1 if fl/ob/cl/sax/bn, 0 otherwise)

    Context features v2 (17-dim, when n_features=17):
        12: pitch_range_position ((pitch - min_pitch) / (max_pitch - min_pitch) for the track)
        13: phrase_position (position in phrase, 0 to 1; phrase = gap > 0.3s)
        14: note_duration_ratio (duration / mean_duration in ±5 notes, capped at 3.0)
        15: cumulative_pitch_direction (running same-direction pitch change / 12)
        16: articulation_ratio (sounding / (sounding + rest) in ±3 note window)

    Combined features (20-dim, when n_features=20):
        12-14: instrument family one-hot (same as 15-dim)
        15: pitch_range_position
        16: phrase_position
        17: note_duration_ratio
        18: cumulative_pitch_direction
        19: articulation_ratio
    """
    features = np.zeros((n_frames, n_features), dtype=np.float32)
    frame_times = np.arange(n_frames) * hop_time

    # Sort notes by onset for pitch_direction / interval_size computation
    if len(notes) > 0:
        sorted_indices = np.argsort(notes[:, 0])
        sorted_notes = notes[sorted_indices]
    else:
        sorted_notes = notes

    # Build per-note prev pitch info (for extended features)
    # note_prev_pitch[i] = pitch of previous note (by onset order), or 0 if first
    note_prev_pitch = np.zeros(len(sorted_notes), dtype=np.float32)
    for i in range(1, len(sorted_notes)):
        note_prev_pitch[i] = sorted_notes[i - 1, 2]  # midi_pitch of previous note

    for note_idx in range(len(sorted_notes)):
        onset, offset, midi_pitch, velocity = sorted_notes[note_idx]
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
            t = frame_times[f] - onset
            features[f, 2] = t / duration
            features[f, 3] = np.log1p(t)
            features[f, 4] = 0.0
            features[f, 5] = 0.0

            # Extended features (only when n_features >= 9)
            if n_features >= 9:
                prev_p = note_prev_pitch[note_idx]
                if prev_p > 0:
                    features[f, 7] = (midi_pitch - prev_p) / 12.0  # pitch_direction
                    features[f, 8] = abs(midi_pitch - prev_p) / 24.0  # interval_size
                # else: 0.0 (no previous note)

            # Velocity feature (only when n_features >= 13)
            if n_features >= 13:
                features[f, 12] = velocity / 127.0

        onset_frame = int(onset / hop_time)
        for f in range(max(0, onset_frame - ONSET_OFFSET_RADIUS),
                       min(n_frames, onset_frame + ONSET_OFFSET_RADIUS + 1)):
            features[f, 4] = 1.0

        offset_frame = int(offset / hop_time)
        for f in range(max(0, offset_frame - ONSET_OFFSET_RADIUS),
                       min(n_frames, offset_frame + ONSET_OFFSET_RADIUS + 1)):
            features[f, 5] = 1.0

    # Compute note_density for extended features
    if n_features >= 9 and len(sorted_notes) > 0:
        window = 0.5  # seconds
        max_density = 10.0
        for f in range(n_frames):
            ft = frame_times[f]
            # Count notes active within [ft - window, ft + window]
            count = np.sum(
                (sorted_notes[:, 0] <= ft + window) &
                (sorted_notes[:, 1] >= ft - window)
            )
            features[f, 6] = min(count / max_density, 1.0)

    # Compute 12-dim extended features (exp077+)
    if n_features >= 12 and len(sorted_notes) > 1:
        n_notes = len(sorted_notes)
        onsets = sorted_notes[:, 0]

        # Pre-compute per-note: rest_before, IOIs, pitch trend
        rest_before = np.zeros(n_notes, dtype=np.float32)
        for i in range(1, n_notes):
            gap = onsets[i] - sorted_notes[i - 1, 1]  # onset_i - offset_{i-1}
            rest_before[i] = np.log1p(max(gap, 0.0))

        # IOIs (inter-onset intervals)
        iois = np.zeros(n_notes, dtype=np.float32)
        for i in range(1, n_notes):
            iois[i] = onsets[i] - onsets[i - 1]

        # Local pitch trend: slope of pitch vs note_index in ±5 note window
        pitches = sorted_notes[:, 2]
        pitch_trend = np.zeros(n_notes, dtype=np.float32)
        for i in range(n_notes):
            lo = max(0, i - 5)
            hi = min(n_notes, i + 6)
            if hi - lo >= 3:
                x = np.arange(hi - lo, dtype=np.float32)
                y = pitches[lo:hi]
                # Linear regression slope
                x_mean = x.mean()
                y_mean = y.mean()
                denom = np.sum((x - x_mean) ** 2)
                if denom > 0:
                    pitch_trend[i] = np.sum((x - x_mean) * (y - y_mean)) / denom / 12.0

        # Rhythmic regularity: IOI std / mean in ±3 note window
        rhythmic_reg = np.zeros(n_notes, dtype=np.float32)
        for i in range(n_notes):
            lo = max(1, i - 3)  # start from 1 since iois[0] is 0
            hi = min(n_notes, i + 4)
            if hi - lo >= 2:
                local_iois = iois[lo:hi]
                local_iois = local_iois[local_iois > 0]
                if len(local_iois) >= 2:
                    mean_ioi = local_iois.mean()
                    if mean_ioi > 0:
                        rhythmic_reg[i] = local_iois.std() / mean_ioi

        # Map note-level features to frames
        for note_idx in range(n_notes):
            onset, offset, midi_pitch, velocity = sorted_notes[note_idx]
            start_frame = max(0, int(onset / hop_time))
            end_frame = min(n_frames, int(offset / hop_time))
            for f in range(start_frame, end_frame):
                features[f, 9] = rest_before[note_idx]
                features[f, 10] = pitch_trend[note_idx]
                features[f, 11] = rhythmic_reg[note_idx]

    # Instrument family features (exp149+): one-hot instrument family at dims 12-14
    # Active when n_features is 15-16 OR >= 20 (combined mode)
    if ((15 <= n_features <= 16) or n_features >= 20) and instrument is not None:
        STRING_INSTS = ("vn", "va", "vc", "erhu")
        BRASS_INSTS = ("tpt", "tbn")
        WOODWIND_INSTS = ("fl", "ob", "cl", "sax", "bn")
        if instrument in STRING_INSTS:
            features[:, 12] = 1.0
        elif instrument in BRASS_INSTS:
            features[:, 13] = 1.0
        elif instrument in WOODWIND_INSTS:
            features[:, 14] = 1.0

    # Context features v2 (exp148+): 5 improved note-level features
    # For n_features 17-19: at dims 12-16 (standalone)
    # For n_features >= 20: at dims 15-19 (after instrument family at 12-14)
    if (17 <= n_features <= 19 or n_features >= 20) and len(sorted_notes) > 1:
        ctx_base = 15 if n_features >= 20 else 12  # offset for combined mode
        n_notes = len(sorted_notes)
        onsets = sorted_notes[:, 0]
        offsets = sorted_notes[:, 1]
        pitches = sorted_notes[:, 2]
        durations = offsets - onsets
        durations = np.maximum(durations, 1e-6)  # avoid division by zero

        # Per-note context features
        pitch_range_pos = np.zeros(n_notes, dtype=np.float32)
        phrase_position = np.zeros(n_notes, dtype=np.float32)
        note_dur_ratio = np.ones(n_notes, dtype=np.float32)
        cumul_pitch_dir = np.zeros(n_notes, dtype=np.float32)
        articulation_ratio = np.zeros(n_notes, dtype=np.float32)

        # 1. pitch_range_position: where this note sits in the track's pitch range
        min_pitch = pitches.min()
        max_pitch = pitches.max()
        pitch_span = max_pitch - min_pitch
        if pitch_span > 0:
            pitch_range_pos = (pitches - min_pitch) / pitch_span
        else:
            pitch_range_pos[:] = 0.5

        # 2. phrase_position: detect phrases by rest gaps > 0.3s
        PHRASE_GAP = 0.3
        phrase_starts = [0]
        for i in range(1, n_notes):
            gap = onsets[i] - offsets[i - 1]
            if gap > PHRASE_GAP:
                phrase_starts.append(i)
        phrase_starts.append(n_notes)  # sentinel
        for p_idx in range(len(phrase_starts) - 1):
            p_start = phrase_starts[p_idx]
            p_end = phrase_starts[p_idx + 1]
            p_len = p_end - p_start
            for i in range(p_start, p_end):
                phrase_position[i] = (i - p_start) / max(p_len - 1, 1)

        # 3. note_duration_ratio: duration / mean_duration in ±5 note window
        for i in range(n_notes):
            lo = max(0, i - 5)
            hi = min(n_notes, i + 6)
            mean_dur = durations[lo:hi].mean()
            if mean_dur > 0:
                note_dur_ratio[i] = min(durations[i] / mean_dur, 3.0)

        # 4. cumulative_pitch_direction: running same-direction pitch change / 12
        # Tracks how far the melody has been moving in one direction (ascending/descending)
        # Resets when direction reverses. Captures musical phrase dynamics.
        cumul = 0.0
        prev_dir = 0  # 1=up, -1=down, 0=same
        for i in range(n_notes):
            if i == 0:
                cumul_pitch_dir[i] = 0.0
                continue
            diff = pitches[i] - pitches[i - 1]
            if abs(diff) < 0.5:  # same pitch
                cumul_pitch_dir[i] = cumul / 12.0
                continue
            cur_dir = 1 if diff > 0 else -1
            if cur_dir == prev_dir or prev_dir == 0:
                cumul += diff
            else:
                cumul = diff  # direction changed, reset
            prev_dir = cur_dir
            cumul_pitch_dir[i] = cumul / 12.0

        # 5. articulation_ratio: sounding / (sounding + rest) in ±3 note window
        for i in range(n_notes):
            lo = max(0, i - 3)
            hi = min(n_notes, i + 4)
            window_notes = sorted_notes[lo:hi]
            total_sounding = np.sum(window_notes[:, 1] - window_notes[:, 0])
            window_span = window_notes[-1, 1] - window_notes[0, 0]
            if window_span > 0:
                articulation_ratio[i] = min(total_sounding / window_span, 1.0)

        # Map to frames (ctx_base offsets: 12 for 17-dim, 15 for 20-dim)
        for note_idx in range(n_notes):
            onset, offset, midi_pitch, velocity = sorted_notes[note_idx]
            start_frame = max(0, int(onset / hop_time))
            end_frame = min(n_frames, int(offset / hop_time))
            for f in range(start_frame, end_frame):
                features[f, ctx_base + 0] = pitch_range_pos[note_idx]
                features[f, ctx_base + 1] = phrase_position[note_idx]
                features[f, ctx_base + 2] = note_dur_ratio[note_idx]
                features[f, ctx_base + 3] = cumul_pitch_dir[note_idx]
                features[f, ctx_base + 4] = articulation_ratio[note_idx]

    return features


def compute_pseudo_dynamics(notes, n_frames, hop_time):
    """Compute 4 rule-based pseudo dynamics features from pure MIDI info.

    These encode musicological priors about dynamics without using audio GT.
    All features are clamped to [-1, 1].

    Returns: (n_frames, 4) float32 array with:
        0: phrase_arch — arch shape within each phrase (0 at edges, 1 at center)
        1: melodic_charge — ascending intervals accumulate positive, descending negative
        2: pitch_height — (midi_pitch - 60) / 24, higher = louder tendency
        3: note_density — local onset density in ±2s window, normalized
    """
    features = np.zeros((n_frames, 4), dtype=np.float32)
    if len(notes) == 0:
        return features

    sorted_indices = np.argsort(notes[:, 0])
    sorted_notes = notes[sorted_indices]
    n_notes = len(sorted_notes)
    onsets = sorted_notes[:, 0]
    offsets = sorted_notes[:, 1]
    pitches = sorted_notes[:, 2]

    # --- 1. Phrase arch ---
    # Detect phrases by gaps > 0.5s (50 frames at 100fps)
    PHRASE_GAP = 0.5  # seconds
    phrase_starts = [0]
    for i in range(1, n_notes):
        gap = onsets[i] - offsets[i - 1]
        if gap > PHRASE_GAP:
            phrase_starts.append(i)
    phrase_starts.append(n_notes)  # sentinel

    # Per-note phrase arch value
    note_arch = np.zeros(n_notes, dtype=np.float32)
    for p_idx in range(len(phrase_starts) - 1):
        p_start = phrase_starts[p_idx]
        p_end = phrase_starts[p_idx + 1]
        p_len = p_end - p_start
        if p_len <= 1:
            note_arch[p_start:p_end] = 0.5
            continue
        for i in range(p_start, p_end):
            # Arch: sin(pi * position) — 0 at edges, 1 at center
            pos = (i - p_start) / (p_len - 1)  # 0 to 1
            note_arch[i] = np.sin(np.pi * pos)

    # --- 2. Melodic charge ---
    # Ascending intervals accumulate positive value, descending negative
    # Uses exponential decay factor per note
    DECAY = 0.95
    note_charge = np.zeros(n_notes, dtype=np.float32)
    charge = 0.0
    for i in range(n_notes):
        if i > 0:
            diff = pitches[i] - pitches[i - 1]
            charge = charge * DECAY + diff / 12.0  # normalize by octave
        note_charge[i] = np.clip(charge, -1.0, 1.0)

    # --- 3. Pitch height ---
    note_pitch_height = np.clip((pitches - 60.0) / 24.0, -1.0, 1.0)

    # --- 4. Note density ---
    # Count onsets within ±2s window for each note
    DENSITY_WINDOW = 2.0  # seconds
    note_density = np.zeros(n_notes, dtype=np.float32)
    for i in range(n_notes):
        t = onsets[i]
        count = np.sum((onsets >= t - DENSITY_WINDOW) & (onsets <= t + DENSITY_WINDOW))
        note_density[i] = count
    # Normalize: divide by max, scale to [0, 1], then shift to [-1, 1]
    max_density = max(note_density.max(), 1.0)
    note_density = note_density / max_density  # [0, 1]
    note_density = note_density * 2.0 - 1.0  # [-1, 1]

    # --- Broadcast note-level features to frames ---
    for note_idx in range(n_notes):
        onset = sorted_notes[note_idx, 0]
        offset = sorted_notes[note_idx, 1]
        start_frame = max(0, int(onset / hop_time))
        end_frame = min(n_frames, int(offset / hop_time))
        for f in range(start_frame, end_frame):
            features[f, 0] = note_arch[note_idx]
            features[f, 1] = note_charge[note_idx]
            features[f, 2] = note_pitch_height[note_idx]
            features[f, 3] = note_density[note_idx]

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
    def __init__(self, data_dir=DATA_DIR, bach10_dir=None,
                 phenicx_dir=None, trios_dir=None,
                 cocochorales_dir=None, slakh_dir=None,
                 eep_dir=None, ccom_huqin_dir=None,
                 filosax_dir=None, quartet_dir=None,
                 instruments=ALL_INSTRUMENTS,
                 split="train", test_ratio=0.2, crop_len=CROP_LEN, seed=42,
                 samples_per_epoch=None, encoder_input_dim=6,
                 pitch_augment=False, pitch_augment_shifts=None,
                 max_cache_size=0,
                 augment=False, augment_pitch_shift_range=3,
                 augment_tempo_range=(0.8, 1.2),
                 augment_amp_jitter_std=0.0,
                 augment_prob=1.0,
                 augment_pitch_prob=-1.0,
                 augment_tempo_prob=-1.0,
                 augment_amp_scale_range=0.0,
                 augment_amp_scale_prob=0.0,
                 augment_note_jitter_frames=0,
                 augment_note_jitter_prob=0.0,
                 pseudo_velocity=False,
                 pseudo_velocity_dropout=0.0,
                 mixup_prob=0.0, mixup_alpha=0.4,
                 mixup_same_family=True,
                 exclude_tracks=None,
                 pseudo_dynamics=False,
                 teacher_preds_path=None,
                 stacking_preds_path=None,
                 use_f0_deviation=False,
                 use_score_velocity=False,
                 score_velocity_path=None,
                 hubert_embeddings_path=None,
                 hubert_dim=32,
                 slow_amp_path=None,
                 harmony_feature_set="none",
                 amp_target_key=None,
                 amp_targets_path=None,
                 amp_prior_path=None,
                 amp_prior_features="none",
                 adsr_prior_path=None):
        self.crop_len = crop_len
        self.split = split
        self.samples_per_epoch = samples_per_epoch  # None = len(tracks)
        self.encoder_input_dim = encoder_input_dim
        self.pseudo_dynamics = pseudo_dynamics
        self.pseudo_velocity = pseudo_velocity
        self.use_f0_deviation = use_f0_deviation
        # R248: score-level MIDI velocity as 21st feature (from source MIDI, NOT audio-derived)
        self.use_score_velocity = use_score_velocity
        self._score_velocity = None
        if use_score_velocity:
            import torch as _torch
            sv_path = score_velocity_path or os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'experiments', 'checkpoints', 'amp_targets_score_vel.pt')
            if os.path.isfile(sv_path):
                self._score_velocity = _torch.load(sv_path, map_location='cpu', weights_only=False)
                print(f'Loaded score-velocity curves: {len(self._score_velocity)} tracks')
            else:
                print(f'WARNING: use_score_velocity=True but {sv_path} not found')
        # Round 201 (V5.5): harmony / music-theory features
        if harmony_feature_set not in HARMONY_FEATURE_DIMS:
            raise ValueError(
                f"harmony_feature_set must be one of {list(HARMONY_FEATURE_DIMS.keys())}, "
                f"got {harmony_feature_set!r}"
            )
        self.harmony_feature_set = harmony_feature_set
        self.harmony_feature_dim = HARMONY_FEATURE_DIMS[harmony_feature_set]
        # Round 221 (V8): alternate amp target for target representation sweep
        self.amp_target_key = amp_target_key  # e.g. "amp_gauss50", "amp_rank", None=raw
        self._amp_targets = None
        if amp_targets_path is not None and os.path.isfile(amp_targets_path):
            import torch as _torch
            raw = _torch.load(amp_targets_path, map_location="cpu", weights_only=False)
            self._amp_targets = raw
            print(f"Loaded amp targets: {len(raw)} tracks, target_key={amp_target_key}")
        elif amp_target_key is not None and amp_target_key != "amp_raw":
            print(f"WARNING: amp_target_key={amp_target_key} but amp_targets_path not found: {amp_targets_path}")
        # exp144: dropout probability for pseudo_velocity dim during training
        # When dropped, dim 12 is set to 0, teaching model to work without velocity
        self.pseudo_velocity_dropout = pseudo_velocity_dropout if split == "train" else 0.0
        self.pitch_augment = pitch_augment and (split == "train")
        self.pitch_augment_shifts = pitch_augment_shifts or [-2, -1, 1, 2]
        # Online augmentation (exp104+): random pitch shift + tempo scaling per __getitem__
        self.augment = augment and (split == "train")
        self.augment_pitch_shift_range = augment_pitch_shift_range
        self.augment_tempo_range = augment_tempo_range
        self.augment_amp_jitter_std = augment_amp_jitter_std
        self.augment_prob = augment_prob
        # exp263a: per-type augmentation probabilities (-1.0 = use augment_prob)
        self.augment_pitch_prob = augment_pitch_prob
        self.augment_tempo_prob = augment_tempo_prob
        self.augment_amp_scale_range = augment_amp_scale_range
        self.augment_amp_scale_prob = augment_amp_scale_prob if split == "train" else 0.0
        self.augment_note_jitter_frames = augment_note_jitter_frames
        self.augment_note_jitter_prob = augment_note_jitter_prob if split == "train" else 0.0
        # Mixup (exp175+): amp-only mixup with same-family pairing
        self.mixup_prob = mixup_prob if split == "train" else 0.0
        self.mixup_alpha = mixup_alpha
        self.mixup_same_family = mixup_same_family

        all_tracks = []

        # Load URMP tracks: datagen/solo/URMP/processed_{inst}/{track}/data.npz
        for inst in instruments:
            inst_dir = os.path.join(data_dir, f"processed_{inst}")
            if not os.path.isdir(inst_dir):
                continue
            for track_dir in sorted(glob.glob(os.path.join(inst_dir, "*"))):
                npz_path = os.path.join(track_dir, "data.npz")
                if os.path.isfile(npz_path):
                    track_name = os.path.basename(track_dir)
                    piece_id = extract_piece_id(track_name, "urmp")
                    all_tracks.append({
                        "path": npz_path,
                        "piece_id": piece_id,
                        "instrument": inst,
                    })

        # Load Bach10 tracks: datagen/solo/Bach10/processed_{inst}/{piece}/data.npz
        if bach10_dir is not None and os.path.isdir(bach10_dir):
            for inst in instruments:
                inst_dir = os.path.join(bach10_dir, f"processed_{inst}")
                if not os.path.isdir(inst_dir):
                    continue
                for track_dir in sorted(glob.glob(os.path.join(inst_dir, "*"))):
                    npz_path = os.path.join(track_dir, "data.npz")
                    if not os.path.isfile(npz_path):
                        continue
                    dir_name = os.path.basename(track_dir)
                    piece_id = extract_piece_id(dir_name, "bach10")
                    all_tracks.append({
                        "path": npz_path,
                        "piece_id": "bach10_" + piece_id,
                        "instrument": inst,
                    })

        # Load PHENICX tracks: datagen/solo/PHENICX/processed_{inst}/{piece}/data.npz
        if phenicx_dir is not None and os.path.isdir(phenicx_dir):
            for inst in instruments:
                inst_dir = os.path.join(phenicx_dir, f"processed_{inst}")
                if not os.path.isdir(inst_dir):
                    continue
                for track_dir in sorted(glob.glob(os.path.join(inst_dir, "*"))):
                    npz_path = os.path.join(track_dir, "data.npz")
                    if not os.path.isfile(npz_path):
                        continue
                    dir_name = os.path.basename(track_dir)
                    piece_id = extract_piece_id(dir_name, "phenicx")
                    all_tracks.append({
                        "path": npz_path,
                        "piece_id": "phenicx_" + piece_id,
                        "instrument": inst,
                    })

        # Load TRIOS tracks: datagen/solo/TRIOS/processed_{inst}/{piece}/data.npz
        if trios_dir is not None and os.path.isdir(trios_dir):
            for inst in instruments:
                inst_dir = os.path.join(trios_dir, f"processed_{inst}")
                if not os.path.isdir(inst_dir):
                    continue
                for track_dir in sorted(glob.glob(os.path.join(inst_dir, "*"))):
                    npz_path = os.path.join(track_dir, "data.npz")
                    if not os.path.isfile(npz_path):
                        continue
                    dir_name = os.path.basename(track_dir)
                    piece_id = extract_piece_id(dir_name, "trios")
                    all_tracks.append({
                        "path": npz_path,
                        "piece_id": "trios_" + piece_id,
                        "instrument": inst,
                    })

        # Load EEP tracks: datagen/solo/EEP/processed_{inst}/{piece}/data.npz
        if eep_dir is not None and os.path.isdir(eep_dir):
            n_before = len(all_tracks)
            for inst in instruments:
                inst_dir = os.path.join(eep_dir, f"processed_{inst}")
                if not os.path.isdir(inst_dir):
                    continue
                for track_dir in sorted(glob.glob(os.path.join(inst_dir, "*"))):
                    npz_path = os.path.join(track_dir, "data.npz")
                    if not os.path.isfile(npz_path):
                        continue
                    dir_name = os.path.basename(track_dir)
                    all_tracks.append({
                        "path": npz_path,
                        "piece_id": "eep_" + dir_name,
                        "instrument": inst,
                    })
            print(f"EEP: loaded {len(all_tracks) - n_before} tracks")

        # Load CCOM-HuQin tracks: datagen/solo/CCOM-HuQin/processed_{inst}/{piece}/data.npz
        if ccom_huqin_dir is not None and os.path.isdir(ccom_huqin_dir):
            n_before = len(all_tracks)
            for inst in instruments:
                inst_dir = os.path.join(ccom_huqin_dir, f"processed_{inst}")
                if not os.path.isdir(inst_dir):
                    continue
                for track_dir in sorted(glob.glob(os.path.join(inst_dir, "*"))):
                    npz_path = os.path.join(track_dir, "data.npz")
                    if not os.path.isfile(npz_path):
                        continue
                    dir_name = os.path.basename(track_dir)
                    all_tracks.append({
                        "path": npz_path,
                        "piece_id": "ccom_" + dir_name,
                        "instrument": inst,
                    })
            print(f"CCOM-HuQin: loaded {len(all_tracks) - n_before} tracks")

        # Load QUARTET tracks: datagen/solo/QUARTET/processed_{inst}/{track}/data.npz
        if quartet_dir is not None and os.path.isdir(quartet_dir):
            n_before = len(all_tracks)
            for inst in instruments:
                inst_dir = os.path.join(quartet_dir, f"processed_{inst}")
                if not os.path.isdir(inst_dir):
                    continue
                for track_dir in sorted(glob.glob(os.path.join(inst_dir, "*"))):
                    npz_path = os.path.join(track_dir, "data.npz")
                    if not os.path.isfile(npz_path):
                        continue
                    dir_name = os.path.basename(track_dir)
                    all_tracks.append({
                        "path": npz_path,
                        "piece_id": "quartet_" + dir_name,
                        "instrument": inst,
                    })
            print(f"QUARTET: loaded {len(all_tracks) - n_before} tracks")

        # Load CocoChorales tracks: datagen/solo/CocoChorales/processed_{inst}/{track}/data.npz
        if cocochorales_dir is not None and os.path.isdir(cocochorales_dir):
            n_before = len(all_tracks)
            for inst in instruments:
                inst_dir = os.path.join(cocochorales_dir, f"processed_{inst}")
                if not os.path.isdir(inst_dir):
                    continue
                for track_dir in sorted(glob.glob(os.path.join(inst_dir, "*"))):
                    npz_path = os.path.join(track_dir, "data.npz")
                    if not os.path.isfile(npz_path):
                        continue
                    dir_name = os.path.basename(track_dir)
                    piece_id = extract_piece_id(dir_name, "cocochorales")
                    all_tracks.append({
                        "path": npz_path,
                        "piece_id": piece_id,
                        "instrument": inst,
                    })
            print(f"CocoChorales: loaded {len(all_tracks) - n_before} tracks")

        # Load Slakh2100 tracks: datagen/solo/Slakh2100/processed_{inst}/{track}/data.npz
        if slakh_dir is not None and os.path.isdir(slakh_dir):
            n_before = len(all_tracks)
            for inst in instruments:
                inst_dir = os.path.join(slakh_dir, f"processed_{inst}")
                if not os.path.isdir(inst_dir):
                    continue
                for track_dir in sorted(glob.glob(os.path.join(inst_dir, "*"))):
                    npz_path = os.path.join(track_dir, "data.npz")
                    if not os.path.isfile(npz_path):
                        continue
                    dir_name = os.path.basename(track_dir)
                    # Slakh dir_name = "Track0xxxx_Sxx"; piece_id = the Track id (so all stems
                    # of same track go to same train/test split)
                    piece_id = "slakh_" + dir_name.split("_")[0]
                    all_tracks.append({
                        "path": npz_path,
                        "piece_id": piece_id,
                        "instrument": inst,
                    })
            print(f"Slakh2100: loaded {len(all_tracks) - n_before} tracks")

        # Load Filosax tracks: datagen/solo/Filosax/processed_sax/{track}/data.npz
        if filosax_dir is not None and os.path.isdir(filosax_dir):
            n_before = len(all_tracks)
            for inst in instruments:
                inst_dir = os.path.join(filosax_dir, f"processed_{inst}")
                if not os.path.isdir(inst_dir):
                    continue
                for track_dir in sorted(glob.glob(os.path.join(inst_dir, "*"))):
                    npz_path = os.path.join(track_dir, "data.npz")
                    if not os.path.isfile(npz_path):
                        continue
                    dir_name = os.path.basename(track_dir)
                    all_tracks.append({
                        "path": npz_path,
                        "piece_id": "filosax_" + dir_name,
                        "instrument": inst,
                    })
            print(f"Filosax: loaded {len(all_tracks) - n_before} tracks")

        # Mark each track as real or synthetic for weighted sampling
        # CocoChorales and Filosax are treated as "external" for weighting
        for t in all_tracks:
            pid = t.get("piece_id", "")
            t["is_real"] = not (pid.startswith("coco_") or pid.startswith("filosax_"))

        # Round 201 (V5.5): build {track_path -> [sibling_paths]} BEFORE the
        # train/test split. Sibling tracks share a piece_id. Harmony features
        # (chord/key) are pooled across siblings — this is MIDI-derivable and
        # introduces no audio leakage since the feature is causal w.r.t. score.
        self._piece_sibling_paths = {}
        if self.harmony_feature_set != "none":
            by_piece = {}
            for t in all_tracks:
                by_piece.setdefault(t["piece_id"], []).append(t["path"])
            for t in all_tracks:
                self._piece_sibling_paths[t["path"]] = by_piece[t["piece_id"]]

        piece_ids = sorted(set(t["piece_id"] for t in all_tracks))
        rng = np.random.RandomState(seed)
        rng.shuffle(piece_ids)
        # [C5 fix, Round 159] Previously `max(1, int(...))` forced >=1 test piece
        # even when test_ratio=0.0, leaking 1 track that depended on shuffle seed.
        n_test = int(round(len(piece_ids) * test_ratio))
        if n_test > 0:
            test_pieces = set(piece_ids[:n_test])
            train_pieces = set(piece_ids[n_test:])
        else:
            test_pieces = set()
            train_pieces = set(piece_ids)

        if split == "train":
            self.tracks = [t for t in all_tracks if t["piece_id"] in train_pieces]
        else:
            self.tracks = [t for t in all_tracks if t["piece_id"] in test_pieces]

        # Exclude specific tracks by basename (exp196+: data cleaning)
        if exclude_tracks:
            exclude_set = set(exclude_tracks)
            n_before = len(self.tracks)
            self.tracks = [t for t in self.tracks
                           if os.path.basename(os.path.dirname(t["path"])) not in exclude_set
                           and os.path.basename(t["path"]).replace(".npz", "") not in exclude_set]
            n_after = len(self.tracks)
            if n_before != n_after:
                print(f"Excluded {n_before - n_after} tracks (from {n_before} to {n_after})")

        # Pitch augmentation: add shifted copies of each track (train only)
        if self.pitch_augment:
            augmented = []
            for t in self.tracks:
                for shift in self.pitch_augment_shifts:
                    aug_track = dict(t)
                    aug_track["pitch_shift"] = shift
                    augmented.append(aug_track)
            n_orig = len(self.tracks)
            self.tracks.extend(augmented)
            print(f"Pitch augmentation: {n_orig} -> {len(self.tracks)} tracks "
                  f"(shifts={self.pitch_augment_shifts})")

        # Build is_real list for weighted sampling (after augmentation)
        self.is_real = [t.get("is_real", True) for t in self.tracks]

        # Build family index for mixup pairing (exp175+)
        if self.mixup_prob > 0:
            STRING_INSTS = {"vn", "va", "vc", "erhu"}
            BRASS_INSTS = {"tpt", "tbn"}
            WOODWIND_INSTS = {"fl", "ob", "cl", "sax", "bn"}
            self._family_indices = {}  # family_name -> list of track indices
            for i, t in enumerate(self.tracks):
                inst = t.get("instrument", "")
                if inst in STRING_INSTS:
                    fam = "string"
                elif inst in BRASS_INSTS:
                    fam = "brass"
                elif inst in WOODWIND_INSTS:
                    fam = "woodwind"
                else:
                    fam = "other"
                self._family_indices.setdefault(fam, []).append(i)
            # Map each track index to its family
            self._track_family = {}
            for fam, indices in self._family_indices.items():
                for i in indices:
                    self._track_family[i] = fam

        # exp217/exp220: Knowledge distillation — load teacher predictions keyed by
        # <dataset>/processed_<inst>/<piece_folder> (see _teacher_track_key). Round 157
        # bug fix: the old basename-only key collided across instruments in Bach10/
        # PHENICX/TRIOS and silently corrupted 15% of training samples.
        self._teacher_preds = None
        if teacher_preds_path is not None and os.path.isfile(teacher_preds_path):
            raw = torch.load(teacher_preds_path, map_location="cpu", weights_only=False)
            self._teacher_preds = {}
            for k, v in raw.items():
                self._teacher_preds[k] = v.float() if isinstance(v, torch.Tensor) else torch.from_numpy(v).float()
            n_unique = len(self._teacher_preds)
            n_matched = sum(1 for t in self.tracks
                            if _teacher_track_key(t["path"]) in self._teacher_preds)
            print(f"Loaded teacher preds: {n_unique} unique entries, "
                  f"matched {n_matched}/{len(self.tracks)} dataset tracks "
                  f"(split={self.split})")
            if self.split == "train":
                # Only enforce on training set — precompute only covers train tracks.
                missing = [_teacher_track_key(t["path"]) for t in self.tracks
                           if _teacher_track_key(t["path"]) not in self._teacher_preds]
                assert not missing, (
                    f"teacher_preds missing {len(missing)} training tracks; "
                    f"first few: {missing[:5]}"
                )

        # exp232: Residual stacking — load base model predictions as extra input features.
        # Predictions are appended as the last feature dimension(s) in frame_features.
        # The stacking .pt file has the same format as teacher_preds:
        #   {track_key: tensor(T,) of amp_pred (linear scale)}
        self._stacking_preds = None
        if stacking_preds_path is not None and os.path.isfile(stacking_preds_path):
            raw = torch.load(stacking_preds_path, map_location="cpu", weights_only=False)
            self._stacking_preds = {}
            for k, v in raw.items():
                self._stacking_preds[k] = v.float() if isinstance(v, torch.Tensor) else torch.from_numpy(v).float()
            n_matched = sum(1 for t in self.tracks
                            if _teacher_track_key(t["path"]) in self._stacking_preds)
            print(f"Loaded stacking preds: {len(self._stacking_preds)} entries, "
                  f"matched {n_matched}/{len(self.tracks)} tracks (split={self.split})")

        # Round 165 (exp244b): HuBERT track-level embeddings as extra frame features.
        # Each track gets a fixed D-dim vector broadcast to every frame.
        self._hubert_embeddings = None
        self._hubert_dim = hubert_dim
        if hubert_embeddings_path is not None and os.path.isfile(hubert_embeddings_path):
            raw = torch.load(hubert_embeddings_path, map_location="cpu", weights_only=False)
            emb_dict = raw.get("embeddings", raw)  # handle both dict-of-tensors and wrapped format
            self._hubert_embeddings = {}
            for k, v in emb_dict.items():
                self._hubert_embeddings[k] = v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            n_matched = sum(1 for t in self.tracks
                            if _teacher_track_key(t["path"]) in self._hubert_embeddings)
            sample_dim = next(iter(self._hubert_embeddings.values())).shape[0]
            print(f"Loaded HuBERT embeddings: {len(self._hubert_embeddings)} entries, "
                  f"matched {n_matched}/{len(self.tracks)} tracks (split={self.split}), "
                  f"dim={sample_dim}")

        # Round 166 (exp250): Precomputed slow amp for Y decomposition.
        # Loaded like teacher_preds: dict of track_key -> np.array(T,).
        self._slow_amp = None
        if slow_amp_path is not None and os.path.isfile(slow_amp_path):
            raw = torch.load(slow_amp_path, map_location="cpu", weights_only=False)
            self._slow_amp = {}
            for k, v in raw.items():
                if isinstance(v, torch.Tensor):
                    self._slow_amp[k] = v.numpy().astype(np.float32)
                else:
                    self._slow_amp[k] = np.asarray(v, dtype=np.float32)
            n_matched = sum(1 for t in self.tracks
                            if _teacher_track_key(t["path"]) in self._slow_amp)
            print(f"Loaded slow amp: {len(self._slow_amp)} entries, "
                  f"matched {n_matched}/{len(self.tracks)} tracks (split={self.split})")

        # Round 229 (V10 Part 2): Per-instrument amp priors as input features
        # amp_prior_features: "none", "inst_mean", "inst_pitch", "inst_envelope", "all"
        self._amp_prior_data = None
        self._amp_prior_features = amp_prior_features
        self._amp_prior_extra_dim = 0
        if amp_prior_path is not None and amp_prior_features != "none" and os.path.isfile(amp_prior_path):
            import torch as _torch
            self._amp_prior_data = _torch.load(amp_prior_path, map_location="cpu", weights_only=False)
            # Compute extra dims based on which priors are used
            if amp_prior_features == "inst_mean":
                self._amp_prior_extra_dim = 1
            elif amp_prior_features == "inst_pitch":
                self._amp_prior_extra_dim = 1
            elif amp_prior_features == "inst_envelope":
                self._amp_prior_extra_dim = 1
            elif amp_prior_features == "all":
                self._amp_prior_extra_dim = 3
            print(f"Loaded amp priors: features={amp_prior_features}, extra_dim={self._amp_prior_extra_dim}")

        # Round 230: ADSR analytical prior as extra input feature (+1 dim)
        self._adsr_prior_data = None
        self._adsr_prior_extra_dim = 0
        if adsr_prior_path is not None and os.path.isfile(adsr_prior_path):
            import torch as _torch
            raw = _torch.load(adsr_prior_path, map_location="cpu", weights_only=False)
            self._adsr_prior_data = {
                "envelopes": {k: (v.numpy() if isinstance(v, _torch.Tensor) else np.asarray(v, dtype=np.float32))
                              for k, v in raw["envelopes"].items()},
                "norm_mean": float(raw["norm_mean"]),
                "norm_std": float(raw["norm_std"]),
            }
            self._adsr_prior_extra_dim = 1
            n_matched = sum(1 for t in all_tracks
                            if _teacher_track_key(t["path"]) in self._adsr_prior_data["envelopes"])
            print(f"Loaded ADSR prior: {len(self._adsr_prior_data['envelopes'])} entries, "
                  f"matched {n_matched}/{len(all_tracks)} tracks, "
                  f"norm_mean={self._adsr_prior_data['norm_mean']:.4f}, "
                  f"norm_std={self._adsr_prior_data['norm_std']:.4f}")

        # Preload: max_cache_size=-1 means preload ALL data into memory at init
        # (server has 1TB RAM, 160K npz files ~few GB total)
        # max_cache_size=0 means unlimited lazy cache (backward compatible)
        # max_cache_size>0 means LRU cache with eviction
        self._max_cache_size = max_cache_size
        self._cache = {}

        if max_cache_size == -1:
            print(f"Preloading {len(self.tracks)} tracks into memory...")
            import time
            t0 = time.time()
            for i in range(len(self.tracks)):
                self._preload_one(i)
                if (i + 1) % 10000 == 0:
                    elapsed = time.time() - t0
                    print(f"  Preloaded {i+1}/{len(self.tracks)} tracks ({elapsed:.1f}s)")
            elapsed = time.time() - t0
            print(f"  Preload complete: {len(self.tracks)} tracks in {elapsed:.1f}s")

    def _compute_track_data(self, idx):
        """Load and compute features for a single track."""
        track = self.tracks[idx]
        data = np.load(track["path"])
        notes = data["notes"].astype(np.float32)
        f0 = data["f0"].astype(np.float32)
        amp = data["amp"].astype(np.float32)
        hop_time = float(data["hop_time"])
        n_frames = len(f0)

        # Apply pitch shift if this is an augmented track
        pitch_shift = track.get("pitch_shift", 0)
        if pitch_shift != 0:
            notes = notes.copy()
            notes[:, 2] = notes[:, 2] + pitch_shift

        frame_features = notes_to_frame_features(notes, n_frames, hop_time,
                                                    n_features=self.encoder_input_dim,
                                                    instrument=track.get("instrument"))

        # exp232: Residual stacking — append base model prediction as extra feature dim.
        # The prediction is log-transformed and roughly normalized.
        if self._stacking_preds is not None:
            track_key = _teacher_track_key(self.tracks[idx]["path"])
            pred = self._stacking_preds.get(track_key)
            if pred is not None:
                pred_np = pred.numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
                if len(pred_np) > n_frames:
                    pred_np = pred_np[:n_frames]
                elif len(pred_np) < n_frames:
                    pred_np = np.pad(pred_np, (0, n_frames - len(pred_np)))
                # Log-transform and normalize: log(pred) is roughly in [-16, 0]
                log_pred = np.log(np.clip(pred_np, 1e-7, None)).astype(np.float32)
                log_pred_norm = (log_pred + 8.0) / 8.0  # center near 0, range ~[-1, 1]
                frame_features[:, 20] = log_pred_norm  # fill dim 20 (assumes encoder_input_dim >= 21)

        # Pseudo-velocity (exp142): compute onset amp from GT amp for each note,
        # broadcast to all frames of that note, use as dim 12 (replacing file velocity).
        # This gives the model note-level loudness info derived from actual audio.
        if self.pseudo_velocity and self.encoder_input_dim >= 13 and len(notes) > 0:
            sorted_indices = np.argsort(notes[:, 0])
            sorted_notes = notes[sorted_indices]
            pseudo_vel = np.zeros(n_frames, dtype=np.float32)
            # Collect onset amps for normalization
            onset_amps = []
            note_ranges = []
            for note_idx in range(len(sorted_notes)):
                onset, offset, midi_pitch, _ = sorted_notes[note_idx]
                onset_frame = int(onset / hop_time)
                onset_frame = max(0, min(n_frames - 1, onset_frame))
                # Use small window around onset for robustness (±2 frames)
                win_lo = max(0, onset_frame - 2)
                win_hi = min(n_frames, onset_frame + 3)
                onset_amp = float(np.mean(amp[win_lo:win_hi]))
                onset_amps.append(onset_amp)
                start_frame = max(0, int(onset / hop_time))
                end_frame = min(n_frames, int(offset / hop_time))
                note_ranges.append((start_frame, end_frame))
            # Normalize onset amps to 0~1 range
            onset_amps = np.array(onset_amps, dtype=np.float32)
            amp_min = onset_amps.min()
            amp_max = onset_amps.max()
            if amp_max - amp_min > 1e-8:
                onset_amps_norm = (onset_amps - amp_min) / (amp_max - amp_min)
            else:
                onset_amps_norm = np.full_like(onset_amps, 0.5)
            # Broadcast to frames
            for note_idx, (start_f, end_f) in enumerate(note_ranges):
                pseudo_vel[start_f:end_f] = onset_amps_norm[note_idx]
            frame_features[:, 12] = pseudo_vel

        # exp209: Pseudo dynamics — 4 rule-based features appended at the end
        if self.pseudo_dynamics:
            pd_feats = compute_pseudo_dynamics(notes, n_frames, hop_time)  # (n_frames, 4)
            frame_features = np.concatenate([frame_features, pd_feats], axis=1)

        # Round 201 (V5.5): harmony / music-theory features appended after pseudo_dynamics
        if self.harmony_feature_set != "none":
            siblings = self._piece_sibling_paths.get(
                self.tracks[idx]["path"], [self.tracks[idx]["path"]]
            )
            h_feats = compute_harmony_features(
                self.tracks[idx]["path"], siblings, n_frames, hop_time,
                self.harmony_feature_set,
            )
            assert h_feats.shape == (n_frames, self.harmony_feature_dim), (
                f"harmony feature shape mismatch: got {h_feats.shape}, "
                f"expected ({n_frames}, {self.harmony_feature_dim})"
            )
            frame_features = np.concatenate([frame_features, h_feats], axis=1)

        # R248: score-velocity from source MIDI as extra feature (dim 20)
        # Normalised to [0,1] by dividing by 127.
        if self.use_score_velocity and self.encoder_input_dim >= 21 and self._score_velocity is not None:
            track_key = _teacher_track_key(self.tracks[idx]["path"])
            curve = self._score_velocity.get(track_key)
            if curve is not None:
                curve_np = curve.numpy() if isinstance(curve, torch.Tensor) else np.asarray(curve)
                if len(curve_np) >= n_frames:
                    curve_np = curve_np[:n_frames]
                else:
                    curve_np = np.pad(curve_np, (0, n_frames - len(curve_np)), mode='edge')
                frame_features[:, 20] = (curve_np.astype(np.float32) / 127.0)

        # exp243f: GT f0 deviation from MIDI pitch as extra feature (dim 20)
        if self.use_f0_deviation and self.encoder_input_dim >= 21:
            # Compute per-frame MIDI pitch from note annotations
            midi_pitch_per_frame = np.zeros(n_frames, dtype=np.float32)
            voiced_mask = np.zeros(n_frames, dtype=bool)
            for note in notes:
                onset, offset, pitch, _ = note
                s = max(0, int(onset / hop_time))
                e = min(n_frames, int(offset / hop_time))
                midi_pitch_per_frame[s:e] = pitch
                voiced_mask[s:e] = True
            # f0 -> MIDI pitch in semitones
            safe_f0 = np.clip(f0, 1.0, None)
            f0_midi = 69.0 + 12.0 * np.log2(safe_f0 / 440.0)
            # Deviation in cents
            cent_dev = (f0_midi - midi_pitch_per_frame) * 100.0
            # Normalize to [-1, 1] by clipping at ±200 cents
            f0_dev_norm = np.clip(cent_dev / 200.0, -1.0, 1.0).astype(np.float32)
            f0_dev_norm[~voiced_mask] = 0.0
            f0_dev_norm[f0 <= 0] = 0.0
            frame_features[:, 20] = f0_dev_norm

        # Round 165 (exp244b): Append HuBERT track-level embedding to each frame.
        # The embedding is a fixed D-dim vector broadcast to all T frames → (T, D).
        if self._hubert_embeddings is not None:
            track_key = _teacher_track_key(self.tracks[idx]["path"])
            hubert_emb = self._hubert_embeddings.get(track_key)
            if hubert_emb is not None:
                # Broadcast (D,) -> (T, D)
                hubert_tile = np.tile(hubert_emb.astype(np.float32), (n_frames, 1))
                frame_features = np.concatenate([frame_features, hubert_tile], axis=1)
            else:
                # Track not found — pad with zeros
                frame_features = np.concatenate(
                    [frame_features, np.zeros((n_frames, self._hubert_dim), dtype=np.float32)],
                    axis=1)

        # Round 229 (V10 Part 2): Append amp prior features
        if self._amp_prior_data is not None and self._amp_prior_features != "none":
            prior_data = self._amp_prior_data
            norm = prior_data["norm_stats"]
            inst = track["instrument"]
            n_envelope_bins = prior_data.get("n_envelope_bins", 20)

            prior_channels = []

            # Sort notes for envelope lookup
            if len(notes) > 0:
                sorted_idx_prior = np.argsort(notes[:, 0])
                sorted_notes_prior = notes[sorted_idx_prior]
            else:
                sorted_notes_prior = notes

            need_inst_mean = self._amp_prior_features in ("inst_mean", "all")
            need_inst_pitch = self._amp_prior_features in ("inst_pitch", "all")
            need_envelope = self._amp_prior_features in ("inst_envelope", "all")

            if need_inst_mean:
                # Prior 1: per-instrument mean log-amp, z-score normalized
                raw_val = prior_data["inst_mean"].get(inst, 0.0)
                normed = (raw_val - norm["prior1_mean"]) / norm["prior1_std"]
                ch = np.full(n_frames, normed, dtype=np.float32)
                prior_channels.append(ch)

            if need_inst_pitch:
                # Prior 2: per-(instrument, octave), z-score normalized
                ch = np.zeros(n_frames, dtype=np.float32)
                for note_idx in range(len(sorted_notes_prior)):
                    onset, offset, midi_pitch, _ = sorted_notes_prior[note_idx]
                    s = max(0, int(onset / hop_time))
                    e = min(n_frames, int(offset / hop_time))
                    octave = int(midi_pitch) // 12
                    raw_val = prior_data["inst_pitch"].get(
                        (inst, octave),
                        prior_data["inst_mean"].get(inst, 0.0)
                    )
                    normed = (raw_val - norm["prior2_mean"]) / norm["prior2_std"]
                    ch[s:e] = normed
                prior_channels.append(ch)

            if need_envelope:
                # Prior 3: per-instrument envelope, z-score normalized
                ch = np.zeros(n_frames, dtype=np.float32)
                envelope = prior_data["inst_envelope"].get(
                    inst, np.zeros(n_envelope_bins, dtype=np.float32)
                )
                for note_idx in range(len(sorted_notes_prior)):
                    onset, offset, _, _ = sorted_notes_prior[note_idx]
                    s = max(0, int(onset / hop_time))
                    e = min(n_frames, int(offset / hop_time))
                    dur = e - s
                    if dur < 1:
                        continue
                    for f in range(s, e):
                        pos = (f - s) / max(dur - 1, 1)
                        bin_idx = min(int(pos * n_envelope_bins), n_envelope_bins - 1)
                        raw_val = envelope[bin_idx]
                        normed = (raw_val - norm["prior3_mean"]) / norm["prior3_std"]
                        ch[f] = normed
                prior_channels.append(ch)

            if prior_channels:
                prior_arr = np.stack(prior_channels, axis=1)  # (n_frames, n_priors)
                frame_features = np.concatenate([frame_features, prior_arr], axis=1)

        # Round 230: ADSR analytical prior as extra input feature (+1 dim, z-scored)
        if self._adsr_prior_data is not None:
            track_key = _teacher_track_key(self.tracks[idx]["path"])
            adsr_env = self._adsr_prior_data["envelopes"].get(track_key)
            if adsr_env is not None:
                if len(adsr_env) > n_frames:
                    adsr_env = adsr_env[:n_frames]
                elif len(adsr_env) < n_frames:
                    adsr_env = np.pad(adsr_env, (0, n_frames - len(adsr_env)),
                                      constant_values=self._adsr_prior_data["norm_mean"])
                # z-score normalize using global stats
                adsr_normed = ((adsr_env - self._adsr_prior_data["norm_mean"])
                               / self._adsr_prior_data["norm_std"]).astype(np.float32)
            else:
                adsr_normed = np.zeros(n_frames, dtype=np.float32)
            frame_features = np.concatenate(
                [frame_features, adsr_normed[:, None]], axis=1)

        f0_bins = f0_to_cent_bins(f0, notes, hop_time)

        # Round 203 (V5.5): note_to_frame_map for hierarchical note-level supervision
        # Maps each frame to its note index (-1 for unvoiced frames)
        note_to_frame_map = np.full(n_frames, -1, dtype=np.int64)
        if len(notes) > 0:
            sorted_idx = np.argsort(notes[:, 0])
            sorted_notes_for_map = notes[sorted_idx]
            for i in range(len(sorted_notes_for_map)):
                onset, offset = sorted_notes_for_map[i, 0], sorted_notes_for_map[i, 1]
                start_f = max(0, int(onset / hop_time))
                end_f = min(n_frames, int(offset / hop_time))
                for f in range(start_f, end_f):
                    note_to_frame_map[f] = i

        # Round 206 (V6): per-piece log-amp statistics for normalization ablation
        eps_logamp = 1e-7
        voiced_piece = f0 > 0
        log_amp_full = np.log(amp + eps_logamp)
        if voiced_piece.sum() > 0:
            voiced_log = log_amp_full[voiced_piece]
            piece_logamp_mean = float(voiced_log.mean())
            piece_logamp_std = float(max(voiced_log.std(), 1e-6))
            piece_logamp_min = float(voiced_log.min())
            piece_logamp_max = float(voiced_log.max())
            piece_logamp_median = float(np.median(voiced_log))
            q75, q25 = float(np.percentile(voiced_log, 75)), float(np.percentile(voiced_log, 25))
            piece_logamp_iqr = max(q75 - q25, 1e-6)
        else:
            piece_logamp_mean = 0.0
            piece_logamp_std = 1.0
            piece_logamp_min = -10.0
            piece_logamp_max = 0.0
            piece_logamp_median = 0.0
            piece_logamp_iqr = 1.0

        result = {
            "frame_features": frame_features,
            "f0": f0,
            "amp": amp,
            "f0_bins": f0_bins,
            "notes": notes,
            "hop_time": hop_time,
            "instrument": track["instrument"],
            "instrument_id": INSTRUMENT_TO_ID.get(track["instrument"], 0),
            "note_to_frame_map": note_to_frame_map,
            # Round 206: per-piece log-amp stats for normalization ablation
            "piece_logamp_mean": piece_logamp_mean,
            "piece_logamp_std": piece_logamp_std,
            "piece_logamp_min": piece_logamp_min,
            "piece_logamp_max": piece_logamp_max,
            "piece_logamp_median": piece_logamp_median,
            "piece_logamp_iqr": piece_logamp_iqr,
        }

        # Round 166 (exp250): Include precomputed slow amp if available
        if self._slow_amp is not None:
            track_key = _teacher_track_key(track["path"])
            amp_slow = self._slow_amp.get(track_key)
            if amp_slow is not None:
                # Align length with amp
                if len(amp_slow) > n_frames:
                    amp_slow = amp_slow[:n_frames]
                elif len(amp_slow) < n_frames:
                    amp_slow = np.pad(amp_slow, (0, n_frames - len(amp_slow)))
                result["amp_slow"] = amp_slow
            else:
                # Fallback: use original amp as slow (no filtering)
                result["amp_slow"] = amp.copy()

        # Round 221 (V8): alternate amp target from precomputed targets
        if self._amp_targets is not None and self.amp_target_key is not None:
            track_key = _teacher_track_key(track["path"])
            track_targets = self._amp_targets.get(track_key)
            if track_targets is not None and self.amp_target_key in track_targets:
                amp_target = np.asarray(track_targets[self.amp_target_key], dtype=np.float32)
                if len(amp_target) > n_frames:
                    amp_target = amp_target[:n_frames]
                elif len(amp_target) < n_frames:
                    amp_target = np.pad(amp_target, (0, n_frames - len(amp_target)))
                result["amp_target"] = amp_target
            else:
                # Fallback: use raw amp as target
                result["amp_target"] = amp.copy()
        elif self.amp_target_key == "amp_raw" or self.amp_target_key is None:
            # Control: target == raw amp
            result["amp_target"] = amp.copy()

        return result

    def _preload_one(self, idx):
        """Preload a single track into cache (used during init)."""
        self._cache[idx] = self._compute_track_data(idx)

    def _load(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        result = self._compute_track_data(idx)
        self._cache[idx] = result
        # Evict oldest entries if LRU mode (max_cache_size > 0)
        if self._max_cache_size > 0 and len(self._cache) > self._max_cache_size:
            # Remove a random key (simpler than OrderedDict for dict)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        return result

    def __len__(self):
        if self.samples_per_epoch is not None:
            return self.samples_per_epoch
        return len(self.tracks)

    def __getitem__(self, idx):
        real_idx = idx % len(self.tracks)
        data = self._load(real_idx)
        notes = data["notes"]
        f0 = data["f0"]
        amp = data["amp"]
        f0_bins = data["f0_bins"]
        hop_time = data["hop_time"]
        n_frames = len(f0)
        # Online augmentation (exp104+): modify notes, recompute frame features
        # exp263a: per-type probabilities. If augment_pitch_prob >= 0, use it;
        # otherwise fall back to augment_prob (backward compatible).
        if self.augment:
            notes_modified = False
            pitch_prob = self.augment_pitch_prob if self.augment_pitch_prob >= 0 else self.augment_prob
            tempo_prob = self.augment_tempo_prob if self.augment_tempo_prob >= 0 else self.augment_prob
            notes = notes.copy()

            # Pitch shift: ±N semitones (only affects input features, not f0/amp GT)
            if np.random.random() < pitch_prob:
                shift = np.random.randint(-self.augment_pitch_shift_range,
                                           self.augment_pitch_shift_range + 1)
                notes[:, 2] = np.clip(notes[:, 2] + shift, 0, 127)
                notes_modified = True

            # Tempo augmentation: scale onset/offset times
            if np.random.random() < tempo_prob:
                tempo_factor = np.random.uniform(*self.augment_tempo_range)
                notes[:, 0] *= tempo_factor
                notes[:, 1] *= tempo_factor
                notes_modified = True

            # exp263a: Note boundary jitter (±N frames random offset per note)
            if self.augment_note_jitter_frames > 0 and np.random.random() < self.augment_note_jitter_prob:
                jf = self.augment_note_jitter_frames
                for i in range(len(notes)):
                    onset_jitter = np.random.randint(-jf, jf + 1) * hop_time
                    offset_jitter = np.random.randint(-jf, jf + 1) * hop_time
                    notes[i, 0] = max(0, notes[i, 0] + onset_jitter)
                    notes[i, 1] = max(notes[i, 0] + hop_time, notes[i, 1] + offset_jitter)
                notes_modified = True

            if notes_modified:
                # Recompute frame features with augmented notes (f0/amp/f0_bins unchanged)
                frame_features = notes_to_frame_features(
                    notes, n_frames, hop_time, n_features=self.encoder_input_dim,
                    instrument=data.get("instrument"))
                # Recompute pseudo dynamics for augmented notes
                if self.pseudo_dynamics:
                    pd_feats = compute_pseudo_dynamics(notes, n_frames, hop_time)
                    frame_features = np.concatenate([frame_features, pd_feats], axis=1)
                # Round 201: harmony features are score-level context, NOT altered by
                # local pitch/tempo augmentation → reuse the cached columns from the
                # original _compute_track_data output (last harmony_feature_dim cols).
                if self.harmony_feature_set != "none" and self.harmony_feature_dim > 0:
                    cached_ff = data["frame_features"]
                    h_feats = cached_ff[:, -self.harmony_feature_dim:]
                    frame_features = np.concatenate([frame_features, h_feats], axis=1)
            else:
                frame_features = data["frame_features"]

            # Amp jitter: add Gaussian noise to amp target (exp137+)
            if self.augment_amp_jitter_std > 0:
                amp = amp.copy()
                amp += np.random.randn(*amp.shape).astype(np.float32) * self.augment_amp_jitter_std
                amp = np.clip(amp, 0.0, None)  # amp must be non-negative

            # exp263a: Amp scaling — global offset in log space (simulates recording level)
            if self.augment_amp_scale_range > 0 and np.random.random() < self.augment_amp_scale_prob:
                amp = amp.copy()
                log_offset = np.random.uniform(-self.augment_amp_scale_range, self.augment_amp_scale_range)
                amp = amp * np.exp(log_offset).astype(np.float32)

            # DO NOT recompute f0_bins — keep original GT labels unchanged
        else:
            frame_features = data["frame_features"]

        # Round 166 (exp250): Load amp_slow if available
        amp_slow = data.get("amp_slow")

        # Round 221 (V8): Load alternate amp target if available
        amp_target = data.get("amp_target")

        # Round 203: note_to_frame_map for hierarchical supervision
        note_to_frame_map = data["note_to_frame_map"]

        crop_start = 0
        if n_frames > self.crop_len:
            if self.split == "train":
                crop_start = np.random.randint(0, n_frames - self.crop_len)
            else:
                crop_start = 0
            end = crop_start + self.crop_len
            frame_features = frame_features[crop_start:end]
            f0 = f0[crop_start:end]
            amp = amp[crop_start:end]
            f0_bins = f0_bins[crop_start:end]
            note_to_frame_map = note_to_frame_map[crop_start:end]
            if amp_slow is not None:
                amp_slow = amp_slow[crop_start:end]
            if amp_target is not None:
                amp_target = amp_target[crop_start:end]
        elif n_frames < self.crop_len:
            pad = self.crop_len - n_frames
            frame_features = np.pad(frame_features, ((0, pad), (0, 0)))
            f0 = np.pad(f0, (0, pad))
            amp = np.pad(amp, (0, pad))
            f0_bins = np.pad(f0_bins, (0, pad), constant_values=80)
            note_to_frame_map = np.pad(note_to_frame_map, (0, pad), constant_values=-1)
            if amp_slow is not None:
                amp_slow = np.pad(amp_slow, (0, pad))
            if amp_target is not None:
                amp_target = np.pad(amp_target, (0, pad))

        # Round 203: Remap note indices to crop-local and compute per-note mean amp
        # After cropping, note_to_frame_map may have non-contiguous indices (e.g. 5,6,7)
        # Remap to 0,1,2 and compute note_mean_amp for these crop-local notes
        unique_notes = np.unique(note_to_frame_map)
        unique_notes = unique_notes[unique_notes >= 0]  # exclude -1 (unvoiced)
        n_crop_notes = len(unique_notes)
        # Build remap: old_index -> new_index
        remap = np.full(note_to_frame_map.max() + 2 if len(note_to_frame_map) > 0 else 1,
                        -1, dtype=np.int64)
        for new_idx, old_idx in enumerate(unique_notes):
            remap[old_idx] = new_idx
        # Apply remap to note_to_frame_map
        remapped = np.full_like(note_to_frame_map, -1)
        valid_mask = note_to_frame_map >= 0
        if valid_mask.any():
            remapped[valid_mask] = remap[note_to_frame_map[valid_mask]]
        note_to_frame_map = remapped
        # Compute per-note mean log-amp from cropped amp
        # When per_crop_amp_norm is used, subtract the per-crop voiced mean so that
        # note-level targets are in the same normalized space as frame-level loss.
        # This avoids scale mismatch: encoder hidden states encode relative dynamics
        # but note targets were absolute log-amp. (Methodology fix W1)
        eps = 1e-7
        note_mean_amp = np.zeros(max(n_crop_notes, 1), dtype=np.float32)
        note_valid_mask = np.zeros(max(n_crop_notes, 1), dtype=np.float32)
        log_amp_crop = np.log(amp + eps)
        # Per-crop voiced mean for normalization (matches compute_baseline_loss behavior)
        voiced_frames = (note_to_frame_map >= 0)  # voiced = assigned to a note
        if voiced_frames.any():
            crop_voiced_mean = log_amp_crop[voiced_frames].mean()
        else:
            crop_voiced_mean = 0.0
        for ni in range(n_crop_notes):
            mask = (note_to_frame_map == ni)
            if mask.any():
                # Normalized: subtract per-crop voiced mean (consistent with per_crop_amp_norm)
                note_mean_amp[ni] = log_amp_crop[mask].mean() - crop_voiced_mean
                note_valid_mask[ni] = 1.0
        # exp144: pseudo_velocity dropout — zero out dim 12 with probability p
        if (self.pseudo_velocity and self.pseudo_velocity_dropout > 0
                and self.encoder_input_dim >= 13
                and np.random.random() < self.pseudo_velocity_dropout):
            frame_features = frame_features.copy()
            frame_features[:, 12] = 0.0

        # exp175: Amp-only mixup with same-family pairing
        # Only mix amp target and continuous input features; DO NOT mix f0_bins
        if self.mixup_prob > 0 and np.random.random() < self.mixup_prob:
            # Find a partner track from the same instrument family
            if self.mixup_same_family:
                fam = self._track_family.get(real_idx, "other")
                candidates = self._family_indices.get(fam, [])
                if len(candidates) > 1:
                    partner_idx = real_idx
                    while partner_idx == real_idx:
                        partner_idx = candidates[np.random.randint(len(candidates))]
                else:
                    partner_idx = np.random.randint(len(self.tracks))
            else:
                partner_idx = np.random.randint(len(self.tracks))

            # Load partner data
            partner_data = self._load(partner_idx)
            p_ff = partner_data["frame_features"]
            p_amp = partner_data["amp"]
            p_f0_bins = partner_data["f0_bins"]
            p_n_frames = len(partner_data["f0"])

            # Crop partner (random crop for train)
            if p_n_frames > self.crop_len:
                p_start = np.random.randint(0, p_n_frames - self.crop_len)
                p_end = p_start + self.crop_len
                p_ff = p_ff[p_start:p_end]
                p_amp = p_amp[p_start:p_end]
                p_f0_bins = p_f0_bins[p_start:p_end]
            elif p_n_frames < self.crop_len:
                pad = self.crop_len - p_n_frames
                p_ff = np.pad(p_ff, ((0, pad), (0, 0)))
                p_amp = np.pad(p_amp, (0, pad))
                p_f0_bins = np.pad(p_f0_bins, (0, pad), constant_values=80)

            # Sample mixup lambda from Beta distribution
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            # Mix amp target only (f0_bins stays from original — classification target)
            frame_features = frame_features.copy()
            amp = amp * lam + p_amp * (1 - lam)

            # Mix continuous input features (dims 0-11, 15-19 for 20-dim)
            # but NOT f0_bins (discrete), and NOT instrument family one-hot (dims 12-14)
            mixed_ff = frame_features.copy()
            # Mix base continuous features: dims 0-5 (is_voiced, pitch, position, etc.)
            for d in range(min(6, self.encoder_input_dim)):
                mixed_ff[:, d] = frame_features[:, d] * lam + p_ff[:, d] * (1 - lam)
            # Mix extended features dims 6-11 if present
            for d in range(6, min(12, self.encoder_input_dim)):
                mixed_ff[:, d] = frame_features[:, d] * lam + p_ff[:, d] * (1 - lam)
            # Skip dims 12-14 (instrument family one-hot — keep original)
            # Mix context features v2 dims 15-19 if present
            for d in range(15, min(20, self.encoder_input_dim)):
                mixed_ff[:, d] = frame_features[:, d] * lam + p_ff[:, d] * (1 - lam)
            frame_features = mixed_ff

        # exp217/exp220: Knowledge distillation — load teacher prediction for this track and crop
        teacher_pred = None
        teacher_valid = 0
        if self._teacher_preds is not None:
            track_key = _teacher_track_key(self.tracks[real_idx]["path"])
            t_full = self._teacher_preds.get(track_key)
            if t_full is not None:
                t_full_np = t_full.numpy() if isinstance(t_full, torch.Tensor) else np.asarray(t_full)
                # Apply same crop as f0/amp (n_frames may have been padded)
                if len(t_full_np) >= crop_start + self.crop_len:
                    teacher_pred = t_full_np[crop_start:crop_start + self.crop_len].astype(np.float32)
                    teacher_valid = 1
                else:
                    # Pad to crop_len
                    src = t_full_np[crop_start:crop_start + self.crop_len].astype(np.float32)
                    pad = self.crop_len - len(src)
                    teacher_pred = np.concatenate([src, np.zeros(pad, dtype=np.float32)])
                    teacher_valid = 1
        if teacher_pred is None:
            teacher_pred = np.zeros(self.crop_len, dtype=np.float32)

        result = {
            "frame_features": torch.from_numpy(frame_features),
            "f0": torch.from_numpy(f0),
            "amp": torch.from_numpy(amp),
            "f0_bins": torch.from_numpy(f0_bins),
            "notes": notes,
            "hop_time": hop_time,
            "instrument_id": data["instrument_id"],
            "track_idx": real_idx,
            "crop_start": crop_start,
            "teacher_pred": torch.from_numpy(teacher_pred),
            "teacher_valid": teacher_valid,
            # Round 203: note_to_frame_map and note-level amp for hierarchical supervision
            "note_to_frame_map": torch.from_numpy(note_to_frame_map),
            "note_mean_amp": torch.from_numpy(note_mean_amp),
            "note_valid_mask": torch.from_numpy(note_valid_mask),
            # Round 206: per-piece log-amp stats for normalization ablation
            "piece_logamp_mean": data["piece_logamp_mean"],
            "piece_logamp_std": data["piece_logamp_std"],
            "piece_logamp_min": data["piece_logamp_min"],
            "piece_logamp_max": data["piece_logamp_max"],
            "piece_logamp_median": data["piece_logamp_median"],
            "piece_logamp_iqr": data["piece_logamp_iqr"],
        }
        # Round 166 (exp250): Include amp_slow in batch
        if amp_slow is not None:
            result["amp_slow"] = torch.from_numpy(amp_slow)
        # Round 221 (V8): Include alternate amp target in batch
        if amp_target is not None:
            result["amp_target"] = torch.from_numpy(amp_target)
        return result


def collate_fn(batch):
    """Custom collate that handles variable-length notes."""
    out = {
        "frame_features": torch.stack([b["frame_features"] for b in batch]),
        "f0": torch.stack([b["f0"] for b in batch]),
        "amp": torch.stack([b["amp"] for b in batch]),
        "f0_bins": torch.stack([b["f0_bins"] for b in batch]),
        "instrument_id": torch.tensor([b["instrument_id"] for b in batch], dtype=torch.long),
        "track_idx": torch.tensor([b["track_idx"] for b in batch], dtype=torch.long),
        "crop_start": torch.tensor([b["crop_start"] for b in batch], dtype=torch.long),
    }
    if "teacher_pred" in batch[0]:
        out["teacher_pred"] = torch.stack([b["teacher_pred"] for b in batch])
        out["teacher_valid"] = torch.tensor([b["teacher_valid"] for b in batch], dtype=torch.float32)
    # Round 166 (exp250): slow amp target
    if "amp_slow" in batch[0]:
        out["amp_slow"] = torch.stack([b["amp_slow"] for b in batch])
    # Round 221 (V8): alternate amp target
    if "amp_target" in batch[0]:
        out["amp_target"] = torch.stack([b["amp_target"] for b in batch])
    # Round 203: note_to_frame_map (same crop_len for all, stackable)
    if "note_to_frame_map" in batch[0]:
        out["note_to_frame_map"] = torch.stack([b["note_to_frame_map"] for b in batch])
        # note_mean_amp and note_valid_mask have variable lengths — pad to max
        max_notes = max(b["note_mean_amp"].shape[0] for b in batch)
        note_mean_amp_padded = torch.zeros(len(batch), max_notes)
        note_valid_padded = torch.zeros(len(batch), max_notes)
        for i, b in enumerate(batch):
            n = b["note_mean_amp"].shape[0]
            note_mean_amp_padded[i, :n] = b["note_mean_amp"]
            note_valid_padded[i, :n] = b["note_valid_mask"]
        out["note_mean_amp"] = note_mean_amp_padded
        out["note_valid_mask"] = note_valid_padded
    # Round 206: per-piece log-amp stats for normalization ablation
    if "piece_logamp_mean" in batch[0]:
        out["piece_logamp_mean"] = torch.tensor([b["piece_logamp_mean"] for b in batch], dtype=torch.float32)
        out["piece_logamp_std"] = torch.tensor([b["piece_logamp_std"] for b in batch], dtype=torch.float32)
        out["piece_logamp_min"] = torch.tensor([b["piece_logamp_min"] for b in batch], dtype=torch.float32)
        out["piece_logamp_max"] = torch.tensor([b["piece_logamp_max"] for b in batch], dtype=torch.float32)
        out["piece_logamp_median"] = torch.tensor([b["piece_logamp_median"] for b in batch], dtype=torch.float32)
        out["piece_logamp_iqr"] = torch.tensor([b["piece_logamp_iqr"] for b in batch], dtype=torch.float32)
    return out
