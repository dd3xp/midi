"""
TRIOS 数据预处理脚本
只处理非钢琴/非打击的单声部乐器。

输入: dataset/TRIOS Dataset/
输出: datagen/solo/TRIOS/processed_{inst}/{piece}_{inst_name}/data.npz
  格式与 URMP/Bach10 完全一致: notes(N,4), f0(T,), amp(T,), hop_time, sr

TRIOS 曲目和乐器:
  brahms:    horn, violin (+ piano skip)
  lussier:   bassoon, trumpet (+ piano skip)
  mozart:    clarinet, viola (+ piano skip)
  schubert:  cello, violin (+ piano skip)
  take_five: saxophone (+ piano, kick, ride, snare skip)
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import pretty_midi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.solo.preprocess import (
    extract_amp, estimate_velocity, synthesize,
    HOP_TIME, TARGET_SR,
)

TRIOS_DIR = os.path.join(PROJECT_ROOT, "dataset", "TRIOS")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "datagen", "solo", "TRIOS")

# TRIOS instrument name -> project abbreviation
INST_MAP = {
    "violin": "vn",
    "viola": "va",
    "cello": "vc",
    "clarinet": "cl",
    "horn": "hn",
    "flute": "fl",
    "oboe": "ob",
    "bassoon": "bn",
    "trumpet": "tpt",
    "trombone": "tbn",
    "saxophone": "sax",
}

# Skip piano, drums, and percussion
SKIP_INSTRUMENTS = {"piano", "drums", "bass", "kick", "ride", "snare", "mix"}


def extract_f0_pyin(audio, sr, hop_time):
    """Extract f0 using pYIN."""
    hop_length = int(sr * hop_time)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio, fmin=50, fmax=2000,
        sr=sr, hop_length=hop_length,
        fill_na=0.0
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    return f0


def midi_to_notes(midi_path):
    """Extract notes from MIDI. Returns (N, 4) [onset, offset, pitch, velocity]."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    all_notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            # Keep original velocity from MIDI (TRIOS has real velocity 45-75)
            all_notes.append([note.start, note.end, note.pitch, float(note.velocity)])
    if not all_notes:
        return np.zeros((0, 4), dtype=np.float32)
    all_notes.sort(key=lambda x: x[0])
    return np.array(all_notes, dtype=np.float32)


def process_track(piece_dir, piece_name, inst_name, inst_abbr):
    """Process one instrument track."""
    audio_path = os.path.join(piece_dir, f"{inst_name}.wav")
    midi_path = os.path.join(piece_dir, f"{inst_name}.mid")

    if not os.path.isfile(audio_path):
        return None
    if not os.path.isfile(midi_path):
        print(f"  [SKIP] No MIDI: {piece_name}/{inst_name}")
        return None

    # Load audio
    audio, orig_sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)

    # Extract f0 using pYIN
    f0 = extract_f0_pyin(audio, TARGET_SR, HOP_TIME)

    # Extract amp
    amp = extract_amp(audio, TARGET_SR, HOP_TIME)

    # Extract notes from MIDI (TRIOS has real velocity)
    notes = midi_to_notes(midi_path)
    if len(notes) == 0:
        print(f"  [SKIP] No notes: {piece_name}/{inst_name}")
        return None

    # Align lengths
    min_len = min(len(f0), len(amp))
    f0 = f0[:min_len]
    amp = amp[:min_len]

    # Keep MIDI velocity as-is (TRIOS has real performance velocity 45-75)
    # No need to override with RMS estimate

    # Output
    track_id = f"{piece_name}_{inst_name}"
    out_dir = os.path.join(OUTPUT_BASE, f"processed_{inst_abbr}", track_id)
    os.makedirs(out_dir, exist_ok=True)

    np.savez(
        os.path.join(out_dir, "data.npz"),
        notes=notes,
        f0=f0,
        amp=amp,
        hop_time=np.float32(HOP_TIME),
        sr=np.int32(TARGET_SR),
    )

    # Synth for verification
    synth_audio = synthesize(f0, amp, HOP_TIME, TARGET_SR)
    sf.write(os.path.join(out_dir, "synth.wav"), synth_audio, TARGET_SR)

    duration = len(f0) * HOP_TIME
    print(f"  [OK] {track_id}: {len(notes)} notes, {min_len} frames, {duration:.1f}s, "
          f"vel=[{notes[:,3].min():.0f}-{notes[:,3].max():.0f}]")
    return {
        "track_id": track_id,
        "instrument": inst_abbr,
        "n_notes": len(notes),
        "n_frames": min_len,
        "duration_sec": duration,
    }


def main():
    print("=" * 60)
    print("TRIOS 预处理 (solo instruments only, skip piano/drums)")
    print(f"输入: {TRIOS_DIR}")
    print(f"输出: {OUTPUT_BASE}")
    print("=" * 60)

    results = []
    for piece_name in sorted(os.listdir(TRIOS_DIR)):
        piece_dir = os.path.join(TRIOS_DIR, piece_name)
        if not os.path.isdir(piece_dir):
            continue

        print(f"\n--- {piece_name} ---")
        for fname in sorted(os.listdir(piece_dir)):
            if not fname.endswith(".wav"):
                continue
            inst_name = fname.replace(".wav", "")

            # Skip piano, drums, percussion, synth versions, mix
            if inst_name in SKIP_INSTRUMENTS:
                print(f"  [SKIP] {inst_name} (not solo instrument)")
                continue
            if "_syn" in inst_name:
                continue

            if inst_name not in INST_MAP:
                print(f"  [SKIP] Unknown instrument: {inst_name}")
                continue

            inst_abbr = INST_MAP[inst_name]
            result = process_track(piece_dir, piece_name, inst_name, inst_abbr)
            if result:
                results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"总计: {len(results)} 条音轨")
    if results:
        by_inst = {}
        for r in results:
            inst = r["instrument"]
            if inst not in by_inst:
                by_inst[inst] = []
            by_inst[inst].append(r)
        for inst, inst_results in sorted(by_inst.items()):
            total_dur = sum(r["duration_sec"] for r in inst_results)
            total_notes = sum(r["n_notes"] for r in inst_results)
            print(f"  {inst}: {len(inst_results)} tracks, {total_notes} notes, {total_dur:.1f}s ({total_dur/60:.1f}min)")

        total_dur = sum(r["duration_sec"] for r in results)
        total_notes = sum(r["n_notes"] for r in results)
        print(f"\n总时长: {total_dur:.1f}s ({total_dur/60:.1f}min)")
        print(f"总音符: {total_notes}")


if __name__ == "__main__":
    main()
