"""
PHENICX-Anechoic 数据预处理脚本
将分离的单乐器音轨处理为 (notes, f0, amp) 三元组。

输入: dataset/PHENICX-Anechoic/
输出: datagen/solo/PHENICX/processed_{inst}/{piece}_{inst_name}/data.npz
  格式与 URMP/Bach10 完全一致: notes(N,4), f0(T,), amp(T,), hop_time, sr

PHENICX 特点:
- 音频文件按编号命名 (violin1.wav, horn2.wav)，对应同一声部的多个演奏者
- 标注文件按声部命名 (violin.txt, horn.txt)，多个演奏者共享同一标注
- 标注格式: onset,offset,note_name (txt)
- 有些声部是复音的（如弦乐组齐奏时有和弦），需跳过
- 无 f0 标注，使用 pYIN 从音频提取

乐器名 -> 缩写映射:
  violin -> vn, viola -> va, cello -> vc, doublebass -> db,
  flute -> fl, oboe -> ob, clarinet -> cl, bassoon -> bn,
  horn -> hn, trumpet -> tpt
"""

import os
import sys
import re
import numpy as np
import librosa
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.solo.preprocess import (
    extract_amp, estimate_velocity, synthesize,
    HOP_TIME, TARGET_SR,
)

PHENICX_DIR = os.path.join(PROJECT_ROOT, "dataset", "PHENICX")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "datagen", "solo", "PHENICX")

# PHENICX instrument base name -> project abbreviation
INST_BASE_MAP = {
    "violin": "vn",
    "viola": "va",
    "cello": "vc",
    "doublebass": "db",
    "flute": "fl",
    "oboe": "ob",
    "clarinet": "cl",
    "bassoon": "bn",
    "horn": "hn",
    "trumpet": "tpt",
}

PIECES = ["mozart", "beethoven", "bruckner", "mahler"]


def get_inst_base(inst_name):
    """Strip trailing digits from instrument name. e.g. 'violin3' -> 'violin'."""
    return re.sub(r'\d+$', '', inst_name)


def get_inst_abbr(inst_name):
    """Map PHENICX instrument name (e.g. 'violin3', 'horn1') to abbreviation."""
    base = get_inst_base(inst_name)
    return INST_BASE_MAP.get(base)


def note_name_to_midi(name):
    """Convert note name (e.g. 'A#3', 'Db4') to MIDI pitch number."""
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    name = name.strip()
    i = 0
    base = note_map.get(name[i].upper())
    if base is None:
        return None
    i += 1
    while i < len(name) and name[i] in '#b':
        if name[i] == '#':
            base += 1
        elif name[i] == 'b':
            base -= 1
        i += 1
    octave = int(name[i:])
    return (octave + 1) * 12 + base


def txt_to_notes(txt_path):
    """Parse PHENICX txt annotation. Format: onset,offset,note_name per line.
    Returns (N, 4) array [onset, offset, midi_pitch, velocity=0]."""
    notes = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            onset = float(parts[0])
            offset = float(parts[1])
            pitch = note_name_to_midi(parts[2])
            if pitch is not None:
                notes.append([onset, offset, float(pitch), 0.0])
    if not notes:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.array(notes, dtype=np.float32)
    # Sort by onset
    arr = arr[arr[:, 0].argsort()]
    return arr


def is_monophonic_notes(notes, tolerance=0.05):
    """Check if note array is monophonic (no overlapping notes).
    tolerance=50ms to handle annotation imprecision in PHENICX."""
    for i in range(len(notes) - 1):
        if notes[i, 1] > notes[i + 1, 0] + tolerance:
            return False
    return True


def extract_f0_pyin(audio, sr, hop_time):
    """Extract f0 using pYIN (librosa)."""
    hop_length = int(sr * hop_time)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio, fmin=50, fmax=2000,
        sr=sr, hop_length=hop_length,
        fill_na=0.0
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    return f0


def process_track(piece, inst_name, inst_abbr):
    """Process one instrument track from one piece."""
    audio_path = os.path.join(PHENICX_DIR, "audio", piece, f"{inst_name}.wav")
    if not os.path.isfile(audio_path):
        return None

    # Find annotation: annotations use base name (no number)
    base_name = get_inst_base(inst_name)
    txt_path = os.path.join(PHENICX_DIR, "annotations", piece, f"{base_name}.txt")
    if not os.path.isfile(txt_path):
        print(f"  [SKIP] No annotation: {piece}/{inst_name} (looked for {base_name}.txt)")
        return None

    # Parse notes from txt and check monophonic
    notes = txt_to_notes(txt_path)
    if len(notes) == 0:
        print(f"  [SKIP] No notes: {piece}/{inst_name}")
        return None
    if not is_monophonic_notes(notes):
        print(f"  [SKIP] Polyphonic: {piece}/{inst_name} ({len(notes)} notes)")
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

    # Align lengths
    min_len = min(len(f0), len(amp))
    f0 = f0[:min_len]
    amp = amp[:min_len]

    # Estimate velocity from RMS
    notes = estimate_velocity(notes, amp, HOP_TIME)

    # Output
    track_id = f"{piece}_{inst_name}"
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
    print(f"  [OK] {track_id}: {len(notes)} notes, {min_len} frames, {duration:.1f}s")
    return {
        "track_id": track_id,
        "instrument": inst_abbr,
        "n_notes": len(notes),
        "n_frames": min_len,
        "duration_sec": duration,
    }


def main():
    print("=" * 60)
    print("PHENICX-Anechoic 预处理")
    print(f"输入: {PHENICX_DIR}")
    print(f"输出: {OUTPUT_BASE}")
    print("=" * 60)

    results = []
    for piece in PIECES:
        print(f"\n--- {piece} ---")
        audio_dir = os.path.join(PHENICX_DIR, "audio", piece)
        if not os.path.isdir(audio_dir):
            print(f"  [SKIP] No audio dir for {piece}")
            continue

        for wav_file in sorted(os.listdir(audio_dir)):
            if not wav_file.endswith(".wav"):
                continue
            inst_name = wav_file.replace(".wav", "")
            inst_abbr = get_inst_abbr(inst_name)
            if inst_abbr is None:
                print(f"  [SKIP] Unknown instrument: {inst_name}")
                continue

            result = process_track(piece, inst_name, inst_abbr)
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
