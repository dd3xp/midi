"""
Bach10 数据预处理脚本
将 Bach10 的 4 个单乐器声部处理为 (notes, f0, amp) 三元组，格式与 URMP 一致。

Bach10 格式:
- GTF0s.mat: 帧级 f0（MIDI 编号），4行 x T列，10ms hop
  行顺序: Violin(1), Clarinet(2), Saxophone(3), Bassoon(4)
- *.txt: 音符标注，4列: onset_ms(audio), onset_ms(MIDI), midi_pitch, channel
  channel: 1=violin, 2=clarinet, 3=saxophone, 4=bassoon
- 每个声部有独立的 wav 文件

输出: datagen/solo/Bach10/processed_{instrument}/{piece}/data.npz + synth.wav + notes.mid
"""

import sys
import os
import glob
import shutil
import numpy as np
import librosa
import soundfile as sf
import scipy.io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.solo.preprocess import (
    extract_amp, estimate_velocity, synthesize, notes_to_midi,
    HOP_TIME, TARGET_SR,
)

BACH10_DIR = os.path.join(PROJECT_ROOT, "dataset", "Bach10")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "datagen", "solo", "Bach10")

# Part ID -> (instrument abbreviation, wav suffix, instrument name)
PARTS = {
    1: ("vn", "violin", "小提琴"),
    2: ("cl", "clarinet", "单簧管"),
    3: ("sax", "saxphone", "萨克斯"),  # Bach10 uses "saxphone" (typo in dataset)
    4: ("bn", "bassoon", "巴松管"),
}


def midi_to_hz(midi_num):
    """Convert MIDI number to Hz. 0 -> 0."""
    hz = 440.0 * 2.0 ** ((midi_num - 69.0) / 12.0)
    return np.where(midi_num > 0, hz, 0.0)


def process_piece(piece_dir):
    """Process all 4 parts of one Bach10 piece."""
    piece_name = os.path.basename(piece_dir)
    results = []

    # Load GTF0s (MIDI numbers, 4 x T, 10ms hop)
    f0s_path = glob.glob(os.path.join(piece_dir, "*-GTF0s.mat"))
    if not f0s_path:
        print(f"  [SKIP] No GTF0s.mat found in {piece_name}")
        return results
    f0s_mat = scipy.io.loadmat(f0s_path[0])
    f0s_midi = f0s_mat["GTF0s"]  # (4, T) in MIDI numbers

    # Load txt notes (onset_ms_audio, onset_ms_midi, midi_pitch, channel)
    txt_path = glob.glob(os.path.join(piece_dir, "*.txt"))
    if not txt_path:
        print(f"  [SKIP] No txt annotation found in {piece_name}")
        return results
    txt_data = np.loadtxt(txt_path[0])

    for part_id, (inst_abbr, wav_suffix, inst_name) in PARTS.items():
        # Find audio file
        wav_pattern = os.path.join(piece_dir, f"*-{wav_suffix}.wav")
        wav_files = glob.glob(wav_pattern)
        if not wav_files:
            print(f"  [SKIP] No {wav_suffix}.wav in {piece_name}")
            continue

        track_id = f"{piece_name}_{inst_abbr}"

        # 1. Read audio and resample
        audio, orig_sr = sf.read(wav_files[0])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)

        # 2. Get f0 for this part (convert MIDI -> Hz)
        f0_midi = f0s_midi[part_id - 1]  # (T,)
        f0_hz = midi_to_hz(f0_midi).astype(np.float32)

        # 3. Extract amp
        amp = extract_amp(audio, TARGET_SR, HOP_TIME)

        # 4. Align f0 and amp lengths
        min_len = min(len(f0_hz), len(amp))
        f0_hz = f0_hz[:min_len]
        amp = amp[:min_len]

        # 5. Build notes array from txt
        # txt columns: onset_ms(audio), onset_ms(midi), midi_pitch, channel
        part_notes_mask = txt_data[:, 3] == part_id
        part_txt = txt_data[part_notes_mask]

        if len(part_txt) == 0:
            print(f"  [SKIP] No notes for part {part_id} in {piece_name}")
            continue

        # Convert onset_ms to seconds. We need onset and offset.
        # txt only has onset_ms(audio). To get offset, use next note's onset or end of f0.
        onset_ms = part_txt[:, 0]
        midi_pitches = part_txt[:, 2]

        # Sort by onset
        sort_idx = np.argsort(onset_ms)
        onset_ms = onset_ms[sort_idx]
        midi_pitches = midi_pitches[sort_idx]

        # Compute offsets: each note ends when the next one starts, last note ends at audio end
        onset_sec = onset_ms / 1000.0
        offset_sec = np.zeros_like(onset_sec)
        for i in range(len(onset_sec) - 1):
            offset_sec[i] = onset_sec[i + 1]
        offset_sec[-1] = min_len * HOP_TIME

        velocities = np.zeros_like(onset_sec)  # placeholder, filled below
        notes = np.stack([onset_sec, offset_sec, midi_pitches, velocities], axis=1).astype(np.float32)

        # 5b. Estimate velocity from RMS amplitude at note onsets
        notes = estimate_velocity(notes, amp, HOP_TIME)

        # 6. Save
        track_dir = os.path.join(OUTPUT_BASE, f"processed_{inst_abbr}", piece_name)
        os.makedirs(track_dir, exist_ok=True)

        np.savez(
            os.path.join(track_dir, "data.npz"),
            notes=notes,
            f0=f0_hz,
            amp=amp,
            hop_time=np.float32(HOP_TIME),
            sr=np.int32(TARGET_SR),
        )

        notes_to_midi(notes, os.path.join(track_dir, "notes.mid"))

        synth_audio = synthesize(f0_hz, amp, HOP_TIME, TARGET_SR)
        sf.write(os.path.join(track_dir, "synth.wav"), synth_audio, TARGET_SR)

        duration = min_len * HOP_TIME
        result = {
            "track_id": track_id,
            "instrument": inst_abbr,
            "n_notes": len(notes),
            "n_frames": min_len,
            "duration_sec": duration,
            "f0_min": f0_hz[f0_hz > 0].min() if (f0_hz > 0).any() else 0,
            "f0_max": f0_hz.max(),
            "amp_min": amp.min(),
            "amp_max": amp.max(),
        }
        results.append(result)
        print(f"  [{inst_abbr}] {result['n_notes']} notes, {result['n_frames']} frames, "
              f"{result['duration_sec']:.1f}s, "
              f"f0=[{result['f0_min']:.1f}, {result['f0_max']:.1f}] Hz, "
              f"amp=[{result['amp_min']:.4f}, {result['amp_max']:.4f}]")

    return results


def main():
    print("=" * 60)
    print("Bach10 数据预处理")
    print(f"输入: {BACH10_DIR}")
    print(f"输出: {OUTPUT_BASE}")
    print("=" * 60)

    if os.path.exists(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)
        print(f"已清除旧数据: {OUTPUT_BASE}")
    os.makedirs(OUTPUT_BASE)

    pieces = sorted(glob.glob(os.path.join(BACH10_DIR, "[0-9]*")))
    print(f"\n找到 {len(pieces)} 首曲目\n")

    all_results = []
    for piece_dir in pieces:
        piece_name = os.path.basename(piece_dir)
        print(f"\n处理: {piece_name}")
        results = process_piece(piece_dir)
        all_results.extend(results)

    # Summary
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    by_inst = {}
    for r in all_results:
        inst = r["instrument"]
        if inst not in by_inst:
            by_inst[inst] = []
        by_inst[inst].append(r)

    for inst, results in sorted(by_inst.items()):
        total_dur = sum(r["duration_sec"] for r in results)
        total_notes = sum(r["n_notes"] for r in results)
        print(f"  {inst}: {len(results)} tracks, {total_notes} notes, {total_dur:.1f}s ({total_dur/60:.1f}min)")

    print(f"\n总计: {len(all_results)} tracks")
    print(f"输出目录: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
