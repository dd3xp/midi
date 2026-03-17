"""
URMP 数据预处理脚本
将指定乐器的音轨处理为 (notes, f0, amp) 三元组，用于 MIDI→连续演奏表达 模型训练。

输入: URMP 数据集目录
输出: 每条音轨一个文件夹，包含:
  - data.npz: notes(N,4), f0(T,), amp(T,), hop_time, sr
  - synth.wav: 用 f0+amp 正弦波合成的音频（用于验证）

用法:
  python preprocess.py                  # 处理所有支持的乐器
  python preprocess.py vn               # 只处理小提琴
  python preprocess.py vn fl            # 处理小提琴和长笛
  python preprocess.py tpt              # 只处理小号
"""

import sys
import os
import glob
import numpy as np
import librosa
import soundfile as sf


# ============ 配置 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
URMP_DIR = os.path.join(PROJECT_ROOT, "dataset", "URMP")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "datagen")
HOP_TIME = 0.01    # 10ms，与 URMP F0s 标注一致
TARGET_SR = 16000   # 下采样到 16kHz，减少计算量

# URMP 中的单音乐器: 缩写 -> 中文名
SUPPORTED_INSTRUMENTS = {
    "vn": "小提琴",
    "va": "中提琴",
    "vc": "大提琴",
    "db": "低音提琴",
    "fl": "长笛",
    "ob": "双簧管",
    "cl": "单簧管",
    "bn": "巴松管",
    "tpt": "小号",
    "hn": "圆号",
    "tbn": "长号",
    "tba": "大号",
}


def hz_to_midi(f0_hz):
    """将频率(Hz)转换为 MIDI 音高编号，0 Hz 返回 0。"""
    with np.errstate(divide='ignore', invalid='ignore'):
        midi = 69 + 12 * np.log2(f0_hz / 440.0)
    midi = np.where(f0_hz > 0, midi, 0.0)
    return midi


def parse_f0s(filepath):
    """
    解析 URMP F0s 标注文件。
    格式: time_sec \t f0_hz (每行一帧，约 10ms hop)
    返回: times (T,), f0 (T,) in Hz
    """
    data = np.loadtxt(filepath)
    times = data[:, 0]
    f0 = data[:, 1]
    return times, f0


def parse_notes(filepath):
    """
    解析 URMP Notes 标注文件。
    格式: onset_sec \t f0_hz \t duration_sec
    返回: notes (N, 4) [onset, offset, midi_pitch, velocity]
    URMP 没有 velocity 信息，统一设为 80。
    """
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    onsets = data[:, 0]
    f0_hz = data[:, 1]
    durations = data[:, 2]
    offsets = onsets + durations
    midi_pitches = np.round(hz_to_midi(f0_hz)).astype(np.float32)
    velocities = np.full_like(onsets, 80.0)  # placeholder

    notes = np.stack([onsets, offsets, midi_pitches, velocities], axis=1).astype(np.float32)
    return notes


def extract_amp(audio, sr, hop_time):
    """
    从音频提取帧级 RMS 振幅。
    hop_time: 帧间隔（秒）
    返回: amp (T,)
    """
    hop_length = int(sr * hop_time)
    # librosa.feature.rms 返回 (1, T)
    rms = librosa.feature.rms(y=audio, frame_length=hop_length * 2, hop_length=hop_length)[0]
    return rms.astype(np.float32)


def find_tracks(urmp_dir, instrument):
    """
    扫描 URMP 目录，找到指定乐器的所有音轨。
    返回: list of dict, 每个包含 audio_path, f0s_path, notes_path, track_id
    """
    tracks = []
    for piece_dir in sorted(glob.glob(os.path.join(urmp_dir, "*"))):
        if not os.path.isdir(piece_dir):
            continue
        piece_name = os.path.basename(piece_dir)

        audio_files = glob.glob(os.path.join(piece_dir, f"AuSep_*_{instrument}_*.wav"))
        for audio_path in sorted(audio_files):
            fname = os.path.basename(audio_path)
            parts = fname.replace(".wav", "").split("_")
            track_num = parts[1]

            f0s_pattern = os.path.join(piece_dir, f"F0s_{track_num}_{instrument}_*.txt")
            notes_pattern = os.path.join(piece_dir, f"Notes_{track_num}_{instrument}_*.txt")

            f0s_files = glob.glob(f0s_pattern)
            notes_files = glob.glob(notes_pattern)

            if not f0s_files or not notes_files:
                print(f"  [SKIP] Missing F0s or Notes for {fname}")
                continue

            track_id = f"{piece_name}_track{track_num}_{instrument}"
            tracks.append({
                "audio_path": audio_path,
                "f0s_path": f0s_files[0],
                "notes_path": notes_files[0],
                "track_id": track_id,
            })

    return tracks


def synthesize(f0, amp, hop_time, sr):
    """用正弦波合成：f0 控制音高，amp 控制振幅。"""
    hop_samples = int(sr * hop_time)
    n_frames = len(f0)
    total_samples = n_frames * hop_samples
    audio = np.zeros(total_samples, dtype=np.float64)

    phase = 0.0
    for i in range(n_frames):
        freq = f0[i]
        amplitude = amp[i]
        start = i * hop_samples
        end = start + hop_samples

        if freq > 0 and amplitude > 0:
            t = np.arange(hop_samples) / sr
            segment = amplitude * np.sin(2 * np.pi * freq * t + phase)
            audio[start:end] = segment
            phase += 2 * np.pi * freq * hop_samples / sr
            phase %= 2 * np.pi
        else:
            phase = 0.0

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.8
    return audio.astype(np.float32)


def process_track(track_info, output_dir):
    """处理单条音轨，输出到独立文件夹（data.npz + synth.wav）。"""
    track_id = track_info["track_id"]

    # 1. 解析 F0s 标注
    f0_times, f0 = parse_f0s(track_info["f0s_path"])

    # 2. 解析 Notes 标注
    notes = parse_notes(track_info["notes_path"])

    # 3. 读取音频并下采样
    audio, orig_sr = sf.read(track_info["audio_path"])
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # 转单声道
    if orig_sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)

    # 4. 提取 amp(t)
    amp = extract_amp(audio, TARGET_SR, HOP_TIME)

    # 5. 对齐 f0 和 amp 的长度（取较短的）
    min_len = min(len(f0), len(amp))
    f0 = f0[:min_len].astype(np.float32)
    amp = amp[:min_len]
    f0_times = f0_times[:min_len]

    # 6. 保存到独立文件夹
    track_dir = os.path.join(output_dir, track_id)
    os.makedirs(track_dir, exist_ok=True)

    np.savez(
        os.path.join(track_dir, "data.npz"),
        notes=notes,
        f0=f0,
        amp=amp,
        hop_time=np.float32(HOP_TIME),
        sr=np.int32(TARGET_SR),
    )

    # 合成验证音频
    audio = synthesize(f0, amp, HOP_TIME, TARGET_SR)
    sf.write(os.path.join(track_dir, "synth.wav"), audio, TARGET_SR)

    duration = len(f0) * HOP_TIME
    return {
        "track_id": track_id,
        "n_notes": len(notes),
        "n_frames": min_len,
        "duration_sec": duration,
        "f0_min": f0[f0 > 0].min() if (f0 > 0).any() else 0,
        "f0_max": f0.max(),
        "amp_min": amp.min(),
        "amp_max": amp.max(),
    }


def process_instrument(instrument, urmp_dir, output_base):
    """处理单个乐器的所有音轨。"""
    inst_name = SUPPORTED_INSTRUMENTS[instrument]
    output_dir = os.path.join(output_base, f"processed_{instrument}")

    print("=" * 60)
    print(f"处理乐器: {inst_name} ({instrument})")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    tracks = find_tracks(urmp_dir, instrument)
    print(f"\n找到 {len(tracks)} 条 {inst_name} 音轨\n")

    if not tracks:
        print(f"未找到 {inst_name} 音轨，跳过。\n")
        return []

    results = []
    for i, track in enumerate(tracks):
        print(f"[{i+1}/{len(tracks)}] 处理 {track['track_id']}...")
        try:
            result = process_track(track, output_dir)
            results.append(result)
            print(f"  -> {result['n_notes']} notes, {result['n_frames']} frames, "
                  f"{result['duration_sec']:.1f}s, "
                  f"f0=[{result['f0_min']:.1f}, {result['f0_max']:.1f}] Hz, "
                  f"amp=[{result['amp_min']:.4f}, {result['amp_max']:.4f}]")
        except Exception as e:
            print(f"  [ERROR] {e}")

    if results:
        print(f"\n--- {inst_name} 汇总 ---")
        total_frames = sum(r["n_frames"] for r in results)
        total_notes = sum(r["n_notes"] for r in results)
        total_duration = sum(r["duration_sec"] for r in results)
        all_f0_min = min(r["f0_min"] for r in results if r["f0_min"] > 0)
        all_f0_max = max(r["f0_max"] for r in results)
        all_amp_min = min(r["amp_min"] for r in results)
        all_amp_max = max(r["amp_max"] for r in results)

        print(f"成功处理: {len(results)}/{len(tracks)} 条音轨")
        print(f"总音符数: {total_notes}")
        print(f"总帧数:   {total_frames}")
        print(f"总时长:   {total_duration:.1f}s ({total_duration/60:.1f}min)")
        print(f"f0 范围:  [{all_f0_min:.1f}, {all_f0_max:.1f}] Hz")
        print(f"amp 范围: [{all_amp_min:.6f}, {all_amp_max:.6f}]")
        print(f"输出目录: {output_dir}")

    print()
    return results


def main():
    # 解析命令行参数：指定乐器，不指定则处理全部
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if args:
        instruments = []
        for inst in args:
            if inst not in SUPPORTED_INSTRUMENTS:
                print(f"不支持的乐器: {inst}")
                print(f"支持的乐器: {', '.join(f'{k}({v})' for k, v in SUPPORTED_INSTRUMENTS.items())}")
                return
            instruments.append(inst)
    else:
        instruments = list(SUPPORTED_INSTRUMENTS.keys())

    print(f"将处理以下乐器: {', '.join(f'{i}({SUPPORTED_INSTRUMENTS[i]})' for i in instruments)}\n")

    all_results = {}
    for instrument in instruments:
        results = process_instrument(instrument, URMP_DIR, OUTPUT_BASE)
        if results:
            all_results[instrument] = results

    # 总汇总
    if len(all_results) > 1:
        print("=" * 60)
        print("全部乐器汇总")
        print("=" * 60)
        for inst, results in all_results.items():
            inst_name = SUPPORTED_INSTRUMENTS[inst]
            total_duration = sum(r["duration_sec"] for r in results)
            print(f"  {inst_name} ({inst}): {len(results)} 条音轨, "
                  f"{sum(r['n_notes'] for r in results)} 音符, "
                  f"{total_duration:.1f}s")


if __name__ == "__main__":
    main()
