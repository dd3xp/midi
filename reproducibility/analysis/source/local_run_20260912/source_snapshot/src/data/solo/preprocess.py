"""
数据预处理 -- 共享工具函数 + 统一入口

共享函数: extract_amp, estimate_velocity, synthesize, hz_to_midi, notes_to_midi
统一入口: python preprocess.py          -> 处理全部数据集 (urmp, bach10, phenicx, trios)
         python preprocess.py urmp      -> 只处理 URMP
         python preprocess.py bach10 phenicx -> 处理 Bach10 和 PHENICX
"""

import sys
import os
import numpy as np
import librosa
import midiutil


# ============ 全局常量 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
HOP_TIME = 0.01    # 10ms
TARGET_SR = 16000   # 下采样到 16kHz


# ============ 共享工具函数 ============

def hz_to_midi(f0_hz):
    """将频率(Hz)转换为 MIDI 音高编号，0 Hz 返回 0。"""
    with np.errstate(divide='ignore', invalid='ignore'):
        midi = 69 + 12 * np.log2(f0_hz / 440.0)
    midi = np.where(f0_hz > 0, midi, 0.0)
    return midi


def extract_amp(audio, sr, hop_time):
    """
    从音频提取帧级 RMS 振幅。
    hop_time: 帧间隔（秒）
    返回: amp (T,)
    """
    hop_length = int(sr * hop_time)
    rms = librosa.feature.rms(y=audio, frame_length=hop_length * 2, hop_length=hop_length)[0]
    return rms.astype(np.float32)


def estimate_velocity(notes, amp, hop_time, global_max_rms=None):
    """
    [DEPRECATED] 从 RMS 振幅估算 velocity 会导致循环论证：
    velocity 从 amp 提取 → 模型用 velocity 预测 amp → 模型在学自己的输入。
    且实际 MIDI 文件很少有靠谱的 velocity 标注，部署时无法使用。

    现在所有 notes 的 velocity 统一设为 0（不使用），
    frame_features 中也不再包含 velocity 维度。
    """
    notes[:, 3] = 0.0
    return notes


def notes_to_midi(notes, output_path):
    """将 notes (N,4) [onset, offset, midi_pitch, velocity] 转为单轨 MIDI 文件。"""
    midi = midiutil.MIDIFile(1)
    midi.addTempo(0, 0, 120)
    bps = 120 / 60  # beats per second
    for onset, offset, pitch, vel in notes:
        start_beat = onset * bps
        duration_beat = (offset - onset) * bps
        if duration_beat <= 0:
            continue
        midi.addNote(0, 0, int(pitch), start_beat, duration_beat, int(vel))
    with open(output_path, "wb") as f:
        midi.writeFile(f)


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


# ============ 统一入口 ============

DATASET_NAMES = ["urmp", "bach10", "phenicx", "trios", "eep", "ccom_huqin", "quartet"]


def main():
    args = [a.lower() for a in sys.argv[1:] if not a.startswith("-")]

    if args:
        datasets = []
        for name in args:
            if name not in DATASET_NAMES:
                print(f"未知数据集: {name}")
                print(f"支持的数据集: {', '.join(DATASET_NAMES)}")
                return
            datasets.append(name)
    else:
        datasets = DATASET_NAMES

    print(f"将处理以下数据集: {', '.join(datasets)}\n")

    for name in datasets:
        print(f"\n{'='*60}")
        print(f"  处理数据集: {name.upper()}")
        print(f"{'='*60}\n")

        if name == "urmp":
            from src.data.solo.preprocess_urmp import main as urmp_main
            urmp_main()
        elif name == "bach10":
            from src.data.solo.preprocess_bach10 import main as bach10_main
            bach10_main()
        elif name == "phenicx":
            from src.data.solo.preprocess_phenicx import main as phenicx_main
            phenicx_main()
        elif name == "trios":
            from src.data.solo.preprocess_trios import main as trios_main
            trios_main()
        elif name == "eep":
            from src.data.solo.preprocess_eep import main as eep_main
            eep_main()
        elif name == "ccom_huqin":
            from src.data.solo.preprocess_ccom_huqin import main as ccom_main
            ccom_main()
        elif name == "quartet":
            from src.data.solo.preprocess_quartet import main as quartet_main
            quartet_main()


if __name__ == "__main__":
    main()
