"""
用 f0 + amp 合成音频，验证提取的数据是否合理。
用法: python synthesize.py [track_folder]
默认合成第一条音轨。
"""

import sys
import os
import numpy as np
import soundfile as sf


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

    # 归一化防止爆音
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.8

    return audio.astype(np.float32)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not args:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        data_dir = os.path.join(project_root, "datagen")
        # 找第一个文件夹
        folders = sorted([d for d in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, d))])
        if not folders:
            print("未找到数据")
            return
        track_folder = os.path.join(data_dir, folders[0])
    else:
        track_folder = args[0]

    npz_path = os.path.join(track_folder, "data.npz")
    data = np.load(npz_path)
    f0 = data["f0"]
    amp = data["amp"]
    hop_time = float(data["hop_time"])
    sr = int(data["sr"])

    track_name = os.path.basename(track_folder)
    print(f"合成: {track_name}")
    print(f"  帧数={len(f0)}, 时长={len(f0)*hop_time:.1f}s, sr={sr}")

    audio = synthesize(f0, amp, hop_time, sr)

    out_path = os.path.join(track_folder, "synth.wav")
    sf.write(out_path, audio, sr)
    print(f"  输出: {out_path}")


if __name__ == "__main__":
    main()
