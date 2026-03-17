"""
查看 .npz 文件内容。
用法: python view_npz.py [文件路径] [--export]
  --export: 导出为 CSV 文件，方便用 Excel 查看或画图
"""

import sys
import os
import numpy as np


def view(npz_path, export=False):
    data = np.load(npz_path)
    notes = data["notes"]
    f0 = data["f0"]
    amp = data["amp"]
    hop_time = float(data["hop_time"])
    sr = int(data["sr"])

    print(f"文件: {os.path.basename(npz_path)}")
    print(f"采样率: {sr} Hz, 帧间隔: {hop_time*1000:.1f}ms")
    print(f"总帧数: {len(f0)}, 总时长: {len(f0)*hop_time:.1f}s")
    print(f"音符数: {len(notes)}")
    print()

    # 显示音符信息
    print("=== 音符列表 (前20个) ===")
    print(f"{'#':>3}  {'onset':>7}  {'offset':>7}  {'dur':>5}  {'pitch':>5}  {'note':>5}  {'vel':>3}")
    print("-" * 50)

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for i, (onset, offset, pitch, vel) in enumerate(notes[:20]):
        dur = offset - onset
        midi_int = int(pitch)
        name = f"{note_names[midi_int % 12]}{midi_int // 12 - 1}"
        print(f"{i+1:>3}  {onset:>7.3f}  {offset:>7.3f}  {dur:>5.3f}  {midi_int:>5}  {name:>5}  {int(vel):>3}")

    if len(notes) > 20:
        print(f"  ... 还有 {len(notes)-20} 个音符")

    print()
    print("=== f0/amp 统计 ===")
    voiced = f0 > 0
    print(f"f0:  min={f0[voiced].min():.1f} Hz, max={f0.max():.1f} Hz, voiced={voiced.mean():.1%}")
    print(f"amp: min={amp.min():.6f}, max={amp.max():.6f}, mean={amp.mean():.6f}")

    if export:
        base = os.path.splitext(npz_path)[0]

        # 导出帧级数据
        frames_path = base + "_frames.csv"
        times = np.arange(len(f0)) * hop_time
        with open(frames_path, 'w') as f:
            f.write("time_sec,f0_hz,amp\n")
            for t, f0_val, amp_val in zip(times, f0, amp):
                f.write(f"{t:.4f},{f0_val:.2f},{amp_val:.6f}\n")
        print(f"\n帧数据已导出: {frames_path}")

        # 导出音符数据
        notes_path = base + "_notes.csv"
        with open(notes_path, 'w') as f:
            f.write("onset,offset,duration,midi_pitch,note_name,velocity\n")
            for onset, offset, pitch, vel in notes:
                midi_int = int(pitch)
                name = f"{note_names[midi_int % 12]}{midi_int // 12 - 1}"
                f.write(f"{onset:.4f},{offset:.4f},{offset-onset:.4f},{midi_int},{name},{int(vel)}\n")
        print(f"音符数据已导出: {notes_path}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    export = "--export" in flags

    if not args:
        # 默认查看第一个文件
        data_dir = "c:/lxr/study/final/datagen/processed_violin"
        import glob
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not files:
            print("未找到 .npz 文件")
            return
        npz_path = files[0]
    else:
        npz_path = args[0]
    view(npz_path, export=export)


if __name__ == "__main__":
    main()
