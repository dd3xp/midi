"""
验证预处理后的数据质量。
检查: 文件完整性、数值范围、notes 与 f0 对齐质量。
"""

import os
import glob
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "datagen")


def verify_single(npz_path):
    """验证单个 .npz 文件。"""
    data = np.load(npz_path)
    notes = data["notes"]    # (N, 4) [onset, offset, midi_pitch, velocity]
    f0 = data["f0"]          # (T,)
    amp = data["amp"]        # (T,)
    hop_time = float(data["hop_time"])
    sr = int(data["sr"])

    issues = []

    # 基本检查
    if notes.shape[1] != 4:
        issues.append(f"notes shape wrong: {notes.shape}")
    if len(f0) != len(amp):
        issues.append(f"f0/amp length mismatch: {len(f0)} vs {len(amp)}")
    if len(f0) == 0:
        issues.append("empty f0")

    # 数值范围检查
    if np.any(np.isnan(f0)):
        issues.append(f"f0 has {np.isnan(f0).sum()} NaN values")
    if np.any(np.isnan(amp)):
        issues.append(f"amp has {np.isnan(amp).sum()} NaN values")
    if np.any(f0 < 0):
        issues.append(f"f0 has negative values")
    if np.any(amp < 0):
        issues.append(f"amp has negative values")

    # MIDI pitch 范围检查 (小提琴 G3~E7, MIDI 55~100 左右，留余量)
    valid_pitches = notes[:, 2]
    if np.any(valid_pitches < 40) or np.any(valid_pitches > 110):
        out_of_range = valid_pitches[(valid_pitches < 40) | (valid_pitches > 110)]
        issues.append(f"MIDI pitch out of range: {out_of_range}")

    # Notes 时间检查
    if np.any(notes[:, 1] <= notes[:, 0]):
        issues.append("some notes have offset <= onset")
    if np.any(notes[:, 0] < 0):
        issues.append("some notes have negative onset")

    # 对齐质量: 检查 notes 的 onset 是否在 f0 时间范围内
    total_time = len(f0) * hop_time
    if notes[-1, 1] > total_time + 0.5:
        issues.append(f"last note offset ({notes[-1,1]:.1f}s) exceeds audio ({total_time:.1f}s)")

    # 对齐质量: 在 note 活跃区间内，f0 应该大部分 > 0
    voiced_frames = 0
    total_note_frames = 0
    for onset, offset, _, _ in notes:
        start_frame = int(onset / hop_time)
        end_frame = int(offset / hop_time)
        end_frame = min(end_frame, len(f0))
        if start_frame < end_frame:
            segment = f0[start_frame:end_frame]
            voiced_frames += (segment > 0).sum()
            total_note_frames += len(segment)

    voicing_rate = voiced_frames / max(total_note_frames, 1)

    return {
        "file": os.path.basename(npz_path),
        "n_notes": len(notes),
        "n_frames": len(f0),
        "duration": len(f0) * hop_time,
        "f0_voiced_ratio": (f0 > 0).sum() / len(f0),
        "note_voicing_rate": voicing_rate,
        "issues": issues,
    }


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"找到 {len(files)} 个 .npz 文件\n")

    all_results = []
    for f in files:
        result = verify_single(f)
        all_results.append(result)
        status = "OK" if not result["issues"] else "WARN"
        print(f"[{status}] {result['file']}: "
              f"{result['n_notes']} notes, {result['n_frames']} frames, "
              f"{result['duration']:.1f}s, "
              f"voiced={result['f0_voiced_ratio']:.1%}, "
              f"note_voicing={result['note_voicing_rate']:.1%}")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"  !! {issue}")

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)

    total_files = len(all_results)
    files_with_issues = sum(1 for r in all_results if r["issues"])
    total_duration = sum(r["duration"] for r in all_results)
    total_notes = sum(r["n_notes"] for r in all_results)
    avg_voicing = np.mean([r["note_voicing_rate"] for r in all_results])

    print(f"总文件数:       {total_files}")
    print(f"有问题的文件:   {files_with_issues}")
    print(f"总音符数:       {total_notes}")
    print(f"总时长:         {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"平均 note 内 voiced 比例: {avg_voicing:.1%}")

    if avg_voicing < 0.7:
        print("\n[WARNING] note 内 voiced 比例偏低，可能存在对齐问题")
    else:
        print("\n[OK] 数据质量良好，可以用于训练")


if __name__ == "__main__":
    main()
