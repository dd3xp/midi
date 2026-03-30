"""
验证预处理后的数据质量。
检查: 文件完整性、数值范围、notes 与 f0 对齐质量、合成音频与原始音频对比。
"""

import os
import glob
import numpy as np
import librosa


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "datagen")
URMP_DIR = os.path.join(PROJECT_ROOT, "dataset", "solo", "URMP")


BACH10_DIR = os.path.join(PROJECT_ROOT, "dataset", "solo", "Bach10_v1.1")

# Bach10 instrument abbreviation -> wav suffix
BACH10_INST_MAP = {"vn": "violin", "cl": "clarinet", "sax": "saxphone", "bn": "bassoon"}


def find_original_audio(npz_path):
    """根据输出文件夹名反推原始音频路径。支持 URMP 和 Bach10。"""
    track_dir = os.path.basename(os.path.dirname(npz_path))

    # Try URMP: {piece_name}_track{N}_{instrument}
    parts = track_dir.rsplit("_track", 1)
    if len(parts) == 2:
        piece_name = parts[0]
        num_inst = parts[1]  # "1_vn"
        num, instrument = num_inst.split("_", 1)
        piece_dir = os.path.join(URMP_DIR, piece_name)
        if os.path.isdir(piece_dir):
            pattern = os.path.join(piece_dir, f"AuSep_{num}_{instrument}_*.wav")
            matches = glob.glob(pattern)
            if matches:
                return matches[0]

    # Try Bach10: processed_{inst_abbr}/{piece_name}/data.npz
    # e.g. processed_vn/01-AchGottundHerr/data.npz
    #   -> dataset/solo/Bach10_v1.1/01-AchGottundHerr/01-AchGottundHerr-violin.wav
    parent_dir = os.path.basename(os.path.dirname(os.path.dirname(npz_path)))
    for inst_abbr, wav_suffix in BACH10_INST_MAP.items():
        if parent_dir == f"processed_{inst_abbr}":
            piece_name = track_dir  # track_dir is already the piece name
            piece_dir = os.path.join(BACH10_DIR, piece_name)
            if os.path.isdir(piece_dir):
                pattern = os.path.join(piece_dir, f"*-{wav_suffix}.wav")
                matches = glob.glob(pattern)
                if matches:
                    return matches[0]
            break

    return None


def compare_audio(synth_path, original_path, sr=16000):
    """比较合成音频和原始音频，返回相关性指标。"""
    synth, _ = librosa.load(synth_path, sr=sr)
    original, _ = librosa.load(original_path, sr=sr)

    # 对齐长度
    min_len = min(len(synth), len(original))
    synth = synth[:min_len]
    original = original[:min_len]

    # 计算包络相关性（比波形相关性更有意义）
    hop = int(sr * 0.01)
    synth_env = librosa.feature.rms(y=synth, frame_length=hop * 2, hop_length=hop)[0]
    orig_env = librosa.feature.rms(y=original, frame_length=hop * 2, hop_length=hop)[0]

    min_frames = min(len(synth_env), len(orig_env))
    synth_env = synth_env[:min_frames]
    orig_env = orig_env[:min_frames]

    # 皮尔逊相关系数
    if np.std(synth_env) == 0 or np.std(orig_env) == 0:
        return 0.0
    corr = np.corrcoef(synth_env, orig_env)[0, 1]
    return float(corr)


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

    # MIDI pitch 范围检查 (留宽范围，适配各种乐器)
    valid_pitches = notes[:, 2]
    if np.any(valid_pitches < 20) or np.any(valid_pitches > 120):
        out_of_range = valid_pitches[(valid_pitches < 20) | (valid_pitches > 120)]
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

    # 合成音频 vs 原始音频对比
    env_corr = None
    synth_path = os.path.join(os.path.dirname(npz_path), "synth.wav")
    original_path = find_original_audio(npz_path)
    if os.path.exists(synth_path) and original_path:
        try:
            env_corr = compare_audio(synth_path, original_path, sr=sr)
            if env_corr < 0.5:
                issues.append(f"envelope correlation low: {env_corr:.3f}")
        except Exception as e:
            issues.append(f"audio comparison failed: {e}")

    return {
        "file": os.path.relpath(npz_path, DATA_DIR),
        "n_notes": len(notes),
        "n_frames": len(f0),
        "duration": len(f0) * hop_time,
        "f0_voiced_ratio": (f0 > 0).sum() / len(f0),
        "note_voicing_rate": voicing_rate,
        "env_corr": env_corr,
        "issues": issues,
    }


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.npz"), recursive=True))
    print(f"找到 {len(files)} 个 .npz 文件\n")

    all_results = []
    for f in files:
        result = verify_single(f)
        all_results.append(result)
        status = "OK" if not result["issues"] else "WARN"
        corr_str = f", env_corr={result['env_corr']:.3f}" if result["env_corr"] is not None else ""
        print(f"[{status}] {result['file']}: "
              f"{result['n_notes']} notes, {result['n_frames']} frames, "
              f"{result['duration']:.1f}s, "
              f"voiced={result['f0_voiced_ratio']:.1%}, "
              f"note_voicing={result['note_voicing_rate']:.1%}"
              f"{corr_str}")
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

    corrs = [r["env_corr"] for r in all_results if r["env_corr"] is not None]
    avg_corr = np.mean(corrs) if corrs else None

    print(f"总文件数:       {total_files}")
    print(f"有问题的文件:   {files_with_issues}")
    print(f"总音符数:       {total_notes}")
    print(f"总时长:         {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"平均 note 内 voiced 比例: {avg_voicing:.1%}")
    if avg_corr is not None:
        print(f"平均包络相关性:  {avg_corr:.3f} (共 {len(corrs)} 条有对比)")

    if avg_voicing < 0.7:
        print("\n[WARNING] note 内 voiced 比例偏低，可能存在对齐问题")
    elif avg_corr is not None and avg_corr < 0.5:
        print("\n[WARNING] 包络相关性偏低，合成质量可能有问题")
    else:
        print("\n[OK] 数据质量良好，可以用于训练")


if __name__ == "__main__":
    main()
