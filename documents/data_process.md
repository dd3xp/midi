# 数据处理方法

## 单乐器

### 原始数据集

#### URMP

URMP 数据集，数据集自带单乐器的 wav 文件以及对应的帧级 f0 标注（每帧 10ms）、单乐器的 MIDI 文件所对应的 notes。数据集包含 13 种单音乐器，本次工作使用其中 9 种：小提琴(vn)、中提琴(va)、大提琴(vc)、长笛(fl)、双簧管(ob)、单簧管(cl)、萨克斯(sax)、小号(tpt)、长号(tbn)。共 133 条音轨。

预处理脚本：`python src/data/preprocess.py [乐器缩写...]`，不指定则处理全部。

#### Bach10

Bach10 数据集，包含 10 首四声部巴赫合唱，每个声部由不同乐器单独录制：Soprano=violin, Alto=clarinet, Tenor=saxophone, Bass=bassoon。数据集提供：

- 各声部独立录音 wav 文件（44.1kHz）
- `GTF0s.mat`：帧级 f0 ground truth（MIDI 编号，4 声部 × T 帧，10ms hop），由 YIN 算法检测后人工校正
- `*.txt`：音符标注（onset_ms, onset_ms_midi, midi_pitch, channel），channel 1=vn, 2=cl, 3=sax, 4=bn

预处理时将 f0 从 MIDI 编号转为 Hz，amp 从原始 wav 提取（方法同 URMP），音符 offset 由下一音符 onset 推算。共 40 条音轨（10 首 × 4 乐器）。

预处理脚本：`python src/data/preprocess_bach10.py`

### 预处理数据集

位于 `data.npz`，内有四个字段：

- **notes**：(N, 4) 数组，每行为 [onset, offset, midi_pitch, velocity]。直接从 URMP 的 Notes 标注文件解析，onset 和 duration 转为 onset 和 offset，f0(Hz) 转为 MIDI pitch，velocity 统一设为 80
- **f0**：帧级基频序列，URMP 自带，每帧 10ms
- **amp**：帧级 RMS(均方根) 振幅，从单乐器 wav 中提取。计算方法为 librosa 库的 `feature.rms()`，窗口长度 $N$ 为 320 样本（0.02s），每帧移动 $H$ 为 160 样本（0.01s），与 f0 帧率一致。公式：$\text{amp}(t) = \sqrt{\frac{1}{N}\sum_{i=0}^{N-1} x(t \cdot H + i)^2}$，其中 $x(n)$ 为原始音频的第 $n$ 个采样点
- **hop_time**：帧间隔，标量，0.01 秒
- **sr**：采样率，标量，16000Hz

### 预处理数据验证

由于只有 amp 是原始数据集里面没有的，因此这里只验证了 amp：计算 `synth.wav` 与原始分离音频的 RMS 包络的皮尔逊相关系数。RMS 包络即每帧 RMS 值组成的曲线，反映音量随时间的变化。相关系数公式：

$$r = \frac{\sum_{i}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i}(x_i - \bar{x})^2 \cdot \sum_{i}(y_i - \bar{y})^2}}$$

其中 $x$ 为合成音频的 RMS 包络，$y$ 为原始音频的 RMS 包络。阈值 $0.5$ 以上为合格，但大多数都是 $0.95$ 以上的，运行 `python src/data/verify_data.py` 可以验证。

URMP 和 Bach10 均通过验证，173 条音轨全部合格，平均包络相关性 0.987。

### 数据目录结构

```
dataset/solo/
  URMP/           # 原始 URMP 数据（44 首曲目）
  Bach10_v1.1/    # 原始 Bach10 数据（10 首曲目）

datagen/solo/
  URMP/           # URMP 预处理输出
    processed_vn/ processed_va/ processed_vc/
    processed_fl/ processed_ob/ processed_cl/
    processed_sax/ processed_tpt/ processed_tbn/
  Bach10/         # Bach10 预处理输出
    {piece_name}_{inst_abbr}/data.npz
```

