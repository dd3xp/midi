# 数据处理方法

## 单乐器

### 原始数据集

URMP 数据集，数据集自带单乐器的 wav 文件以及对应的帧级 f0 标注（每帧 10ms）、单乐器的 MIDI 文件所对应的 notes。其中单乐器包含小提琴，小号，长笛等，本次工作使用小提琴，小号，长笛。

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

其中 $x$ 为合成音频的 RMS 包络，$y$ 为原始音频的 RMS 包络。阈值 $0.5$ 以上为合格，但大多数都是 $0.95$ 以上的，运行 `python src/data/preprocess` 可以验证。

```验证结果
PS C:\lxr\study\final> python src/data/verify_data.py
找到 74 个 .npz 文件

[OK] processed_fl\03_Dance_fl_cl_track1_fl\data.npz: 254 notes, 9708 frames, 97.1s, voiced=45.9%, note_voicing=91.7%, env_corr=0.988
[OK] processed_fl\04_Allegro_fl_fl_track1_fl\data.npz: 778 notes, 15713 frames, 157.1s, voiced=70.8%, note_voicing=94.9%, env_corr=0.987
[OK] processed_fl\04_Allegro_fl_fl_track2_fl\data.npz: 414 notes, 15713 frames, 157.1s, voiced=49.2%, note_voicing=92.7%, env_corr=0.966
[OK] processed_fl\08_Spring_fl_vn_track1_fl\data.npz: 110 notes, 3502 frames, 35.0s, voiced=70.2%, note_voicing=95.8%, env_corr=0.988
[OK] processed_fl\14_Waltz_fl_fl_cl_track1_fl\data.npz: 294 notes, 9277 frames, 92.8s, voiced=85.7%, note_voicing=98.1%, env_corr=0.968
[OK] processed_fl\14_Waltz_fl_fl_cl_track2_fl\data.npz: 215 notes, 9277 frames, 92.8s, voiced=61.6%, note_voicing=97.3%, env_corr=0.994
[OK] processed_fl\17_Nocturne_vn_fl_cl_track2_fl\data.npz: 108 notes, 9561 frames, 95.6s, voiced=84.5%, note_voicing=99.3%, env_corr=0.996
[OK] processed_fl\18_Nocturne_vn_fl_tpt_track2_fl\data.npz: 108 notes, 9561 frames, 95.6s, voiced=84.5%, note_voicing=99.3%, env_corr=0.996
[OK] processed_fl\28_Fugue_fl_ob_cl_bn_track1_fl\data.npz: 245 notes, 17456 frames, 174.6s, voiced=64.7%, note_voicing=96.8%, env_corr=0.988
[OK] processed_fl\29_Fugue_fl_fl_ob_cl_track1_fl\data.npz: 234 notes, 17199 frames, 172.0s, voiced=60.4%, note_voicing=97.5%, env_corr=0.959
[OK] processed_fl\29_Fugue_fl_fl_ob_cl_track2_fl\data.npz: 254 notes, 17199 frames, 172.0s, voiced=77.9%, note_voicing=98.8%, env_corr=0.994
[OK] processed_fl\30_Fugue_fl_fl_ob_sax_track1_fl\data.npz: 234 notes, 17199 frames, 172.0s, voiced=60.4%, note_voicing=97.5%, env_corr=0.959
[OK] processed_fl\30_Fugue_fl_fl_ob_sax_track2_fl\data.npz: 254 notes, 17199 frames, 172.0s, voiced=77.9%, note_voicing=98.8%, env_corr=0.994
[OK] processed_fl\37_Rondeau_fl_vn_va_cl_track1_fl\data.npz: 484 notes, 12817 frames, 128.2s, voiced=76.8%, note_voicing=95.9%, env_corr=0.976
[OK] processed_fl\40_Miserere_fl_fl_ob_cl_bn_track1_fl\data.npz: 31 notes, 3992 frames, 39.9s, voiced=74.1%, note_voicing=99.3%, env_corr=0.980
[OK] processed_fl\40_Miserere_fl_fl_ob_cl_bn_track2_fl\data.npz: 33 notes, 3992 frames, 39.9s, voiced=69.4%, note_voicing=99.3%, env_corr=0.995
[OK] processed_fl\41_Miserere_fl_fl_ob_sax_bn_track1_fl\data.npz: 31 notes, 3992 frames, 39.9s, voiced=74.1%, note_voicing=99.3%, env_corr=0.980
[OK] processed_fl\41_Miserere_fl_fl_ob_sax_bn_track2_fl\data.npz: 33 notes, 3992 frames, 39.9s, voiced=69.4%, note_voicing=99.3%, env_corr=0.995
[OK] processed_tpt\05_Entertainer_tpt_tpt_track1_tpt\data.npz: 258 notes, 8707 frames, 87.1s, voiced=80.0%, note_voicing=95.4%, env_corr=0.991
[OK] processed_tpt\05_Entertainer_tpt_tpt_track2_tpt\data.npz: 214 notes, 8707 frames, 87.1s, voiced=70.9%, note_voicing=95.7%, env_corr=0.995
[OK] processed_tpt\07_GString_tpt_tbn_track1_tpt\data.npz: 284 notes, 26972 frames, 269.7s, voiced=89.9%, note_voicing=99.0%, env_corr=0.996
[OK] processed_tpt\09_Jesus_tpt_vn_track1_tpt\data.npz: 218 notes, 19884 frames, 198.8s, voiced=86.7%, note_voicing=98.3%, env_corr=0.998
[OK] processed_tpt\10_March_tpt_sax_track1_tpt\data.npz: 296 notes, 10275 frames, 102.7s, voiced=86.5%, note_voicing=96.6%, env_corr=0.995
[OK] processed_tpt\15_Surprise_tpt_tpt_tbn_track1_tpt\data.npz: 122 notes, 5303 frames, 53.0s, voiced=63.4%, note_voicing=95.0%, env_corr=0.996
[OK] processed_tpt\15_Surprise_tpt_tpt_tbn_track2_tpt\data.npz: 103 notes, 5303 frames, 53.0s, voiced=47.7%, note_voicing=94.6%, env_corr=0.995
[OK] processed_tpt\15_Surprise_tpt_tpt_tbn_track3_tpt\data.npz: 65 notes, 5303 frames, 53.0s, voiced=42.5%, note_voicing=95.1%, env_corr=0.997
[OK] processed_tpt\16_Surprise_tpt_tpt_sax_track1_tpt\data.npz: 122 notes, 5303 frames, 53.0s, voiced=63.4%, note_voicing=95.0%, env_corr=0.996
[OK] processed_tpt\16_Surprise_tpt_tpt_sax_track2_tpt\data.npz: 103 notes, 5303 frames, 53.0s, voiced=47.7%, note_voicing=94.6%, env_corr=0.995
[OK] processed_tpt\18_Nocturne_vn_fl_tpt_track3_tpt\data.npz: 94 notes, 9561 frames, 95.6s, voiced=77.8%, note_voicing=98.1%, env_corr=0.996
[OK] processed_tpt\20_Pavane_tpt_vn_vc_track1_tpt\data.npz: 214 notes, 13335 frames, 133.3s, voiced=86.3%, note_voicing=99.4%, env_corr=0.997
[OK] processed_tpt\31_Slavonic_tpt_tpt_hn_tbn_track1_tpt\data.npz: 272 notes, 8176 frames, 81.8s, voiced=72.9%, note_voicing=94.8%, env_corr=0.995
[OK] processed_tpt\31_Slavonic_tpt_tpt_hn_tbn_track2_tpt\data.npz: 247 notes, 8176 frames, 81.8s, voiced=74.0%, note_voicing=94.3%, env_corr=0.995
[OK] processed_tpt\33_Elise_tpt_tpt_hn_tbn_track1_tpt\data.npz: 69 notes, 4769 frames, 47.7s, voiced=63.9%, note_voicing=97.8%, env_corr=0.994
[OK] processed_tpt\33_Elise_tpt_tpt_hn_tbn_track2_tpt\data.npz: 80 notes, 4769 frames, 47.7s, voiced=63.1%, note_voicing=97.7%, env_corr=0.994
[OK] processed_tpt\34_Fugue_tpt_tpt_hn_tbn_track1_tpt\data.npz: 246 notes, 17415 frames, 174.1s, voiced=71.8%, note_voicing=97.5%, env_corr=0.997
[OK] processed_tpt\34_Fugue_tpt_tpt_hn_tbn_track2_tpt\data.npz: 254 notes, 17415 frames, 174.1s, voiced=75.1%, note_voicing=97.7%, env_corr=0.996
[OK] processed_tpt\42_Arioso_tpt_tpt_hn_tbn_tba_track1_tpt\data.npz: 353 notes, 25500 frames, 255.0s, voiced=77.4%, note_voicing=98.6%, env_corr=0.995
[OK] processed_tpt\42_Arioso_tpt_tpt_hn_tbn_tba_track2_tpt\data.npz: 207 notes, 25500 frames, 255.0s, voiced=63.7%, note_voicing=98.7%, env_corr=0.997
[OK] processed_tpt\43_Chorale_tpt_tpt_hn_tbn_tba_track1_tpt\data.npz: 116 notes, 5343 frames, 53.4s, voiced=70.3%, note_voicing=96.9%, env_corr=0.994
[OK] processed_tpt\43_Chorale_tpt_tpt_hn_tbn_tba_track2_tpt\data.npz: 119 notes, 5343 frames, 53.4s, voiced=78.9%, note_voicing=97.3%, env_corr=0.995
[OK] processed_vn\01_Jupiter_vn_vc_track1_vn\data.npz: 92 notes, 6264 frames, 62.6s, voiced=88.8%, note_voicing=99.0%, env_corr=0.992
[OK] processed_vn\02_Sonata_vn_vn_track1_vn\data.npz: 82 notes, 4603 frames, 46.0s, voiced=86.5%, note_voicing=98.7%, env_corr=0.997
[OK] processed_vn\02_Sonata_vn_vn_track2_vn\data.npz: 50 notes, 4603 frames, 46.0s, voiced=84.2%, note_voicing=98.6%, env_corr=0.997
[OK] processed_vn\08_Spring_fl_vn_track2_vn\data.npz: 94 notes, 3502 frames, 35.0s, voiced=83.5%, note_voicing=97.3%, env_corr=0.983
[OK] processed_vn\09_Jesus_tpt_vn_track2_vn\data.npz: 482 notes, 19884 frames, 198.8s, voiced=86.3%, note_voicing=97.6%, env_corr=0.993
[OK] processed_vn\12_Spring_vn_vn_vc_track1_vn\data.npz: 429 notes, 13092 frames, 130.9s, voiced=92.8%, note_voicing=98.0%, env_corr=0.980
[OK] processed_vn\12_Spring_vn_vn_vc_track2_vn\data.npz: 339 notes, 13092 frames, 130.9s, voiced=80.1%, note_voicing=98.1%, env_corr=0.989
[OK] processed_vn\13_Hark_vn_vn_va_track1_vn\data.npz: 76 notes, 4669 frames, 46.7s, voiced=88.2%, note_voicing=98.8%, env_corr=0.990
[OK] processed_vn\13_Hark_vn_vn_va_track2_vn\data.npz: 73 notes, 4669 frames, 46.7s, voiced=87.2%, note_voicing=98.8%, env_corr=0.996
[OK] processed_vn\17_Nocturne_vn_fl_cl_track1_vn\data.npz: 138 notes, 9561 frames, 95.6s, voiced=92.7%, note_voicing=99.2%, env_corr=0.994
[OK] processed_vn\18_Nocturne_vn_fl_tpt_track1_vn\data.npz: 136 notes, 9561 frames, 95.6s, voiced=92.7%, note_voicing=99.2%, env_corr=0.994
[OK] processed_vn\19_Pavane_cl_vn_vc_track2_vn\data.npz: 211 notes, 13335 frames, 133.3s, voiced=68.5%, note_voicing=98.1%, env_corr=0.992
[OK] processed_vn\20_Pavane_tpt_vn_vc_track2_vn\data.npz: 211 notes, 13335 frames, 133.3s, voiced=68.5%, note_voicing=98.1%, env_corr=0.992
[OK] processed_vn\24_Pirates_vn_vn_va_vc_track1_vn\data.npz: 133 notes, 4979 frames, 49.8s, voiced=80.8%, note_voicing=96.7%, env_corr=0.991
[OK] processed_vn\24_Pirates_vn_vn_va_vc_track2_vn\data.npz: 87 notes, 4979 frames, 49.8s, voiced=85.7%, note_voicing=98.2%, env_corr=0.995
[OK] processed_vn\25_Pirates_vn_vn_va_sax_track1_vn\data.npz: 133 notes, 4979 frames, 49.8s, voiced=80.8%, note_voicing=96.7%, env_corr=0.991
[OK] processed_vn\25_Pirates_vn_vn_va_sax_track2_vn\data.npz: 87 notes, 4979 frames, 49.8s, voiced=85.7%, note_voicing=98.2%, env_corr=0.995
[OK] processed_vn\26_King_vn_vn_va_vc_track1_vn\data.npz: 229 notes, 8516 frames, 85.2s, voiced=70.8%, note_voicing=95.4%, env_corr=0.990
[OK] processed_vn\26_King_vn_vn_va_vc_track2_vn\data.npz: 211 notes, 8516 frames, 85.2s, voiced=63.4%, note_voicing=94.2%, env_corr=0.990
[OK] processed_vn\27_King_vn_vn_va_sax_track1_vn\data.npz: 229 notes, 8516 frames, 85.2s, voiced=70.8%, note_voicing=95.4%, env_corr=0.990
[OK] processed_vn\27_King_vn_vn_va_sax_track2_vn\data.npz: 211 notes, 8516 frames, 85.2s, voiced=63.4%, note_voicing=94.2%, env_corr=0.990
[OK] processed_vn\32_Fugue_vn_vn_va_vc_track1_vn\data.npz: 244 notes, 17362 frames, 173.6s, voiced=77.9%, note_voicing=98.5%, env_corr=0.991
[OK] processed_vn\32_Fugue_vn_vn_va_vc_track2_vn\data.npz: 253 notes, 17362 frames, 173.6s, voiced=82.5%, note_voicing=98.3%, env_corr=0.993
[OK] processed_vn\35_Rondeau_vn_vn_va_db_track1_vn\data.npz: 478 notes, 12817 frames, 128.2s, voiced=90.8%, note_voicing=97.3%, env_corr=0.993
[OK] processed_vn\35_Rondeau_vn_vn_va_db_track2_vn\data.npz: 218 notes, 12817 frames, 128.2s, voiced=93.9%, note_voicing=99.2%, env_corr=0.993
[OK] processed_vn\36_Rondeau_vn_vn_va_vc_track1_vn\data.npz: 478 notes, 12817 frames, 128.2s, voiced=90.8%, note_voicing=97.3%, env_corr=0.993
[OK] processed_vn\36_Rondeau_vn_vn_va_vc_track2_vn\data.npz: 218 notes, 12817 frames, 128.2s, voiced=93.9%, note_voicing=99.2%, env_corr=0.993
[OK] processed_vn\37_Rondeau_fl_vn_va_cl_track2_vn\data.npz: 218 notes, 12817 frames, 128.2s, voiced=93.9%, note_voicing=99.2%, env_corr=0.993
[OK] processed_vn\38_Jerusalem_vn_vn_va_vc_db_track1_vn\data.npz: 171 notes, 11910 frames, 119.1s, voiced=93.3%, note_voicing=99.0%, env_corr=0.995
[OK] processed_vn\38_Jerusalem_vn_vn_va_vc_db_track2_vn\data.npz: 179 notes, 11910 frames, 119.1s, voiced=92.2%, note_voicing=98.9%, env_corr=0.991
[OK] processed_vn\39_Jerusalem_vn_vn_va_sax_db_track1_vn\data.npz: 171 notes, 11910 frames, 119.1s, voiced=93.3%, note_voicing=99.0%, env_corr=0.995
[OK] processed_vn\39_Jerusalem_vn_vn_va_sax_db_track2_vn\data.npz: 179 notes, 11910 frames, 119.1s, voiced=92.2%, note_voicing=98.9%, env_corr=0.991
[OK] processed_vn\44_K515_vn_vn_va_va_vc_track1_vn\data.npz: 615 notes, 22520 frames, 225.2s, voiced=72.5%, note_voicing=97.7%, env_corr=0.987
[OK] processed_vn\44_K515_vn_vn_va_va_vc_track2_vn\data.npz: 451 notes, 22520 frames, 225.2s, voiced=69.3%, note_voicing=98.4%, env_corr=0.958

============================================================
汇总
============================================================
总文件数:       74
有问题的文件:   0
总音符数:       15776
总时长:         8093.5s (134.9min)
平均 note 内 voiced 比例: 97.4%
平均包络相关性:  0.990 (共 74 条有对比)
```