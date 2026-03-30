# 项目上下文

## 项目概述
MIDI-to-expression 连续表达生成系统。从 MIDI 音符序列生成帧级 f0（基频）和 amp（振幅），用于单声部管弦乐器（弦乐/管乐/铜管）。
- 目标投稿: AIMC 2026，如果结果够强冲 ISMIR 2026
- 数据集: URMP(133 tracks) + Bach10(40) + PHENICX(3) + TRIOS(9) = 185 tracks, ~4.7h
- 数据目录: datagen/solo/{URMP,Bach10,PHENICX,TRIOS}/processed_{inst}/{track}/data.npz

## 当前状况与核心问题

**f0 预测已经很好**（RPA 94.8%, MAE 23.4 cents），**amp 预测是核心瓶颈**（Corr 0.807）。
经过 55 个实验的系统性探索，amp 卡在 0.80 左右无法突破 0.90。

**amp 难的根本原因**：
1. 弦乐（小提琴等）的 amp 依赖弓法信息（弓压/弓速/触弦点），MIDI 里完全没有这些信息
2. 数据量太小（185 tracks, 4.7h），模型学不到足够的 amp 模式
3. 铜管已达 0.91，但弦乐只有 0.73，严重拉低均值
4. 去掉 instrument embedding 后 amp 从 0.807 降到 0.789（泛化性 vs 精度的 trade-off）

**之前的硬件限制**：本地 RTX 5070（8GB 显存），训练慢、评估卡、频繁蓝屏。

## 硬件升级

**现在有 8×A100 GPU 服务器**（每张 80GB 显存），算力和显存不再是瓶颈。这意味着：
- 可以用更大模型（Transformer Encoder、更深网络）
- 可以加 SynthSOD（47h 合成管弦乐数据）做预训练，数据量翻 10 倍
- 可以大 batch size、多卡并行、训练 1000+ epochs
- 评估不再 OOM，不需要显存限制

## 模型架构（当前最佳 exp039/exp048）
- **MIDI Encoder**: Linear(7→128) → 2层BiGRU(128×2) → 256维 condition
- **f0 Diffusion (Stage 2)**: 1ch ConditionalDDPM + 1D U-Net，DDIM采样50步
- **AmpPredictor**: Linear(256→256) → 2层BiGRU(hidden=128) → 可选Self-Attention → Linear(256→1) → log_amp
- 训练: Stage 1 训 Encoder+Baseline，Stage 2 冻结 Encoder 训 Diffusion+AmpPredictor
- velocity 从音频 RMS 估算（Dannenberg 2006 平方根映射）

## 当前指标
- **f0 RPA**: 94.8%, **f0 MAE**: 23.4 cents, **VDE**: ~7.6
- **Amp Corr**: 0.807 (带instrument embedding, exp039) / 0.789 (不带, exp048)
- 目标: Amp Corr > 0.90, f0 MAE < 20, VDE < 6.0
- 铜管(tpt/tbn) amp已达0.91，弦乐(vn)只有0.73，拉低均值

## 已尝试的 amp 优化（55个实验）
- ✅ 有效: velocity估算(+0.13), instrument embedding(+0.013), 更大AmpPredictor(+0.02)
- ❌ 无效: amp diffusion, f0-conditioning, pitch-anchor, correlation loss, model soup, TCN, note_position, residual diffusion, encoder unfreezing, augmentation, knowledge distillation
- 核心瓶颈: 弦乐amp依赖弓法信息（MIDI里没有），数据量小(185 tracks)

## 下一步方向（利用 A100 服务器）

### 提升 amp（优先级最高，目标 > 0.90）
1. **全局统计特征替代 instrument embedding**: 从 notes 算统计量（mean_pitch, pitch_range, note_density 等）拼到 frame_features，让模型自动推断乐器特征而不需要显式指定乐器
2. **SynthSOD(47h) 预训练+微调**: 在合成数据上预训练学通用 amp 模式，再在真实数据上微调。数据量翻 10 倍
3. **更大模型**: Transformer Encoder 替换 GRU，更深的 AmpPredictor，利用 A100 显存
4. **大 batch size + 多卡并行**: batch 64+，减少过拟合
5. 以上方向可以组合尝试

### 提升 f0（次优先）
- f0 MAE 从 22 降到 < 20
- VDE 从 7.6 降到 < 6.0

### 论文准备
- 和 MIDI-DDSP 做对比实验（投 ISMIR 必须）
- listening test（至少 10 人评分）
- 最终 figures 和 tables

## 自动化实验循环
- `run_experiments.sh` 驱动 worker+supervisor agent 交替
- worker: 准备实验→写训练脚本→记录结果
- supervisor: 审查结果→诊断问题→规划下一步
- 实验配置: experiments/configs/exp{ID}.yaml
- 结果: experiments/results/exp{ID}.json
- 日志: experiments/log.md

## 用户偏好
- 使用中文交流
- 离开时让实验自动跑，Claude自主做决策
- 不要轻言放弃，从失败中总结经验

