# 模型架构调研

本文档整理了与本项目任务（MIDI → 帧级 f0(t) + amp(t)）相关的模型架构。

---

## 一、核心参考（与本任务直接相关）

### 1. DDSP — Differentiable Digital Signal Processing

**论文**: Engel et al., ICLR 2020
**核心思想**: 将传统信号处理模块（谐波振荡器、滤波器、混响）嵌入可微分框架，用神经网络预测合成器控制参数而非直接预测波形。

**音频表示（参数空间）**:
- **f0(t)**: 基频 Hz，由预训练 CREPE 从音频中提取，帧率 250Hz（4ms）
- **l(t)**: 响度 dB，A-加权功率谱取对数
- **z(t)**: 16 维隐变量，由 30 维 MFCC 经单层 GRU 编码，表示残余音色信息

**解码器架构 (RnnFcDecoder)**:
```
f0  → FC Stack (512, 3层, ReLU) ─┐
l   → FC Stack (512, 3层, ReLU) ─┼→ Concat → GRU (512) → FC Stack (512, 3层)
z   → FC Stack (512, 3层, ReLU) ─┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ↓                     ↓                     ↓
            amplitude (1/帧)    harmonic dist (100/帧)    noise mags (65/帧)
                    │                     │                     │
                    ↓                     ↓                     ↓
            ┌─ Harmonic Additive Synth ──┐    Filtered Noise Synth
            │  100个正弦波叠加           │    白噪声 + 时变FIR滤波
            └────────────┬───────────────┘              │
                         └──────── 求和 → Reverb IR 卷积 → 输出波形
```

**损失函数**: 多尺度频谱损失（Multi-Scale Spectral Loss）
- 6 个 FFT 尺寸：2048, 1024, 512, 256, 128, 64
- 每个尺度：L1(线性幅度) + L1(对数幅度)

**关键特点**:
- 约 6M 参数，仅需约 10 分钟单乐器数据即可训练
- f0 和 loudness 不由模型预测，而是从音频中提取作为输入
- 信号处理的归纳偏置大幅降低了网络学习难度

---

### 2. MIDI-DDSP — Hierarchical MIDI-to-Audio

**论文**: Wu et al., ICLR 2022 (Oral)
**核心思想**: 将 MIDI→音频 分解为三层，每层引入可解释的瓶颈表示。
**与本项目的关系**: **最直接相关**——本项目的任务恰好对应其前两层（MIDI → 表达参数 → 帧级 f0/amp）。

**三层分级架构**:

| 层级 | 粒度 | 表示 | 模型 |
|------|------|------|------|
| Notes | 音符级 | MIDI pitch, onset, offset, velocity | 输入 |
| Performance | 音符级 | 6 个表达控制参数 | Expression Generator (GRU) |
| Synthesis | 帧级 | f0, amplitude, harmonics, noise | Synthesis Generator (BiLSTM + GRU / 膨胀卷积) |

**6 个音符级表达参数**:

| 参数 | 含义 |
|------|------|
| Volume | 音符整体响度 |
| Volume fluctuation | 响度随时间的变化幅度 |
| Volume peak position | 响度峰值在音符内的位置 |
| Vibrato | 颤音程度（从 f0 变化中提取） |
| Brightness | 音色亮度（谐波质心） |
| Attack noise | 起奏噪声量 |

**Expression Generator（音符级）**:
- 自回归 GRU，输入音符序列，预测每个音符的 6 个表达参数
- 表达参数离散化后用交叉熵损失训练

**Synthesis Generator（帧级）**:
- 输入构造：音符级表达参数 + pitch → 按音符时长展开到帧级，拼接 onset/offset 标志 + 音符内位置编码 → 线性映射到 256 维
- **f0 预测**：BiLSTM + 2 层 GRU (256)，自回归预测相对于 MIDI pitch 的偏移量，**量化为分类任务**（交叉熵损失），dropout=0.5
- **幅度/谐波/噪声预测**：膨胀卷积网络（WaveNet 风格），多尺度频谱损失 + 对抗损失（LSGAN + feature matching）

**训练**:
- 自底向上分阶段训练：先 DDSP 自编码器 → 再 Synthesis Generator → 最后 Expression Generator
- 推理自顶向下：MIDI → Expression → Synthesis → DDSP → Audio
- 数据集：URMP，3.75 小时，13 种弦乐/管乐器
- 训练时间：单块 RTX 8000 约 18 小时

**关键发现**:
- f0 用分类而非回归效果更好
- 对抗损失解决 one-to-many 映射导致的过度平滑问题
- 分层设计允许用户在任意层级介入控制

---

### 3. Jonason et al. 2023 — DDSP 吉他多弦合成

**论文**: Jonason et al., arXiv:2309.07658, DAFx 2024
**核心思想**: 将吉他的多声部问题分解为 6 根弦各自独立的单声部 DDSP 合成。

**架构创新**:
- 每根弦独立使用一个 DDSP 合成器（谐波振荡器 + 滤波噪声）
- 从 MIDI 预测每根弦的控制参数（f0, loudness），6 根弦输出求和
- 用 MLP + GRU 做参数预测

**关键发现**:
- **分类 > 回归**：将 pitch/loudness 离散化为分类 bins，效果优于连续回归
- 最简单的单阶段系统（MIDI 直接到合成参数）反而效果最好

---

### 4. Deep Performer — Score-to-Audio Transformer

**论文**: Dong et al., ICASSP 2022, arXiv:2202.06034
**核心思想**: 端到端乐谱→音频合成，用 Transformer 直接生成 mel 频谱图。

**架构**:
- **Encoder**: 3 层 Transformer（128 hidden, 2 heads, 512 FFN），编码音符级输入
- **Decoder**: 6 层 Transformer，自回归生成 mel 频谱帧
- **Polyphonic Mixer**: 桥接音符级→帧级，将同一时间帧内所有活跃音符的 embedding 求和
- **Note-wise Positional Encoding**: 音符内位置编码（区分 attack/sustain/release）
- **HiFi-GAN** vocoder 将 mel 转为波形

**三阶段流水线**: 对齐模型（预测表达性时值）→ 合成模型（生成 mel）→ Vocoder

**数据**: Bach 小提琴 6.5h + MAESTRO 钢琴

**与本项目的关系**: Polyphonic Mixer 和 Note-wise Positional Encoding 可借鉴用于多声部扩展。

---

## 二、表达性演奏建模（符号域，间接相关）

### 5. Performance RNN

**论文**: Oore et al., 2018, arXiv:1808.03715
**任务**: 生成带有表现力的钢琴 MIDI 演奏。

**事件词表（413 个 token）**:
- 128 NOTE-ON + 128 NOTE-OFF + 125 TIME-SHIFT (8ms 步进) + 32 VELOCITY

**架构**: 3 层 LSTM，每层 512 单元，自回归预测下一个 token。

**意义**: 证明了 LSTM 在演奏表达序列建模上的有效性，8ms 时间量化能捕获表达性微时值。

---

### 6. ScorePerformer — 分层风格编码

**论文**: ISMIR 2023
**任务**: 从乐谱预测表达性 MIDI 演奏参数（IOI、onset 偏移、时值、velocity）。

**架构**:
- Encoder-Decoder Transformer + 多层级 MMD-VAE 风格建模头
- 在全局、小节、拍、onset 四个层级建模演奏风格
- 用 SPMuple 表示（含局部 tempo 编码）
- 风格向量可映射到演奏标记（forte, piano 等）实现引导式渲染

**代码**: https://github.com/ilya16/ScorePerformer

---

### 7. Peransformer — 低信息量的表达渲染

**论文**: arXiv:2510.10175, 2025.10
**任务**: 仅从 MIDI 音符（不需详细音乐标注）预测 IOI、duration、velocity。

**架构**:
- Transformer Encoder + 回归输出头（3 维：IOI, duration, velocity）
- **Score-aware Discriminator** 对抗训练
- 数据集：ASAP-MIDI（音符级对齐的乐谱-演奏配对）

**意义**: 证明低信息量输入 + 对抗训练可接近需要详细标注的系统性能。

---

### 8. Pianist Transformer — 大规模预训练

**论文**: arXiv:2512.02652, 2025.12
**任务**: 乐谱→表达性钢琴 MIDI 演奏。

**架构**:
- **非对称 Transformer**：10 层 Encoder（深）+ 2 层 Decoder（轻，快速自回归）
- 每个音符 8 个 token [Pitch, IOI, Velocity, Duration, Pedal×4]，Encoder 将 8 个 embedding 压缩为 1 个 → **64× 注意力计算量压缩**
- 两阶段训练：(1) 10 万+ 小时无标注 MIDI 做 masked denoising 预训练 (2) 监督微调

**意义**: 大规模预训练在音乐领域的可行性验证。

---

## 三、扩散模型方向

### 9. DExter — 扩散模型做表达渲染

**论文**: arXiv:2406.14850, 2024
**任务**: 从乐谱生成表达性演奏参数。

**架构**:
- 条件扩散概率模型（DDPM）
- 将乐谱信息（onset, duration, pitch, voice）和演奏参数（beat period, velocity, timing, articulation, pedal）表示为 **2D 矩阵**（类似频谱图）
- 额外条件：中层感知特征（melodiousness, rhythmic complexity, dissonance 等）
- 支持风格迁移和感知引导生成

**与本项目的关系**: 扩散模型可以建模 one-to-many 映射（同一乐谱可有多种合理演奏），但增加了训练复杂度。

---

### 10. RenderBox — Diffusion Transformer 端到端

**论文**: arXiv:2502.07711, 2025.2
**任务**: 多乐器乐谱→表达性音频。

**架构**:
- **Diffusion Transformer (DiT)** 在自编码器隐空间中运行
- 多模态条件：文本提示（粗粒度风格）+ MIDI token（细粒度音符控制），通过 **cross-attention 联合条件**
- 课程学习：从简单合成逐步到表达性演奏

---

## 四、其他端到端参考

### 11. Tan et al. 2020 — GM-VAE 钢琴合成

**论文**: arXiv:2006.09833, ICML ML4MD Workshop 2020
**架构**: Gaussian Mixture VAE 在 mel 频谱图上操作，不同混合分量对应不同表达风格。WaveGlow vocoder 转波形。
**数据**: MAESTRO v2.0.0

### 12. MIDI-VALLE — 神经编解码语言模型

**论文**: arXiv:2507.08530, ISMIR 2025
**架构**: 改编 VALL-E（TTS 模型），用 Octuple MIDI 分词 + Piano-Encodec 音频编码，零样本风格适配。
**代码**: https://github.com/nii-yamagishilab/MIDI-VALLE

---

## 五、本项目模型设计

### 5.1 问题定义与动机

本项目的任务是从离散的 MIDI 音符序列（每个音符包含 onset、offset、pitch、velocity）生成帧级的连续演奏信号 f0(t) 和 amp(t)，帧率为 100Hz（每帧 10ms）。

这个任务的核心挑战在于它是一个 **one-to-many 映射**：同一段 MIDI 乐谱可以被不同的演奏家以不同的方式演绎——有人颤音深而慢，有人颤音浅而快；有人起奏柔和，有人起奏果断——这些都是合理的演奏。MIDI-DDSP [Wu et al., 2022, §2] 在其 Synthesis Generator 的训练中明确遇到了这个问题，并指出 one-to-many 映射会导致确定性模型的输出**过度平滑**（over-smoothing），因为模型在多个合理目标之间取均值。他们的解决方案是引入对抗损失（LSGAN + feature matching）来缓解这一问题。

本项目提出另一种思路来处理 one-to-many 问题：使用条件扩散模型（Conditional DDPM）。与确定性模型不同，扩散模型学习的是条件分布 P(f0, amp | MIDI) 本身，而非该分布的某个统计量。每次采样都从随机噪声出发，通过迭代去噪生成一种具体的、合理的演奏轨迹。DExter [arXiv:2406.14850, 2024, §9] 已经在**符号域**的演奏参数（IOI、velocity、timing 等离散量）上验证了条件扩散模型用于表达性演奏生成的可行性。本项目将这一思路扩展到**帧级连续信号**（f0 和 amp），这是目前尚未被探索的方向。

需要说明的是，"确定性模型会把颤音平均掉"这一推断目前**没有论文直接通过实验验证**。MIDI-DDSP 报告了 one-to-many 导致的过度平滑现象，但它的实验对象是幅度/谐波分布，而非 f0 颤音轨迹。本项目的实验 3（§5.6）正是要验证这一假设：如果实验结果确实显示扩散模型在颤音段优于确定性 baseline，则支持这一推断；如果差别不大，这本身也是一个有价值的发现。

### 5.2 整体架构

整个系统由三个模块组成：一个共享的 MIDI Encoder 负责将离散音符转化为帧级条件特征，一个确定性 BiGRU 模型作为 baseline，一个条件扩散模型作为本项目提出的方法。两个预测模型共享同一个 MIDI Encoder，以确保对比实验的公平性。

训练阶段的数据流如下：MIDI 音符序列经过 MIDI Encoder 得到帧级条件特征 C(t)；对于 baseline 模型，C(t) 直接输入 BiGRU 预测 f0 和 amp；对于扩散模型，ground truth 的 f0(t) 和 amp(t) 被加上随机噪声得到 x_t，然后 1D U-Net 以 x_t、C(t) 和扩散时间步 t 为输入，预测所加的噪声。

推理阶段：baseline 模型只需一次前向传播即可得到输出；扩散模型则从纯高斯噪声出发，以 C(t) 为条件，经过多步迭代去噪，最终得到生成的 f0(t) 和 amp(t)。由于每次采样的起始噪声不同，扩散模型每次生成的结果也不同。

### 5.3 模块一：MIDI Encoder

MIDI Encoder 的作用是将离散的音符事件转化为与输出等长的帧级条件序列 C(t)。这个模块被 baseline 和扩散模型共享。

**第一步是音符到帧级的展开。** 对于序列中的每一帧（对应 10ms 时间窗口），根据当前时间找到活跃音符（如果有的话），提取 7 维特征：是否在音符内（is_voiced，0 或 1）、归一化 MIDI 音高（midi_pitch / 127）、归一化力度（velocity / 127）、音符内相对位置（position_in_note，从 0 到 1 线性变化）、距 onset 的时间（取 log 变换）、onset 标志（onset 前后 3 帧内为 1）、offset 标志（offset 前后 3 帧内为 1）。不在任何音符内的帧，所有特征为 0。

其中，**音符内相对位置**（position_in_note）这一设计参考了两个来源。MIDI-DDSP [Wu et al., 2022, §2] 在其 Synthesis Generator 的输入中使用了一个"scalar note positioning code"，它是一个 0 到 1 的标量，表示当前帧在音符内的相对位置，目的是让模型区分 attack、sustain 和 release 阶段的不同行为。Deep Performer [Dong et al., ICASSP 2022, §4] 提出了"Note-wise Positional Encoding"实现类似目的，让 Transformer 解码器知道当前帧处于音符的哪个阶段。本项目采用与 MIDI-DDSP 相同的简单线性编码。

**onset/offset 标志**同样参考了 MIDI-DDSP [Wu et al., 2022, §2]。在其 Synthesis Generator 中，二值的 onset 和 offset 指示器被拼接到帧级输入中，帮助模型识别音符边界。

**第二步是上下文编码。** 将 7 维帧级特征通过一个线性层映射到 128 维，再通过一个双向 GRU（BiGRU）编码上下文信息，输出 256 维的帧级条件特征 C(t)。使用 BiGRU 的原因是演奏表达存在上下文依赖——例如，前一个音符的结束方式会影响下一个音符的起奏风格，后续的旋律走向也可能影响当前音符的力度处理。MIDI-DDSP [Wu et al., 2022, §2] 的 Synthesis Generator 使用了 BiLSTM + GRU 的组合来建模这种时序依赖。本项目简化为单层 BiGRU，因为本项目的帧级输入已经包含了音符级信息（pitch、velocity 等），不需要像 MIDI-DDSP 那样从更抽象的表达参数中恢复这些信息。

### 5.4 模块二：Baseline 模型（确定性 BiGRU）

Baseline 模型的作用是提供一个确定性的对照方法。它在 MIDI Encoder 输出的条件特征 C(t) 上再加一层 BiGRU（隐藏维度 512），然后通过两个独立的输出头分别预测 f0 和 amp。

**f0 使用分类预测而非回归。** 这一决策有两个独立的实验依据。MIDI-DDSP [Wu et al., 2022, §2] 在其 Synthesis Generator 中将 f0 的预测设计为分类任务：将相对于 MIDI 音高的偏移量化为离散 bin，用交叉熵损失训练，而非直接回归连续 f0 值。Jonason et al. [arXiv:2309.07658, 2023, §3] 在 DDSP 吉他合成的实验中独立验证了这一结论：他们对比了分类（classification）和回归（regression）两种方式预测 pitch 和 loudness 控制特征，结果分类方式在音频质量上优于回归。这两篇工作在不同乐器（管弦乐器 vs 吉他）、不同任务细节上得出了一致的结论，说明分类优于回归在连续控制参数预测中具有一定的普适性。

具体实现上，将 f0 相对于 MIDI 音高的 cent 偏移量化为 81 个 bin：-200 到 +200 cents 范围内每 5 cents 一个 bin（共 80 个），加 1 个 unvoiced bin（表示当前帧无基频）。±200 cents 的范围足以覆盖正常演奏中的颤音（通常 ±50 cents 以内）和小幅滑音。推理时，取 softmax 输出的加权期望（soft argmax）得到连续 f0 值。

**amp 使用回归预测。** 振幅是一个单调的、物理意义明确的量（越大越响），不存在 f0 那样的多峰分布问题，因此直接用 sigmoid 输出归一化振幅，MSE 损失训练即可。

总损失为两个任务的加权和：L_baseline = L_CE(f0) + λ · L_MSE(amp)，其中 λ 为平衡系数，需要通过实验调节。

### 5.5 模块三：条件扩散模型（Proposed）

#### 5.5.1 为什么选择扩散模型

如 §5.1 所述，MIDI 到演奏信号的映射本质上是 one-to-many 的。现有工作中处理这一问题有两种思路：MIDI-DDSP [Wu et al., 2022, §2] 使用了对抗训练（GAN）来避免过度平滑；DExter [arXiv:2406.14850, 2024, §9] 则使用条件扩散模型来建模演奏参数的条件分布。

本项目选择扩散模型而非 GAN，原因有二。第一，扩散模型的训练比 GAN 更稳定，不存在模式坍缩和训练不稳定的问题，这对于数据量有限（135 分钟）的场景尤为重要。第二，扩散模型天然支持多次采样，可以从同一个 MIDI 条件生成多种不同的演奏，这种多样性采样能力本身就是一个有价值的实验对象和应用特性。

#### 5.5.2 扩散过程

采用标准的 DDPM（Denoising Diffusion Probabilistic Model）框架。定义目标信号 x₀ = [f0(t), amp(t)]，形状为 (T, 2)，其中 T 为帧数。

**前向过程**按照预定义的噪声调度 β₁, β₂, ..., β_T（本项目使用余弦调度，T=1000）逐步向 x₀ 添加高斯噪声。利用重参数化技巧，可以直接从 x₀ 得到任意时间步 t 的噪声版本：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$。当 t 接近 T 时，$\bar{\alpha}_t$ 趋近于 0，x_t 几乎变成纯噪声。

**反向过程**从纯噪声 x_T 出发，用神经网络 ε_θ 在每一步预测当前的噪声成分，然后去除一部分噪声，逐步恢复出干净的信号。每一步都以 MIDI 条件 C(t) 为额外输入，确保生成的信号符合给定的乐谱。

**训练目标**是标准的噪声预测损失：

$$L_{diffusion} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, C) \|^2 \right]$$

即在所有时间步 t、所有训练样本 x₀、所有随机噪声 ε 上，最小化预测噪声与实际噪声的均方误差。

#### 5.5.3 去噪网络：条件 1D U-Net

去噪网络的架构采用 1D U-Net。选择 U-Net 的原因是它通过下采样-上采样结构和 skip connection，能够同时捕捉不同尺度的时序特征：下采样路径在压缩的时间尺度上感知乐句级的轮廓和力度走向，上采样路径借助 skip connection 恢复帧级的细节（如颤音的具体波形）。DiffWave [Kong et al., ICLR 2021] 率先将 1D 卷积网络用于扩散音频生成，证明了 1D 卷积架构在处理时序音频信号时的有效性，不过 DiffWave 使用的是膨胀卷积而非 U-Net 结构。本项目采用 U-Net 而非膨胀卷积，是因为 f0+amp 信号的帧率（100Hz）远低于音频采样率（16kHz），序列长度适中，U-Net 的下采样不会导致信息过度丢失。

网络的输入是噪声信号 x_t（2 通道，对应 f0 和 amp）与 MIDI 条件 C(t)（256 通道）在通道维度的拼接，共 258 通道。网络包含 3 个下采样块、1 个中间块和 3 个上采样块，通道数依次为 128→256→512→512→256→128。每个块内包含残差卷积块（ResBlock）和可选的自注意力层。最后通过一个 Conv1d 将通道数映射回 2，输出预测的噪声 ε。

**扩散时间步 t 的注入**采用 FiLM（Feature-wise Linear Modulation）机制。WaveGrad [Chen et al., 2020] 在其扩散音频生成模型中使用了 FiLM 条件注入，将噪声级别通过 feature-wise 仿射变换注入中间层。本项目参考这一做法：将时间步 t 通过正弦位置编码和 MLP 映射为一个向量，然后在每个 ResBlock 中对归一化后的特征做仿射变换 h = γ(t) · Norm(h) + β(t)，其中 γ 和 β 由 t 的嵌入向量通过线性层生成。这告诉网络当前的噪声强度，使其据此调整去噪力度。

**MIDI 条件的注入**采用最直接的通道拼接方式：将 C(t) 与 x_t 在输入端 concat。这比 cross-attention 实现简单且计算量小，适合本项目的序列长度和数据规模。RenderBox [arXiv:2502.07711, 2025, §10] 使用了 cross-attention 做 MIDI 条件注入，但它面对的是更复杂的多模态条件（文本+MIDI）和更长的序列，需要更灵活的注入方式。

#### 5.5.4 数据归一化

扩散模型假设目标信号近似服从标准高斯分布，因此需要对 f0 和 amp 进行归一化：

**f0 的归一化**：首先将 Hz 转换为相对于当前音符 MIDI 音高的 cent 偏移（1 semitone = 100 cents），然后除以 200，使大部分值落在 [-1, 1] 范围内。使用 cent 偏移而非绝对 Hz 的原因是：不同音高的音符，相同的颤音深度对应不同的 Hz 变化量（高音区更大），但对应相同的 cent 偏移。这一表示方式与 MIDI-DDSP [Wu et al., 2022, §2] 的 f0 偏移预测一致。无声帧（f0=0）单独标记，不参与归一化。

**amp 的归一化**：取对数后做 z-score 标准化（减均值除以标准差），使其近似服从 N(0,1)。

#### 5.5.5 采样加速

标准 DDPM 需要 T=1000 步采样，速度较慢。可以使用 DDIM（Denoising Diffusion Implicit Models，Song et al., 2020）将采样步数减少到 50-100 步，在几乎不损失质量的前提下大幅提速。DDIM 通过将随机微分方程替换为确定性常微分方程，使得采样过程可以跳步执行。

### 5.6 训练策略与评估方案

训练策略（数据划分、分阶段训练、超参数、损失函数设计）和评估方案（保真度指标、合理性指标、多样性指标、听觉评估）的详细内容见 [training.md](training.md)。

### 5.7 预估模型规模与训练资源

| 模块 | 参数量估算 |
|------|-----------|
| MIDI Encoder（Linear + BiGRU） | ~400K |
| Baseline（BiGRU + 输出头） | ~2M |
| 1D U-Net（去噪网络） | ~3-5M |

训练数据约 135 分钟（约 810K 帧），模型参数 3-5M（单个模型），数据与参数的比例合理。作为参考，DDSP [Engel et al., 2020, §1] 约 6M 参数，仅用约 10 分钟数据即可训练出高质量的单乐器合成器，说明在有强归纳偏置的条件下，小数据量可以支撑数百万参数的模型。本项目的数据量（135 分钟）远大于 DDSP 的最低需求。MIDI-DDSP [Wu et al., 2022, §2] 使用 3.75 小时 URMP 数据训练整个三层系统（包括 DDSP 自编码器），在单块 RTX 8000 上约 18 小时完成。本项目不包含 DDSP 合成层，模型更小，预计训练时间更短。详细的计算资源估算见 [training.md §七](training.md)。

### 5.8 与现有工作的关系总结

本项目的设计决策均有文献依据，但在整体组合方式上是新的：

- **分层建模框架**来自 MIDI-DDSP [Wu et al., 2022]，但本项目去掉了 DDSP 合成器层，直接以 f0+amp 为最终输出
- **帧级 MIDI 编码**（音符内位置编码 + onset/offset 标志 + BiGRU 上下文）综合参考了 MIDI-DDSP [Wu et al., 2022] 和 Deep Performer [Dong et al., 2022]
- **f0 分类预测**（baseline 中使用）由 MIDI-DDSP [Wu et al., 2022] 和 Jonason et al. [2023] 独立验证
- **条件扩散模型用于演奏表达生成**的思路来自 DExter [2024]，但 DExter 工作在符号域（离散参数），本项目将其扩展到帧级连续信号域
- **1D 卷积去噪网络**参考了 DiffWave [Kong et al., 2021] 的 1D 架构设计
- **FiLM 条件注入**参考了 WaveGrad [Chen et al., 2020]

**本项目的新颖之处**在于：将条件扩散模型应用于帧级连续演奏信号（f0 + amp）的生成——这个具体问题（扩散 × 帧级 f0/amp × MIDI 条件）目前在已有文献中没有被直接研究过。DExter 做了扩散 × 符号域演奏参数；MIDI-DDSP 做了 MIDI 条件 × 帧级 f0/amp 但用的是确定性模型+GAN；本项目将这两条线索结合。

### 5.9 论文结构（AIMC 2026）

```
1. Introduction
   - MIDI 的局限性：无法表达连续演奏细节（颤音、力度变化等）
   - 核心问题：MIDI → f0+amp 是 one-to-many 映射，确定性模型存在过度平滑
   - 贡献：提出用条件扩散模型生成帧级连续演奏信号，对比确定性方法

2. Related Work
   - 分层建模与 DDSP：DDSP [Engel et al., 2020], MIDI-DDSP [Wu et al., 2022]
   - 表达性演奏建模：Performance RNN [Oore et al., 2018], ScorePerformer [ISMIR 2023]
   - 扩散模型在音乐中的应用：DExter [2024], RenderBox [2025]

3. Method
   - 3.1 问题形式化与 one-to-many 动机
   - 3.2 MIDI Encoder（共享模块）
   - 3.3 Baseline: 确定性 BiGRU + f0 分类 + amp 回归
   - 3.4 Proposed: 条件 DDPM + 1D U-Net

4. Experiments
   - 4.1 数据集与实验设置（URMP, 3 种乐器, 74 tracks, 135 分钟）
   - 4.2 质量对比（RPA, MAE, Amp Corr, VDE）
   - 4.3 多样性分析（多次采样的 f0 方差 + 可视化）
   - 4.4 可视化案例（颤音段、跳进段、起奏段）

5. Conclusion
   - 总结扩散模型在帧级演奏信号生成中的优劣
   - 未来工作：多声部扩展、接入下游音色合成系统
```
