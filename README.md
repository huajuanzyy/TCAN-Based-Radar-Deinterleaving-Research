# 项目定位

本仓库面向任务书中的未知辐射源数量雷达脉冲分选问题。当前代码继承了早期 TCAN 论文复现仓库中的基础能力，包括 synthetic PDW 生成、DTOA 预处理、binary 输入预处理、focal loss、信号稀疏仿真、非理想接收条件仿真，以及 TCAN sequence-labeling 模型。

本仓库的长期目标不是停留在固定类别数的监督式分类，而是逐步转向脉冲级 embedding learning 与 non-parametric clustering。后续可以使用 TCAN 或 Transformer encoder 生成上下文脉冲嵌入，再使用聚类方法在未知源数条件下完成分选。

## Phase 1A：Turing Synthetic Radar Dataset Loader

Phase 1A 只实现 Turing Synthetic Radar Dataset 的数据加载与字段适配。本阶段不实现 triplet loss、不实现 HDBSCAN、不实现源数估计，也不实现 embedding clustering 训练目标。

TSRD 字段到内部 PDW 格式的映射如下：

```text
TOA              -> TOA
Pulse Width      -> PW
Centre Frequency -> RF
Angle of Arrival -> AOA
Amplitude        -> PA
```

TSRD adapter 返回：

```text
pdw_array: [TOA, PW, RF, AOA, PA]
labels: 当前 pulse train 内部的整数 emitter labels
```

loader 会按 `TOA` 升序稳定排序，并用相同顺序重排 labels。labels 会在当前 pulse train 内部重新映射为连续整数，因此不同文件中的相同 label ID 不应被默认理解为同一个物理辐射源。

当前支持的 feature set：

```text
4d DTOA input:   [DTOA, PW, RF, AOA]
5d DTOA input:   [DTOA, PW, RF, AOA, PA]
4d binary input: [binary_presence, PW, RF, AOA]
5d binary input: [binary_presence, PW, RF, AOA, PA]
```

synthetic mode 仍然是默认模式：

```bash
python train.py --data-source synthetic --input-format dtoa --epochs 2
python train.py --data-source synthetic --input-format binary --epochs 2
```

TSRD mode 需要一个本地 pulse train 文件，并且文件中包含可识别的字段名：

```bash
python train.py --data-source tsrd --tsrd-path <path_to_file> --input-format dtoa --feature-set 5d --epochs 2
```

如果使用 `--data-source tsrd` 但没有提供 `--tsrd-path`，程序会给出明确的参数错误。TSRD 数据文件应保留在本地，不应提交到 Git。

# Radar Pulse Deinterleaving Research Project

本仓库面向复杂电磁环境下的多辐射源雷达脉冲分选任务。

本仓库基于已有 TCAN 论文复现代码继续扩展，但目标不再局限于固定类别数的监督式 sequence labeling，而是逐步转向：

1. 接入公开雷达脉冲去交错数据集；
2. 构建脉冲级上下文嵌入表示；
3. 使用 triplet loss / supervised contrastive loss 进行度量学习；
4. 使用 HDBSCAN / DBSCAN / 层次聚类进行未知源数分选；
5. 使用 V-measure、ARI、AMI、Homogeneity、Completeness、源数估计误差等指标评估分选效果。

当前从原 TCAN 复现仓库继承了以下基础能力：

- synthetic PDW generation
- DTOA input
- binary input
- focal loss
- signal sparsity
- nonideal receiving conditions

下一阶段优先接入 Turing Synthetic Radar Dataset，并先完成数据加载与格式适配。

## Legacy TCAN Reproduction Components

用于复现论文：

> Deinterleaving of Intercepted Radar Pulse Streams via Temporal Convolutional Attention Network

当前代码支持两个输入版本：

- Phase 1：基于 DTOA 输入的 TCAN 最小可运行复现流程。
- Phase 2：在保留 DTOA pipeline 的基础上，新增 binary input 支持。
- Phase 3：在保留 DTOA/binary input pipeline 的基础上，新增 focal loss 支持。
- Phase 4：新增 signal sparsity 仿真与训练入口。
- Phase 5：新增 nonideal receiving conditions，包括 measurement error、random pulse loss 和 spurious pulses。

这个版本的重点不是完整复现论文中的全部实验，而是先把最核心的端到端训练流程跑通：

1. 生成多个传统雷达源的脉冲描述字数据。
2. 将多个雷达源的脉冲按到达时间排序，形成交错脉冲流。
3. 将 `TOA` 转换为 `DTOA`。
4. 支持使用 `[DTOA, PW, RF, DOA]` 或 `[binary_presence, PW, RF, DOA]` 作为模型输入特征。
5. 使用 TCAN 模型进行逐脉冲序列标注。
6. 输出每个脉冲位置对应的雷达源类别。
7. 计算并打印 recall per class、average recall 和 confusion matrix。

## Phase 1: DTOA-based TCAN Pipeline

Phase 1 已实现：

- 四类传统雷达源仿真：
  - 固定 PRI 雷达
  - 抖动 PRI 雷达
  - 参差 PRI 雷达
  - 驻留与切换 PRI 雷达
- 每个脉冲包含以下 PDW 字段：
  - `TOA`
  - `PW`
  - `RF`
  - `DOA`
  - `label`
- 多雷达源脉冲合并后按 `TOA` 升序排序。
- DTOA 预处理：
  - `DTOA_i = TOA_i - TOA_{i-1}`
  - 第一个脉冲的 `DTOA` 设为 `0`
- 输入特征为 `[DTOA, PW, RF, DOA]`。
- 对每个特征维度做 min-max 归一化。
- 构造固定长度序列窗口。
- 使用 TCAN 进行 sequence labeling，而不是整条序列分类。
- 使用 PyTorch `CrossEntropyLoss`，并在所有脉冲位置上计算损失。
- 评估指标包括：
  - 每类 recall
  - 平均 recall
  - 混淆矩阵

Phase 1 本身不包含 binary input；binary input 已在 Phase 2 中作为新增功能实现。

当前仍未实现：

- MFR 雷达仿真
- CDIF、SDIF、PRI Transform
- 论文级大规模实验
- 多组对比图和复杂可视化

## Phase 2: Binary Input Support

Phase 2 在不重写 DTOA pipeline 的前提下，新增 binary input 训练方式。

binary input 会将 `TOA` 按采样间隔 `Ts` 离散成 time bins。每个 time bin 的输入特征为：

```text
[binary_presence, PW, RF, DOA]
```

其中：

- `binary_presence = 1` 表示该 time bin 中存在脉冲。
- `binary_presence = 0` 表示该 time bin 中没有脉冲。
- 如果 time bin 中没有脉冲，则 `PW/RF/DOA` 都填 `0`。
- 如果 time bin 中有脉冲，则将该脉冲的 `PW/RF/DOA` 归一化后填入对应位置。

binary input 的标签设计与 DTOA input 不同：

- DTOA input 只对真实脉冲位置做序列标注，类别数为 `4`，标签为 `0, 1, 2, 3`。
- binary input 包含没有脉冲的空 time bins，因此增加 background class，类别数为 `5`。
- binary input 的类别编号连续，满足 `CrossEntropyLoss` 要求：

```text
background = 0
radar label 0 -> class 1
radar label 1 -> class 2
radar label 2 -> class 3
radar label 3 -> class 4
```

binary input 评估时会同时打印：

- overall average recall：包含 background class 0 和 radar classes 1-4。
- pulse-only average recall：只对 radar classes 1-4 求平均，不把 background class 0 计入平均值。

当前 binary input 的简化限制：

- 如果多个脉冲落入同一个 time bin，当前版本暂时保留最早 `TOA` 的那个脉冲。
- 后续可以进一步实现碰撞统计、多脉冲聚合策略或更细粒度采样。

## Phase 3: Focal Loss Support

Phase 3 新增 focal loss，同时保留标准 cross entropy loss。

训练脚本支持：

```text
--loss ce
--loss focal
--gamma
--alpha-mode none 或 inverse_freq
```

focal loss 的逐位置定义为：

```text
L = - alpha_c * (1 - p_c)^gamma * log(p_c)
```

其中：

- `p_c` 是真实类别 `c` 的预测概率。
- `gamma` 控制对简单样本的抑制强度。
- `alpha_c` 是类别权重。

当前支持的 alpha 模式：

- `none`：不使用类别权重。
- `inverse_freq`：根据训练窗口中的标签频次计算反频率权重。

`inverse_freq` 的权重计算方式为：

```text
alpha_c = total_count / (num_classes * count_c)
```

如果某个类别在训练窗口中没有出现，则该类别权重设为 `0`。

focal loss 和 cross entropy loss 都在所有序列位置上计算。训练时仍然保持：

```text
logits: [B, T, C] -> [B*T, C]
labels: [B, T]    -> [B*T]
```

当前 focal loss 只作为损失函数加入；MFR 和额外 baseline 不在当前实现范围内。

## Phase 4: Signal Sparsity

Phase 4 新增 signal sparsity 仿真，用于模拟雷达主瓣扫描导致的周期性可见和长时间不可见。

signal sparsity 的形式是：

```text
visible pulse segment -> long missing interval -> visible pulse segment -> long missing interval
```

它和 random pulse loss 不同：

- random pulse loss 是随机删除单个脉冲。
- signal sparsity 是删除连续时间区间内的脉冲，形成 discontinuous but periodic intercepted pulse segments。

当前实现使用统一 scan gate：

```text
scan_period = visible_duration * (1 + gap_ratio)
phase = (TOA + phase_offset) % scan_period
```

如果：

```text
phase < visible_duration
```

则保留该脉冲；否则删除该脉冲。

支持的 sparsity ratio：

```text
none: 不启用 signal sparsity
1:3 : gap_ratio = 3，理论保留比例约 25%
1:5 : gap_ratio = 5，理论保留比例约 16.7%
1:8 : gap_ratio = 8，理论保留比例约 11.1%
```

signal sparsity 的位置在 pipeline 中非常重要。当前流程是：

```text
generate PDW stream
-> apply signal sparsity
-> DTOA 或 binary preprocessing
-> windowing
-> train TCAN
```

也就是说，sparsity 作用在原始 PDW 上，而不是在 DTOA 或 binary preprocessing 之后再删除样本。

运行 DTOA sparsity smoke tests：

```bash
python train.py --input-format dtoa --loss focal --sparsity-ratio none --epochs 2
python train.py --input-format dtoa --loss focal --sparsity-ratio 1:3 --epochs 2
python train.py --input-format dtoa --loss focal --sparsity-ratio 1:5 --epochs 2
python train.py --input-format dtoa --loss focal --sparsity-ratio 1:8 --epochs 2
```

运行 binary sparsity smoke tests：

```bash
python train.py --input-format binary --loss focal --sparsity-ratio none --epochs 2
python train.py --input-format binary --loss focal --sparsity-ratio 1:3 --epochs 2
python train.py --input-format binary --loss focal --sparsity-ratio 1:5 --epochs 2
python train.py --input-format binary --loss focal --sparsity-ratio 1:8 --epochs 2
```

Phase 4 当前限制：

- 所有 emitter 暂时使用统一 scan gate。
- 没有实现 MFR 或 baseline 方法。

## Phase 5: Nonideal Conditions

Phase 5 在保留 DTOA input、binary input、focal loss 和 signal sparsity 的基础上，新增三类接收端非理想条件。

nonideal conditions 的应用位置是 PDW 层，位于 signal sparsity 之后、DTOA/binary preprocessing 之前：

```text
generate PDW stream
-> optional signal sparsity
-> nonideal conditions
-> DTOA 或 binary preprocessing
-> windowing
-> train TCAN
```

### Measurement Error

measurement error 用于模拟接收机测量误差。当前实现对以下 PDW 字段添加高斯噪声：

```text
TOA, PW, RF, DOA
```

对应命令行参数：

```text
--toa-error-std
--pw-error-std
--rf-error-std
--doa-error-std
```

`PW` 加噪后会裁剪到正数。`TOA` 加噪后会重新按 `TOA` 排序，并同步重排 labels。

### Random Pulse Loss

random pulse loss 按概率随机删除单个脉冲：

```text
--pulse-loss-rate
```

它和 signal sparsity 不同：

- signal sparsity 删除连续时间区间内的脉冲，形成周期性可见/不可见片段。
- random pulse loss 随机删除独立脉冲，不保证形成连续缺失区间。

### Spurious Pulses

spurious pulses 用于插入虚假脉冲：

```text
--spurious-rate
```

当前实现会按当前脉冲数量的一定比例插入虚假脉冲。虚假脉冲的 `TOA` 在当前观测时间范围内均匀采样，`PW/RF/DOA` 在当前 PDW 全局范围内均匀采样。插入后会重新按 `TOA` 排序并同步 labels。

spurious class 标签设计：

```text
DTOA input:
  radar labels: 0, 1, 2, 3
  spurious label: 4

Binary input:
  background label: 0
  radar labels: 1, 2, 3, 4
  spurious label: 5
```

因此类别数为：

```text
DTOA without spurious: 4
DTOA with spurious:    5
Binary without spurious: 5
Binary with spurious:    6
```

nonideal smoke tests：

```bash
python train.py --input-format dtoa --loss focal --epochs 2 --toa-error-std 2.0
python train.py --input-format dtoa --loss focal --epochs 2 --pulse-loss-rate 0.10
python train.py --input-format dtoa --loss focal --epochs 2 --spurious-rate 0.10

python train.py --input-format binary --loss focal --epochs 2 --toa-error-std 2.0
python train.py --input-format binary --loss focal --epochs 2 --pulse-loss-rate 0.10
python train.py --input-format binary --loss focal --epochs 2 --spurious-rate 0.10
```

原始无 nonideal 条件命令仍然可运行：

```bash
python train.py --input-format dtoa --loss focal --epochs 2
python train.py --input-format binary --loss focal --epochs 2
```

当前限制：

- measurement error 只使用独立高斯噪声。
- random pulse loss 只使用统一删除概率。
- spurious pulses 只从全局 PDW 范围均匀采样。
- spurious pulses 暂未模拟更复杂的物理来源或干扰机制。

## 项目结构

```text
TCAN/
  README.md
  AGENTS.md
  train.py
  src/
    data_simulator.py
    losses.py
    nonideal.py
    preprocessing.py
    sparsity.py
    model_tcan.py
    metrics.py
    utils.py
```

各文件作用：

- `train.py`
  - 主训练入口。
  - 负责生成数据、划分训练集和测试集、预处理、构造窗口、训练模型、评估结果。

- `src/data_simulator.py`
  - 生成四类传统雷达源的仿真 PDW 数据。
  - 每个脉冲输出 `TOA, PW, RF, DOA, label`。
  - 将不同雷达源的脉冲合并，并按 `TOA` 排序。

- `src/preprocessing.py`
  - 将 `TOA` 转换为 `DTOA`。
  - 构造输入特征 `[DTOA, PW, RF, DOA]`。
  - 构造 binary input 特征 `[binary_presence, PW, RF, DOA]`。
  - 执行 min-max 归一化。
  - 生成固定长度序列窗口。

- `src/sparsity.py`
  - 在原始 PDW 层面应用周期性可见/不可见 scan gate。
  - 用于模拟 signal sparsity，而不是随机脉冲丢失。

- `src/nonideal.py`
  - 在原始 PDW 层面模拟 measurement error、random pulse loss 和 spurious pulses。
  - 所有 nonideal conditions 都发生在 DTOA/binary preprocessing 之前。

- `src/losses.py`
  - 实现 cross entropy 与 focal loss 的统一构造入口。
  - 支持 focal loss 的 `gamma` 和 `inverse_freq` alpha。

- `src/model_tcan.py`
  - 实现 TCAN 模型。
  - 包含 TCN residual block、dilated causal convolution、weight normalization、ReLU、dropout、residual connection、1x1 channel matching、自注意力模块和最终分类层。

- `src/metrics.py`
  - 计算 confusion matrix。
  - 计算每类 recall。
  - 计算 average recall。

- `src/utils.py`
  - 设置随机种子。
  - 自动选择 CPU 或 GPU。

## 数据格式

仿真生成的 PDW 数据包含 5 个字段：

```text
[TOA, PW, RF, DOA, label]
```

其中：

- `TOA`：time of arrival，到达时间。
- `PW`：pulse width，脉冲宽度。
- `RF`：radio frequency，射频。
- `DOA`：direction of arrival，到达角。
- `label`：雷达源类别标签，取值为 `0, 1, 2, 3`。

多个雷达源生成后会被合并为一个 interleaved pulse stream，并按照 `TOA` 升序排序。

## DTOA 预处理

对于按 `TOA` 排序后的脉冲流，DTOA 定义为：

```text
DTOA_i = TOA_i - TOA_{i-1}
```

第一个脉冲没有前一个脉冲，因此设为：

```text
DTOA_0 = 0
```

模型最终使用的每个脉冲输入特征为：

```text
[DTOA, PW, RF, DOA]
```

即输入维度 `D = 4`。

每个特征维度使用 min-max 归一化：

```text
x_norm = (x - x_min) / (x_max - x_min + 1e-8)
```

## Tensor Shape

本项目是逐脉冲序列标注任务。

输入 tensor shape：

```text
[B, T, D]
```

其中：

- `B`：batch size 或窗口数量。
- `T`：序列长度。
- `D`：输入特征维度，当前为 `4`。

标签 tensor shape：

```text
[B, T]
```

模型输出 logits shape：

```text
[B, T, C]
```

其中：

- `C`：类别数。DTOA input 下为 `4`；binary input 下因为包含 background class，所以为 `5`。

这意味着模型会对每一个脉冲位置输出一个类别预测，而不是只对整条序列输出一个类别。

## 损失函数

当前版本使用标准 PyTorch `CrossEntropyLoss`。

训练时将 logits 和 labels reshape 为：

```text
logits: [B, T, C] -> [B*T, C]
labels: [B, T]    -> [B*T]
```

然后在所有脉冲位置上计算交叉熵损失。

当前版本支持 `CrossEntropyLoss` 和 focal loss，可通过 `--loss ce` 或 `--loss focal` 选择。

## 运行方式

请先确保环境中已安装：

- Python
- NumPy
- PyTorch

然后在项目根目录运行 DTOA input：

```bash
python train.py --input-format dtoa --epochs 2
```

运行 binary input：

```bash
python train.py --input-format binary --epochs 2 --ts 10.0
```

运行 focal loss：

```bash
python train.py --input-format dtoa --loss focal --epochs 2
python train.py --input-format binary --loss focal --epochs 2
```

运行 signal sparsity：

```bash
python train.py --input-format dtoa --loss focal --sparsity-ratio 1:3 --epochs 2
python train.py --input-format binary --loss focal --sparsity-ratio 1:3 --epochs 2
```

脚本会自动选择运行设备：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

如果当前默认 Python 环境没有安装 PyTorch，可以切换到已安装 PyTorch 的环境后再运行。例如本机可使用：

```powershell
D:\Anaconda3\envs\pytorch\python.exe train.py --input-format dtoa --epochs 2
```

训练脚本当前支持的主要参数：

```text
--input-format dtoa 或 binary
--ts          binary time bin 的采样间隔
--window-size 固定序列窗口长度
--batch-size  batch size
--epochs      训练轮数
--loss        ce 或 focal
--gamma       focal loss 的 focusing strength
--alpha-mode  none 或 inverse_freq
--sparsity-ratio none、1:3、1:5 或 1:8
--visible-duration 每个可见扫描段的持续时间
--sparsity-phase-offset signal sparsity 的相位偏移
--toa-error-std TOA 高斯测量误差标准差
--pw-error-std  PW 高斯测量误差标准差
--rf-error-std  RF 高斯测量误差标准差
--doa-error-std DOA 高斯测量误差标准差
--pulse-loss-rate 随机脉冲丢失概率
--spurious-rate 虚假脉冲插入比例
```

## 仿真数据可视化

为了检查 Phase 1 中四类传统雷达源的脉冲仿真、混合脉冲流和 DTOA 预处理结果，可以运行独立可视化脚本：

```bash
python visualize_simulation.py
```

该脚本不会训练模型，也不会影响 `train.py` 的最小训练闭环。它会自动生成一小段仿真数据，并将图片保存到：

```text
outputs/figures/
```

当前会生成以下基础图：

- `emitter_fixed_pri.png`：固定 PRI 雷达的 `TOA-DTOA/PRI` 图，用于观察等间隔脉冲。
- `emitter_jitter_pri.png`：抖动 PRI 雷达的 `TOA-DTOA/PRI` 图，用于观察 PRI 围绕基准值抖动。
- `emitter_stagger_pri.png`：参差 PRI 雷达的 `TOA-DTOA/PRI` 图，用于观察周期性 PRI 轮换。
- `emitter_dwell_switch_pri.png`：驻留与切换 PRI 雷达的 `TOA-DTOA/PRI` 图，用于观察分段 PRI 切换。
- `true_pri_by_emitter.png`：按真实 `label` 分组后，对每个 emitter 内部的 TOA 单独排序并计算 same-emitter interval。这张图展示单个雷达源自己的真实 PRI 模式。
- `interleaved_label_vs_toa.png`：四个雷达源混合并按 TOA 排序后的 `TOA-label` 散点图，用于观察 interleaved pulse stream 的真实类别分布。
- `mixed_stream_adjacent_dtoa.png`：混合脉冲流按 TOA 排序后的相邻脉冲 DTOA。该 DTOA 是模型输入中的 DTOA 形式，但相邻两个脉冲可能来自不同雷达，因此它不是单个雷达源的真实 PRI，也不应当当作论文 Fig. 6(a) 的复现图。
- `dtoa_sequence.png`：混合脉冲流的 DTOA 随 pulse index 的变化。
- `dtoa_histogram.png`：混合脉冲流 DTOA 的统计直方图。

运行可视化脚本时，终端还会打印每个 `label` 的：

- pulse count
- min TOA
- max TOA
- mean same-emitter interval

这些统计量用于检查不同雷达源的观测时间是否大致对齐，避免某个雷达源在观测后半段单独存在，从而造成混合 DTOA 图右侧出现不合理的平台或尾部结构。

## 运行后会看到什么

运行 `train.py` 后会打印：

```text
Using device: cuda
Selected input format: dtoa
Selected loss function: ce
Selected sparsity ratio: none
Visible duration: ...
Measurement error stds: ...
Pulse loss rate: ...
Spurious pulse rate: ...
PDW columns: ('TOA', 'PW', 'RF', 'DOA', 'label')
Pulse count before sparsity: ...
Pulse count after sparsity: ...
Retained pulse ratio: ...
Pulse count before nonideal conditions: ...
Pulse count after measurement error: ...
Pulse count after pulse loss: ...
Pulse count after spurious insertion: ...
Total pulses: ...
Train input tensor shape [B, T, D]: ...
Train label tensor shape [B, T]: ...
Test input tensor shape [B, T, D]: ...
Test label tensor shape [B, T]: ...
Number of classes: ...
Train class counts: ...
Model output tensor shape [B, T, C]: ...
Epoch ...
Evaluation class counts: ...
Recall per class:
Average recall:
Confusion matrix:
```

其中最重要的是确认：

- 输入是 `[B, T, D]`
- 标签是 `[B, T]`
- 输出是 `[B, T, C]`
- 损失函数覆盖所有 `B*T` 个脉冲位置
- 评估结果按逐脉冲预测计算

## 与论文完整实验的差异

当前版本是最小复现 pipeline，与论文完整实验相比仍有明显简化：

- 雷达参数是简化设置，并不声明与论文参数完全一致。
- 只实现四类传统雷达源，没有实现 MFR 场景。
- 当前支持 DTOA input 和基础 binary input，但 binary input 仍采用简单 time-bin 离散化策略。
- 当前支持 cross entropy loss 和 focal loss。
- 当前支持基础 signal sparsity，但所有 emitter 暂时共用同一个 scan gate。
- 当前支持基础 nonideal receiving conditions：measurement error、random pulse loss 和 spurious pulses。
- 没有实现 CDIF、SDIF、PRI Transform 等对比方法。
- 没有进行论文级多场景、多噪声、多密度实验。
- 没有生成完整对比图表。

## Phase 1B: TSRD Windowing and Raw-feature Clustering Baseline

Phase 1B 的目标是先建立未知源数分选的评价链路，而不是训练深度模型。当前阶段使用 TSRD 的原始 PDW 派生特征做 clustering baseline，用来检查窗口构造、聚类输出、源数估计和聚类指标是否能够端到端跑通。后续 TCAN embedding 或 Transformer embedding 可以替换这里的 raw feature 输入，但评价代码应继续复用。

window 是按 pulse count 构造的固定长度片段：

```text
window_size: 每个窗口包含的 pulse 数量
stride: 相邻窗口起点之间相隔的 pulse 数量
max_windows: 最多评估多少个窗口
```

例如 `window_size=1024, stride=1024` 表示按 1024 个 pulse 做不重叠切分；如果 `stride < window_size`，则会产生重叠窗口。每个窗口都会返回：

```text
X_window: [window_size, feature_dim]
y_window: [window_size]
metadata: source file, start index, end index, true source count
```

当前支持两种 raw feature set：

```text
4d: [DTOA, PW, RF, AOA]
5d: [DTOA, PW, RF, AOA, PA]
```

其中 DTOA 在完整 pulse train 按 TOA 排序后计算，第一个 pulse 的 DTOA 置为 0。每个窗口在聚类前使用 `StandardScaler` 独立标准化。

当前 baseline 方法包括：

```text
dbscan: 非参数聚类，可以输出噪声点 -1，但对 eps 和 min_samples 敏感。
hdbscan: 可选方法，如果本地未安装 hdbscan 包，会打印清楚提示并跳过。
agglomerative_oracle: 使用真实 source count 作为 n_clusters 的层次聚类，仅用于 sanity check。
```

这里不能使用分类 accuracy 作为主要指标，因为未知源数分选中的 cluster ID 没有固定语义。例如预测 cluster 0 与真实 label 17 对齐并不重要，重要的是同源 pulse 是否被聚到一起、不同源 pulse 是否被分开。因此当前使用聚类指标：

```text
homogeneity
completeness
v_measure
adjusted_rand_index
adjusted_mutual_info
```

source count 的估计方式是统计预测聚类标签中的不同 cluster 数量，但不把 `-1` 噪声点算作一个 emitter：

```text
estimated_source_count = number of unique predicted labels excluding -1
source_count_error = estimated_source_count - true_source_count
abs_source_count_error = abs(source_count_error)
noise_ratio = predicted label 为 -1 的 pulse 比例
```

运行 DBSCAN baseline：

```powershell
python run_tsrd_clustering_baseline.py --tsrd-path E:\Datasets\TSRD\scan\train_scan\config_0.h5 --feature-set 5d --window-size 1024 --max-windows 3 --method dbscan
```

运行 oracle source-count 层次聚类 baseline：

```powershell
python run_tsrd_clustering_baseline.py --tsrd-path E:\Datasets\TSRD\scan\train_scan\config_0.h5 --feature-set 5d --window-size 1024 --max-windows 3 --method agglomerative_oracle
```

如果当前环境中没有安装 HDBSCAN，可以先跳过该方法；DBSCAN 与 `agglomerative_oracle` 已足够验证 Phase 1B 的窗口与评价链路。

## Phase 2A: TCAN Encoder Embedding Extraction

Phase 2A 开始把项目从 fixed-class classifier 逐步转向 pulse-level embeddings。原来的 TCAN 分类路径仍然保留：`model(X)` 继续输出每个 pulse 的分类 logits。新增的 encoder 路径可以输出每个 pulse 的上下文 embedding：

```text
输入 X: [B, T, D]
输出 embeddings: [B, T, E]
```

其中 `B` 是 batch size，`T` 是 window 内 pulse 数量，`D` 是输入 feature 维度，`E` 是 embedding 维度，默认 `E=64`，可通过 `--embedding-dim` 设置。对单个 window 做聚类时，脚本会使用：

```text
embeddings: [T, E]
```

Phase 1B 的 raw-feature clustering 是直接对 `[DTOA, PW, RF, AOA]` 或 `[DTOA, PW, RF, AOA, PA]` 聚类。Phase 2A 则先把这些 window features 输入 TCAN encoder，得到 pulse-level embeddings，再对 embeddings 执行 DBSCAN、HDBSCAN 或 `agglomerative_oracle`，并复用已有聚类指标。

当前阶段不训练 embedding。如果没有提供 checkpoint，脚本会使用随机初始化的 TCAN encoder，并打印警告：

```text
Warning: using randomly initialized TCAN encoder. Metrics are for pipeline testing only.
```

因此当前指标只用于验证链路是否跑通，不代表最终分选性能。下一阶段会在这个 scaffold 上加入 triplet loss 或 supervised contrastive loss，让 embeddings 具备“同源更近、异源更远”的度量结构。

运行随机初始化 TCAN embedding + DBSCAN：

```powershell
python run_tsrd_embedding_clustering.py --tsrd-path E:\Datasets\TSRD\scan\train_scan\config_0.h5 --feature-set 5d --window-size 1024 --max-windows 3 --embedding-dim 64 --method dbscan
```

运行随机初始化 TCAN embedding + oracle source-count 层次聚类：

```powershell
python run_tsrd_embedding_clustering.py --tsrd-path E:\Datasets\TSRD\scan\train_scan\config_0.h5 --feature-set 5d --window-size 1024 --max-windows 3 --embedding-dim 64 --method agglomerative_oracle
```

如果有兼容 checkpoint，可以额外传入：

```powershell
--checkpoint path\to\checkpoint.pt
```

## Phase 2B: Triplet Metric Learning

Phase 2B 使用 triplet margin loss 训练 TCAN encoder 的 pulse-level embedding 空间。目标是让同一个 emitter 的 pulse embeddings 更近，不同 emitter 的 pulse embeddings 更远。标准 triplet 由三部分组成：

```text
anchor: 当前参考 pulse
positive: 与 anchor 来自同一个 label 的 pulse
negative: 与 anchor 来自不同 label 的 pulse
```

loss 形式为：

```text
max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

这种训练方式适合 unknown-emitter-count deinterleaving，因为模型不需要学习固定类别 ID，而是学习一个可聚类的距离空间。后续在新 pulse train 中，即使 emitter 数量未知，也可以先提取 embeddings，再用 DBSCAN、HDBSCAN 或层次聚类完成分选。

重要限制：TSRD 的 label 只在当前 pulse train 内有意义。不同文件中的相同 label ID 不能默认理解为同一个物理 emitter。因此当前实现只在同一个 window 内构造 triplets，不跨文件、不跨 window 采样 positive。

当前 triplet sampling 是 random sampling：

```text
anchor 和 positive 来自同一 label
negative 来自不同 label
样本数少于 2 的 label 不能作为 anchor-positive 类
unique labels 少于 2 的 window 会被跳过
```

尚未实现 batch-hard mining、semi-hard mining 或 supervised contrastive loss。

训练 triplet encoder：

```powershell
python train_tsrd_triplet.py --tsrd-path E:\Datasets\TSRD\scan\train_scan\config_0.h5 --feature-set 5d --window-size 1024 --max-windows 10 --embedding-dim 64 --epochs 2 --num-triplets-per-window 256
```

训练时 embeddings 会先做 L2 normalize，再计算 triplet loss。checkpoint 默认保存到：

```text
checkpoints/
```

checkpoint 文件不应提交到 Git。

使用训练后的 checkpoint 做 embedding clustering：

```powershell
python run_tsrd_embedding_clustering.py --tsrd-path E:\Datasets\TSRD\scan\train_scan\config_0.h5 --feature-set 5d --window-size 1024 --max-windows 3 --embedding-dim 64 --checkpoint checkpoints\<checkpoint_file>.pt --method dbscan
```

embedding clustering 阶段同样会对 TCAN 输出的 embeddings 做 L2 normalize，然后再执行聚类和指标计算。

## Phase 2C: Systematic Embedding Evaluation

Phase 2C 提供统一评估脚本，用同一个 TSRD 文件、同一批 pulse windows、同一种 clustering 方法和同一套指标，对比不同表示方式的分选效果：

```text
raw: 直接使用 [DTOA, PW, RF, AOA] 或 [DTOA, PW, RF, AOA, PA]
random_embedding: 使用随机初始化 TCAN encoder 输出 embeddings
triplet_embedding: 加载 triplet 训练后的 TCAN checkpoint 输出 embeddings
```

这样可以区分三件事：

```text
raw feature baseline 本身有多强
随机 TCAN embedding 管线是否正常
triplet metric learning 是否改善 embedding clustering
```

统一评估脚本：

```powershell
python run_embedding_evaluation.py --tsrd-path E:\Datasets\TSRD\scan\train_scan\config_0.h5 --feature-set 5d --window-size 1024 --max-windows 3 --cluster-method dbscan --methods raw,random_embedding
```

如果已经有 triplet checkpoint：

```powershell
python run_embedding_evaluation.py --tsrd-path E:\Datasets\TSRD\scan\train_scan\config_0.h5 --feature-set 5d --window-size 1024 --max-windows 3 --cluster-method dbscan --methods raw,triplet_embedding --checkpoint checkpoints\<checkpoint_file>.pt
```

如果 `--methods` 包含 `random_embedding`，脚本会提示该方法使用未训练 TCAN encoder，指标只用于 sanity check。如果 `--methods` 包含 `triplet_embedding` 但没有提供 `--checkpoint`，脚本会直接给出参数错误。

每个 method 都会打印每个 window 的指标，并按 method 打印均值：

```text
homogeneity
completeness
v_measure
adjusted_rand_index
adjusted_mutual_info
abs_source_count_error
noise_ratio
```

如果需要保存结果，可以使用：

```powershell
--output-csv results\embedding_eval.csv
```

CSV 会同时包含每个 window 的结果和每个 method 的 mean 行。`results/`、`outputs/`、`checkpoints/` 和数据文件不应提交到 Git。

## Phase 3A: Multi-file Clustering Parameter Search

Phase 3A 用于检查单文件、少窗口上看到的提升是否能在多个 TSRD h5 文件和不同 clustering 参数下保持稳定。单个文件上的高指标可能来自局部场景较简单、窗口选择偶然或参数刚好适配，因此需要多文件、多窗口评估。

当前脚本支持比较：

```text
raw: 直接使用 DTOA/PDW raw features
triplet_embedding: 加载 triplet checkpoint 后使用 TCAN embeddings
```

并支持两类聚类参数搜索：

```text
DBSCAN: eps x min_samples
HDBSCAN: min_cluster_size x min_samples
```

参数含义：

```text
eps: DBSCAN 中两个点被视为邻域点的距离半径，过小会产生大量噪声，过大容易把不同源合并。
min_samples: DBSCAN/HDBSCAN 中形成核心密度区域所需的最小样本数，越大越保守。
min_cluster_size: HDBSCAN 中允许形成稳定 cluster 的最小规模，越大越倾向于忽略小簇。
```

source-count error 的解释：

```text
source_count_error = estimated_source_count - true_source_count
abs_source_count_error = abs(source_count_error)
```

其中 `estimated_source_count` 不会把 `-1` 噪声点算作 emitter。负数表示低估源数，正数表示聚类过分裂或估出了过多源。

运行 raw feature 的 DBSCAN 参数搜索：

```powershell
python run_clustering_param_search.py --tsrd-dir E:\Datasets\TSRD\scan\train_scan --file-glob "config_*.h5" --max-files 3 --max-windows-per-file 3 --feature-set 5d --window-size 1024 --representation raw --method dbscan --eps-grid 0.2,0.5,0.8 --min-samples-grid 3,5
```

运行 triplet embedding 的 DBSCAN 参数搜索：

```powershell
python run_clustering_param_search.py --tsrd-dir E:\Datasets\TSRD\scan\train_scan --file-glob "config_*.h5" --max-files 3 --max-windows-per-file 3 --feature-set 5d --window-size 1024 --representation triplet_embedding --checkpoint checkpoints\<checkpoint_file>.pt --method dbscan --eps-grid 0.2,0.5,0.8 --min-samples-grid 3,5
```

如果需要保存每组参数的聚合结果：

```powershell
--output-csv outputs\param_search.csv
```

选择 best parameter setting 的规则是：

```text
1. mean_v_measure 最大
2. 如果接近，则 mean_abs_source_count_error 更小
3. 再比较 mean_noise_ratio 更小
```

当前阶段不做 cluster merging、cluster splitting、source-count correction 或任何 post-processing。

## Phase 3B: Cluster Post-processing and Sequence Reconstruction

Phase 3B 在已有 `triplet_embedding + DBSCAN/HDBSCAN` 初始聚类结果基础上，增加保守的 cluster post-processing。目标不是重新训练 encoder，也不是引入新的 clustering 方法，而是把 cluster label 映射回原始 pulse window 后，根据 embedding 与 PDW 统计量做小幅修正，并比较 before/after metrics。

这个阶段只借鉴多接收机 TDOA mapping 文献中的两个思想：

```text
cluster-to-pulse mapping
iterative refinement
```

当前实现不生成 TDOA，不使用多接收机信息，不实现 SSC-DBSCAN。

后处理包含三个部分：

```text
reassign: 只尝试把 -1 noise points 等边界点重新分配给非常可信的最近 cluster。
merge: 只合并 embedding centroid 与 PDW 分布都非常接近，且合并后 compactness 不明显变差的簇。
split: 当前是保守 scaffold，默认不实际拆分；后续可加入内部 DBSCAN/HDBSCAN 和稳定性检查。
```

后处理决策不使用 ground-truth labels。真实 labels 只用于 before/after evaluation。

运行示例：

```powershell
python run_cluster_postprocessing.py --tsrd-path E:\Datasets\TSRD\scan\train_scan\config_0.h5 --feature-set 5d --window-size 1024 --max-windows-per-file 3 --checkpoint checkpoints\<checkpoint_file>.pt --cluster-method dbscan --eps 0.5 --min-samples 5 --enable-reassign --enable-merge
```

脚本会对每个 window 打印：

```text
Before post-processing:
v_measure, ARI, AMI, homogeneity, completeness, estimated_source_count, abs_source_count_error, noise_ratio

After post-processing:
同样指标
```

并输出本窗口被重分配的 pulse index 与被合并的 cluster 对。最后会打印 mean before/after metrics。如果指定：

```powershell
--output-csv outputs\postprocessing_eval.csv
```

则会保存每个 window 的 before/after 结果。`outputs/`、`checkpoints/` 和 TSRD 数据文件不应提交到 Git。

## Phase 3C: Source-count-aware Clustering Refinement

Phase 3C 专门处理 source-count underestimation。前一阶段的保守 post-processing 基本不破坏指标，但对 `estimated_source_count` 偏低和 `noise_ratio` 偏高的改善很小。因此这一阶段不继续做简单 merge/reassign，而是增加：

```text
error diagnosis
noise subcluster recovery
conservative split of high-dispersion clusters
before/after source-count-aware evaluation
```

普通 boundary reassignment 只能把少量 `-1` 点分配回已有 cluster，通常不能增加 cluster 数，因此无法解决“真实源被整体标为 noise”或“多个真实源被合并到一个大 cluster”的问题。Phase 3C 的两个 refinement 方向分别对应这两类错误：

```text
noise_subcluster_recovery:
  只在 noise points 内部做二次 DBSCAN/HDBSCAN。
  只有稳定、足够大、compactness 足够低、PDW 分布一致的子簇才恢复为新 cluster。

split_dispersion_clusters:
  只对 size 足够大且 embedding compactness 较高的 cluster 做内部二次聚类。
  只有能产生多个稳定子簇且拆分后 compactness 改善时才接受。
```

诊断模块会使用 true labels 输出：

```text
true-label to cluster contingency table
cluster to true-label composition table
noise true-label distribution
major_error_type: missing_as_noise / over_merged / over_split / mixed / clean
```

这些 true-label-based 信息只用于 diagnosis 和 evaluation，不能用于 refinement decision。实际 recovery/splitting 只使用 embeddings、PDW features 和初始 cluster labels。

单文件测试：

```powershell
python run_source_count_refinement.py --tsrd-path E:\Datasets\TSRD\scan\train_scan\config_0.h5 --feature-set 5d --window-size 1024 --max-windows-per-file 3 --checkpoint checkpoints\<checkpoint_file>.pt --cluster-method dbscan --eps 0.5 --min-samples 3 --enable-noise-recovery --enable-split
```

多文件小规模测试：

```powershell
python run_source_count_refinement.py --tsrd-dir E:\Datasets\TSRD\scan\train_scan --file-glob "config_*.h5" --max-files 3 --max-windows-per-file 3 --feature-set 5d --window-size 1024 --checkpoint checkpoints\<checkpoint_file>.pt --cluster-method dbscan --eps 0.5 --min-samples 3 --enable-noise-recovery --enable-split
```

常用 refinement 参数：

```text
--noise-eps
--noise-min-samples
--min-recovered-cluster-size
--recovery-compactness-threshold
--recovery-pdw-threshold
--split-eps
--split-min-samples
--min-split-cluster-size
--min-split-subcluster-size
--split-compactness-threshold
```

当前限制：

```text
secondary clustering 仍是启发式；
阈值尚未系统搜索；
不做 source-count correction；
不使用 true labels 做 refinement；
不重新训练 TCAN encoder。
```

## Phase 4A: Batch-hard Triplet Metric Learning

Phase 4A 不继续增加 clustering 后处理规则，而是回到 TCAN embedding 空间本身。Phase 3C 的 source-count-aware refinement 能提升 V-measure、ARI 和 AMI，但 noise ratio 已经较低而 source-count error 仍然偏高，说明主要问题更可能是多个真实 emitter 在 embedding 空间中被压得太近，导致 DBSCAN/HDBSCAN 形成 over-merged clusters。

Phase 2B 的 random triplet sampling 每次随机选择 positive 和 negative。它能建立基本的“同源近、异源远”结构，但很多随机 triplet 很容易，loss 很快变小，却不一定处理最容易造成聚类合并的边界样本。

Batch-hard triplet mining 在同一个 local label context 内为每个 anchor 选择更有信息量的样本：

```text
hardest positive:
  与 anchor 同 label、距离最远的 pulse embedding

hardest negative:
  与 anchor 不同 label、距离最近的 pulse embedding

loss:
  max(0, d(anchor, hardest_positive) - d(anchor, hardest_negative) + margin)
```

这样训练会直接惩罚“同一 emitter 内部太分散”和“不同 emitter 之间太接近”的情况，有助于拉开容易被聚成同一簇的 emitter 边界，从而减少 over-merged clusters。实现仍然不改变聚类阶段，也不使用 true labels 做 clustering/refinement 决策；labels 只在 metric learning 训练中作为监督信号使用。

TSRD label 有一个重要限制：label 只在当前 pulse train/window 内有意义。不同文件或不同 pulse train 中相同整数 label 不能默认视为同一个物理 emitter。因此当前 batch-hard 实现对 `[B, T, E]` embeddings 按 window 独立 mining，不跨 window 混合 label。

训练旧 random triplet 流程：

```powershell
python train_tsrd_triplet.py --tsrd-path E:\Datasets\TSRD\scan\train_scan\config_0.h5 --feature-set 5d --window-size 1024 --max-windows 10 --embedding-dim 64 --epochs 2 --triplet-mining random
```

训练 batch-hard triplet：

```powershell
python train_tsrd_triplet.py --tsrd-path E:\Datasets\TSRD\scan\train_scan\config_0.h5 --feature-set 5d --window-size 1024 --max-windows 10 --embedding-dim 64 --epochs 2 --triplet-mining batch_hard
```

batch-hard 训练日志会打印：

```text
batch_hard_triplet_loss
valid_anchors
skipped_batches
```

其中 `valid_anchors` 是同时存在同 label positive 和不同 label negative 的 anchor 数量。对于正常的 multi-emitter TSRD window，它通常应接近 `batch_size * window_size`；如果一个 batch 中 window 只有单一 label，或某些 label 只有一个 pulse，则对应 anchor 会被跳过。

评估训练后的 checkpoint：

```powershell
python run_embedding_evaluation.py --tsrd-path E:\Datasets\TSRD\scan\train_scan\config_0.h5 --feature-set 5d --window-size 1024 --max-windows 3 --cluster-method dbscan --methods raw,triplet_embedding --checkpoint checkpoints\<batch_hard_checkpoint>.pt
```

建议用同一组 DBSCAN/HDBSCAN 参数分别评估 random triplet checkpoint 与 batch-hard triplet checkpoint，并比较：

```text
V-measure
ARI
AMI
homogeneity
completeness
abs source-count error
noise ratio
```

当前阶段不实现 supervised contrastive loss，不新增后处理规则，不实现 cluster merge/split，也不提交 checkpoint、outputs 或 h5 数据文件。
