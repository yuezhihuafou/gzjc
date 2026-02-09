# 故障诊断实验平台使用说明

本模块实现了基于 CWRU 数据集的模块化故障诊断实验平台，支持多种特征提取方法和模型训练。

## 1. 模块结构

```
d:\guzhangjiance\
├── core/
│   ├── features.py       # 特征提取模块 (FFT, 机理特征, 声音特征)
│   ├── models.py         # 模型定义模块 (随机森林, Transformer占位符)
│   └── __init__.py
├── tools/
│   └── load_cwru.py      # 数据加载工具
├── run_experiment.py     # 实验运行主脚本
└── README_experiment.md  # 本文档
```

## 2. 支持的特征类型

| 特征类型 (`--feature_type`) | 说明 | 维度 (示例) |
|---------------------------|------|------------|
| `mechanism` | **特征倍频信号误差**。基于轴承物理参数计算故障特征频率 (BPFO, BPFI 等) 及其倍频处的幅值。 | 低维 (12维) |
| `sound` | **声音密度和能量**。目前使用振动信号的功率谱密度 (PSD) 和 RMS 值作为代理。待声音转换算法完成后，可在 `core/features.py` 中替换实现。 | 中维 (129维) |
| `fft` | **FFT 幅值和相位**。全频谱幅值和相位曲线，用于对比基准。 | 高维 (2050维) |

## 3. 运行实验

使用 `run_experiment.py` 脚本运行实验。

### 3.1 训练随机森林模型 (机理特征)
这是任务中要求的“特征倍频信号误差”输入。

```bash
python run_experiment.py --feature_type mechanism --segment_length 2048
```

### 3.2 训练随机森林模型 (声音特征)
这是任务中要求的“声音密度和声音能量曲线”输入。

```bash
python run_experiment.py --feature_type sound --segment_length 2048
```

### 3.3 训练随机森林模型 (FFT 对比)
这是任务中要求的对比基准。

```bash
python run_experiment.py --feature_type fft --segment_length 2048
```

## 4. 代码扩展指南

### 如何添加 Transformer 模型？
1. 打开 `core/models.py`。
2. 找到 `TransformerModel` 类。
3. 在 `train` 和 `predict` 方法中实现 PyTorch 或 TensorFlow 的 Transformer 模型逻辑。

### 如何替换声音特征算法？
1. 打开 `core/features.py`。
2. 找到 `SoundMetricsExtractor` 类。
3. 修改 `extract` 方法，将目前的 PSD/RMS 计算替换为姚飞提供的声音转换算法。

### 如何修改轴承参数？
1. 打开 `core/features.py`。
2. 修改 `MechanismFeatureExtractor` 类中的 `__init__` 方法，更新 `d`, `D`, `n`, `alpha` 等参数。
