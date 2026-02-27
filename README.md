# 机械故障诊断系统

基于深度学习的轴承故障诊断系统，使用**李群双通道特征**和**ArcFace损失函数**实现高精度故障分类和未知故障检测（Open-set Recognition）。

## ✨ 核心特性

- **李群双通道特征**: 能量曲线（Energy）+ 密度曲线（Density），对频率漂移具有鲁棒性
- **ArcFace度量学习**: 在特征空间将同类故障压缩，异类故障拉开距离
- **Open-set识别**: 支持未知故障类型的检测
- **多数据源支持**: CWRU、XJTU、声音能量曲线数据
- **声音API集成**: 支持将音频文件转换为李群能量密度曲线

## 🚀 快速开始

### 1. 环境配置

```bash
# Windows
setup_conda_env.bat

# Linux/Ubuntu
bash setup_conda_env.sh
```

### 2. 激活环境

```bash
# Windows
activate_env.bat

# Linux/Ubuntu
source activate_env.sh
```

### 3. 数据准备

#### 使用CWRU数据集（推荐）

```bash
# 预处理CWRU数据，生成signals.npy和labels.npy
python tools/load_cwru.py --data_dir CWRU-dataset-main --output_dir cwru_processed
```

#### 使用声音能量曲线数据

```bash
# 使用声音API转换音频文件（需要配置API）
# 详见: tools/docs/README_声音API使用指南.md
```

#### 使用声音API缓存数据（NPZ格式，推荐）

**最短流程（3步，JSON-first）**：

```bash
# 步骤1: API转换（生成JSON，按bearing_id分桶）
# 1A) 如果你已有音频文件（wav/mp3/flac...），直接调用 API 转 JSON
python tools/sound_api/convert_sound_api.py --audio_dir <你的音频目录> --output_root datasets/sound_api
# 输出到: datasets/sound_api/output_json/{bearing_id}/...
#
# 1B) 如果你只有 XJTU 的 MC 数据（.f + sidecar .json），推荐走“从 .f 直接到 API JSON（不落盘 wav）”：
python tools/sound_api/convert_mc_to_api_json.py --mc_dir datasets/output_xjtu_mc --output_root datasets/sound_api --resume --workers 8
# 这一步会递归扫描 *.f，要求同名 sidecar *.json 存在；成功后同样输出到 output_json/{bearing_id}/

# 步骤2: 构建NPZ缓存（递归扫描，优先JSON回退xlsx）
python tools/build_sound_api_cache.py --json_dir datasets/sound_api/output_json --xlsx_dir datasets/sound_api/output_xlsx --output_dir datasets/sound_api/cache_npz --workers 8
# 输出到: datasets/sound_api/cache_npz/{bearing_id}/{t:06d}.npz
# 同时写出索引: datasets/sound_api/cache_npz/index.jsonl（用于追溯与完整性对账）

# 步骤3: 开始训练（支持hi/risk/cls任务）
python experiments/train.py --data_source sound_api_cache --task hi --batch_size 8 --epochs 50
python experiments/train.py --data_source sound_api_cache --task risk --horizon 10 --batch_size 8 --epochs 50
```

**说明**：
- **JSON-first原则**：推荐使用JSON格式（包含metadata），xlsx仅作为调试回退（需 `--write-xlsx`）
- **目录规范**：所有产物统一落在 `datasets/sound_api/`，按 `bearing_id` 分桶，**不会在 tools/sound_api/ 下落任何数据**（避免IDE卡顿）
- **训练阶段只读NPZ**：不直接读取JSON/xlsx，只读 `datasets/sound_api/cache_npz/`
- **时间序号说明**：训练用 `t` 为稳定重编号，仅用于序列建模；`orig_t` 保留原始采集顺序，用于追溯与人工校验
- 支持的任务：`hi`（健康指数回归）、`risk`（风险预测）、`arcface`（分类，需要标签）
- 使用bearing-level split，保证同一bearing的所有样本在同一子集
- 自动质量门禁（长度3000、finite、std、non_zero_ratio、volume>=0等），失败文件记录到 `datasets/sound_api/logs/bad_files_cache.txt`
- t 连续性硬校验：不连续的 bearing 记录到 `datasets/sound_api/logs/bad_bearings.txt`
- 端到端完整性校验：`python tools/sound_api/verify_mc_pipeline.py --mc_dir datasets/output_xjtu_mc`
- 详细规范见：[docs/OUTPUT_SPEC_LIEGROUP_V1.md](docs/OUTPUT_SPEC_LIEGROUP_V1.md)

**旧流程（兼容保留，仅用于回退）**：
```bash
# 注意：以下为 legacy 路径，新项目请使用上面的 JSON-first 流程

# 1. 生成索引文件（从旧版 xlsx 输出）
python tools/build_sound_api_index.py --data_dir datasets/sound_api_output

# 2. 转换为NPZ缓存（从xlsx）
python tools/cache_sound_api_to_npz.py --index datasets/sound_api_output/index.csv --workers 8
```

### 4. 开始训练

```bash
# 使用CWRU数据训练
python experiments/train.py --data_source cwru --batch_size 32 --epochs 100

# 使用声音数据训练（样本数较少，建议小batch）
python experiments/train.py --data_source sound --batch_size 8 --epochs 50

# 使用声音API缓存数据训练（支持hi/risk任务）
python experiments/train.py --data_source sound_api_cache --task hi --batch_size 8 --epochs 50
python experiments/train.py --data_source sound_api_cache --task risk --horizon 10 --batch_size 8 --epochs 50
```

### 5. 模型推理

```bash
# 推理和Open-set检测
python experiments/inference.py --data_source cwru --threshold 0.4
```

## 📁 项目结构

```
guzhangjiance/
├── dl/                          # 深度学习核心模块（主线代码）
│   ├── data_loader.py           # 李群数据集加载器（支持CWRU和声音数据）
│   ├── model.py                 # ResNet-18 1D骨干网络
│   ├── loss.py                  # ArcFace损失函数
│   ├── sound_data_loader.py     # 声音数据专用加载器
│   ├── sound_api_data_loader.py # 声音API数据加载器
│   └── sound_api_cache_dataset.py # 声音API NPZ缓存数据集（支持hi/risk任务）
│
├── experiments/                 # 实验脚本
│   ├── train.py                 # 训练脚本（主线）
│   ├── inference.py             # 推理和Open-set检测（主线）
│   ├── run_experiment.py        # 传统特征+随机森林实验
│   └── view_analysis_results.py # 查看分析结果
│
├── core/                        # 核心特征提取模块
│   ├── features.py              # FFT、机理、声音特征提取器
│   └── models.py                # 随机森林等传统模型
│
├── tools/                       # 工具脚本
│   ├── load_cwru.py             # CWRU数据加载和预处理
│   ├── load_sound.py            # 声音能量曲线数据加载
│   ├── load_xjtu.py             # XJTU数据加载
│   ├── build_sound_api_index.py # 构建声音API输出索引（新增）
│   ├── cache_sound_api_to_npz.py # Excel转NPZ缓存（新增）
│   ├── sound_api/               # 声音转能量密度曲线API工具
│   │   └── convert_sound_api.py # API转换脚本
│   └── docs/                    # 工具使用文档
│       ├── README_声音API使用指南.md
│       └── README_数据转换使用指南.md
│
├── legacy/                      # 历史代码（参考用）
│   ├── analyze_sound_curves.py  # 声音曲线分析
│   ├── compare_fft_vs_lie_group.py  # FFT vs 李群对比
│   └── dual_channel_model_implementation.py  # TensorFlow原型
│
├── docs/                        # 项目文档
│   ├── PROJECT_STRUCTURE.md     # 项目结构详细说明
│   ├── SOUND_DATA_STRUCTURE.md  # 声音数据结构说明
│   ├── README_SOUND_ANALYSIS.md # 声音分析完整指南
│   ├── README_SOUND_TRAINING.md # 声音数据训练指南
│   ├── INSTALL.md               # 安装指南
│   └── CONDA_ENV_GUIDE.md      # Conda环境管理
│
├── datasets/                    # 数据集目录
│   ├── CWRU-dataset-main/       # CWRU原始数据
│   ├── cwru_processed/          # CWRU预处理数据（signals.npy, labels.npy）
│   ├── xjtu_dataset/           # XJTU数据集
│   ├── 声音能量曲线数据/        # 声音能量曲线xlsx文件
│   └── sound_api/              # 声音API数据（新规范：所有产物统一管理）
│       ├── output_json/        # JSON主产物（按bearing_id分桶）
│       │   └── {bearing_id}/
│       ├── output_xlsx/        # xlsx调试产物（默认不写，--write-xlsx才写）
│       │   └── {bearing_id}/
│       ├── cache_npz/          # NPZ缓存（训练唯一入口）
│       │   └── {bearing_id}/
│       │       └── {t:06d}.npz
│       └── logs/               # 日志与统计报表
│           ├── bad_files_cache.txt
│           ├── bad_bearings.txt
│           └── conversion_report.json
│
├── deploy_ubuntu/               # Ubuntu部署相关
│   ├── deploy_to_ubuntu.sh      # 部署脚本
│   └── REMOTE_TRAINING_GUIDE.md # 远程训练指南
│
├── environment.yml              # Conda环境配置（CPU）
├── environment_gpu.yml          # Conda环境配置（GPU）
└── requirements.txt             # Python依赖
```

## 🔧 核心功能模块

### 1. 数据加载 (`dl/data_loader.py`)

- **LieGroupDataset**: 加载预处理后的双通道李群特征
- **支持数据源**: CWRU、声音能量曲线
- **自动标准化**: 按通道独立的Z-Score标准化
- **数据划分**: 自动划分训练/验证/测试集

### 2. 模型架构 (`dl/model.py`)

- **ResNet18_1D_Backbone**: 修改版1D ResNet-18
- **输入**: `(batch_size, 2, sequence_length)`
- **输出**: `(batch_size, 512)` 特征向量
- **特点**: 去除分类层，输出纯特征用于度量学习

### 3. 损失函数 (`dl/loss.py`)

- **ArcMarginProduct**: ArcFace损失实现
- **特点**: L2归一化 + 角度margin + 缩放因子
- **优势**: 在特征空间实现类内压缩、类间分离

### 4. 训练流程 (`experiments/train.py`)

- 支持多种数据源：CWRU、声音数据、声音API缓存
- 支持多种任务：ArcFace分类、健康指数回归（HI）、风险预测（Risk）
- 自动保存最优模型权重
- 分别保存backbone和任务头
- 实时显示训练/验证指标
- 支持按bearing-level split（保证同一bearing的所有样本在同一子集）

### 5. 推理与Open-set检测 (`experiments/inference.py`)

- 仅加载backbone（适合边缘部署）
- 基于余弦相似度的分类
- 支持未知故障检测（阈值判定）
- t-SNE可视化特征空间

## 📊 数据说明

### CWRU数据集

- **格式**: 预处理后的`.npy`文件（`signals.npy`, `labels.npy`）
- **形状**: `(N, 2, L)` - N个样本，2通道（能量+密度），L为序列长度
- **预处理**: 使用`tools/load_cwru.py`从原始`.mat`文件转换

### 声音能量曲线数据

- **格式**: Excel文件（`.xlsx`），每个文件包含多个Sheet
- **结构**: 每个Sheet包含3列（频率、能量、密度），共3000个点
- **频率范围**: 20 Hz - 20000 Hz
- **样本数**: 94个样本（4个正常，90个故障）
- **加载**: 使用`tools/load_sound.py`或`dl/sound_data_loader.py`

### 声音API缓存数据（NPZ格式）

- **格式**: NPZ文件（`.npz`），每个文件对应一个样本
- **目录结构**: `datasets/sound_api/cache_npz/{bearing_id}/{t:06d}.npz`
- **数据格式**: `x` 形状为 `(2, 3000)`，`x[0]=log1p(volume)`, `x[1]=density`
- **时间序号说明**：
  - `t`（训练用）：稳定重编号（0..T-1），仅用于序列建模
  - `orig_t`（追溯用）：保留原始采集顺序，用于追溯与人工校验
- **预处理（推荐）**: 使用 `tools/build_sound_api_cache.py` 一键构建（JSON-first）
- **预处理（旧流程）**: 使用 `tools/build_sound_api_index.py` + `tools/cache_sound_api_to_npz.py`（legacy）
- **加载**: 使用`dl/sound_api_cache_dataset.py`，支持hi/risk/cls任务
- **详细规范**: 见[docs/OUTPUT_SPEC_LIEGROUP_V1.md](docs/OUTPUT_SPEC_LIEGROUP_V1.md)

### 李群转换参数

- **频率点数**: 3000
- **频率范围**: 20 - 20000 Hz
- **通道0（能量）**: FFT幅度谱，反映信号能量强度
- **通道1（密度）**: 能量浓度分布，对频率漂移鲁棒（SE(3)不变性）

## 📖 详细文档

### 核心文档

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 项目结构详细说明
- [cursor.md](cursor.md) - 项目规范文档（技术设计）

### 数据相关

- [docs/SOUND_DATA_STRUCTURE.md](docs/SOUND_DATA_STRUCTURE.md) - 声音数据结构说明
- [docs/README_SOUND_ANALYSIS.md](docs/README_SOUND_ANALYSIS.md) - 声音分析完整指南
- [docs/README_SOUND_TRAINING.md](docs/README_SOUND_TRAINING.md) - 声音数据训练指南
- [docs/OUTPUT_SPEC_LIEGROUP_V1.md](docs/OUTPUT_SPEC_LIEGROUP_V1.md) - 声音API输出规范（Excel→NPZ缓存）

### 工具使用

- [tools/docs/README_声音API使用指南.md](tools/docs/README_声音API使用指南.md) - 声音API使用说明
- [tools/docs/README_数据转换使用指南.md](tools/docs/README_数据转换使用指南.md) - 数据转换指南

### 环境配置

- [docs/INSTALL.md](docs/INSTALL.md) - 安装指南
- [docs/CONDA_ENV_GUIDE.md](docs/CONDA_ENV_GUIDE.md) - Conda环境管理
- [docs/CUDA_WINDOWS_GUIDE.md](docs/CUDA_WINDOWS_GUIDE.md) - Windows CUDA配置

### 实验相关

- [docs/README_experiment.md](docs/README_experiment.md) - 传统特征实验说明
- [docs/PHASE_1_SUMMARY.md](docs/PHASE_1_SUMMARY.md) - Phase 1实验总结

## 🎯 使用示例

### 训练模型

```bash
# CWRU数据训练
python experiments/train.py \
    --data_source cwru \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-3 \
    --split_ratio 0.7 0.15 0.15

# 声音数据训练
python experiments/train.py \
    --data_source sound \
    --batch_size 8 \
    --epochs 50

# 声音API缓存数据训练（健康指数回归）
python experiments/train.py \
    --data_source sound_api_cache \
    --task hi \
    --batch_size 8 \
    --epochs 50

# 声音API缓存数据训练（风险预测）
python experiments/train.py \
    --data_source sound_api_cache \
    --task risk \
    --horizon 10 \
    --batch_size 8 \
    --epochs 50
```

### 模型推理

```bash
# 推理和Open-set检测
python experiments/inference.py \
    --data_source cwru \
    --checkpoint checkpoints/backbone.pth \
    --threshold 0.4 \
    --batch_size 32
```

### 数据预处理

```bash
# 预处理CWRU数据
python tools/load_cwru.py \
    --data_dir CWRU-dataset-main \
    --output_dir cwru_processed \
    --segment_length 2048
```

### 传统特征实验

```bash
# 运行随机森林实验
python experiments/run_experiment.py \
    --feature_type fft \
    --data_dir CWRU-dataset-main
```

## 📝 依赖安装

### 使用Conda（推荐）

```bash
# CPU版本
conda env create -f environment.yml

# GPU版本（需要CUDA）
conda env create -f environment_gpu.yml
```

### 使用pip

```bash
pip install -r requirements.txt
```

## 🔬 技术特点

### 李群特征优势

- **SE(3)不变性**: 对频率漂移具有鲁棒性
- **双通道互补**: 能量曲线提供直观特征，密度曲线提供形状信息
- **信息冗余度低**: 能量和密度相关系数约0.31

### ArcFace优势

- **类内压缩**: 同类故障在特征空间聚集
- **类间分离**: 异类故障在特征空间拉开距离
- **Open-set支持**: 通过余弦相似度阈值检测未知故障

## 🚧 项目状态

### 主线代码（当前维护）

- ✅ `dl/data_loader.py` - 数据加载器
- ✅ `dl/model.py` - 模型定义
- ✅ `dl/loss.py` - 损失函数
- ✅ `experiments/train.py` - 训练脚本
- ✅ `experiments/inference.py` - 推理脚本

### 传统特征实验（稳定基线）

- ✅ `core/features.py` - 特征提取器
- ✅ `core/models.py` - 传统模型
- ✅ `experiments/run_experiment.py` - 实验脚本

### 历史代码（参考用）

- 📚 `legacy/` - 声音分析和TensorFlow原型代码

## 📞 相关资源

- **项目结构**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **项目规范**: [cursor.md](cursor.md)
- **部署指南**: [deploy_ubuntu/REMOTE_TRAINING_GUIDE.md](deploy_ubuntu/REMOTE_TRAINING_GUIDE.md)

---

**版本**: 2.0  
**最后更新**: 2026-01-15
