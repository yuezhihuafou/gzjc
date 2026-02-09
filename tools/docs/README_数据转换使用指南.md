# CWRU 数据转换使用指南

本文档说明如何使用 `load_cwru.py` 和 `数据库转换.py` 进行数据加载、预处理与转换。

## 目录

- [基本功能](#基本功能)
- [单通道模式](#单通道模式)
- [多通道模式](#多通道模式)
- [高级选项](#高级选项)
- [Python 编程接口](#python-编程接口)

---

## 基本功能

### 1. 查看数据集信息

快速扫描数据集并显示统计信息，不进行实际导出：

```bash
python tools\load_cwru.py CWRU-dataset-main --info
```

**输出示例：**
```
扫描到 161 个 .mat 文件

==================================================
CWRU 数据集统计信息
==================================================
数据集路径: D:\guzhangjiance\CWRU-dataset-main
文件总数: 161

类别分布:
  B        (label=1): 40 个文件
  IR       (label=2): 40 个文件
  Normal   (label=0): 4 个文件
  OR       (label=3): 77 个文件
```

---

## 单通道模式

### 2. 标准单通道转换（推荐基线）

默认使用驱动端（DE）单通道，分段长度 2048，重叠 50%：

```bash
python tools\数据库转换.py --data_dir CWRU-dataset-main --output_dir output
```

**适用场景：**
- 快速建立诊断基线
- 与现有单通道模型兼容
- 数据量适中（~35,000 样本）

**输出：**
- `output/*.f`：每个文件为一个 2048 点的信号片段（float32 二进制）
- 文件命名：`{原文件名}_{序号}.f`

### 3. 自定义分段长度

```bash
# 短窗口（256 点）
python tools\数据库转换.py --data_dir CWRU-dataset-main --output_dir output_256 --segment_length 256

# 长窗口（4096 点）
python tools\数据库转换.py --data_dir CWRU-dataset-main --output_dir output_4096 --segment_length 4096
```

### 4. 不分段导出（原始整段）

导出每个 .mat 文件的完整信号（不切片）：

```bash
python tools\数据库转换.py --data_dir CWRU-dataset-main --output_dir output_full --no_seg
```

**注意：** 每个 .mat 文件生成一个样本，数量较少（~161 个）。

---

## 多通道模式

### 5. 双通道转换（DE + FE）

同时导出驱动端与风扇端信号，每个样本包含 2 个通道：

```bash
python tools\数据库转换.py --data_dir CWRU-dataset-main --output_dir output_mc --segment_length 256 --multi_channel --sensor_locations DE FE
```

**输出：**
- 形状：每个 `.f` 文件包含 `2 × 256 = 512` 个 float32 值（通道连续）
- 样本数：~280,000 个（双通道样本量更大）
- 数据布局：`[DE_0, DE_1, ..., DE_255, FE_0, FE_1, ..., FE_255]`

### 6. 三通道转换（DE + FE + BA）

```bash
python tools\数据库转换.py --data_dir CWRU-dataset-main --output_dir output_mc3 --segment_length 512 --multi_channel --sensor_locations DE FE BA
```

**注意：** 若某文件缺少某通道（如 BA），会用零填充。

### 7. 多通道拆分输出

每个通道单独写一个文件（便于单通道分析或对比）：

```bash
python tools\数据库转换.py --data_dir CWRU-dataset-main --output_dir output_split --segment_length 256 --multi_channel --sensor_locations DE FE --split_channels
```

**输出：**
- `{原文件名}_{序号}_DE.f`：驱动端信号
- `{原文件名}_{序号}_FE.f`：风扇端信号

---

## 高级选项

### 8. 筛选特定故障类型

仅导出内圈（IR）和外圈（OR）故障：

```bash
python tools\load_cwru.py CWRU-dataset-main output_ir_or --segment-length 2048 --fault-types IR OR
```

### 9. 不进行标准化

默认会对信号标准化（零均值、单位方差），若需原始值：

```bash
python tools\load_cwru.py CWRU-dataset-main output_raw --segment-length 2048 --no-normalize
```

### 10. 导出 NumPy 格式（用于深度学习）

使用 `load_cwru.py` 直接导出 `.npy` 和元数据：

```bash
python tools\load_cwru.py CWRU-dataset-main cwru_processed --segment-length 2048
```

**输出：**
- `cwru_processed/signals.npy`：形状 `(N, 2048)`
- `cwru_processed/labels.npy`：形状 `(N,)`，值为 0/1/2/3
- `cwru_processed/metadata.json`：每个样本的元数据

---

## Python 编程接口

### 11. 在代码中使用加载器

#### 单通道加载

```python
from tools.load_cwru import CWRUDataLoader

loader = CWRUDataLoader('CWRU-dataset-main')

# 加载所有数据（分段）
X, y, meta = loader.load_all(
    segment_length=2048,
    overlap=0.5,
    normalize=True
)

print(X.shape)  # (N, 2048)
print(y.shape)  # (N,)
```

#### 多通道加载

```python
from tools.load_cwru import CWRUDataLoader

loader = CWRUDataLoader('CWRU-dataset-main')

# 双通道加载（DE + FE）
X, y, meta = loader.load_all(
    segment_length=512,
    overlap=0.5,
    normalize=True,
    multi_channel=True,
    sensor_locations=['DE', 'FE']
)

print(X.shape)  # (N, 2, 512)
print(meta[0]['channels'])  # ['DE', 'FE']
```

### 12. PyTorch Dataset

```python
from tools.load_cwru import CWRUDataset
from torch.utils.data import DataLoader

# 单通道
dataset = CWRUDataset(
    root_dir='CWRU-dataset-main',
    segment_length=2048,
    normalize=True
)

# 多通道
dataset_mc = CWRUDataset(
    root_dir='CWRU-dataset-main',
    segment_length=512,
    normalize=True,
    multi_channel=True,
    sensor_locations=['DE', 'FE']
)

dataloader = DataLoader(dataset_mc, batch_size=32, shuffle=True)

for X, y in dataloader:
    print(X.shape)  # (32, 2, 512) - (batch, channels, length)
    print(y.shape)  # (32,)
    break
```

---

## 参数说明

### `数据库转换.py` 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | CWRU 数据集根目录 | `CWRU-dataset-main` |
| `--output_dir` | 输出目录 | `output` |
| `--segment_length` | 分段长度 | `2048` |
| `--no_seg` | 禁用分段，导出原始整段 | `False` |
| `--multi_channel` | 启用多通道模式 | `False` |
| `--sensor_locations` | 指定通道：`DE FE BA` | `None`（全部） |
| `--split_channels` | 每通道单独输出一个文件 | `False`（展平） |

### `load_cwru.py` 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--info` | 仅显示统计信息 | `False` |
| `--segment-length` | 分段长度 | `2048` |
| `--overlap` | 分段重叠比例 | `0.5` |
| `--fault-types` | 筛选故障类型 | `None` |
| `--no-normalize` | 不标准化 | `False` |

---
