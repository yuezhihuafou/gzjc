# CWRU 轴承故障数据集加载工具

本工具用于批量读取 [CWRU (Case Western Reserve University) 轴承数据集](https://csegroups.case.edu/bearingdatacenter/)，自动解析元数据，便于集成到故障诊断智能模型中。

## 功能特点

- ✅ 递归扫描所有 `.mat` 文件
- ✅ 自动解析故障类型、采样率、负载等元数据
- ✅ 支持信号分段切片（生成固定长度样本）
- ✅ 支持导出为 NumPy 格式
- ✅ 提供 PyTorch Dataset 类（可直接用于 DataLoader）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速使用

### 命令行

```bash
# 查看数据集统计信息
python tools/load_cwru.py CWRU-dataset-main --info

# 导出为 NumPy 格式（默认分段长度 2048，重叠 50%）
python tools/load_cwru.py CWRU-dataset-main ./output

# 自定义分段长度
python tools/load_cwru.py CWRU-dataset-main ./output --segment-length 1024 --overlap 0.3

# 只导出特定故障类型
python tools/load_cwru.py CWRU-dataset-main ./output --fault-types Normal IR OR
```

### Python 代码

```python
from tools.load_cwru import CWRUDataLoader, FAULT_NAMES

# 创建加载器
loader = CWRUDataLoader('CWRU-dataset-main')

# 查看类别分布
print(loader.get_class_distribution())
# {'Normal': 4, 'B': 28, 'IR': 28, 'OR': 36}

# 加载所有数据（分段）
X, y, meta = loader.load_all(
    segment_length=2048,  # 每个样本长度
    overlap=0.5,          # 重叠比例
    normalize=True        # 标准化
)
print(f"样本数: {len(y)}, 形状: {X.shape}")

# 筛选特定故障类型
X, y, meta = loader.load_all(
    segment_length=2048,
    fault_types=['IR', 'OR'],  # 只加载内圈和外圈故障
    sampling_rates=[12000]     # 只加载 12kHz 采样率数据
)

# 导出为 NumPy 文件
loader.export_numpy('./output', segment_length=2048)
```

### PyTorch 集成

```python
from torch.utils.data import DataLoader
from tools.load_cwru import CWRUDataset

# 创建 Dataset
dataset = CWRUDataset(
    'CWRU-dataset-main',
    segment_length=2048,
    normalize=True
)

# 创建 DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练循环
for X, y in train_loader:
    # X: (batch_size, 1, 2048) - 信号
    # y: (batch_size,) - 标签
    pass
```

## 数据结构

### 故障类型标签

| 标签 | 故障类型 | 说明 |
|------|----------|------|
| 0 | Normal | 正常状态 |
| 1 | Ball (B) | 滚动体故障 |
| 2 | Inner Race (IR) | 内圈故障 |
| 3 | Outer Race (OR) | 外圈故障 |

### 元数据字段

每个样本的元数据包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `filepath` | str | 原始文件路径 |
| `sampling_rate` | int | 采样率 (12000 或 48000 Hz) |
| `sensor_location` | str | 传感器位置 (DE=驱动端, FE=风扇端) |
| `fault_type` | str | 故障类型 (Normal, B, IR, OR) |
| `fault_label` | int | 故障标签 (0-3) |
| `fault_diameter` | float | 故障直径 (mm) |
| `load_hp` | int | 负载 (0-3 HP) |
| `rpm` | int | 转速 (RPM) |
| `or_position` | str | 外圈故障位置 (@3, @6, @12) |

## 输出文件

导出后会生成以下文件：

```
output/
├── signals.npy      # 信号数组 (N, segment_length)
├── labels.npy       # 标签数组 (N,)
└── metadata.json    # 元数据列表
```

## 后续集成建议

### 1. 简单模型（传统机器学习）

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 加载数据
X = np.load('output/signals.npy')
y = np.load('output/labels.npy')

# 提取简单特征（均值、标准差、最大值等）
features = np.column_stack([
    X.mean(axis=1),
    X.std(axis=1),
    X.max(axis=1),
    X.min(axis=1),
])

# 训练
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print(f"准确率: {clf.score(X_test, y_test):.4f}")
```

### 2. 深度学习模型（1D-CNN）

```python
import torch
import torch.nn as nn

class BearingCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 使用
model = BearingCNN(num_classes=4)
```

## 参考资料

- [CWRU Bearing Data Center](https://csegroups.case.edu/bearingdatacenter/)
- [数据集说明文档](CWRU-dataset-main/README.md)
