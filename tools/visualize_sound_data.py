#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化声音能量曲线数据，展示数据结构
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT_DIR = Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.load_sound import SoundDataLoader

def visualize_sample_data():
    """可视化单个样本的数据结构"""
    loader = SoundDataLoader()
    sample_names = sorted(loader.get_available_files())
    
    # 选择一个正常样本和一个故障样本
    normal_sample = [s for s in sample_names if 'Normal' in s][0]
    fault_sample = [s for s in sample_names if 'Normal' not in s][0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('声音能量曲线数据结构示例', fontsize=16, fontweight='bold')
    
    for idx, (sample_name, title_prefix) in enumerate([(normal_sample, '正常样本'), (fault_sample, '故障样本')]):
        curves = loader.load_sound_curves(sample_name)
        if curves is None:
            continue
        
        freq = curves['frequency']
        volume = curves['volume']
        density = curves['density']
        
        # 左图：能量曲线
        axes[idx, 0].plot(freq, volume, 'b-', linewidth=1.5, alpha=0.7)
        axes[idx, 0].set_xlabel('频率 (Hz)', fontsize=11)
        axes[idx, 0].set_ylabel('能量 (Volume)', fontsize=11)
        axes[idx, 0].set_title(f'{title_prefix} - 能量曲线\n样本: {sample_name}', fontsize=12)
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].set_xlim(0, 10000)  # 只显示前10kHz
        
        # 右图：密度曲线
        axes[idx, 1].plot(freq, density, 'r-', linewidth=1.5, alpha=0.7)
        axes[idx, 1].set_xlabel('频率 (Hz)', fontsize=11)
        axes[idx, 1].set_ylabel('密度 (Density)', fontsize=11)
        axes[idx, 1].set_title(f'{title_prefix} - 密度曲线\n样本: {sample_name}', fontsize=12)
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_xlim(0, 10000)
    
    plt.tight_layout()
    output_path = 'sound_data_structure_example.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    plt.close()

def create_data_structure_doc():
    """创建数据结构说明文档"""
    doc = """# 声音能量曲线数据结构说明

## 一、数据来源和格式

### 1.1 原始数据文件
- **位置**: `声音能量曲线数据/` 目录
- **格式**: Excel (.xlsx) 文件，共 11 个文件
- **结构**: 每个 xlsx 文件包含多个 **Sheet**，每个 Sheet 对应一个样本

### 1.2 Sheet 命名规则
- Sheet 名称格式: `{文件名}.wav`
- 示例: `97_Normal_0.wav`, `234_0.wav`, `108_3.wav`

### 1.3 数据列结构
每个 Sheet 包含 3 列数据（跳过前 2 行：文件名和列标题）：

| 列索引 | 列名 | 单位 | 说明 |
|--------|------|------|------|
| 0 | 频率 (Frequency) | Hz | 频率值，范围约 20-20000 Hz |
| 1 | 音量/能量 (Volume/Energy) | - | 能量曲线值，反映信号冲击与瞬时功率 |
| 2 | 密度 (Density) | - | 密度曲线值，反映流形空间的熵或混乱度 |

---

## 二、加载后的数据结构

### 2.1 Python 字典格式

使用 `SoundDataLoader.load_sound_curves()` 加载后，返回一个字典：

```python
curves = {
    'frequency': np.array,  # 形状: (3000,), dtype: float32
    'volume': np.array,     # 形状: (3000,), dtype: float32
    'density': np.array     # 形状: (3000,), dtype: float32
}
```

### 2.2 数据统计信息

基于 94 个样本的统计：

**能量曲线 (Volume)**:
- 均值: ~111.33
- 标准差: ~117.14
- 范围: 0.13 - 1011.63
- 中位数: ~84.51

**密度曲线 (Density)**:
- 均值: ~8.28
- 标准差: ~4.62
- 范围: 0.10 - 80.14
- 中位数: ~7.47

**数据长度**:
- 所有样本长度一致: **3000 个点**
- 频率范围: 20.02 - 19976.99 Hz

---

## 三、故障类型分布

| 故障类型 | 样本数 | 说明 |
|---------|--------|------|
| Normal | 4 | 正常状态 |
| B (Ball) | 19 | 滚动体故障 |
| IR (Inner Race) | 16 | 内圈故障 |
| OR (Outer Race) | 55 | 外圈故障 |
| **总计** | **94** | - |

---

## 四、深度学习输入格式转换

### 4.1 原始格式 → 双通道格式

**原始格式** (3个独立数组):
```python
frequency: (3000,)  # 频率轴（用于可视化，训练时不使用）
volume:    (3000,)  # 能量曲线
density:   (3000,)  # 密度曲线
```

**转换后格式** (双通道张量):
```python
x = np.stack([volume, density], axis=0)  # 形状: (2, 3000)
```

### 4.2 批处理格式

在 DataLoader 中，多个样本组成 batch：

```python
batch_x: (batch_size, 2, 3000)  # 例如: (8, 2, 3000)
batch_y: (batch_size,)           # 例如: (8,)
```

其中：
- `batch_x[:, 0, :]` = 所有样本的能量曲线
- `batch_x[:, 1, :]` = 所有样本的密度曲线

### 4.3 标准化处理

在训练前，会对每个样本进行 **按通道独立的 Z-Score 标准化**：

```python
# 在训练集上计算统计量
channel_mean = [mean(volume_train), mean(density_train)]  # (2,)
channel_std  = [std(volume_train),  std(density_train)]   # (2,)

# 对每个样本标准化
x_norm[0, :] = (volume - channel_mean[0]) / channel_std[0]
x_norm[1, :] = (density - channel_mean[1]) / channel_std[1]
```

---

## 五、物理含义解释

### 5.1 能量曲线 (Volume/Energy)

- **定义**: 绝对频域幅度，反映信号在特定频率处的能量强度
- **特点**: 
  - 对**绝对频率位置**敏感
  - 容易受转速漂移影响
  - 直观反映能量分布
- **故障特征**:
  - 正常: 能量均匀分散，高频主导
  - 内圈故障: 低频峰值增加 (BPFI 频率)
  - 外圈故障: 中频能量增加 (BPFO 频率)
  - 滚动体故障: 宽频段扰动

### 5.2 密度曲线 (Density)

- **定义**: 相对能量浓度，反映能量分布的形状
- **特点**:
  - 对**能量分布形状**敏感
  - 对频率漂移**鲁棒** (SE(3) 不变性)
  - 适合故障模式识别
- **故障特征**:
  - 正常: 平坦分布
  - 内圈故障: 低频集中
  - 外圈故障: 中频尖峰
  - 滚动体故障: 多峰分布

### 5.3 为什么需要双通道？

1. **信息互补**: 能量和密度的相关系数 r ≈ 0.31（中等相关），信息冗余度低
2. **鲁棒性**: 能量提供直观特征，密度提供频率漂移鲁棒性
3. **故障区分**: 不同故障类型在两通道上的表现不同，有助于分类

---

## 六、数据加载流程

### 6.1 使用 SoundDataLoader

```python
from tools.load_sound import SoundDataLoader

# 初始化
loader = SoundDataLoader(sound_data_dir='声音能量曲线数据')

# 获取所有可用样本
available = loader.get_available_files()  # 返回 94 个样本名

# 加载单个样本
curves = loader.load_sound_curves('97_Normal_0')
# 返回: {'frequency': array, 'volume': array, 'density': array}
```

### 6.2 使用深度学习 DataLoader

```python
from dl.sound_data_loader import get_sound_dataloaders

# 获取训练/验证/测试 DataLoader
train_loader, val_loader, test_loader = get_sound_dataloaders(
    batch_size=8,
    split_ratio=(0.7, 0.15, 0.15)
)

# 使用
for x, y in train_loader:
    # x: (batch_size, 2, 3000) - 双通道数据
    # y: (batch_size,) - 标签
    pass
```

---

## 七、数据可视化示例

运行以下命令生成可视化图表：

```bash
python visualize_sound_data.py
```

将生成 `sound_data_structure_example.png`，展示：
- 正常样本的能量和密度曲线
- 故障样本的能量和密度曲线
- 对比分析

---

## 八、注意事项

1. **数据长度**: 所有样本长度固定为 3000 点，无需填充或截断
2. **标签匹配**: 样本名称需要与 `cwru_processed/metadata.json` 中的文件名匹配才能获取标签
3. **标准化**: 必须在训练集上计算统计量，然后在所有数据集上使用相同的统计量
4. **小样本**: 只有 94 个样本，建议使用较小的 batch_size (8-16) 和更多训练轮数

---

## 九、相关文件

- `tools/load_sound.py`: 底层数据加载器
- `dl/sound_data_loader.py`: 深度学习数据加载器
- `explore_sound_data.py`: 数据探索脚本
- `visualize_sound_data.py`: 数据可视化脚本
"""
    
    with open('SOUND_DATA_STRUCTURE.md', 'w', encoding='utf-8') as f:
        f.write(doc)
    print("数据结构说明文档已保存到: SOUND_DATA_STRUCTURE.md")

if __name__ == '__main__':
    print("生成数据结构可视化...")
    visualize_sample_data()
    print("\n生成数据结构说明文档...")
    create_data_structure_doc()
    print("\n完成！")

