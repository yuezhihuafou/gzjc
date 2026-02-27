# 使用声音能量曲线数据进行训练和验证

## 快速开始

### 1. 训练模型

使用声音能量曲线数据训练 ArcFace 模型：

```bash
python experiments/train.py --data_source sound --batch_size 8 --epochs 50
```

**参数说明：**
- `--data_source sound`: 使用声音能量曲线数据（默认是 `cwru`）
- `--batch_size 8`: 批大小（声音数据只有94个样本，建议用8-16）
- `--epochs 50`: 训练轮数
- `--lr 1e-3`: 学习率（可选，默认1e-3）
- `--split_ratio 0.7 0.15 0.15`: 数据集划分比例（可选）

**输出：**
- 模型权重保存在 `checkpoints/backbone.pth` 和 `checkpoints/arcface_head.pth`
- 训练过程中会显示每个 epoch 的训练/验证损失和准确率

### 2. 验证和推理

训练完成后，使用训练好的模型进行推理：

```bash
python experiments/inference.py --data_source sound --batch_size 8 --threshold 0.4
```

**参数说明：**
- `--data_source sound`: 使用声音数据
- `--batch_size 8`: 批大小
- `--threshold 0.4`: 余弦相似度阈值，低于此值判为 Unknown（可选，默认0.4）
- `--checkpoint checkpoints/backbone.pth`: 模型权重路径（可选）

**输出：**
- 显示测试集上的分类准确率
- 显示被判为 Unknown 的样本数量
- 显示已知类样本的分类准确率

## 数据说明

### 声音能量曲线数据
- **数据源**: `声音能量曲线数据/` 目录下的 xlsx 文件
- **样本数**: 94 个
- **数据格式**: 每个样本包含：
  - `frequency`: 频率数组 (约3000点)
  - `volume`: 能量曲线 (约3000点)
  - `density`: 密度曲线 (约3000点)
- **标签来源**: 从 `cwru_processed/metadata.json` 自动匹配

### 数据预处理
- 自动从训练集计算通道均值/标准差
- 对每个样本进行 Z-Score 标准化（按通道独立）
- 自动划分训练集/验证集/测试集（默认 70%/15%/15%）

## 与 CWRU 数据的对比

| 特性 | CWRU 数据 | 声音数据 |
|------|-----------|----------|
| 样本数 | ~35,000 | 94 |
| 数据格式 | (N, 2, L) numpy数组 | xlsx 文件 |
| 批大小建议 | 128 | 8-16 |
| 训练时间 | 较长 | 较短 |
| 适用场景 | 大规模训练 | 快速验证/小样本学习 |

## 使用示例

### 完整训练流程

```bash
# 1. 训练模型（使用声音数据）
python experiments/train.py --data_source sound --batch_size 8 --epochs 50

# 2. 推理验证
python experiments/inference.py --data_source sound --batch_size 8
```

### 对比实验

```bash
# 使用 CWRU 数据训练
python experiments/train.py --data_source cwru --batch_size 128 --epochs 30

# 使用声音数据训练
python experiments/train.py --data_source sound --batch_size 8 --epochs 50
```

## 注意事项

1. **小样本问题**: 声音数据只有94个样本，建议：
   - 使用较小的 batch_size (8-16)
   - 适当增加训练轮数
   - 考虑数据增强或迁移学习

2. **标签匹配**: 如果某些样本无法从 metadata.json 匹配到标签，会被自动过滤

3. **模型保存**: 训练过程中会自动保存最优模型到 `checkpoints/` 目录

4. **设备选择**: 脚本会自动检测并使用 GPU（如果可用）

## 故障排查

### 问题：找不到声音数据
**解决**: 检查 `声音能量曲线数据/` 目录是否存在，且包含 xlsx 文件

### 问题：标签匹配失败
**解决**: 检查 `cwru_processed/metadata.json` 是否存在，且包含对应的文件名

### 问题：内存不足
**解决**: 减小 batch_size，或使用 CPU 训练（会自动检测）

## 相关文件

- `dl/sound_data_loader.py`: 声音数据加载器
- `tools/load_sound.py`: 底层声音数据读取工具
- `experiments/train.py`: 训练脚本
- `experiments/inference.py`: 推理脚本

