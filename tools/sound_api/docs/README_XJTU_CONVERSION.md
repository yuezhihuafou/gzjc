# XJTU MC双通道数据批量转换工具

## 概述

这套工具用于将XJTU数据集的MC双通道振动信号文件批量转换为能量和密度曲线，以便后续进行深度学习训练。

## 转换流程

```
MC双通道文件(.f + .json)
         ↓
    [步骤1: convert_xjtu_mc_to_audio.py]
         ↓
    WAV音频文件
         ↓
    [步骤2: 声音API处理]
         ↓
能量和密度曲线(.xlsx + .json)
```

## 脚本文件说明

### 1. `convert_xjtu_mc_to_audio.py`

**功能**: 将MC双通道二进制文件(.f)转换为WAV音频文件

**主要特性**:
- 读取.f二进制文件和对应的.json元数据
- 支持多种通道模式:
  - `horizontal`: 只使用水平通道（推荐）
  - `vertical`: 只使用垂直通道
  - `mix`: 混合两个通道
  - `stereo`: 双通道立体声
- 支持两种归一化方法:
  - `minmax`: Min-Max归一化到[-1, 1]（推荐）
  - `zscore`: Z-Score归一化
- 自动生成转换报告

**使用方法**:
```bash
cd D:\guzhangjiance
python tools/convert_xjtu_mc_to_audio.py
```

**交互式选项**:
1. 输入MC文件目录（默认: `datasets/output_xjtu_mc/xjtu`）
2. 输入输出目录（默认: `datasets/output_xjtu_mc/wav_files`）
3. 选择通道模式（默认: horizontal）
4. 选择归一化方法（默认: minmax）

### 2. `batch_convert_xjtu_with_api.py`

**功能**: 完整的两步转换流程（MC -> WAV -> 能量密度曲线）

**主要特性**:
- 整合了MC转WAV和API转换两个步骤
- 自动调用声音能量密度API
- 可选择在转换完成后自动删除临时WAV文件
- 生成JSON和XLSX两种格式的输出
- 完整的转换报告

**使用方法**:
```bash
cd D:\guzhangjiance
python tools/batch_convert_xjtu_with_api.py
```

**交互式选项**:
1. 输入MC文件目录
2. 输入WAV临时目录
3. 输入API输出目录
4. 选择通道模式
5. 是否清理临时WAV文件

### 3. `sound_api/convert_sound_api.py`

**功能**: 调用API将音频文件转换为能量和密度曲线

**说明**: 这个脚本已经存在，`batch_convert_xjtu_with_api.py`会自动调用它的函数。

## 数据结构说明

### 输入数据结构

MC双通道文件格式:
```
datasets/output_xjtu_mc/xjtu/
├── XJTU-SY_1_0.f          # 二进制数据文件
├── XJTU-SY_1_0.json       # 元数据文件
├── XJTU-SY_1_1.f
├── XJTU-SY_1_1.json
...
```

JSON元数据示例:
```json
{
  "filepath": "...",
  "dataset": "XJTU-SY",
  "sampling_rate": 25600,
  "working_condition": "35Hz12kN",
  "speed_hz": 35,
  "load_kn": 12,
  "bearing_name": "Bearing1_1",
  "health_label": 0,
  "channels": ["Horizontal", "Vertical"],
  "data_shape": [2, 2048],
  "data_dtype": "float32"
}
```

### 输出数据结构

能量密度曲线文件:
```
datasets/output_xjtu_mc/api_output/
├── XJTU-SY_1_0.json       # JSON格式（包含频率、能量、密度数组）
├── XJTU-SY_1_0.xlsx       # Excel格式（兼容load_sound.py）
├── XJTU-SY_1_1.json
├── XJTU-SY_1_1.xlsx
...
└── complete_conversion_report.json  # 转换报告
```

## 快速开始示例

### 方案一：分步执行（适合调试）

```bash
# 步骤1: MC文件转WAV
cd D:\guzhangjiance
python tools/convert_xjtu_mc_to_audio.py

# 步骤2: WAV通过API转换（使用sound_api脚本）
cd tools/sound_api
python convert_sound_api.py
# 选择模式2（批量转换）
# 输入目录: D:\guzhangjiance\datasets\output_xjtu_mc\wav_files
```

### 方案二：一键完整转换（推荐）

```bash
cd D:\guzhangjiance
python tools/batch_convert_xjtu_with_api.py
```

使用默认配置，只需要按几次回车即可完成全部转换。

## 注意事项

### 1. API配置

确保API服务可用。默认API配置在 `tools/sound_api/convert_sound_api.py` 中的 `DEFAULT_API_CONFIG`。

如果API配置有变化，可以:
- 方法1: 直接编辑 `DEFAULT_API_CONFIG`
- 方法2: 在运行 `convert_sound_api.py` 时选择"使用自定义配置"

### 2. 内存和磁盘空间

- XJTU数据集文件数量很多（数千个文件）
- WAV文件会占用临时磁盘空间
- 建议在转换完成后清理临时WAV文件（使用cleanup_wav选项）

### 3. 处理时间

- 每个文件需要调用API处理，可能需要较长时间
- 建议先用少量文件测试流程
- 可以使用tqdm进度条监控进度

### 4. 错误处理

- 如果某个文件转换失败，不会影响其他文件
- 所有失败的文件会记录在转换报告中
- 可以查看报告找出失败原因并重新处理

## 转换报告说明

转换完成后会生成报告文件:

### `conversion_report.json`（步骤1报告）
```json
{
  "success": 1000,
  "failed": 10,
  "files": [
    {
      "file": "...",
      "output": "...",
      "status": "success",
      "metadata": {...}
    },
    ...
  ]
}
```

### `complete_conversion_report.json`（完整流程报告）
```json
{
  "mc_to_wav": {
    "success": 1000,
    "failed": 10
  },
  "wav_to_api": {
    "success": 950,
    "failed": 50
  },
  "total_files": 1000,
  "files": [...]
}
```

## 故障排查

### 问题1: 找不到MC文件

**错误**: `错误: 在 xxx 中未找到匹配 XJTU-SY_*.f 的文件`

**解决**:
- 检查输入目录路径是否正确
- 确认目录中包含 `.f` 文件

### 问题2: API调用失败

**错误**: `[失败] API调用失败，状态码: xxx`

**解决**:
- 检查API服务是否运行
- 检查网络连接
- 检查API配置（URL、headers等）
- 查看API错误信息

### 问题3: 内存不足

**错误**: `MemoryError` 或系统变慢

**解决**:
- 分批处理文件
- 减少同时处理的文件数量
- 使用cleanup_wav选项及时清理临时文件

## 下一步

转换完成后，能量密度曲线数据可以用于:

1. **数据加载**: 使用 `tools/load_sound.py` 加载XLSX文件
2. **可视化分析**: 使用 `visualize_sound_data.py` 可视化曲线
3. **深度学习训练**: 整合到 `dl/data_loader.py` 进行训练

## 相关文档

- `docs/README_SOUND_ANALYSIS.md` - 声音分析详细文档
- `PROJECT_STRUCTURE.md` - 项目整体结构
- `SOUND_DATA_STRUCTURE.md` - 声音数据结构说明

## 联系与支持

如有问题，请查看项目文档或提Issue。
