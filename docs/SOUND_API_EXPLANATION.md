# Sound API 目录说明和转换流程

## 📁 sound_api 目录里放的是什么？

`sound_api` 目录位于 `tools/sound_api/`，是一个**声音转能量密度曲线的API工具集**。

### 目录内容

```
tools/sound_api/
├── convert_sound_api.py          # 核心API调用脚本
│   ├── 功能：调用外部API，将WAV音频文件转换为能量密度曲线
│   ├── 输入：WAV音频文件
│   └── 输出：JSON和XLSX格式的能量密度曲线数据
│
├── convert_mc_to_wav.py         # MC文件转WAV脚本
│   ├── 功能：将XJTU MC双通道文件(.f)转换为WAV音频
│   ├── 输入：.f文件 + .json元数据文件
│   └── 输出：WAV音频文件
│
├── batch_convert_xjtu.py        # 批量转换脚本（完整流程）
│   └── 功能：整合MC->WAV->API的完整转换流程
│
├── sound_api_output/            # API输出目录
│   ├── XJTU-SY_1_0.xlsx         # 转换后的Excel文件
│   ├── XJTU-SY_1_0.json         # 转换后的JSON文件
│   └── ...                      # 其他转换结果
│
└── docs/                        # 相关文档
    ├── README_XJTU_CONVERSION.md
    └── ...
```

### 核心功能

1. **API调用模块** (`convert_sound_api.py`)
   - 调用外部API: `http://115.236.25.110:8003/hardware/device/open-api/calculate-sound`
   - 将WAV音频文件上传到API
   - 接收并解析API返回的能量密度曲线数据
   - 保存为JSON和XLSX格式

2. **MC转WAV模块** (`convert_mc_to_wav.py`)
   - 读取XJTU MC双通道文件（.f格式）
   - 解析双通道数据
   - 转换为标准WAV音频格式

3. **批量转换模块** (`batch_convert_xjtu.py`)
   - 整合完整转换流程
   - 批量处理多个文件
   - 错误处理和进度显示

---

## 🔄 转换流程：从什么格式转换到什么格式？

### 完整转换链路

```
XJTU MC双通道文件 (.f + .json)
    ↓
    [步骤1: MC转WAV]
    ↓
WAV音频文件 (.wav)
    ↓
    [步骤2: API调用]
    ↓
能量密度曲线数据 (.xlsx + .json)
```

### 详细说明

#### 输入格式：XJTU MC双通道文件

**文件类型**:
- `.f` 文件：双通道振动数据（二进制格式）
- `.json` 文件：元数据（采样率、通道信息等）

**位置**: `datasets/output_xjtu_mc/xjtu/`

**示例**:
```
XJTU-SY_1_0.f      # 双通道振动数据
XJTU-SY_1_0.json   # 元数据
```

**数据特点**:
- 双通道（水平+垂直）
- 原始振动信号
- 需要转换为音频格式才能调用API

---

#### 中间格式：WAV音频文件

**文件类型**: `.wav`（标准音频格式）

**位置**: `datasets/output_xjtu_mc/wav_temp/`（临时目录）

**转换过程**:
1. 读取`.f`文件的二进制数据
2. 解析双通道数据
3. 归一化处理（min-max或z-score）
4. 转换为WAV格式（采样率20000Hz）

**特点**:
- 这是临时文件，转换完成后可以删除
- 用于上传到API

---

#### 输出格式：能量密度曲线数据

**文件类型**:
- `.xlsx`：Excel格式（用于查看和验证）
- `.json`：JSON格式（用于程序读取）

**位置**: `tools/sound_api/sound_api_output/`

**数据结构**:

**Excel格式** (`.xlsx`):
```
行0: 文件名（如：XJTU-SY_1_0）
行1: 列标题（频率 | 能量 | 密度）
行2开始: 数据
  列0: 频率 (Hz)      - 范围: 20-20000 Hz
  列1: 能量 (Volume)  - 反映信号冲击与瞬时功率
  列2: 密度 (Density)  - 反映流形空间的熵或混乱度
```

**JSON格式** (`.json`):
```json
{
  "frequency": [20.0, 20.1, ..., 19977.0],  // 3000个点
  "volume": [0.5, 0.6, ..., 141.13],        // 能量曲线
  "density": [0.1, 0.2, ..., 47.48]         // 密度曲线
}
```

**数据特点**:
- 每个文件包含3000个数据点
- 频率范围：20-20000 Hz
- 能量和密度曲线具有物理意义
- 可用于深度学习训练（转换为NPZ缓存）

---

## 🎯 转换目的

### 为什么需要转换？

1. **数据格式统一**
   - XJTU原始数据是双通道振动信号（.f格式）
   - 需要转换为标准的能量密度曲线格式
   - 与项目中其他声音数据格式保持一致

2. **特征提取**
   - 能量曲线：反映信号冲击与瞬时功率
   - 密度曲线：反映流形空间的熵或混乱度
   - 这两个特征互补，信息冗余度低

3. **深度学习训练**
   - 转换后的数据可以用于训练故障诊断模型
   - 支持健康指数回归（hi）和风险预测（risk）任务
   - 可以转换为NPZ缓存格式用于训练

---

## 📊 转换示例

### 输入文件
```
datasets/output_xjtu_mc/xjtu/
├── XJTU-SY_1_0.f
└── XJTU-SY_1_0.json
```

### 转换过程
```
[步骤1] MC -> WAV
  XJTU-SY_1_0.f → XJTU-SY_1_0.wav (临时文件)

[步骤2] WAV -> API -> 能量密度曲线
  XJTU-SY_1_0.wav → API调用 → 
  ├── XJTU-SY_1_0.xlsx (Excel格式)
  └── XJTU-SY_1_0.json (JSON格式)
```

### 输出文件
```
tools/sound_api/sound_api_output/
├── XJTU-SY_1_0.xlsx
└── XJTU-SY_1_0.json
```

---

## 🔧 当前转换任务

### 目标
- 转换10个XJTU MC文件
- 验证转换质量（能量密度相关性、物理意义）

### 使用脚本
- `convert_xjtu_10.py` 或 `run_convert_xjtu_10.bat`
- 自动跳过已转换的文件
- 使用最新Token: `JSESSIONID=node015xfjqikf7vxkroocnnjvtonn4.node0`

### 输出位置
- `tools/sound_api/sound_api_output/`

---

## 📝 总结

**sound_api目录** = 声音转能量密度曲线的API工具集

**转换流程**:
```
XJTU MC文件 (.f) 
  → WAV音频 (.wav) 
  → API调用 
  → 能量密度曲线 (.xlsx/.json)
```

**目的**: 将XJTU双通道振动数据转换为标准的能量密度曲线格式，用于深度学习训练和故障诊断。

---

**更新日期**: 2026-01-16
