# 🎯 项目快速导航索引

## 📚 关键文档位置

| 文档 | 位置 | 内容 | 适合场景 |
|-----|------|------|--------|
| **流程总览** | [SOUND_API_PROCESS_FLOW.md](./SOUND_API_PROCESS_FLOW.md) | 4个阶段的整体流程和数据流向 | 🎓 学习系统架构 |
| **深度架构** | [SOUND_API_ARCHITECTURE_DEEP_DIVE.md](./SOUND_API_ARCHITECTURE_DEEP_DIVE.md) | 详细的代码函数和算法说明 | 🔧 开发和调试 |
| **API使用** | `tools/sound_api/docs/README_声音API使用指南.md` | API调用方法和配置 | 💡 快速上手 |
| **CWRU加载** | `tools/load_cwru.py` (头部注释) | CWRU数据集加载说明 | 📖 数据集文档 |
| **XJTU加载** | `tools/load_xjtu.py` (头部注释) | XJTU数据集加载说明 | 📖 数据集文档 |

---

## 🏗️ 核心模块速查

### 模块1: 数据预处理

```
[核心脚本] tools/数据库转换.py
├─ 函数: convert_to_binary_files()
├─ 功能: CSV/MAT → 二进制.f + JSON元数据
└─ 输出: output/{cwru,xjtu}/*.f + *.json
```

**使用命令：**
```bash
# XJTU数据转换
python tools/数据库转换.py \
    --dataset_type xjtu \
    --xjtu_dir xjtu_dataset/XJTU-SY_Bearing_Datasets \
    --output_dir output_xjtu \
    --segment_length 512 \
    --multi_channel

# CWRU数据转换
python tools/数据库转换.py \
    --dataset_type cwru \
    --cwru_dir CWRU-dataset-main \
    --output_dir output_cwru \
    --fault_types IR OR
```

### 模块2: MC→WAV转换

```
[核心脚本] tools/sound_api/convert_mc_to_wav.py
├─ 函数: batch_convert_mc_to_wav()
├─ 功能: .f二进制 + JSON → WAV音频
├─ 参数: channel_mode={horizontal|vertical|stereo|mix}
└─ 输出: wav_output_dir/*.wav
```

**关键函数：**
```python
load_binary_signal(f_file, json_file)   # 加载.f+JSON
normalize_signal(signal, method)         # 归一化
convert_to_wav(data, output_file)       # 单文件转换
batch_convert_mc_to_wav(input_dir, ...) # 批量处理
```

### 模块3: WAV→API转换

```
[核心脚本] tools/sound_api/convert_sound_api.py
├─ 函数: test_sound_api()
├─ 功能: WAV文件 + API参数 → 能量密度曲线
├─ API: http://115.236.25.110:8003/hardware/device/open-api/calculate-sound
└─ 输出: JSON/XLSX格式
```

**关键函数：**
```python
get_default_config()              # 获取默认API配置
test_sound_api(wav_file, ...)    # 单个API调用
parse_api_response(response)      # 响应解析（多格式支持）
save_to_json_with_metadata(...)   # 保存结果
```

### 模块4: 完整自动化流程

```
[整合脚本] tools/sound_api/batch_convert_xjtu.py
├─ 函数: convert_xjtu_mc_to_energy_density()
├─ 流程: MC → WAV → API → JSON/XLSX
└─ 特性: 自动元数据追踪、错误处理、临时文件清理
```

**使用命令：**
```bash
python tools/sound_api/batch_convert_xjtu.py \
    --mc_input_dir output_xjtu/xjtu \
    --wav_output_dir temp_wav \
    --api_output_dir api_output \
    --channel_mode horizontal \
    --cleanup_wav
```

---

## 🔄 完整工作流

### 场景1: XJTU数据 → 能量曲线（一键处理）

```bash
# ─────────────────────────────────────────────────
# 步骤1: 原始数据 → MC二进制格式
# ─────────────────────────────────────────────────
python tools/数据库转换.py \
    --dataset_type xjtu \
    --xjtu_dir xjtu_dataset/XJTU-SY_Bearing_Datasets \
    --output_dir output_xjtu \
    --segment_length 512 \
    --multi_channel

# 输出:
# output_xjtu/
# ├─ xjtu/
# │  ├─ XJTU-SY_Bearing1_1_1_0.f       (512*2*4 bytes = 4KB)
# │  ├─ XJTU-SY_Bearing1_1_1_0.json
# │  └─ ...

# ─────────────────────────────────────────────────
# 步骤2: MC → WAV → API → 能量曲线
# ─────────────────────────────────────────────────
python tools/sound_api/batch_convert_xjtu.py \
    --mc_input_dir output_xjtu/xjtu \
    --wav_output_dir temp_wav \
    --api_output_dir api_output \
    --channel_mode horizontal \
    --cleanup_wav

# 输出:
# api_output/
# ├─ XJTU-SY_Bearing1_1_1_0.json       (能量密度曲线)
# ├─ XJTU-SY_Bearing1_1_1_0.xlsx       (可选)
# └─ ...
```

### 场景2: CWRU数据 → 能量曲线

```bash
# 步骤1: CWRU数据转换
python tools/数据库转换.py \
    --dataset_type cwru \
    --cwru_dir CWRU-dataset-main \
    --output_dir output_cwru \
    --segment_length 2048 \
    --multi_channel \
    --sensor_locations DE FE \
    --fault_types IR OR

# 步骤2: MC → WAV → API
python tools/sound_api/batch_convert_xjtu.py \
    --mc_input_dir output_cwru/cwru \
    --wav_output_dir temp_wav_cwru \
    --api_output_dir api_output_cwru \
    --channel_mode stereo \
    --cleanup_wav
```

### 场景3: 仅测试单个文件

```bash
# 创建WAV文件
python tools/sound_api/convert_mc_to_wav.py \
    --input_file output_xjtu/xjtu/XJTU-SY_Bearing1_1_1_0.f \
    --json_file output_xjtu/xjtu/XJTU-SY_Bearing1_1_1_0.json \
    --output_file test.wav

# 调用API
python tools/sound_api/convert_sound_api.py \
    --test-single test.wav \
    --output-dir test_output

# 查看结果
cat test_output/XJTU-SY_Bearing1_1_1_0.json | jq .
```

---

## 💾 关键数据结构

### 元数据JSON示例（XJTU）

```json
{
  "data": {
    "frequency": [20.0, 20.667, ..., 19999.333],
    "volume": [-60.5, -58.2, ..., -20.3],
    "density": [0.12, 0.15, ..., 0.88]
  },
  "metadata": {
    "bearing_id": "Bearing1_1",
    "t": 1,
    "source_path": "/path/to/original/data",
    "api_url": "http://115.236.25.110:8003/...",
    "api_params": {
      "freq1": "20",
      "freq2": "20000",
      "freqCount": "3000"
    },
    "created_at": "2025-12-25T15:30:45"
  }
}
```

### 元数据JSON示例（.f文件配套）

```json
{
  "dataset": "XJTU-SY",
  "bearing_name": "Bearing1_1",
  "file_number": 1,
  "working_condition": "35Hz12kN",
  "speed_hz": 35,
  "load_kn": 12,
  "sampling_rate": 25600,
  "health_label": 0,
  "channels": ["Horizontal", "Vertical"],
  "data_shape": [2, 512],
  "data_dtype": "float32",
  "data_length": 1024,
  "segment_index": 0,
  "binary_file": "XJTU-SY_Bearing1_1_1_0.f",
  "label": 0,
  "is_fft_data": false
}
```

---

## ⚙️ 常用参数速查

### 数据库转换参数

| 参数 | 默认值 | 说明 | 例子 |
|-----|-------|------|------|
| `--dataset_type` | cwru | 数据集类型 | cwru/xjtu/both |
| `--segment_length` | 2048 | 分段长度 | 512/1024/2048 |
| `--overlap` | 0.5 | 重叠比例 | 0.3/0.5/0.7 |
| `--multi_channel` | False | 多通道模式 | - |
| `--sampling_rates` | None | 采样率过滤 | 12000 48000 |
| `--fault_types` | None | 故障类型过滤 | Normal B IR OR |
| `--health_ratio` | 0.3 | XJTU健康比例 | 0.1-0.5 |

### MC到WAV转换参数

| 参数 | 默认值 | 说明 | 例子 |
|-----|-------|------|------|
| `channel_mode` | horizontal | 通道模式 | horizontal/vertical/stereo/mix |
| `normalize_method` | minmax | 归一化方法 | minmax/zscore |
| `cleanup_wav` | False | 清理临时文件 | - |

---

## 🐛 常见问题速查

| 问题 | 原因 | 解决方案 |
|-----|------|--------|
| API超时 | 网络慢或API服务重 | 增加timeout参数到120秒 |
| 内存不足 | 一次加载过多文件 | 减小segment_length或使用流式处理 |
| 响应格式错误 | API版本不同 | 检查parse_api_response中的格式支持 |
| WAV文件音量异常 | 归一化参数不合适 | 尝试normalize_method='zscore' |
| 频率点数不一致 | API返回数据质量问题 | 检查日志，跳过该文件 |

---

## 📊 性能基准

| 操作 | 数据量 | 耗时 | 并行化潜力 |
|-----|-------|------|---------|
| 数据转换(CSV→.f) | 1000文件 | ~5秒 | 低 (I/O密集) |
| MC→WAV | 1000文件 | ~10秒 | 低 (I/O密集) |
| API调用 | 1000文件 | ~1000秒 | 高 (网络密集) |
| 响应解析 | 1000文件 | ~5秒 | 中 (CPU密集) |

---

## 🔗 相关工具集成

### 与load_sound.py集成

```python
from tools.load_sound import load_sound_api_data

# 加载API输出
frequency, volume, density, meta = load_sound_api_data(
    'api_output/XJTU-SY_1_0.json'
)

# 用于特征提取
from core.features import SoundMetricsExtractor

extractor = SoundMetricsExtractor()
features = extractor.extract({
    'frequency': frequency,
    'volume': volume,
    'density': density
})
```

### 与特征提取集成

```python
from core.features import SoundMetricsExtractor

# 创建提取器
extractor = SoundMetricsExtractor()

# 从API输出提取特征
for json_file in api_output_dir:
    with open(json_file) as f:
        data = json.load(f)
    
    features = extractor.extract(data['data'])
    # 特征: {
    #   'peak_db': float,
    #   'avg_db': float,
    #   'peak_density': float,
    #   'avg_density': float,
    #   'energy': float,
    #   'entropy': float,
    #   ...
    # }
```

---

## 📞 快速参考卡

### API配置（默认值）

```
URL: http://115.236.25.110:8003/hardware/device/open-api/calculate-sound
频率范围: 20-20000 Hz
频率点数: 3000
采样率参数: 192000
超时时间: 60秒
重试次数: 3次
```

### 文件命名约定

```
.f二进制:     {DATASET}_{ID}_{INDEX}.f
元数据JSON:   {DATASET}_{ID}_{INDEX}.json
WAV音频:      {DATASET}_{ID}_{INDEX}_{CHANNEL}.wav
API输出:      {DATASET}_{ID}_{INDEX}.json
```

### 数据形状速查

```
CSV原始:      (32768, 2)        # 行×列
MC文件:       (2, 32768)        # 通道×采样点
分段后:       (N, 2, 512)       # 段×通道×采样点
WAV格式:      (32768,) mono     # 采样点 (单声道)
               (32768, 2) stereo # 采样点×通道 (立体声)
API输出:      (3000,)           # 频率点数
```

---

快速导航完成！选择适合你的场景，参考对应的文档和命令即可开始使用。🚀
