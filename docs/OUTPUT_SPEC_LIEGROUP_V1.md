# 李群转换输出规范 v1

## 概述

本文档定义了声音 API 输出的 JSON/Excel 文件格式，以及训练阶段使用的 NPZ 缓存格式。

**核心原则（JSON-first）**：
- **JSON 为主产物**：默认仅保存 JSON 格式（包含 metadata），xlsx 仅作为调试回退（需 `--write-xlsx`）
- **训练阶段必须使用 NPZ 缓存**：不直接读取 JSON 或 Excel，只读 `datasets/sound_api/cache_npz/`
- **目录规范**：所有产物统一落在 `datasets/sound_api/`，按 `bearing_id` 分桶
- **禁止在 tools/sound_api/ 下落任何数据**：避免 IDE 索引卡顿
- 所有数据转换在离线预处理阶段完成

## JSON 文件格式（主产物）

### 目录结构

```
datasets/sound_api/
├── output_json/          # JSON 主产物（训练不读，用于缓存生成/追溯）
│   ├── {bearing_id_1}/
│   │   ├── XJTU-SY_1_0_t000001.json
│   │   └── ...
│   └── {bearing_id_2}/
│       └── ...
├── output_xlsx/          # xlsx 调试产物（默认不写，--write-xlsx 才写）
│   ├── {bearing_id_1}/
│   └── ...
├── cache_npz/            # NPZ 缓存（训练唯一入口）
│   ├── {bearing_id_1}/
│   │   ├── 000000.npz
│   │   └── ...
│   └── ...
└── logs/                  # 日志和统计报表
    ├── bad_files_cache.txt
    ├── bad_bearings.txt
    ├── warnings.txt      # 原始 t 质量告警（renumber 模式）
    └── conversion_report.json
```

### JSON 文件格式

每个 JSON 文件包含以下结构：

```json
{
  "data": {
    "frequency": [20.0, 20.1, ...],  // 频率数组（3000个点）
    "volume": [0.5, 0.6, ...],       // 音量/能量数组（3000个点）
    "density": [0.1, 0.2, ...]       // 密度数组（3000个点）
  },
  "metadata": {
    "bearing_id": "1_0",              // 轴承标识（必填，可为null）
    "t": 1,                           // 时间序号（可选，可为null）
    "source_path": "/path/to/audio.wav",  // 源音频文件路径
    "api_url": "http://...",          // API URL（可选，可为null）
    "api_params": {...},              // API 参数（可选，可为null）
    "created_at": "2024-01-01T12:00:00"  // 时间戳
  }
}
```

**注意**：
- `metadata` 字段使用稳定的 schema，所有字段都存在（可为 null）
- `api_params` 字段名从 `form_data_params` 改为 `api_params`（更简洁）
- 时间戳字段名从 `timestamp` 改为 `created_at`（更明确）

### 文件命名规范

每个 JSON 文件对应一个样本，命名格式：

```
XJTU-SY_{bearing_id}_t{时间序号}.json
```

或

```
XJTU-SY_{bearing_id}_{时间序号}.json
```

示例：
- `XJTU-SY_1_0_t000001.json`
- `XJTU-SY_1_0_000001.json`
- `XJTU-SY_1_0.json`（无时间序号时，按文件名排序后自动分配）

## Excel 文件格式（调试回退）

### 文件命名规范

每个 Excel 文件对应一个样本，命名格式与 JSON 相同（仅扩展名不同）：

```
XJTU-SY_{bearing_id}_t{时间序号}.xlsx
```

### 文件内容格式

每个 Excel 文件包含一个 sheet，格式如下：

| 行号 | 内容 | 说明 |
|------|------|------|
| 1 | `XJTU-SY_{bearing_id}` | 轴承标识（文本） |
| 2 | `frequency, volume, density` | 列标题（可选） |
| 3+ | 数据行 | 三列数据：频率、音量、密度 |

**数据要求**：
- 从第 3 行开始读取数据
- 至少 3000 行数据
- 三列：`frequency` (Hz), `volume`, `density`

### bearing_id 解析规则（优先级1→4）

1. **优先级1：从文件名解析**：正则表达式 `XJTU-SY_(.+?)(?:_t?\d+)?\.(json|xlsx)`
2. **优先级2：从JSON metadata解析**：读取 `metadata.bearing_id` 字段
3. **优先级3：从xlsx第1行解析**：读取第 1 行第 1 列，匹配 `XJTU-SY_(.+)`
4. **优先级4：使用文件名基础部分**：如果仍无法解析，使用文件名（不含扩展名）作为 bearing_id

**冲突处理**：文件名与 metadata 不一致时，默认以文件名为准，同时记录警告。

### 时间序号 t 解析规则（优先级1→4）

1. **优先级1：从文件名解析**：
   - 格式 `_t000123`：提取数字部分
   - 格式 `_000123`：提取 6 位数字
2. **优先级2：从JSON metadata解析**：读取 `metadata.t` 字段
3. **优先级3：从xlsx第1行解析**（如果支持）
4. **优先级4：自动分配**：如果文件名和metadata中无序号，则按 bearing_id 分组，按文件名排序后分配 `t=0..T-1`

**一致性要求**：
- 每个 bearing 的 t 必须连续递增（0, 1, 2, ..., T-1）
- 不满足则标记为问题文件

## NPZ 缓存格式

### 目录结构

```
datasets/sound_api/cache_npz/
├── {bearing_id_1}/
│   ├── 000000.npz
│   ├── 000001.npz
│   └── ...
├── {bearing_id_2}/
│   ├── 000000.npz
│   └── ...
└── ...
```

### NPZ 文件内容

每个 NPZ 文件包含以下字段：

| 字段名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `x` | `float32` | `(2, 3000)` | 双通道数据，x[0]=log1p(volume), x[1]=density |
| `frequency` | `float32` | `(3000,)` | 频率数组（可选，元数据） |
| `bearing_id` | `str` | - | 轴承标识（可选，元数据） |
| `t` | `int` | - | 时间序号（重编号后，用于训练） |
| `orig_t` | `int` | - | 原始时间序号（保留原始采集顺序，用于追溯，可选） |
| `source_path` | `str` | - | 源文件路径（可选，元数据） |

**关于 `t` 和 `orig_t` 的区别**：
- **`t`（训练用）**：稳定重编号（0..T-1），保证连续性，用于序列建模和训练
- **`orig_t`（追溯用）**：保留原始采集顺序（如文件名中的时间戳），用于追溯与人工校验

**关键变换**：
- `x[0] = log1p(volume)`：对音量取 log1p 变换
- `x[1] = density`：直接使用密度
- 数据类型：`float32`
- 固定长度：3000（不足则截断，超出则取前 3000 行）

## 索引文件

### cache_npz/index.jsonl（推荐）

当运行 `tools/build_sound_api_cache.py` 构建 NPZ 缓存后，会在缓存目录写出：

- `datasets/sound_api/cache_npz/index.jsonl`

每行一个 JSON 记录，用于**追溯与完整性对账**（不用于训练读取）：

```json
{"t": 12, "orig_t": 146413, "path": "XJTU-SY_1000_146413.json", "bearing_id": "1000"}
```

- **`t`（训练用）**：稳定重编号（0..T-1），仅用于序列建模/训练
- **`orig_t`（追溯用）**：保留原始采集顺序（如文件名中的时间戳），用于追溯与人工校验
- **`path`**：源文件名（JSON/xlsx 的文件名）
- **`bearing_id`**：轴承标识

### index.csv（legacy）

索引文件格式（CSV）：

```csv
path,bearing_id,t
datasets/sound_api_output/XJTU-SY_1_0_t000001.xlsx,1_0,1
datasets/sound_api_output/XJTU-SY_1_0_t000002.xlsx,1_0,2
...
```

**用途**：
- 记录所有有效 Excel 文件的位置
- 提供 bearing_id 和 t 的映射关系
- 用于生成 NPZ 缓存

## 训练数据集

### 数据加载流程（最短流程，JSON-first）

1. **API转换**（生成JSON，默认不写xlsx，按bearing_id分桶）：
   ```bash
   # 使用 convert_sound_api.py 转换音频文件为JSON（包含metadata）
   python tools/sound_api/convert_sound_api.py --audio_dir datasets/audio --output_root datasets/sound_api
   # 如果你只有 XJTU 的 MC 数据（.f + sidecar .json），推荐使用不落盘 wav 的直连脚本：
   # python tools/sound_api/convert_mc_to_api_json.py --mc_dir datasets/output_xjtu_mc --output_root datasets/sound_api --resume --workers 8
   # 默认只保存JSON到 datasets/sound_api/output_json/{bearing_id}/
   # 使用 --write-xlsx 才同时保存xlsx到 datasets/sound_api/output_xlsx/{bearing_id}/（仅用于调试）
   # 注意：不会在 tools/sound_api/ 下落任何数据，避免IDE卡顿
   ```

2. **构建NPZ缓存**（递归扫描，优先JSON回退xlsx）：
   ```bash
   # 一键从 output_json 构建 NPZ 缓存（递归扫描）
   python tools/build_sound_api_cache.py --json_dir datasets/sound_api/output_json --xlsx_dir datasets/sound_api/output_xlsx --output_dir datasets/sound_api/cache_npz --workers 8
   # 递归扫描 output_json 和 output_xlsx，优先读取JSON，如果不存在则回退到同名xlsx
   # 自动质量门禁（长度3000、finite、std、non_zero_ratio、volume>=0等）
   # 失败文件记录到 datasets/sound_api/logs/bad_files_cache.txt
   # t 不连续的 bearing 记录到 datasets/sound_api/logs/bad_bearings.txt
   
   # 可选：使用不同的 t 分配策略
   # renumber（推荐）：按文件名排序强制重编号 t=0..T-1，保证连续性
   python tools/build_sound_api_cache.py --t-policy renumber
   # fill_gaps：保留显式 t，只为缺失的 t 补齐最小缺口
   python tools/build_sound_api_cache.py --t-policy fill_gaps
   
   # 可选：调整原始 t 质量告警阈值（默认 30%）
   # 如果某 bearing 的原始 t 存在"乱序/重复/跳号"且问题比例超过阈值，会写入 warnings.txt
   python tools/build_sound_api_cache.py --orig-t-warning-threshold 0.5  # 50% 才告警
   ```

3. **开始训练**：
   ```bash
   # 训练（支持hi/risk/cls任务）
   python experiments/train.py --data_source sound_api_cache --task hi --batch_size 8 --epochs 50
   python experiments/train.py --data_source sound_api_cache --task risk --horizon 10 --batch_size 8 --epochs 50
   # 默认缓存目录: datasets/sound_api/cache_npz
   # 训练阶段只读 NPZ，不读 JSON 或 xlsx
   ```

**旧流程（兼容保留，用于回退）**：
   ```bash
   # 1. 生成索引（如果使用xlsx）
   python tools/build_sound_api_index.py --data_dir datasets/sound_api_output
   
   # 2. 转换为 NPZ 缓存（从xlsx）
   python tools/cache_sound_api_to_npz.py --index datasets/sound_api_output/index.csv
   ```

2. **训练阶段**：
   - 从 NPZ 缓存加载数据
   - 使用 `dl/sound_api_cache_dataset.py` 中的 `SoundAPICacheDataset`
   - 支持 bearing-level split（按 bearing_id 分组切分）

### 支持的任务

#### 1. 健康指数回归 (HI)

- **标签计算**：`y_hi = 1 - t/(T-1)`
- **边界情况**：`T=1` 时，`y_hi = 1.0`
- **损失函数**：MSE 或 Huber Loss
- **评估指标**：MAE, RMSE

#### 2. 风险预测 (Risk)

- **标签计算**：给定 `horizon=K`，`tf = floor(0.3*T)`，`y = 1 if t+K >= tf else 0`
- **损失函数**：BCEWithLogitsLoss
- **评估指标**：ROC-AUC, PR-AUC, Accuracy

#### 3. 分类任务 (CLS)

- **当前状态**：无真实标签，返回占位标签 `y=0`
- **说明**：仅用于占位，不可用于实际训练
- **后续扩展**：可读取 `labels.csv` 获取真实标签

## 标准化

训练阶段对数据进行**按通道独立的 Z-Score 标准化**：

```python
x_norm[c] = (x[c] - mean[c]) / (std[c] + 1e-8)
```

其中 `mean` 和 `std` 在**训练集**上按通道统计得到，并在 val/test 上复用。

## 注意事项

1. **JSON-first原则**：推荐使用JSON格式（包含metadata），xlsx仅作为调试回退（需 `--write-xlsx`）
2. **目录规范**：所有产物统一落在 `datasets/sound_api/`，按 `bearing_id` 分桶
3. **禁止在 tools/sound_api/ 下落任何数据**：避免 IDE 索引卡顿
4. **训练阶段只读 NPZ**：不直接读取 JSON/xlsx，必须使用 NPZ 缓存（`datasets/sound_api/cache_npz/`）
5. **volume >= 0 检查**：质量门禁会拒绝包含负值的 volume（方案2：判坏样本）
6. **volume 必须 log1p**：在转换为 NPZ 时，必须对 volume 应用 `log1p` 变换
7. **t 连续性硬校验**：缓存构建完成后会校验每个 bearing 的 t 是否连续递增，不通过则记录到 `datasets/sound_api/logs/bad_bearings.txt`
8. **bearing-level split**：训练时按 bearing_id 分组切分，保证同一 bearing 的所有样本在同一子集
9. **固定随机种子**：使用固定随机种子（默认 42）保证可复现性
10. **质量门禁**：`build_sound_api_cache.py` 会自动执行质量检查（长度3000、finite、std、non_zero_ratio、volume>=0等），失败文件记录到 `datasets/sound_api/logs/bad_files_cache.txt`
11. **断点续跑**：目标 npz 存在则跳过，支持并行写入（原子写入：临时文件 + rename，修复了 bug）
12. **pandas/openpyxl 懒加载**：仅在需要 xlsx 回退时导入，减少默认路径依赖
13. **递归扫描输入**：`build_sound_api_cache.py` 会递归扫描 `output_json` 和 `output_xlsx` 目录

## 质量门禁

`build_sound_api_cache.py` 在构建NPZ缓存时会自动执行质量检查：

- **长度检查**：frequency/volume/density 必须等于 3000
- **有限值检查**：所有值必须是 finite（不允许 inf 或 nan）
- **volume >= 0 检查**：volume 不允许有负值（方案2：判坏样本）
- **非零比例检查**：volume 的非零比例必须 >= 0.05
- **标准差检查**：log1p(volume) 和 density 的标准差必须 > 1e-6（不能是常数）

不满足质量门禁的文件会记录到 `datasets/sound_api/logs/bad_files_cache.txt`，不会生成NPZ文件。

## t 连续性硬校验

`build_sound_api_cache.py` 在构建NPZ缓存完成后会自动执行 t 连续性校验：

- 对每个 bearing，收集写出的 t 列表
- 校验是否等于 0..T-1（连续递增，无重复）
- 不通过则输出到 `datasets/sound_api/logs/bad_bearings.txt` 并在 stdout 报告

## 文件清单

### 推荐使用（新流程）
- `tools/sound_api/convert_sound_api.py`：API转换脚本（JSON-first，默认只写JSON）
- `tools/build_sound_api_cache.py`：一键构建NPZ缓存（优先JSON回退xlsx，质量门禁，并行处理）
- `dl/sound_api_cache_dataset.py`：NPZ 数据集类（支持hi/risk/cls任务，bearing-level split）
- `dl/data_loader.py`：数据加载器（`get_sound_api_cache_dataloaders`，打印分配概况）
- `experiments/train.py`：训练脚本（支持sound_api_cache和task=hi/risk，arcface保持不变）

### 兼容保留（旧流程，用于回退）
- `tools/build_sound_api_index.py`：构建索引脚本（从xlsx）
- `tools/cache_sound_api_to_npz.py`：Excel → NPZ 转换脚本（从xlsx）
