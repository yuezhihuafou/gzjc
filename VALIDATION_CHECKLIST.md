# 新流程验收清单（JSON-first → NPZ缓存 → 训练）

## 目标
验证"声音API → JSON → NPZ缓存 → 训练（HI/Risk）"的完整链路是否可以稳定运行，数据是否正确。

## 前置条件
- 环境已配置（依赖已安装）：`numpy, pandas, openpyxl, torch, sklearn, tqdm, requests`
- 至少有 1-3 个样本的 JSON 或 xlsx 文件（可用现有的 `tools/sound_api/sound_api_output/*.json` 测试）

---

## 验收步骤

### 第1步：API转换（可选，若已有JSON可跳过）
```bash
# 如果你有音频文件，转换它们
python tools/sound_api/convert_sound_api.py --audio_dir datasets/audio --output_root datasets/sound_api

# 验收点：
# ✓ 输出到 datasets/sound_api/output_json/{bearing_id}/*.json
# ✓ 不会在 tools/sound_api/ 下生成数据
# ✓ JSON 包含 data 和 metadata 字段
```

### 第2步：构建NPZ缓存（核心）
```bash
# 一键构建（使用默认的 renumber 策略）
python tools/build_sound_api_cache.py --workers 4

# 验收点1：输出目录结构
✓ datasets/sound_api/cache_npz/{bearing_id}/000000.npz, 000001.npz, ...
✓ bearing 目录数 >= 1
✓ 总 NPZ 文件数 >= 输入文件数（质量门禁通过的数量）

# 验收点2：日志与统计
✓ 打印"成功: X 个文件, 跳过: Y, 失败: Z"
✓ 打印"所有 bearing 的 t 都连续递增 ✓" 或 "发现 N 个 bearing 的 t 不连续"
✓ 如有失败，datasets/sound_api/logs/bad_files_cache.txt 存在且内容正确
✓ 如有不连续，datasets/sound_api/logs/bad_bearings.txt 存在且内容正确
✓ 如果原始 t 有质量问题（乱序/重复/跳号），打印告警并生成 datasets/sound_api/logs/warnings.txt

# 验收点3：断点续跑
# 再次运行相同命令
python tools/build_sound_api_cache.py --workers 4
✓ 打印"跳过: X 个文件（已存在）"，X 应等于上次成功数
✓ 不会重新生成已有的 npz 文件（检查文件修改时间不变）

# 验收点4：NPZ 文件内容
# 手动抽查一个 npz 文件
python -c "
import numpy as np
data = np.load('datasets/sound_api/cache_npz/{某个bearing}/000000.npz', allow_pickle=True)
print('字段:', list(data.keys()))
print('x.shape:', data['x'].shape)
print('x.dtype:', data['x'].dtype)
print('bearing_id:', data['bearing_id'])
print('t (训练用):', data['t'])
if 'orig_t' in data:
    print('orig_t (追溯用):', data['orig_t'])
"
✓ 字段包含: x, frequency, bearing_id, t, source_path
✓ 如果有原始 t，还应包含 orig_t 字段
✓ x.shape = (2, 3000)
✓ x.dtype = float32
✓ t 为重编号（0..T-1），orig_t 保留原始值

# 或使用测试脚本
python tools/test_orig_t_in_npz.py
✓ 打印 NPZ 文件结构和字段
✓ 验证 t 和 orig_t 的区别
```

### 第3步：训练启动（HI任务）
```bash
python experiments/train.py --data_source sound_api_cache --task hi --batch_size 8 --epochs 2

# 验收点1：数据加载
✓ 打印"找到 N 个样本"
✓ 打印"数据集划分概况 (Bearing-level Split)"
✓ 打印训练集/验证集/测试集的样本数和 bearing 数
✓ 打印"通道均值"和"通道标准差"

# 验收点2：训练执行
✓ Epoch 1 能启动并打印 "Train Loss: X.XXXX"
✓ 不报错（形状不匹配、缺失字段等）
✓ 模型权重保存到 checkpoints/backbone.pth 和 checkpoints/hi_head.pth

# 验收点3：输出文件
✓ experiments/outputs/plots/hi_bearing_{bearing_id}.png 生成
✓ experiments/outputs/metrics.csv 生成且包含 test_loss, test_mae, test_rmse
```

### 第4步：训练启动（Risk任务）
```bash
python experiments/train.py --data_source sound_api_cache --task risk --horizon 10 --batch_size 8 --epochs 2

# 验收点：
✓ 数据加载与划分正常（同上）
✓ Epoch 1 能启动
✓ 打印"Val AUC: X.XXXX | Val PR-AUC: X.XXXX"
✓ 模型权重保存到 checkpoints/risk_head.pth
✓ experiments/outputs/plots/risk_bearing_{bearing_id}.png 生成
✓ metrics.csv 包含 test_auc, test_pr_auc
```

### 第5步：ArcFace旧流程不受影响（回归测试）
```bash
# 如果你有 cwru_processed/ 数据，验证 ArcFace 仍可用
python experiments/train.py --data_source cwru --task arcface --batch_size 32 --epochs 2

# 验收点：
✓ 能正常启动并训练（不报错）
✓ 保存 checkpoints/arcface_head.pth
✓ 说明新流程没有破坏旧功能
```

---

## 已知问题与排查
### 问题1：构建缓存时报错"未找到任何 JSON 或 xlsx 文件"
**原因**：输入目录为空或路径错误  
**解决**：
- 检查 `datasets/sound_api/output_json/` 是否存在且包含 `.json` 文件
- 或将现有的测试数据（`tools/sound_api/sound_api_output/*.json`）移动到 `datasets/sound_api/output_json/{bearing_id}/`

### 问题2：训练时报错"未找到任何样本在目录"
**原因**：NPZ缓存未生成或路径错误  
**解决**：
- 先运行步骤2（构建缓存）
- 检查 `datasets/sound_api/cache_npz/` 是否包含子目录和 `.npz` 文件

### 问题3：t 不连续（bad_bearings.txt 有记录）
**原因**：使用了 fill_gaps 策略但显式 t 有缺口  
**解决**：
- 使用默认的 `--t-policy renumber` 策略（强制重编号）
- 或手动检查并修正源文件的 t 标注

---

## 成功标志
- ✅ 能从 JSON/xlsx 生成 NPZ 缓存（无报错或报错清晰可定位）
- ✅ t 连续性校验通过（bad_bearings.txt 为空或少量可接受）
- ✅ 训练启动能打印 bearing-level split 概况
- ✅ HI/Risk 任务能正常训练并生成预测曲线图
- ✅ ArcFace 旧流程仍可用（回归测试通过）
- ✅ 所有产物只落在 `datasets/sound_api/`，tools 目录下无数据堆积
