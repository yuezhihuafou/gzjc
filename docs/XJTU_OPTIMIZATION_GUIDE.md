# XJTU转换优化指南

## 🎯 优化目标

1. **降低开销**: 限制转换数量，跳过已转换文件
2. **提高效率**: 自动清理临时文件，减少存储占用
3. **更新Token**: 使用最新的API认证Token
4. **进度显示**: 使用进度条显示转换进度

## 📋 优化内容

### 1. 限制转换数量

**问题**: 之前脚本尝试转换全部285,696个文件，耗时2.5小时

**优化**: 
- 默认只转换10个文件
- 可通过参数调整数量

```python
convert_limited_xjtu_files(limit=10)  # 只转换10个
```

### 2. 跳过已转换文件

**优化**: 自动检查并跳过已转换的文件

```python
skip_existing=True  # 跳过已转换的文件
```

**检查逻辑**:
- 检查输出目录中是否已存在对应的`.json`和`.xlsx`文件
- 如果存在，跳过转换，节省时间和API调用

### 3. 更新API Token

**新Token**: `JSESSIONID=node015xfjqikf7vxkroocnnjvtonn4.node0`

**更新方式**:
1. 使用优化脚本（自动更新）
2. 或手动更新`convert_sound_api.py`中的配置

### 4. 清理临时文件

**优化**: 转换成功后自动清理临时WAV文件

```python
cleanup_wav=True  # 清理临时WAV文件，降低存储开销
```

**节省空间**: 每个WAV文件约几MB，清理后可以节省大量存储空间

### 5. 进度显示

**优化**: 使用`tqdm`显示转换进度

```
API转换进度: 100%|████████| 10/10 [00:30<00:00, 3.33file/s]
```

## 🚀 使用方法

### 方法1: 使用优化脚本（推荐）

```bash
cd D:\guzhangjiance\tools\sound_api
python convert_xjtu_optimized.py
```

### 方法2: 作为模块调用

```python
from tools.sound_api.convert_xjtu_optimized import convert_limited_xjtu_files

result = convert_limited_xjtu_files(
    mc_input_dir=r'D:\guzhangjiance\datasets\output_xjtu_mc\xjtu',
    wav_output_dir=r'D:\guzhangjiance\datasets\output_xjtu_mc\wav_temp',
    api_output_dir=r'D:\guzhangjiance\tools\sound_api\sound_api_output',
    limit=10,
    skip_existing=True,
    cleanup_wav=True
)
```

## 📊 优化效果对比

### 优化前
- ❌ 尝试转换全部285,696个文件
- ❌ 耗时约2.5小时
- ❌ 所有API调用失败（Token过期）
- ❌ 不清理临时文件，占用大量存储

### 优化后
- ✅ 只转换10个文件（可配置）
- ✅ 预计耗时约1-2分钟
- ✅ 使用最新Token，成功率提高
- ✅ 自动清理临时文件，节省存储
- ✅ 跳过已转换文件，避免重复工作

## 🔧 配置说明

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `limit` | 10 | 转换文件数量限制 |
| `skip_existing` | True | 是否跳过已转换的文件 |
| `cleanup_wav` | True | 是否清理临时WAV文件 |
| `channel_mode` | 'horizontal' | 通道模式 |
| `normalize_method` | 'minmax' | 归一化方法 |

### Token配置

**当前Token**: `JSESSIONID=node015xfjqikf7vxkroocnnjvtonn4.node0`

**更新位置**: `tools/sound_api/convert_xjtu_optimized.py` 中的 `get_optimized_api_config()` 函数

## 📝 使用示例

### 转换10个文件（默认）

```bash
python convert_xjtu_optimized.py
```

### 转换20个文件

修改脚本中的 `limit=10` 为 `limit=20`

### 不清理临时文件（用于调试）

修改脚本中的 `cleanup_wav=True` 为 `cleanup_wav=False`

## ⚠️ 注意事项

1. **Token有效期**: Token可能会过期，如果API调用失败，需要更新Token
2. **存储空间**: 虽然会清理临时文件，但最终输出文件仍需要存储空间
3. **API限制**: 注意API的调用频率限制，避免过于频繁的请求

## 🔄 后续优化建议

1. **批量处理**: 可以进一步优化为批量API调用（如果API支持）
2. **断点续传**: 记录转换进度，支持中断后继续
3. **并行处理**: 如果API支持，可以考虑并行转换多个文件
4. **错误重试**: 添加失败重试机制

---

**更新日期**: 2026-01-16  
**Token**: JSESSIONID=node015xfjqikf7vxkroocnnjvtonn4.node0
