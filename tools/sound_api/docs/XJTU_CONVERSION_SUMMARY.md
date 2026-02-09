# XJTU MC双通道批量转换脚本完成总结

## ✅ 已完成的工作

### 1. 核心转换脚本

已创建三个核心脚本文件：

#### `convert_xjtu_mc_to_audio.py`
- **功能**: MC双通道文件(.f) -> WAV音频文件
- **状态**: ✅ 已完成并测试通过
- **特性**:
  - 读取二进制.f文件和.json元数据
  - 支持4种通道模式（horizontal/vertical/mix/stereo）
  - 支持2种归一化方法（minmax/zscore）
  - 自动生成转换报告
  - 批量处理功能
  - 进度条显示

#### `batch_convert_xjtu_with_api.py`
- **功能**: 完整两步转换流程（MC -> WAV -> 能量密度曲线）
- **状态**: ✅ 已完成
- **特性**:
  - 整合MC转WAV和API处理
  - 自动调用声音能量密度API
  - 支持临时文件清理
  - 生成JSON和XLSX双格式输出
  - 完整转换报告
  - 错误处理和重试机制

#### `test_xjtu_conversion.py`
- **功能**: 单文件转换流程测试
- **状态**: ✅ 已完成并测试通过
- **测试结果**:
  - ✅ MC文件加载：成功
  - ✅ WAV转换：成功
  - ⚠️  API调用：返回401（需要有效认证）

### 2. 测试结果

```
[步骤1] 加载MC文件
  ✓ 加载成功
  数据形状: (2, 2048)
  数据类型: float32
  采样率: 25600 Hz

[步骤2] 转换为WAV音频
  ✓ 转换成功
  输出文件: XJTU-SY_1_0.wav
  文件大小: 4.04 KB

[步骤3] 调用API
  ⚠ 认证失败（401）- 需要配置有效的API认证
```

### 3. 文档

已创建完整的使用文档：

#### `README_XJTU_CONVERSION.md`
- 工具概述和转换流程
- 各脚本详细说明
- 数据结构说明
- 快速开始指南
- 故障排查
- 相关文档索引

## 📋 使用指南

### 方案一：分步转换（适合调试）

```bash
# 步骤1: MC文件转WAV
cd D:\guzhangjiance
python tools/convert_xjtu_mc_to_audio.py

# 按提示输入:
# - MC文件目录（默认: datasets/output_xjtu_mc/xjtu）
# - 输出目录（默认: datasets/output_xjtu_mc/wav_files）
# - 通道模式（推荐: 1-horizontal）
# - 归一化方法（推荐: 1-minmax）

# 步骤2: WAV通过API转换
cd tools/sound_api
python convert_sound_api.py
# 选择模式2（批量转换）
# 输入WAV文件目录
```

### 方案二：一键转换（推荐）

```bash
cd D:\guzhangjiance
python tools/batch_convert_xjtu_with_api.py

# 按提示输入各项配置，或直接回车使用默认值
```

### 方案三：测试单个文件

```bash
cd D:\guzhangjiance
python tools/test_xjtu_conversion.py
```

## ⚙️ 配置说明

### API配置

在使用批量转换前，需要确保API配置正确：

**位置**: `tools/sound_api/convert_sound_api.py`

**默认配置**:
```python
DEFAULT_API_CONFIG = {
    'api_url': 'http://115.236.25.110:8003/hardware/device/open-api/calculate-sound',
    'headers': {
        "Cookie": "JSESSIONID=...",  # 需要有效的session
        ...
    },
    'form_data_params': {
        'freq1': '20',
        'freq2': '20000',
        'freqCount': '3000',
        'sampleFrq': '192000',
        ...
    }
}
```

**如何更新配置**:

1. **方法1** - 直接编辑配置
   - 打开 `tools/sound_api/convert_sound_api.py`
   - 修改 `DEFAULT_API_CONFIG` 字典

2. **方法2** - 从cURL导入
   - 运行 `python tools/sound_api/convert_sound_api.py`
   - 选择"方式2: 从cURL命令导入"
   - 粘贴从Apipost复制的cURL命令

3. **方法3** - 手动输入
   - 运行脚本时选择"方式3: 手动输入"
   - 输入API URL和Token

## 📊 数据规模估算

### XJTU数据集规模
- MC文件数量：~142,910 个文件
- 每个文件包含：2通道 × 2048点 × float32
- 单个文件大小：~16 KB（二进制）

### 转换后文件大小
- WAV文件：~4 KB/文件
- JSON输出：~180 KB/文件
- XLSX输出：~100 KB/文件

### 总计估算
- 临时WAV文件：~560 MB
- API输出文件：~39 GB

### 处理时间估算
假设：
- MC->WAV: ~0.01秒/文件
- API处理: ~1秒/文件（取决于网络和服务器）

总时间：
- MC->WAV: ~24分钟
- API处理: ~40小时（建议分批处理）

## 🚀 下一步建议

### 1. 配置API认证
- 从Apipost或浏览器获取有效的Cookie/Token
- 更新 `DEFAULT_API_CONFIG` 中的认证信息

### 2. 小批量测试
```bash
# 建议先处理少量文件测试
# 可以手动复制100个文件到测试目录
mkdir datasets\output_xjtu_mc\test_sample
# 复制部分文件...
python tools/batch_convert_xjtu_with_api.py
```

### 3. 分批处理全部数据
由于文件数量巨大，建议分批处理：

```python
# 可以修改 batch_convert_xjtu_with_api.py
# 添加文件数量限制或分批逻辑

# 例如：只处理前1000个文件
binary_files = list(input_path.glob(file_pattern))[:1000]
```

### 4. 监控和日志
- 每批处理完检查转换报告
- 对失败的文件单独重新处理
- 保存每批次的报告用于追溯

### 5. 数据集成
转换完成后，可以：
- 使用 `tools/load_sound.py` 加载XLSX文件
- 整合到 `dl/data_loader.py` 用于深度学习
- 参考CWRU的处理方式创建 `xjtu_processed/` 目录

## 📁 文件清单

```
tools/
├── convert_xjtu_mc_to_audio.py          # MC转WAV脚本
├── batch_convert_xjtu_with_api.py       # 完整转换流程
├── test_xjtu_conversion.py              # 测试脚本
├── README_XJTU_CONVERSION.md            # 详细文档
├── XJTU_CONVERSION_SUMMARY.md           # 本文件
└── sound_api/
    └── convert_sound_api.py             # API转换脚本（已存在）
```

## ⚠️ 注意事项

1. **API认证**: 必须配置有效的认证信息才能调用API
2. **磁盘空间**: 确保有足够空间存储临时文件和输出文件
3. **网络稳定**: API调用需要稳定的网络连接
4. **处理时间**: 全量数据处理需要很长时间，建议后台运行
5. **错误处理**: 出现错误不会中断整个流程，查看报告了解失败原因

## 🔧 故障排查

### 问题1: 找不到MC文件
```
错误: 在 xxx 中未找到匹配 XJTU-SY_*.f 的文件
```
**解决**: 检查目录路径，确认包含.f文件

### 问题2: API 401认证失败
```
[认证失败] API需要认证，状态码: 401
```
**解决**: 更新API配置中的Cookie或Token

### 问题3: API超时
```
[超时] 请求超时（60秒）
```
**解决**: 
- 检查网络连接
- 增加timeout参数
- 检查API服务状态

### 问题4: 内存不足
**解决**:
- 分批处理文件
- 使用cleanup_wav选项清理临时文件
- 关闭其他占用内存的程序

## ✅ 完成状态

- [x] MC文件加载功能
- [x] WAV转换功能
- [x] API调用集成
- [x] 批量处理功能
- [x] 进度显示
- [x] 错误处理
- [x] 转换报告生成
- [x] 测试脚本
- [x] 完整文档

## 📞 技术支持

如有问题：
1. 查看 `README_XJTU_CONVERSION.md` 详细文档
2. 运行 `test_xjtu_conversion.py` 诊断问题
3. 查看生成的转换报告文件
4. 检查项目其他文档（PROJECT_STRUCTURE.md等）

---

**创建日期**: 2026-01-15
**版本**: 1.0
**状态**: ✅ 已完成，可以开始使用
