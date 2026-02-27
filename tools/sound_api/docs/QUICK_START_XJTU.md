# XJTU MC双通道转换 - 快速开始

## 🚀 快速上手（5分钟）

### 步骤1: 测试单个文件（验证环境）

```bash
cd D:\guzhangjiance
python tools\test_xjtu_conversion.py
```

**预期输出**:
- ✅ MC文件加载成功
- ✅ WAV转换成功
- ⚠️  API调用（可能需要配置认证）

### 步骤2: 配置API认证（如果需要）

编辑文件: `tools\sound_api\convert_sound_api.py`

找到 `DEFAULT_API_CONFIG`，更新Cookie/Token：

```python
DEFAULT_API_CONFIG = {
    'headers': {
        "Cookie": "JSESSIONID=你的session",  # 更新这里
        ...
    },
    ...
}
```

### 步骤3: 批量转换

```bash
cd D:\guzhangjiance
python tools\batch_convert_xjtu_with_api.py
```

**然后按回车使用默认配置！**

---

## 📂 输入输出位置

### 默认输入
```
datasets\output_xjtu_mc\xjtu\
├── XJTU-SY_1_0.f
├── XJTU-SY_1_0.json
├── XJTU-SY_1_1.f
├── XJTU-SY_1_1.json
...（共142910个文件）
```

### 默认输出
```
datasets\output_xjtu_mc\api_output\
├── XJTU-SY_1_0.json       # 能量密度曲线（JSON）
├── XJTU-SY_1_0.xlsx       # 能量密度曲线（Excel）
├── XJTU-SY_1_1.json
├── XJTU-SY_1_1.xlsx
...
└── complete_conversion_report.json  # 转换报告
```

---

## 🎯 三种使用方式

### 方式1: 一键全自动（推荐新手）

```bash
python tools\batch_convert_xjtu_with_api.py
# 全部按回车，使用默认配置
```

### 方式2: 仅转WAV（不调用API）

```bash
python tools\convert_xjtu_mc_to_audio.py
# 适合：只想生成WAV文件，稍后再处理
```

### 方式3: 分步执行（灵活控制）

```bash
# 步骤A: MC -> WAV
python tools\convert_xjtu_mc_to_audio.py

# 步骤B: WAV -> 能量密度（使用API）
cd tools\sound_api
python convert_sound_api.py
# 选择模式2（批量转换）
```

---

## ⚙️ 常用配置选项

### 通道模式选择

| 模式 | 说明 | 推荐 |
|------|------|------|
| horizontal | 只用水平通道 | ✅ 推荐 |
| vertical | 只用垂直通道 | - |
| mix | 混合两通道 | - |
| stereo | 双声道 | - |

**建议**: 使用 `horizontal` 模式，与CWRU数据集保持一致

### 归一化方法

| 方法 | 说明 | 推荐 |
|------|------|------|
| minmax | [-1,1]范围 | ✅ 推荐 |
| zscore | 标准化 | - |

**建议**: 使用 `minmax` 方法，WAV标准格式

---

## 📊 处理进度监控

### 查看实时进度

脚本运行时会显示进度条：
```
转换中: 45%|████████      | 450/1000 [00:45<00:55, 10.00it/s]
```

### 查看转换报告

```bash
# 查看JSON报告
type datasets\output_xjtu_mc\api_output\complete_conversion_report.json
```

报告内容示例：
```json
{
  "mc_to_wav": {
    "success": 950,
    "failed": 50
  },
  "wav_to_api": {
    "success": 900,
    "failed": 50
  }
}
```

---

## 🔧 常见问题

### Q1: API返回401错误
**A**: 需要配置有效的Cookie/Token
- 从浏览器或Apipost获取
- 更新 `DEFAULT_API_CONFIG`

### Q2: 处理太慢怎么办？
**A**: 分批处理
- 先处理100个文件测试
- 手动复制部分文件到测试目录
- 或修改脚本限制文件数量

### Q3: 磁盘空间不够
**A**: 使用清理选项
- 运行时选择"删除临时WAV文件"
- 或手动定期清理 `wav_temp` 目录

### Q4: 中断后如何继续？
**A**: 自动跳过已处理文件
- 查看输出目录，已有的文件不会重复处理
- 或删除失败文件重新处理

---

## 📈 性能参考

### 单文件处理时间
- MC -> WAV: ~0.01秒
- API处理: ~1秒（网络相关）

### 批量处理建议
- 小批测试: 100文件 (~2分钟)
- 中批处理: 1000文件 (~20分钟)
- 全量处理: 142910文件 (~40小时)

**建议**: 分批处理，每批1000-5000个文件

---

## 📝 下一步

转换完成后：

1. **验证数据**
   ```bash
   python tools\load_sound.py
   # 加载生成的XLSX文件验证
   ```

2. **可视化检查**
   ```bash
   python visualize_sound_data.py
   # 可视化能量密度曲线
   ```

3. **整合训练**
   - 参考 `dl/data_loader.py`
   - 创建XJTU数据集的DataLoader
   - 开始深度学习训练

---

## 📚 相关文档

- **详细文档**: `tools/README_XJTU_CONVERSION.md`
- **完成总结**: `tools/XJTU_CONVERSION_SUMMARY.md`
- **项目结构**: `PROJECT_STRUCTURE.md`
- **API文档**: `tools/sound_api/README.md`

---

**提示**: 第一次使用建议先运行 `test_xjtu_conversion.py` 测试环境！
