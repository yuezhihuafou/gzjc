# XJTU转换快速开始（优化版）

## 🚀 快速使用

### 方法1: 使用批处理文件（最简单）

```bash
# 双击运行或在命令行执行
run_convert_xjtu_10.bat
```

### 方法2: 直接运行Python

```bash
cd D:\guzhangjiance
python convert_xjtu_10.py
```

## ✅ 优化特性

1. **限制数量**: 只转换10个文件（可修改）
2. **跳过已转换**: 自动检查并跳过已转换的文件
3. **新Token**: 使用 `JSESSIONID=node015xfjqikf7vxkroocnnjvtonn4.node0`
4. **清理临时文件**: 自动删除WAV文件，节省存储
5. **进度显示**: 显示转换进度

## 📊 预期结果

- **处理时间**: 约1-2分钟（10个文件）
- **成功转换**: 最多10个文件
- **输出位置**: `tools/sound_api/sound_api_output/`

## 🔧 修改转换数量

编辑脚本中的这一行：
```python
all_files = sorted(list(mc_path.glob("XJTU-SY_*.f")))[:10]  # 改为 [:20] 转换20个
```

## 📝 验证转换结果

转换完成后，运行验证脚本：
```bash
python validate_xjtu.py
```

---

**Token**: JSESSIONID=node015xfjqikf7vxkroocnnjvtonn4.node0
