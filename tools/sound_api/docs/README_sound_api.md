# 声音API测试工具使用说明

## 问题排查

如果您遇到 `ModuleNotFoundError` 错误，请按以下步骤排查：

### 1. 检查当前环境

```bash
# 查看当前Python路径
python -c "import sys; print(sys.executable)"

# 查看conda环境
conda env list
```

### 2. 安装依赖

如果您使用的是项目本地环境（`.conda`），需要在该环境中安装依赖：

```bash
# 激活.conda环境（如果存在）
conda activate D:\guzhangjiance\.conda

# 或者直接安装到当前环境
pip install -r requirements.txt
```

### 3. 快速安装脚本所需的依赖

```bash
pip install requests pandas tqdm openpyxl
```

### 4. 验证安装

```bash
python -c "import requests, pandas, tqdm, openpyxl; print('所有依赖已安装')"
```

## 使用方法

### 交互式运行

```bash
python tools/test_sound_api.py
```

脚本会提示您输入：
1. API URL（从Apipost文档获取）
2. Token（如果需要认证）
3. 选择模式（单文件测试或批量转换）
4. 音频文件路径

### 非交互式运行（需要修改代码）

如果需要非交互式运行，可以修改脚本，使用命令行参数或配置文件。

## 常见问题

### Q: ModuleNotFoundError: No module named 'requests'
**A:** 运行 `pip install requests` 安装依赖

### Q: EOFError when reading a line
**A:** 这是正常的，脚本需要在交互式终端中运行，需要用户输入

### Q: .conda环境无法激活
**A:** 如果.conda环境有问题，可以使用base环境，或创建新的conda环境：
```bash
conda create -n guzhangjiance python=3.9
conda activate guzhangjiance
pip install -r requirements.txt
```


