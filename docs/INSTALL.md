# 安装指南

## 🎯 推荐方式：使用Conda环境（长期维护）

**优势**:
- ✅ 环境存储在conda默认位置，不依赖项目目录
- ✅ 便于长期使用和维护
- ✅ 支持多环境管理（CPU/GPU）
- ✅ 自动处理依赖冲突

### Windows 环境

#### CPU版本（默认，适合没有NVIDIA GPU）

```bash
# 1. 创建conda环境（一键完成）
setup_conda_env.bat

# 2. 激活环境
activate_env.bat

# 3. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

#### GPU版本（如果有NVIDIA GPU）

```bash
# 1. 创建GPU环境
setup_conda_env.bat gpu

# 2. 激活环境
activate_env.bat gpu

# 3. 验证CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Windows GPU要求**:
- NVIDIA GPU（支持CUDA）
- NVIDIA驱动（最新版本）
- PyTorch会自动安装CUDA运行时，无需单独安装CUDA Toolkit

### Ubuntu/Linux 环境

#### CPU版本

```bash
# 1. 给脚本执行权限
chmod +x setup_conda_env.sh activate_env.sh

# 2. 创建CPU环境
bash setup_conda_env.sh

# 3. 激活环境
source activate_env.sh
```

#### GPU版本（推荐，用于训练）

```bash
# 1. 给脚本执行权限
chmod +x setup_conda_env.sh activate_env.sh

# 2. 创建GPU环境
bash setup_conda_env.sh gpu

# 3. 激活环境
source activate_env.sh gpu

# 4. 验证CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**详细说明**: 查看 [CONDA_ENV_GUIDE.md](CONDA_ENV_GUIDE.md)

---

## 备选方式：使用Python虚拟环境

### Windows 环境（CPU 训练）

```bash
# 1. 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 PyTorch (CPU 版本)
pip install torch torchvision torchaudio
```

### Ubuntu 环境（GPU 训练，4070 显卡）

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装基础依赖
pip install --upgrade pip
pip install numpy scipy pandas scikit-learn matplotlib seaborn tqdm openpyxl joblib

# 3. 安装 PyTorch (CUDA 版本)
# 首先检查 CUDA 版本: nvidia-smi 或 nvcc --version

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 验证安装
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 依赖说明

### 必需依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| numpy | >=1.20.0 | 数值计算 |
| scipy | >=1.7.0 | 科学计算（信号处理） |
| pandas | >=1.3.0 | 数据处理（Excel 读取） |
| torch | >=1.12.0 | 深度学习框架 |
| scikit-learn | >=1.0.0 | 传统机器学习（随机森林） |
| matplotlib | >=3.4.0 | 数据可视化 |
| seaborn | >=0.11.0 | 高级可视化 |
| tqdm | >=4.62.0 | 进度条 |
| openpyxl | >=3.0.0 | Excel 文件读取 |
| joblib | >=1.0.0 | 模型保存/加载 |

### PyTorch 安装说明

**重要**：PyTorch 需要根据你的 CUDA 版本选择：

1. **CPU 版本**（Windows 集成显卡）：
   ```bash
   pip install torch torchvision torchaudio
   ```

2. **CUDA 11.8**（大多数 Ubuntu 系统）：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **CUDA 12.1**（较新的系统）：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

**检查 CUDA 版本**：
```bash
# 方法1: 查看驱动信息
nvidia-smi

# 方法2: 查看 CUDA 编译器版本
nvcc --version
```

---

## 验证安装

运行以下命令验证所有依赖是否正确安装：

```bash
python -c "
import numpy as np
import scipy
import pandas as pd
import torch
import sklearn
import matplotlib
import seaborn
import tqdm
import openpyxl
import joblib

print('✓ 所有依赖安装成功！')
print(f'  NumPy: {np.__version__}')
print(f'  SciPy: {scipy.__version__}')
print(f'  Pandas: {pd.__version__}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## 常见问题

### 问题1: PyTorch 安装失败

**解决**：
- 检查 Python 版本（需要 3.8+）
- 检查 pip 版本：`pip install --upgrade pip`
- 使用官方源：`pip install torch --index-url https://download.pytorch.org/whl/cu118`

### 问题2: openpyxl 安装失败

**解决**：
```bash
pip install --upgrade pip setuptools wheel
pip install openpyxl
```

### 问题3: CUDA 不可用

**检查**：
1. NVIDIA 驱动是否安装：`nvidia-smi`
2. CUDA 是否安装：`nvcc --version`
3. PyTorch CUDA 版本是否匹配：`python -c "import torch; print(torch.version.cuda)"`

**解决**：
- 重新安装匹配的 PyTorch 版本
- 参考：https://pytorch.org/get-started/locally/

---

## 最小化安装（仅核心功能）

如果只需要运行深度学习训练，最小依赖：

```bash
pip install numpy torch tqdm openpyxl pandas
```

如果只需要运行传统机器学习实验：

```bash
pip install numpy scipy scikit-learn pandas openpyxl
```

---

## 开发环境（可选）

如果需要运行所有脚本（包括 legacy 代码）：

```bash
# 额外安装 TensorFlow (legacy 代码需要)
pip install tensorflow>=2.8.0

# Jupyter Notebook (用于数据分析)
pip install jupyter ipython
```

