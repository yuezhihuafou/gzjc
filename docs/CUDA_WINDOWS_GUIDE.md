# Windows上使用CUDA/GPU指南

## 📋 概述

**重要说明**: CUDA不仅限于Ubuntu/Linux，Windows也可以使用CUDA进行GPU训练！

本项目支持在Windows上使用NVIDIA GPU进行深度学习训练。

## ✅ Windows GPU支持

### 系统要求

1. **NVIDIA GPU**
   - 支持CUDA的NVIDIA显卡（如GTX/RTX系列）
   - 检查方法：`nvidia-smi`（需要安装NVIDIA驱动）

2. **NVIDIA驱动**
   - 安装最新版本的NVIDIA驱动
   - 下载地址：https://www.nvidia.com/Download/index.aspx
   - 检查方法：`nvidia-smi`

3. **PyTorch CUDA版本**
   - PyTorch会自动安装CUDA运行时
   - **不需要单独安装CUDA Toolkit**（PyTorch已包含）

## 🚀 快速开始（Windows GPU）

### 步骤1: 检查GPU

```bash
# 打开命令提示符或PowerShell
nvidia-smi
```

**预期输出**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 5xx.xx       Driver Version: 5xx.xx       CUDA Version: 12.x  |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:01:00.0 On |                  N/A |
...
```

如果看到GPU信息，说明驱动已安装，可以使用GPU。

### 步骤2: 创建GPU环境

```bash
# 在项目根目录运行
setup_conda_env.bat gpu
```

### 步骤3: 激活环境

```bash
activate_env.bat gpu
```

### 步骤4: 验证CUDA

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**预期输出**:
```
PyTorch: 2.x.x+cu118
CUDA available: True
GPU: NVIDIA GeForce RTX 4070
```

## 🔍 为什么通常说CUDA是Ubuntu的？

### 历史原因

1. **服务器环境**: 大多数深度学习服务器运行Linux/Ubuntu
2. **性能优势**: Linux上GPU性能通常略好（驱动开销更小）
3. **兼容性**: Linux上CUDA支持更成熟
4. **文档习惯**: 很多教程默认以Linux为例

### 实际情况

**Windows也完全支持CUDA！**

- ✅ Windows 10/11 完全支持CUDA
- ✅ PyTorch在Windows上支持CUDA
- ✅ 大多数NVIDIA GPU在Windows上可用
- ✅ 性能差异很小（通常<5%）

## 📊 Windows vs Ubuntu GPU对比

| 特性 | Windows | Ubuntu |
|------|---------|--------|
| CUDA支持 | ✅ 完全支持 | ✅ 完全支持 |
| 驱动安装 | 简单（自动更新） | 需要手动安装 |
| 性能 | 略低（~2-5%） | 略高 |
| 易用性 | 更友好 | 需要命令行 |
| 开发体验 | 更好（IDE支持） | 需要配置 |
| 服务器部署 | 不常见 | 常见 |

## ⚙️ 配置说明

### Windows GPU环境配置

使用 `environment_gpu.yml` 创建GPU环境：

```yaml
name: guzhangjiance-gpu
dependencies:
  - pip:
    - torch>=1.12.0 --index-url https://download.pytorch.org/whl/cu118
    - torchvision>=0.13.0 --index-url https://download.pytorch.org/whl/cu118
    - torchaudio>=0.12.0 --index-url https://download.pytorch.org/whl/cu118
```

### CUDA版本选择

PyTorch支持多个CUDA版本：

- **CUDA 11.8** (推荐，兼容性好)
  ```bash
  --index-url https://download.pytorch.org/whl/cu118
  ```

- **CUDA 12.1** (较新，需要较新的驱动)
  ```bash
  --index-url https://download.pytorch.org/whl/cu121
  ```

**如何选择**:
1. 检查驱动支持的CUDA版本：`nvidia-smi`
2. 通常选择CUDA 11.8（兼容性最好）
3. 如果驱动很新，可以尝试CUDA 12.1

## 🔧 故障排查

### 问题1: nvidia-smi命令不存在

**原因**: 未安装NVIDIA驱动

**解决**:
1. 访问 https://www.nvidia.com/Download/index.aspx
2. 选择你的GPU型号
3. 下载并安装最新驱动
4. 重启电脑

### 问题2: CUDA available: False

**可能原因**:
1. PyTorch安装的是CPU版本
2. GPU驱动版本太旧
3. PyTorch CUDA版本与驱动不匹配

**解决**:
```bash
# 1. 确认安装了GPU版本环境
conda activate guzhangjiance-gpu

# 2. 检查PyTorch版本
python -c "import torch; print(torch.__version__)"
# 应该看到类似: 2.x.x+cu118

# 3. 如果看到CPU版本，重新安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 检查驱动版本
nvidia-smi
```

### 问题3: 性能不如预期

**可能原因**:
1. Windows电源模式设置为"省电"
2. GPU被其他程序占用
3. 显存不足

**解决**:
1. 设置电源模式为"高性能"
2. 关闭其他使用GPU的程序
3. 减小batch size

## 💡 最佳实践

### Windows GPU训练建议

1. **使用GPU环境**
   ```bash
   setup_conda_env.bat gpu
   activate_env.bat gpu
   ```

2. **验证GPU可用**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **监控GPU使用**
   ```bash
   # 在另一个终端窗口
   nvidia-smi -l 1  # 每秒刷新一次
   ```

4. **设置合适的batch size**
   - 根据GPU显存调整
   - RTX 4070通常可以设置batch_size=32或更大

## 📚 相关文档

- `CONDA_ENV_GUIDE.md` - Conda环境管理指南
- `INSTALL.md` - 安装指南
- `environment_gpu.yml` - GPU环境配置

## 🎯 总结

- ✅ **Windows完全支持CUDA/GPU训练**
- ✅ 使用 `setup_conda_env.bat gpu` 创建GPU环境
- ✅ 不需要单独安装CUDA Toolkit（PyTorch已包含）
- ✅ 只需要NVIDIA驱动即可
- ✅ 性能与Ubuntu差异很小

**推荐**: 如果你有NVIDIA GPU，在Windows上也可以使用GPU版本进行训练！

---

**创建日期**: 2026-01-15  
**版本**: 1.0
