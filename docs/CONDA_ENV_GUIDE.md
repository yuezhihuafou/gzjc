# Conda环境管理指南

## 📋 概述

本项目使用conda环境管理依赖，环境存储在conda的默认位置（不在项目目录下），便于长期使用和维护。

## 🚀 快速开始

### Windows系统

#### 1. 创建环境（首次使用）

```bash
# CPU版本（默认，适合没有NVIDIA GPU的Windows）
setup_conda_env.bat

# GPU版本（如果有NVIDIA GPU和CUDA支持）
setup_conda_env.bat gpu
```

**注意**: Windows也可以使用CUDA/GPU版本，前提是：
- 有NVIDIA GPU
- 安装了NVIDIA驱动
- 安装了CUDA Toolkit（可选，PyTorch会自带CUDA运行时）

#### 2. 激活环境

```bash
# 方式1: 使用项目脚本（推荐）
activate_env.bat

# 方式2: 直接使用conda命令
conda activate guzhangjiance
```

#### 3. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Linux/Ubuntu系统

#### 1. 创建环境（首次使用）

```bash
# 给脚本执行权限
chmod +x setup_conda_env.sh activate_env.sh

# CPU版本
bash setup_conda_env.sh

# GPU版本（Ubuntu服务器）
bash setup_conda_env.sh gpu
```

#### 2. 激活环境

```bash
# 方式1: 使用项目脚本（推荐）
source activate_env.sh

# 方式2: 直接使用conda命令
conda activate guzhangjiance
```

## 📁 环境配置文件

### environment.yml（CPU版本）
- **用途**: Windows或CPU训练
- **PyTorch**: CPU版本
- **环境名**: `guzhangjiance`

### environment_gpu.yml（GPU版本）
- **用途**: Windows或Ubuntu GPU训练（需要NVIDIA GPU）
- **PyTorch**: CUDA版本（支持11.8或12.1）
- **环境名**: `guzhangjiance-gpu`
- **说明**: Windows和Ubuntu都可以使用，只要有NVIDIA GPU和驱动

## 🔧 环境管理命令

### 查看所有环境

```bash
conda env list
```

### 激活环境

```bash
# Windows
conda activate guzhangjiance

# Linux
source activate guzhangjiance
```

### 退出环境

```bash
conda deactivate
```

### 更新环境

```bash
# 激活环境后
conda env update -f environment.yml --prune
```

### 导出环境

```bash
# 导出当前环境配置
conda env export > environment_backup.yml
```

### 删除环境

```bash
conda env remove -n guzhangjiance
```

## 📦 安装额外包

### 在环境中安装新包

```bash
# 激活环境
conda activate guzhangjiance

# 使用conda安装
conda install package_name

# 或使用pip安装
pip install package_name
```

### 更新environment.yml

安装新包后，建议更新配置文件：

```bash
conda env export > environment.yml
```

## 🎯 使用场景

### 场景1: 日常开发（Windows）

```bash
# 1. 激活环境
activate_env.bat

# 2. 运行项目
python experiments/train.py
```

### 场景2: GPU训练（Windows或Ubuntu）

**Windows**（如果有NVIDIA GPU）:
```bash
# 1. 检查GPU（可选）
nvidia-smi

# 2. 创建GPU环境（如果还没有）
setup_conda_env.bat gpu

# 3. 激活GPU环境
activate_env.bat gpu

# 4. 验证CUDA
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# 5. 开始训练
python experiments/train.py
```

**Ubuntu**:
```bash
# 1. 创建GPU环境（如果还没有）
bash setup_conda_env.sh gpu

# 2. 激活GPU环境
source activate_env.sh gpu

# 3. 验证CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 4. 开始训练
python experiments/train.py
```

**注意**: Windows也可以使用GPU训练！只需要有NVIDIA GPU和驱动即可。详见 `docs/CUDA_WINDOWS_GUIDE.md`

### 场景3: 多环境切换

```bash
# 切换到CPU环境
conda activate guzhangjiance

# 切换到GPU环境
conda activate guzhangjiance-gpu
```

## ⚙️ 环境位置

Conda环境默认存储在：

- **Windows**: `C:\Users\<用户名>\anaconda3\envs\` 或 `C:\Users\<用户名>\miniconda3\envs\`
- **Linux**: `~/anaconda3/envs/` 或 `~/miniconda3/envs/`

**优势**:
- ✅ 环境不依赖项目目录
- ✅ 可以多个项目共享同一环境
- ✅ 便于长期维护
- ✅ 不会污染项目目录

## 🔍 故障排查

### 问题1: conda命令未找到

**Windows**:
```bash
# 添加到PATH或使用Anaconda Prompt
# 或运行: conda init cmd.exe
```

**Linux**:
```bash
# 初始化conda
source ~/anaconda3/etc/profile.d/conda.sh
# 或添加到 ~/.bashrc
```

### 问题2: 环境创建失败

**解决**:
1. 检查网络连接（需要下载包）
2. 使用国内镜像源：
   ```bash
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
   ```
3. 清理conda缓存：
   ```bash
   conda clean --all
   ```

### 问题3: PyTorch CUDA不可用

**检查**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi  # 检查GPU驱动
```

**解决**:
1. 确认安装了GPU版本环境：`bash setup_conda_env.sh gpu`
2. 检查CUDA版本匹配
3. 重新安装匹配的PyTorch版本

### 问题4: 环境激活后找不到包

**解决**:
```bash
# 确认环境已激活
conda info --envs

# 重新安装包
conda install package_name
# 或
pip install package_name
```

## 📝 最佳实践

### 1. 使用项目脚本

优先使用项目提供的脚本：
- `setup_conda_env.bat/sh` - 创建环境
- `activate_env.bat/sh` - 激活环境

### 2. 定期更新环境

```bash
# 更新所有包
conda update --all

# 或更新特定包
conda update numpy pandas
```

### 3. 备份环境配置

```bash
# 导出当前环境
conda env export > environment_backup.yml

# 提交到版本控制（可选）
git add environment_backup.yml
```

### 4. 使用requirements.txt作为补充

对于pip-only的包，可以继续使用requirements.txt：
```bash
conda activate guzhangjiance
pip install -r requirements.txt
```

## 🔄 迁移环境

### 导出环境

```bash
conda env export > environment_export.yml
```

### 在新机器上创建

```bash
conda env create -f environment_export.yml
```

## 📚 相关文档

- `INSTALL.md` - 详细安装指南
- `docs/CUDA_WINDOWS_GUIDE.md` - Windows上使用CUDA/GPU详细指南 ⭐
- `requirements.txt` - pip依赖列表
- `environment.yml` - conda环境配置（CPU）
- `environment_gpu.yml` - conda环境配置（GPU）

## 💡 提示

1. **环境名称**: 项目使用 `guzhangjiance`（CPU）和 `guzhangjiance-gpu`（GPU）
2. **Python版本**: 固定为3.9，确保兼容性
3. **PyTorch版本**: 根据系统选择CPU或CUDA版本
4. **长期维护**: 环境存储在conda默认位置，不依赖项目目录

---

**创建日期**: 2026-01-15  
**版本**: 1.0
