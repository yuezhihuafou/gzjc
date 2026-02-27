# 根目录文件说明

## 📋 根目录文件清单

根目录只保留**必要的配置文件和核心文档**，其他文件已整理到相应目录。

### 核心文档（2个）

#### `README.md` ⭐
- **用途**: 项目主README，快速入门指南
- **位置**: 根目录（必须）
- **内容**: 快速开始、项目结构、文档索引

#### `PROJECT_STRUCTURE.md` ⭐
- **用途**: 项目整体结构说明
- **位置**: 根目录（核心文档）
- **内容**: 模块说明、使用路径、维护建议

#### `cursor.md`
- **用途**: 项目规范文档
- **位置**: 根目录（项目规范）
- **内容**: 当前阶段的设计规范

### 配置文件（4个）

#### `requirements.txt`
- **用途**: Python依赖列表
- **位置**: 根目录（标准位置）
- **说明**: pip安装依赖

#### `environment.yml`
- **用途**: Conda环境配置（CPU版本）
- **位置**: 根目录（标准位置）
- **说明**: `conda env create -f environment.yml`

#### `environment_gpu.yml`
- **用途**: Conda环境配置（GPU版本）
- **位置**: 根目录（标准位置）
- **说明**: `conda env create -f environment_gpu.yml`

### 环境管理脚本（4个）

#### `setup_conda_env.bat` / `setup_conda_env.sh`
- **用途**: 创建conda环境
- **位置**: 根目录（便于使用）
- **说明**: Windows/Linux环境创建脚本

#### `activate_env.bat` / `activate_env.sh`
- **用途**: 激活conda环境
- **位置**: 根目录（便于使用）
- **说明**: Windows/Linux环境激活脚本

## 📁 目录结构

```
guzhangjiance/
├── README.md                    # 主README ⭐
├── PROJECT_STRUCTURE.md         # 项目结构 ⭐
├── cursor.md                    # 项目规范
├── requirements.txt             # Python依赖
├── environment.yml              # Conda环境（CPU）
├── environment_gpu.yml          # Conda环境（GPU）
├── setup_conda_env.bat/sh       # 环境创建脚本
├── activate_env.bat/sh          # 环境激活脚本
│
├── core/                        # 核心模块
├── dl/                          # 深度学习模块
├── experiments/                 # 实验脚本
├── tools/                       # 工具脚本
├── legacy/                      # 历史代码
├── datasets/                    # 数据集
└── docs/                        # 所有文档
```

## 📚 文档位置

**所有详细文档都在 `docs/` 目录下**：

- `INSTALL.md` - 安装指南
- `CONDA_ENV_GUIDE.md` - Conda环境管理
- `README_SOUND_ANALYSIS.md` - 声音分析
- `README_experiment.md` - 实验说明
- `README_loader.md` - 数据加载器
- `CUDA_WINDOWS_GUIDE.md` - Windows CUDA指南
- `PHASE_1_SUMMARY.md` - Phase 1总结
- 其他技术文档...

## 🎯 整理原则

### 根目录保留的文件
1. ✅ **README.md** - 项目入口文档（必须）
2. ✅ **PROJECT_STRUCTURE.md** - 项目结构说明（核心）
3. ✅ **配置文件** - requirements.txt, environment.yml等
4. ✅ **环境脚本** - setup/activate脚本（便于使用）
5. ✅ **项目规范** - cursor.md（项目规范）

### 移到docs目录的文件
- ❌ 安装指南 → `docs/INSTALL.md`
- ❌ Conda指南 → `docs/CONDA_ENV_GUIDE.md`
- ❌ 技术文档 → `docs/*.md`
- ❌ PDF文档 → `docs/*.pdf`

### 移到tools目录的文件
- ❌ 工具脚本 → `tools/*.py`
- ❌ 可视化脚本 → `tools/visualize_sound_data.py`

## 📝 使用建议

### 新用户
1. 阅读 `README.md` 快速了解项目
2. 查看 `PROJECT_STRUCTURE.md` 了解结构
3. 参考 `docs/INSTALL.md` 安装环境

### 开发者
1. 查看 `cursor.md` 了解项目规范
2. 参考 `docs/` 目录下的详细文档
3. 查看 `PROJECT_STRUCTURE.md` 了解模块

### 维护者
1. 保持根目录整洁
2. 新文档放在 `docs/` 目录
3. 工具脚本放在 `tools/` 目录

---

**更新日期**: 2026-01-15  
**整理状态**: ✅ 已完成
