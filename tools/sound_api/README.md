# 声音转能量密度曲线API工具

本目录包含将音频文件转换为能量和密度曲线的API调用工具。

## 📁 目录结构

```
sound_api/
├── convert_sound_api.py          # 主脚本：API转换工具
├── README.md                      # 本文件
└── docs/                          # 文档目录
    ├── README_声音API使用指南.md   # 详细使用指南
    └── README_sound_api.md        # 问题排查指南
```

## 🚀 快速开始

### 安装依赖

```bash
pip install requests pandas tqdm openpyxl
```

### 使用方法

```bash
# 运行转换工具
python tools/sound_api/convert_sound_api.py
```

## 📖 文档

详细的使用说明请查看：
- [使用指南](./docs/README_声音API使用指南.md) - 完整的使用说明和配置方法
- [问题排查](./docs/README_sound_api.md) - 常见问题和解决方案

## ✨ 功能特点

1. **支持cURL命令导入** - 从Apipost文档直接复制cURL命令，自动解析配置
2. **单文件和批量转换** - 支持单个文件测试和批量处理
3. **兼容现有格式** - 输出格式兼容 `load_sound.py` 的数据格式
4. **多种输出格式** - 支持JSON和XLSX格式输出

## 🔧 API配置

### 方式1：从cURL命令导入（推荐）

1. 打开Apipost文档页面
2. 找到"生成代码"按钮，选择cURL格式
3. 复制完整的cURL命令
4. 运行脚本时选择方式1，粘贴命令

### 方式2：手动输入

手动输入API URL和Token（功能有限）

## 📝 输出格式

转换后的数据会保存为：
- **JSON格式**：`{filename}.json` - 包含frequency、volume、density数组
- **XLSX格式**：`{filename}.xlsx` - 兼容`load_sound.py`的格式

## 🔗 相关工具

- `tools/load_sound.py` - 加载转换后的声音数据
- `core/features.py::SoundMetricsExtractor` - 使用声音数据进行特征提取
