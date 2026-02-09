#!/bin/bash
# ============================================================================
# 机械故障诊断系统 - Conda环境创建脚本 (Linux/Ubuntu)
# ============================================================================
# 使用方法: bash setup_conda_env.sh [cpu|gpu]
# 默认创建CPU版本环境
# ============================================================================

set -e

echo "============================================================================"
echo "机械故障诊断系统 - Conda环境创建"
echo "============================================================================"

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "[错误] 未找到conda命令，请先安装Anaconda或Miniconda"
    echo "下载地址: https://www.anaconda.com/products/distribution"
    exit 1
fi

# 获取环境类型参数
ENV_TYPE=${1:-cpu}

echo ""
echo "环境类型: $ENV_TYPE"
echo ""

# 根据类型选择配置文件
if [ "$ENV_TYPE" = "gpu" ]; then
    ENV_FILE="environment_gpu.yml"
    ENV_NAME="guzhangjiance-gpu"
    echo "[提示] 将创建GPU版本环境（需要CUDA支持）"
else
    ENV_FILE="environment.yml"
    ENV_NAME="guzhangjiance"
    echo "[提示] 将创建CPU版本环境"
fi

# 检查配置文件是否存在
if [ ! -f "$ENV_FILE" ]; then
    echo "[错误] 配置文件不存在: $ENV_FILE"
    exit 1
fi

echo ""
echo "配置文件: $ENV_FILE"
echo "环境名称: $ENV_NAME"
echo ""

# 检查环境是否已存在
if conda env list | grep -q "^$ENV_NAME "; then
    echo "[警告] 环境 $ENV_NAME 已存在"
    read -p "是否删除并重新创建？(y/n): " OVERWRITE
    if [ "$OVERWRITE" = "y" ] || [ "$OVERWRITE" = "Y" ]; then
        echo ""
        echo "正在删除旧环境..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "已取消"
        exit 0
    fi
fi

echo ""
echo "============================================================================"
echo "开始创建conda环境..."
echo "============================================================================"
echo ""

# 创建环境
conda env create -f "$ENV_FILE"

if [ $? -ne 0 ]; then
    echo ""
    echo "[错误] 环境创建失败"
    exit 1
fi

echo ""
echo "============================================================================"
echo "环境创建成功！"
echo "============================================================================"
echo ""
echo "环境名称: $ENV_NAME"
echo ""
echo "激活环境:"
echo "  conda activate $ENV_NAME"
echo ""
echo "或者使用项目提供的激活脚本:"
echo "  source activate_env.sh"
echo ""
echo "验证安装:"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}')\""
echo "  python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\""
echo ""
