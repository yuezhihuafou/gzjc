#!/bin/bash
# ============================================================================
# 激活conda环境脚本 (Linux/Ubuntu)
# ============================================================================
# 使用方法: source activate_env.sh [cpu|gpu]
# 默认激活CPU版本环境
# ============================================================================

ENV_TYPE=${1:-cpu}

if [ "$ENV_TYPE" = "gpu" ]; then
    ENV_NAME="guzhangjiance-gpu"
else
    ENV_NAME="guzhangjiance"
fi

echo "正在激活conda环境: $ENV_NAME"
echo ""

# 初始化conda（如果还没有）
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    # 尝试初始化conda
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
fi

# 激活环境
conda activate "$ENV_NAME"

if [ $? -ne 0 ]; then
    echo "[错误] 环境 $ENV_NAME 不存在"
    echo ""
    echo "请先创建环境:"
    echo "  bash setup_conda_env.sh $ENV_TYPE"
    return 1
fi

echo "[成功] 环境已激活: $ENV_NAME"
echo ""
echo "当前Python版本:"
python --version
echo ""
echo "当前环境路径:"
which python
echo ""
