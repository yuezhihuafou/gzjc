@echo off
chcp 65001 >nul
echo ========================================
echo 创建 IPEX 环境 (Python 3.10 + PyTorch)
echo ========================================
call conda env create -f environment_ipex.yml
if errorlevel 1 (
    echo 创建失败，请检查 conda 是否可用、environment_ipex.yml 是否存在。
    pause
    exit /b 1
)
echo.
echo 正在安装 Intel Extension for PyTorch (IPEX)...
conda run -n ipex pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
if errorlevel 1 (
    echo 使用 Intel 源仍失败，尝试仅 CPU 版 IPEX...
    conda run -n ipex pip install intel-extension-for-pytorch
)
echo.
echo 激活环境: conda activate ipex
echo 测试集验证(核显): python experiments/train.py --data_source sound_api_cache --task hi --cache_dir datasets/sound_api/cache_npz_quick --eval_only --device xpu
pause
