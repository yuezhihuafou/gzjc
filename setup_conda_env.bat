@echo off
REM ============================================================================
REM 机械故障诊断系统 - Conda环境创建脚本 (Windows)
REM ============================================================================
REM 使用方法: setup_conda_env.bat [cpu|gpu]
REM 默认创建CPU版本环境
REM ============================================================================

setlocal enabledelayedexpansion

echo ============================================================================
echo 机械故障诊断系统 - Conda环境创建
echo ============================================================================

REM 检查conda是否安装
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到conda命令，请先安装Anaconda或Miniconda
    echo 下载地址: https://www.anaconda.com/products/distribution
    pause
    exit /b 1
)

REM 获取环境类型参数
set ENV_TYPE=%1
if "%ENV_TYPE%"=="" set ENV_TYPE=cpu

echo.
echo 环境类型: %ENV_TYPE%
echo.

REM 根据类型选择配置文件
if /i "%ENV_TYPE%"=="gpu" (
    set ENV_FILE=environment_gpu.yml
    set ENV_NAME=guzhangjiance-gpu
    echo [提示] 将创建GPU版本环境（需要CUDA支持）
) else (
    set ENV_FILE=environment.yml
    set ENV_NAME=guzhangjiance
    echo [提示] 将创建CPU版本环境
)

REM 检查配置文件是否存在
if not exist "%ENV_FILE%" (
    echo [错误] 配置文件不存在: %ENV_FILE%
    pause
    exit /b 1
)

echo.
echo 配置文件: %ENV_FILE%
echo 环境名称: %ENV_NAME%
echo.

REM 检查环境是否已存在
conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if %errorlevel% equ 0 (
    echo [警告] 环境 %ENV_NAME% 已存在
    set /p OVERWRITE="是否删除并重新创建？(y/n): "
    if /i "!OVERWRITE!"=="y" (
        echo.
        echo 正在删除旧环境...
        conda env remove -n %ENV_NAME% -y
        if %errorlevel% neq 0 (
            echo [错误] 删除环境失败
            pause
            exit /b 1
        )
    ) else (
        echo 已取消
        pause
        exit /b 0
    )
)

echo.
echo ============================================================================
echo 开始创建conda环境...
echo ============================================================================
echo.

REM 创建环境
conda env create -f %ENV_FILE%

if %errorlevel% neq 0 (
    echo.
    echo [错误] 环境创建失败
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo 环境创建成功！
echo ============================================================================
echo.
echo 环境名称: %ENV_NAME%
echo.
echo 激活环境:
echo   conda activate %ENV_NAME%
echo.
echo 或者使用项目提供的激活脚本:
echo   activate_env.bat
echo.
echo 验证安装:
echo   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo.

pause
