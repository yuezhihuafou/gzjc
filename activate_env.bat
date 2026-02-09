@echo off
REM ============================================================================
REM 激活conda环境脚本 (Windows)
REM ============================================================================
REM 使用方法: activate_env.bat [cpu|gpu]
REM 默认激活CPU版本环境
REM ============================================================================

setlocal

set ENV_TYPE=%1
if "%ENV_TYPE%"=="" set ENV_TYPE=cpu

if /i "%ENV_TYPE%"=="gpu" (
    set ENV_NAME=guzhangjiance-gpu
) else (
    set ENV_NAME=guzhangjiance
)

echo 正在激活conda环境: %ENV_NAME%
echo.

REM 初始化conda
call conda init cmd.exe >nul 2>&1

REM 激活环境
call conda activate %ENV_NAME%

if %errorlevel% neq 0 (
    echo [错误] 环境 %ENV_NAME% 不存在
    echo.
    echo 请先创建环境:
    echo   setup_conda_env.bat %ENV_TYPE%
    exit /b 1
)

echo [成功] 环境已激活: %ENV_NAME%
echo.
echo 当前Python版本:
python --version
echo.
echo 当前环境路径:
where python
echo.

REM 启动新的命令行窗口（可选）
REM start cmd /k "conda activate %ENV_NAME% && cd /d %~dp0"
