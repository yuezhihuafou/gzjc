@echo off
REM ================================================================
REM 每日自动转换脚本 - XJTU 数据集 API 转换
REM ================================================================
REM 
REM 使用方法：
REM 1. 通过 Windows 任务计划程序设置每天自动运行此脚本
REM 2. 脚本会自动从上次停止的地方继续（--resume）
REM 3. 每天处理约 5000 个文件，直到配额用尽
REM 4. 预计 57 天完成全部 285,696 个文件
REM
REM ================================================================

cd /d D:\guzhangjiance

echo ==========================================
echo 开始每日自动转换
echo 时间: %date% %time%
echo ==========================================

REM 运行转换脚本
python tools\sound_api\convert_mc_to_api_json.py ^
  --mc_dir D:\guzhangjiance\datasets\xjtu\output_xjtu_mc\xjtu ^
  --output_root D:\guzhangjiance\datasets\sound_api ^
  --channel-mode horizontal ^
  --workers 4 ^
  --qps 10 ^
  --timeout 30 ^
  --retries 2 ^
  --retry-backoff 1.0 ^
  --resume ^
  --print-every 500 ^
  --print-first-failures 3

echo ==========================================
echo 转换完成
echo 时间: %date% %time%
echo ==========================================
echo.
echo 检查进度：查看输出文件夹中的文件数量
echo 配额用尽时脚本会自动停止，明天继续
echo.

pause
