@echo off
chcp 65001 >nul
echo ============================================================
echo JSON retry: only failed .f from bad_files_mc_to_api.txt
echo Update token in auth_example.json before running.
echo ============================================================

python tools/sound_api/convert_mc_to_api_json.py ^
  --mc_dir "D:\guzhangjiance\datasets\xjtu\output_xjtu_mc\xjtu" ^
  --output_root "D:\guzhangjiance\datasets\sound_api" ^
  --auth-file auth_example.json ^
  --retry-failed ^
  --workers 32 ^
  --max-inflight 128 ^
  --qps 15 ^
  --resume ^
  --print-every 200 ^
  --print-first-failures 5

pause
