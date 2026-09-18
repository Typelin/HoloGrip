@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
python "%~dp0..\Program\run_live_hit_and_zone_0821_ZH_TW.py"
if errorlevel 1 pause
endlocal
