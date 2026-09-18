@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
if "%HOLOGRIP_SMOKE_TEST%"=="1" (
  python "%~dp0Program\smoke_test_entrypoints.py" model
  exit /b %ERRORLEVEL%
)
python "%~dp0Program\run_live_hit_and_zone_0821_ZH_TW.py"
if errorlevel 1 pause
endlocal
