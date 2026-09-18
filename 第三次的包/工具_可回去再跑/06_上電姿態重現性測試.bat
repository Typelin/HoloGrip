@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
if "%HOLOGRIP_SMOKE_TEST%"=="1" (
  python -m py_compile "%~dp0..\Program\power_cycle_orientation_test.py"
  exit /b %ERRORLEVEL%
)
start "" pythonw "%~dp0..\Program\power_cycle_orientation_test.py"
exit /b 0
