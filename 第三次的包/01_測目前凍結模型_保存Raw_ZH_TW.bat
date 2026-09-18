@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
if "%HOLOGRIP_SMOKE_TEST%"=="1" (
  python -m py_compile "%~dp0Program\field_v2_timing_robust_zero_shot.py"
  exit /b %ERRORLEVEL%
)
start "" pythonw "%~dp0Program\gui_launcher.pyw" "%~dp0Program\field_v2_timing_robust_zero_shot.py"
exit /b 0
