@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
if "%HOLOGRIP_SMOKE_TEST%"=="1" (
  python "%~dp0Program\live_3d_imu_space_0918_ZH_TW.py" --smoke
  exit /b %ERRORLEVEL%
)
start "" /b pythonw "%~dp0Program\gui_launcher.pyw" "%~dp0Program\live_3d_imu_space_0918_ZH_TW.py"
exit /b 0
