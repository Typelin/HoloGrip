@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
if "%HOLOGRIP_SMOKE_TEST%"=="1" (
  python "%~dp0Program\no_drum_diagnostic_0918_ZH_TW.py" --smoke-test
  exit /b %ERRORLEVEL%
)
start "" pythonw "%~dp0Program\gui_launcher.pyw" "%~dp0Program\no_drum_diagnostic_0918_ZH_TW.py"
endlocal
