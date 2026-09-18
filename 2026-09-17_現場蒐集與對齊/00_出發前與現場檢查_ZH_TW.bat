@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
python "%~dp0Program\field_check_0917.py"
set "RC=%ERRORLEVEL%"
if "%HOLOGRIP_SMOKE_TEST%"=="1" exit /b %RC%
pause
exit /b %RC%
endlocal
