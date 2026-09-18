@echo off
chcp 65001 >nul
if "%HOLOGRIP_SMOKE_TEST%"=="1" (
  echo OPEN_DATA_ENTRY_OK
  exit /b 0
)
start "" "%~dp0Data"
