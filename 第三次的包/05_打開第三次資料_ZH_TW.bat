@echo off
chcp 65001 >nul
if "%HOLOGRIP_SMOKE_TEST%"=="1" exit /b 0
start "" "%~dp0Data"
