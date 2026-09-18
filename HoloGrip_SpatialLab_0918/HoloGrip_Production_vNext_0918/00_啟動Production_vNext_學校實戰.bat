@echo off
chcp 65001 >nul
cd /d "%~dp0"
taskkill /F /IM serial-monitor.exe >nul 2>nul
start "" /D "%~dp0" pythonw.exe "%~dp0school_production_live.py"
exit /b 0
