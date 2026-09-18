@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0Apps\Song_Collection_COM"
python check_round3_readiness_0914_ZH_TW.py
set RC=%ERRORLEVEL%
pause
exit /b %RC%
