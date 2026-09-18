@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0Apps\Song_Collection_COM"
call analyze_round3_validation_0914_ZH_TW.bat %*
endlocal
