@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if "%~1"=="" (
  echo 請把 Round3 場次資料夾拖到這個 BAT 上。
  echo 如果已有電子鼓 MIDI，也可以把 MIDI 當第二個參數。
  pause
  exit /b 1
)
if "%~2"=="" (
  python analyze_round3_validation_0914_ZH_TW.py "%~1"
) else (
  python analyze_round3_validation_0914_ZH_TW.py "%~1" --midi "%~2"
)
set RC=%ERRORLEVEL%
if not "%RC%"=="0" pause
exit /b %RC%
