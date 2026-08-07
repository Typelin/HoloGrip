@echo off
setlocal
set "SCRIPT_DIR=%~dp0"

python "%SCRIPT_DIR%hand_label_triage_0807.py"
if errorlevel 1 (
  echo.
  echo Unable to rebuild hand-label triage outputs.
  pause
  exit /b 1
)

start "" "%SCRIPT_DIR%HoloGrip_同時MIDI影片手別標記_0807_ZH_TW.html"
