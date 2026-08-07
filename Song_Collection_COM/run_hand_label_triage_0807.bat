@echo off
setlocal
set "SCRIPT_DIR=%~dp0"

python "%SCRIPT_DIR%hand_label_triage_0807.py"
if errorlevel 1 (
  echo.
  echo Unable to create hand-label triage outputs.
  pause
  exit /b 1
)

start "" "%SCRIPT_DIR%simultaneous_midi_hand_review_0807.html"
