@echo off
setlocal
set "SCRIPT_DIR=%~dp0"

python "%SCRIPT_DIR%video_hand_review_seed.py"
if errorlevel 1 (
  echo.
  echo Unable to create the video review queue.
  pause
  exit /b 1
)

start "" "%SCRIPT_DIR%video_hand_review_0807.html"
