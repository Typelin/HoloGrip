@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -c "import customtkinter, serial, joblib, sklearn, numpy" 2>nul
if errorlevel 1 (
  echo 缺少套件。請先執行：
  echo   python -m pip install -r requirements.txt
  pause
  exit /b 1
)
python run_live_hit_and_zone_0821_ZH_TW.py
if errorlevel 1 pause