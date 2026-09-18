@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
python -c "import customtkinter, serial, joblib, sklearn, numpy" >nul 2>&1
if errorlevel 1 (
  echo 缺少套件。請先執行：
  echo   python -m pip install -r requirements.txt
  pause
  exit /b 1
)
if /I "%~1"=="--check-launcher" exit /b 0
python run_round3_validation_0914_ZH_TW.py
set RC=%ERRORLEVEL%
if not "%RC%"=="0" pause
exit /b %RC%
