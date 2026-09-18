@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0Apps\Song_Collection_COM"
python -c "import customtkinter, serial" 2>nul
if errorlevel 1 (
  echo 缺少資料蒐集套件，請執行：
  echo   python -m pip install -r requirements.txt
  pause
  exit /b 1
)
python song_collection_com.py
if errorlevel 1 pause
endlocal
