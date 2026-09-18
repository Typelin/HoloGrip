@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
echo.
echo ==========================================
echo HoloGrip 第三次 MIDI + Raw 異常檢查
echo 自動讀取 Data\第三次MIDI 與 Data\第三次Raw 最新檔
echo ==========================================
echo.
if "%HOLOGRIP_SMOKE_TEST%"=="1" (
  python "%~dp0Program\third_midi_anomaly.py" --smoke
  exit /b %ERRORLEVEL%
)
python "%~dp0Program\third_midi_anomaly.py"
set "RC=%ERRORLEVEL%"
echo.
echo 報告已寫入 Data\對齊與異常報告
echo.
pause
exit /b %RC%
