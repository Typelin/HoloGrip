@echo off
setlocal
chcp 65001 >nul
title HoloGrip 雙手套 USB 預檢

echo ==========================================
echo HoloGrip 雙手套 USB 預檢
echo ==========================================
echo.
echo [1/3] 關閉可能殘留、會鎖住 COM 的 Arduino Serial Monitor...
taskkill /IM serial-monitor.exe /F >nul 2>&1

echo [2/3] 掃描 XIAO ESP32-C6 COM...
python "%~dp0usb_dual_glove_check.py"

echo.
echo [3/3] 完成。
echo.
pause
