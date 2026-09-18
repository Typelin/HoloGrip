@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
echo.
echo ==========================================
echo HoloGrip 第三次現場包｜雙手與模型預檢
echo ==========================================
echo.
if "%HOLOGRIP_SMOKE_TEST%"=="1" (
  python "%~dp0Program\third_preflight.py" --smoke
  exit /b %ERRORLEVEL%
)
python "%~dp0Program\third_preflight.py"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [PASS] 可以進入 01。
) else (
  echo [FAIL/WARN] 請先處理上面的問題，不要直接開始正式測試。
)
echo.
pause
exit /b %RC%
