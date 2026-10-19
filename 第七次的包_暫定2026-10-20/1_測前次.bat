@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
set "OPENBLAS_NUM_THREADS=1"
cd /d "%~dp0"
set "HG_PY=%LocalAppData%\Programs\Python\Python312\python.exe"
if exist "%HG_PY%" (
  "%HG_PY%" "Program\seventh_entry.py" live %*
) else (
  py -3.12 "Program\seventh_entry.py" live %*
)
set "HG_RC=%ERRORLEVEL%"
if not "%HG_RC%"=="0" echo Live console stopped with an error. Inspect the message above.
if not "%HOLOGRIP_NO_PAUSE%"=="1" pause
exit /b %HG_RC%
