@echo off
setlocal
set "PYTHONUTF8=1"
start "" pythonw "%~dp0launch_spatial_cockpit.pyw"
exit /b 0
