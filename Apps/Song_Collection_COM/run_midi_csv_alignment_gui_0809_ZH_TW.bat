@echo off
setlocal
set "APP_DIR=%~dp0"
for %%I in ("%APP_DIR%..\..") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"
python "%APP_DIR%midi_csv_alignment_gui_0809_ZH_TW.py"
if errorlevel 1 pause
