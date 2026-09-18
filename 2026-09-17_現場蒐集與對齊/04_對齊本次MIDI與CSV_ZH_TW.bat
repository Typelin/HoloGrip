@echo off
@chcp 65001 >nul
@setlocal
@set "PYTHONUTF8=1"
@if "%HOLOGRIP_SMOKE_TEST%"=="1" (
  @python "%~dp0Program\smoke_test_entrypoints.py" alignment
  @exit /b %ERRORLEVEL%
)
@start "" pythonw "%~dp0Program\gui_launcher.pyw" "%~dp0Program\midi_csv_alignment_gui_0917_ZH_TW.py"
@exit /b 0

