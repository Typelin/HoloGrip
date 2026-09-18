@echo off
@chcp 65001 >nul
@setlocal
@set "PYTHONUTF8=1"
@if "%HOLOGRIP_SMOKE_TEST%"=="1" (
  @python "%~dp0Program\smoke_test_entrypoints.py" formal
  @exit /b %ERRORLEVEL%
)
@start "" pythonw "%~dp0Program\gui_launcher.pyw" "%~dp0Program\formal_validation_0917.py"
@exit /b 0

