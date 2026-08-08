@echo off
setlocal
python "%~dp0analyze_drum_audio_0808.py"
if errorlevel 1 (
  echo.
  echo WAV 分析失敗，請確認 Python、numpy、scipy 與 ffmpeg 已安裝。
)
pause
