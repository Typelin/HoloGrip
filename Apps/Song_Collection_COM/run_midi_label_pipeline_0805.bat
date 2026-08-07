@echo off
setlocal
set "APP_DIR=%~dp0"
for %%I in ("%APP_DIR%..\..") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"
python "%APP_DIR%midi_label_pipeline.py" ^
  --raw-csv "Data\Raw\Song_Collection_COM\S20260805_P01_song01_raw_100hz_20260805_161628.csv" ^
  --midi "Data\External\FlowAudio_20260805\Drum Midi_110BPM (0805).mid" ^
  --out-dir "Data\Derived\Song_Collection_COM\S20260805_P01_song01" ^
  --bpm 110 ^
  --pre-ms 90 ^
  --post-ms 90
pause
