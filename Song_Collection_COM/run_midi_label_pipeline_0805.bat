@echo off
cd /d "%~dp0.."
python "Song_Collection_COM\midi_label_pipeline.py" ^
  --raw-csv "CSV_Data\Song_Collection_COM\S20260805_P01_song01_raw_100hz_20260805_161628.csv" ^
  --midi "流音給\Drum Midi_110BPM (0805).mid" ^
  --out-dir "Derived_Data\Song_Collection_COM\S20260805_P01_song01" ^
  --bpm 110 ^
  --pre-ms 90 ^
  --post-ms 90
pause
