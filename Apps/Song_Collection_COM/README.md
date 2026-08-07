# HoloGrip SONG Collection (COM)

This is the dedicated wired collection entry for the music-session workflow.

Launch `run_song_collection_com.bat`.

New raw CSV files are written to `Data/Raw/Song_Collection_COM/`.

The separate `Apps/UDP_Collection/` entry remains for the legacy Wi-Fi/UDP workflow.

## MIDI label pipeline

Run `run_midi_label_pipeline_0805.bat` for the current session. It reads the
Raw CSV and the MIDI source, applies the confirmed 110 BPM override and the
0805 seven-zone mapping, then writes reviewable derived files under
`Data/Derived/Song_Collection_COM/`.

The source CSV and MIDI are never overwritten. The generated labels remain
`pending_manual_alignment_review` until the selected offset is checked against
the recording or a known sync point.

Run `run_hand_label_triage_0807.bat` to open the manual simultaneous-MIDI
review page without regenerating its inputs. Use
`rebuild_hand_label_triage_0807.bat` only after raw CSV, MIDI, alignment or
triage rules have changed; it rebuilds the reviewable derived files under
`Data/Derived/Song_Collection_COM/`.

## 三頁主工具

1. `HoloGrip_MIDI_CSV對齊檢查_0807_ZH_TW.html`：檢查 MIDI 時間、CSV 活動峰與固定採樣窗口。
2. `HoloGrip_同時MIDI影片手別標記_0807_ZH_TW.html`：播放同時 MIDI 的影片窗口，逐一標記左手、右手或略過。
3. `HoloGrip_資料集最終進度總覽_0807_ZH_TW.html`：載入統計與人工 CSV，查看目前可訓練事件、固定窗口與七鼓點分布。

操作順序：先在第 1 頁確認 MIDI -> CSV 的候選偏移與採樣窗口；接著在第 2 頁載入 `manual_simultaneous_groups.json` 與演奏影片，完成後下載人工手別 CSV；最後在第 3 頁依序載入 `hand_label_triage_summary.json`、`auto_accepted_single_events.csv`、`manual_simultaneous_groups.json` 與第 2 頁下載的人工手別 CSV，查看可用訓練資料量。

舊版單點影片複核工具已移到 `Archive/Legacy_Code/VideoReviewSinglePoint_0807/`，不列入目前主流程。
