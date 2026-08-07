# HoloGrip 資料夾整理完成紀錄

完成日期：2026-08-07

## 已完成的結構

| 類別 | 目前位置 |
|---|---|
| COM、UDP 程式 | `Apps/` |
| COM 與舊 UDP 韌體 | `Firmware/` |
| 原始手套 CSV | `Data/Raw/` |
| MIDI 對齊、分流與標記 | `Data/Derived/` |
| 流音 MIDI、影片、音訊 | `Data/External/FlowAudio_20260805/` |
| 目前研究文件 | `Docs/Current/` |
| 實驗記錄與歷史文件 | `Docs/Logs/`、`Docs/Archive/` |
| 目前與歷史簡報 | `Presentations/Current/`、`Presentations/Archive/` |
| 示範影片 | `Media/Demo/` |
| 延遲測試與舊程式 | `Tests/Latency/`、`Archive/Legacy_Code/` |

## 目前最常用的入口

1. COM 收集：`Apps/Song_Collection_COM/run_song_collection_com.bat`
2. MIDI/CSV 手別審核：`Apps/Song_Collection_COM/run_hand_label_triage_0807.bat`
3. COM 韌體：`Firmware/COM/Gloves_Firmware_INO_COM.ino`
4. MIDI/CSV 進度：`Docs/Current/HoloGrip_MIDI_CSV_手別標記進度與整理計畫_0807_ZH_TW.md`

## 搬遷保留核對

| 資料 | 搬遷前 | 搬遷後 |
|---|---:|---:|
| COM 歌曲原始 CSV | 2 檔，8.68 MB | 2 檔，8.68 MB |
| 舊歌曲 CSV | 2 檔，0.20 MB | 2 檔，0.20 MB |
| UDP CSV | 1 檔 | 1 檔 |
| 推導標記資料 | 12 檔，30.71 MB | 12 檔，30.71 MB |
| 流音 MIDI、影片、音訊 | 3 檔，915.95 MB | 3 檔，915.95 MB |
| 示範影片 | 8 檔，602.24 MB | 8 檔，602.24 MB |

## 版本與驗證

- 搬遷前 Git 基線：`b297e3c`，訊息為 `Snapshot before HoloGrip project reorganization`。
- 所有活躍 Python 路徑已更新為新結構。
- 實體手套尚未在新路徑下重新收集；下次接上 COM4/COM5 時，先進行一次短錄製確認即可。
- 根目錄原有的單一 Python 快取已移至 `Archive/Generated_Cache_0807/`，未刪除任何研究資料。
