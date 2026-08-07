# HoloGrip

HoloGrip 是雙手 IMU 手套的空氣鼓研究專案。目前的主線是以 COM 有線方式收集雙手 100 Hz 原始 CSV，再以電子鼓本次演奏產生的 MIDI 建立 7 鼓點訓練標記。

## 先看這裡

| 需求 | 位置 |
|---|---|
| 目前 MIDI/CSV 手別標記進度 | `Docs/Current/HoloGrip_MIDI_CSV_手別標記進度與整理計畫_0807_ZH_TW.md` |
| COM 收集程式 | `Apps/Song_Collection_COM/run_song_collection_com.bat` |
| MIDI 與左右手人工審核 | `Apps/Song_Collection_COM/run_hand_label_triage_0807.bat` |
| COM 手套韌體 | `Firmware/COM/Gloves_Firmware_INO_COM.ino` |
| 目前原始資料 | `Data/Raw/Song_Collection_COM/` |
| MIDI、影片與音訊來源 | `Data/External/FlowAudio_20260805/` |
| 推導與標記輸出 | `Data/Derived/Song_Collection_COM/` |

## 目前專案結構

```text
Apps/             可執行程式：COM 收集、UDP 收集
Firmware/         ESP32 韌體：COM 與舊 UDP
Data/Raw/         不修改的原始 CSV
Data/Derived/     MIDI 對齊、分流與標記產物
Data/External/    流音提供的 MIDI、影片、音訊
Docs/Current/     目前流程與研究進度
Docs/Logs/        實驗記錄
Docs/Archive/     舊規格與歷史文件
Presentations/    對外或會議簡報
Media/Demo/       示範影片
Tests/Latency/    延遲與感測器測試
Archive/          舊程式與歷史資料
```

## COM 歌曲資料收集

1. Arduino IDE 開啟 `Firmware/COM/Gloves_Firmware_INO_COM.ino`。
2. 左右手分別燒錄，將 `HAND_ID` 設為 `R` 與 `L`。
3. 雙手以 USB Type-C 連至電腦後，執行 `Apps/Song_Collection_COM/run_song_collection_com.bat`。
4. 選擇兩個 COM 埠、雙手歸零，按「開始歌曲收集」，結束後按「停止並關閉 CSV」。

收集程式會將 100 Hz 原始資料寫入 `Data/Raw/Song_Collection_COM/`。UDP 資料會獨立寫入 `Data/Raw/UDP_Collections/`，不會和歌曲資料混在一起。

## MIDI/CSV 標記

1. 執行 `Apps/Song_Collection_COM/run_hand_label_triage_0807.bat`。
2. 載入 `Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/manual_simultaneous_groups.json`。
3. 載入 `Data/External/FlowAudio_20260805/IMG_6060.MOV`。
4. 為同時 MIDI 群組逐一指定左手、右手或略過；高信心單點已由程式分流。

詳細規則與目前可訓練資料數量請看 `Docs/Current/HoloGrip_MIDI_CSV_手別標記進度與整理計畫_0807_ZH_TW.md`。

## 舊路線

`Apps/UDP_Collection/`、`Firmware/UDP_Legacy/`、`Tests/Latency/` 與 `Archive/` 仍完整保留，供舊版 UDP 實驗、延遲測試與歷史比對使用；它們不是目前歌曲收集與標記的預設入口。
