# HoloGrip 三位一體對齊進度

## 任務目標

在 `HoloGrip_MIDI_CSV對齊檢查_0807_ZH_TW.html` 內加入同一時間游標，讓 MIDI 鼓點、CSV 左右手活動度、演奏影片與 WAV 音訊可以在同一個局部窗口中檢查。

## 已確認資料

- MIDI：`Data/External/FlowAudio_20260805/Drum Midi_110BPM (0805).mid`，已整理為 1,820 筆事件。
- CSV：`Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv`，100 Hz，約 373.125 秒。
- 影片：`Data/External/FlowAudio_20260805/IMG_6060.MOV`，需由瀏覽器選取後以本機 URL 播放。
- WAV：`Data/External/FlowAudio_20260805/Drum Audio_110BPM (0805).wav`，48 kHz、雙聲道、約 358.446 秒。

## 判斷邊界

- WAV 先用於檢查整體時間偏移、擊打瞬間與 MIDI 可疑重複，不直接推斷左手／右手。
- CSV 仍是左右手活動度的來源；影片才是雙手同時事件的手別人工真值來源。
- HTML 不會因為檔案在專案目錄旁邊就自動讀取，會預填預設路徑提示，實際內容仍由使用者選檔載入。

## 目前階段

- [x] 盤點原始 MIDI、Raw CSV、MOV、WAV 與既有頁面。
- [x] 建立可重跑的 WAV onset／MIDI 對照分析。
- [x] 將影片與音訊接入 MIDI／CSV 同步播放。
- [x] 完成靜態檢查與實際資料分析；瀏覽器 `file://` 互動驗證受政策阻擋。
- [ ] Git 保存本次進度。

## 實際分析結果

- WAV 最佳偏移候選：`MIDI +0 ms`。
- ±35 ms 命中：`1,632 / 1,820`，約 `89.7%`；中位誤差 `5.455 ms`。
- 同一 MIDI note 在 20 ms 內：`148 對`；快速音訊有兩個 onset 候選 `77 對`，`71 對`無法分辨。

## 驗證結果

- `python -m py_compile Tools/Audio/analyze_drum_audio_0808.py` 通過。
- 主 HTML 4 段 JavaScript 皆通過 Node `vm.Script` 編譯。
- `git diff --check` 通過；僅有 Git 的換行格式提示。

## 殘餘風險

- WAV onset 不能判斷左手／右手，也不能單獨證明是哪一個鼓點。
- 瀏覽器安全政策拒絕自動化工具開啟本機 `file://` 頁面，未取得截圖與實際按鈕操作證據。

## 續跑入口

工作區：`C:\Users\Typelin_Station\Desktop\HoloGrip`

主頁：`Apps\Song_Collection_COM\HoloGrip_MIDI_CSV對齊檢查_0807_ZH_TW.html`

分析腳本與輸出完成後，補上實際 WAV onset 統計與殘餘風險。
