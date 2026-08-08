# HoloGrip WAV 音訊與 MIDI／CSV／影片三位一體分析

## 結論先行

這條路可行，而且 WAV 有實際用途：它可以提供第三個「聲音發生瞬間」證據，用來檢查整體偏移、驗證 MIDI 是否大致真的有擊打，以及篩出同一鼓點在 20 ms 內的可疑事件。

但 WAV 是混音後的雙聲道音訊，不能可靠告訴我們左手或右手。正式手別仍然使用：CSV 活動峰作為候選，影片作為人工確認真值。

## 實際檔案

- WAV：`C:\Users\Typelin_Station\Desktop\HoloGrip\Data\External\FlowAudio_20260805\Drum Audio_110BPM (0805).wav`
- MIDI：`C:\Users\Typelin_Station\Desktop\HoloGrip\Data\External\FlowAudio_20260805\Drum Midi_110BPM (0805).mid`
- 100 Hz CSV：`C:\Users\Typelin_Station\Desktop\HoloGrip\Data\Raw\Song_Collection_COM\S20260805_P01_song01_raw_100hz_20260805_161628.csv`
- 影片：`C:\Users\Typelin_Station\Desktop\HoloGrip\Data\External\FlowAudio_20260805\IMG_6060.MOV`
- 分析腳本：`C:\Users\Typelin_Station\Desktop\HoloGrip\Tools\Audio\analyze_drum_audio_0808.py`
- 快速啟動：`C:\Users\Typelin_Station\Desktop\HoloGrip\Tools\Audio\run_analyze_drum_audio_0808_ZH_TW.bat`
- 分析報告：`C:\Users\Typelin_Station\Desktop\HoloGrip\Data\Derived\Song_Collection_COM\S20260805_P01_song01\audio_alignment_0808\audio_alignment_report_0808.json`
- onset 時間序列：`C:\Users\Typelin_Station\Desktop\HoloGrip\Data\Derived\Song_Collection_COM\S20260805_P01_song01\audio_alignment_0808\audio_onsets_0808.csv`

## WAV 本身

- 格式：PCM 24-bit little-endian。
- 原始取樣率：48,000 Hz。
- 聲道：2 聲道。
- 音訊長度：358.446458 秒。
- 影片長度：377.446667 秒。
- MIDI 最後事件：353,551.136 ms。
- CSV 長度：373,125 ms。

## 對齊結果

分析方式是把 MIDI 每筆事件加上一個候選偏移，再找附近 WAV onset。容許誤差為 ±35 ms；onset 以 5 ms 時間格計算。

- 最佳音訊偏移候選：`MIDI + 0 ms`。
- 次佳候選：`MIDI + 10 ms`，命中率略低。
- 命中：`1,632 / 1,820`，約 `89.7%`。
- 命中事件的中位絕對誤差：`5.455 ms`。
- 命中事件的平均絕對誤差：`7.853 ms`。
- 35 ms 標準 onset：2,754 個。
- 12 ms 快速 onset：3,549 個。

這代表 WAV 與 MIDI 的整體時間關係很接近，`0 ms` 可以作為目前的 WAV 播放候選偏移；不代表 1,820 筆每一筆都已經被音訊證明。

## 七種鼓點的音訊命中

| MIDI note | 鼓點 | 命中／總數 | 命中率 | 中位誤差 |
|---:|---|---:|---:|---:|
| 38 | 小鼓 | 390 / 436 | 89.4% | 2.727 ms |
| 43 | 落地 Tom | 127 / 128 | 99.2% | 5.682 ms |
| 45 | 中音 Tom | 56 / 58 | 96.6% | 8.068 ms |
| 46 | Hi-Hat | 715 / 837 | 85.4% | 8.864 ms |
| 48 | 高音 Tom | 74 / 77 | 96.1% | 6.818 ms |
| 49 | Crash | 34 / 37 | 91.9% | 3.182 ms |
| 51 | Ride | 236 / 247 | 95.5% | 2.273 ms |

Hi-Hat 命中率較低，不應直接解讀為 MIDI 錯誤；它可能受到鈸延音、混音重疊、同時鼓點與 onset 偵測門檻影響。

## 20 ms 同音事件稽核

- 同一 MIDI note 在 20 ms 內的事件對：`148 對`。
- 快速音訊偵測在該事件前後範圍內找到兩個以上 onset 候選：`77 對`，約 `52.0%`。
- 音訊目前無法分辨兩次：`71 對`，約 `48.0%`。

這裡的「有兩個 onset」只代表混音音訊有兩個時間峰候選，可能包含其他鼓點或延音造成的峰，不能直接證明同一個鼓真的被打了兩次。這 148 對仍應由影片逐區確認；音訊可以把優先檢查範圍縮小。

## 你指出的 5:33.085 案例

依目前影片偏移 `+14,500 ms`，影片 `5:33.085` 約對應 MIDI `318,585 ms`。該區實際 MIDI 事件如下：

| 事件 | MIDI 時間 | note | 鼓點 | 與前一事件 |
|---:|---:|---:|---|---:|
| 1692 | 318,568.182 ms | 49 | Crash | — |
| 1693 | 318,585.227 ms | 46 | Hi-Hat | 17.045 ms |
| 1694 | 318,596.591 ms | 46 | Hi-Hat | 11.364 ms |
| 1695 | 318,784.091 ms | 38 | 小鼓 | 187.500 ms |

WAV 在同一時間附近的快速 onset 為 `318,570 ms`；在 `318,585.227 ms` 與 `318,596.591 ms` 這兩個 Hi-Hat MIDI 事件之間，沒有兩個可獨立辨識的音訊 onset。這支持「Crash 有聲音，但兩個 Hi-Hat 需要懷疑」的判斷；仍需以影片確認是否有極短、低力度的實際 Hi-Hat 接觸。

## 前端三位一體實作

主頁已加入同一個時間游標：

`C:\Users\Typelin_Station\Desktop\HoloGrip\Apps\Song_Collection_COM\HoloGrip_MIDI_CSV對齊檢查_0807_ZH_TW.html`

時間換算為：

```text
MIDI 時間 = CSV 時間 - CSV 偏移（目前 +16,000 ms）
影片時間 = MIDI 時間 + 影片偏移（目前 +14,500 ms）
WAV 時間 = MIDI 時間 + WAV 偏移（目前 +0 ms）
```

頁面新增：

- 演奏影片選檔、目前 MIDI 鼓點影片時間、目前 ±採樣窗口播放。
- WAV 選檔、音訊控制列、目前 MIDI 鼓點 WAV 時間。
- WAV 分析 JSON 選檔；載入後會更新最佳偏移、onset 數與命中率。
- `目前位置 −1 ms`、`目前位置 ＋1 ms`，方便細看密集事件。
- 既有的 MIDI、CSV 局部圖會與三位一體播放時間游標同步。
- 預設檔案目錄提示；瀏覽器仍需要使用者按「選擇檔案」，不能由 HTML 自動讀取 Windows 路徑。

### 0808 主顯示區版面更新

目前頁面已重新分成三個閱讀層次：

1. **資料設定**：MIDI、CSV、影片、WAV、分析報告，以及 MIDI→CSV、MIDI→影片、MIDI→WAV 偏移。
2. **主顯示區**：左側影片，右側 MIDI 鼓點與左右手 CSV 活動度局部圖；下方是 WAV 聲音證據。
3. **事件導覽時間軸**：主顯示區下方顯示整首歌的 MIDI 點，點擊任一點會同步更新局部 CSV 圖、影片、WAV 與目前 ±90 ms 窗口。

頁面會先以相對於 HTML 的路徑嘗試載入：

```text
../../Data/External/FlowAudio_20260805/IMG_6060.MOV
../../Data/External/FlowAudio_20260805/Drum Audio_110BPM (0805).wav
```

若瀏覽器的 `file://` 安全政策拒絕自動載入，仍可在資料設定區手動選取檔案。這不是 MIDI、CSV 或同步公式錯誤，而是本機檔案頁面的瀏覽器限制。

## 使用順序

1. 開啟主頁，先確認 MIDI／CSV 對齊偏移。
2. 在三位一體區塊選取 `IMG_6060.MOV`。
3. 選取 `Drum Audio_110BPM (0805).wav`。
4. 可選取 `audio_alignment_report_0808.json`，讓頁面載入實際分析摘要。
5. 選取一個 MIDI 事件或導覽到密集區，按「跳到目前 MIDI 鼓點」。
6. 按「播放目前 ±90 ms 窗口」，同時看 MIDI 點、CSV 左右活動線、影片接觸畫面與 WAV 聲音。
7. 同一 note 在 20 ms 內的事件，最後以影片決定「真實快速連打」或「疑似 MIDI 重複」。

## 目前仍不能自動化的部分

- WAV 不能分辨左手／右手。
- WAV onset 不能獨立證明是哪個鼓點發聲。
- 影片與 CSV 的 `+14,500 ms`／`+16,000 ms` 仍是候選偏移，正式標記前要用可辨識擊打再次確認。
- `file://` 頁面受瀏覽器政策限制，無法由自動化工具取得截圖或按鈕操作證據；頁面本身的 JavaScript 已通過語法檢查。
