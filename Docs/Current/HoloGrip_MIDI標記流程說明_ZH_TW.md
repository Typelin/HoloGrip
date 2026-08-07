# SONG CSV + MIDI 後處理流程

這套流程把流音的原生 MIDI 與雙手 `Raw 100 Hz CSV` 分開保留，再產生可複核的衍生資料。原始檔不會被覆蓋。

## 目前固定規則

使用流音 0805 確認的 `110 BPM`。MIDI 檔內的 120 BPM metadata 不作為本次轉換基準。

| MIDI note | Zone_ID | Zone_Name |
| ---: | ---: | --- |
| 37、38 | 0 | 小鼓 |
| 48 | 1 | 高音 Tom |
| 45 | 2 | 中音 Tom |
| 43 | 3 | 落地 Tom |
| 46 | 4 | Hi-Hat |
| 49 | 5 | Crash |
| 51 | 6 | Ride |

36 大鼓與 44 腳踏 Hi-Hat 是腳部事件，不納入手部 7 類標籤。

## 執行

目前場次可以直接雙擊：

`Apps/Song_Collection_COM/run_midi_label_pipeline_0805.bat`

或在專案根目錄執行：

```powershell
python Apps/Song_Collection_COM/midi_label_pipeline.py `
  --raw-csv Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv `
  --midi "Data/External/FlowAudio_20260805/Drum Midi_110BPM (0805).mid" `
  --out-dir Data/Derived/Song_Collection_COM/S20260805_P01_song01 `
  --bpm 110
```

## 輸出

`Data/Derived/Song_Collection_COM/<session_id>/` 會產生：

- `midi_events.csv`：每個 MIDI 事件與 7 類 Zone_ID。
- `alignment_report.json`：BPM、offset、來源檔 hash、事件統計與複核門檻。
- `candidate_windows.csv`：每個事件的固定前後時間窗，以及密集事件複核欄位。
- `labeled_window_samples.csv`：事件中心的 Raw 感測樣本，供後續模型特徵處理。

## 密集 MIDI 的處理規則

固定窗口目前統一為 `前 90 ms + 後 90 ms`，總長度 180 ms。這不是保證每筆事件都能單獨分離；它是模型輸入的固定尺寸。

- 同一個 12 ms 內的 MIDI 事件會建立同一個 `onset_group_id`，例如同時小鼓與 Ride 會保留為多標籤群組。
- 若前後相鄰群組間隔小於窗口總長度 180 ms，該事件會標記 `overlap_review=true`。
- 只有 `single_label_eligible=true` 的事件才可直接進入第一版單一鼓點模型。
- `multi_label` 或 `overlap_review` 的事件仍會輸出，供人工複核或多標籤模型使用，不會被硬分成一個鼓點。
- `safe_half_window_ms` 是不碰鄰近 MIDI 群組的最大半徑估計，只做 QC 顯示；正式模型仍使用固定 180 ms 輸入。

因此，密集區的解法不是把窗口放大，而是將它標為多標籤／重疊複核，並從單一標籤訓練集排除。200 Hz 可改善時間量化，但同時發生的 MIDI 仍然必須保留為群組。

## 目前實測結果

`S20260805_P01_song01` 的既有衍生輸出（在本次密集判定加入前）：

- Raw CSV：74,592 列。
- MIDI：1,820 個映射事件，7 類均有資料，未知 note 為 0。
- 自動候選 offset：`16,000 ms`。
- 視窗資料：181,897 列（當時使用舊的前 200 ms／後 300 ms）。
- 所有輸出狀態仍是 `pending_manual_alignment_review`。

程式與 BAT 現在統一使用前後各 90 ms；既有衍生輸出不會被本次程式碼修改，重新執行後才會產生新的密集判定欄位。

## 複核門檻

1. 用 WAV 或已知同步擊確認 `16,000 ms` 是否是正確起點。
2. 檢查同時發生的 MIDI note；MIDI 沒有左右手資訊，不能直接把 `hand_candidate` 當成真值。
3. 確認後再進行事件人工複核，最後才輸出正式訓練資料。

不要直接把 `labeled_window_samples.csv` 當成已驗證資料集；它目前是候選訓練視窗。
