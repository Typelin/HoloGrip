# 20 ms MIDI 與影片稽核工具進度

## 本次目標

建立獨立的 HTML，專門檢查 MIDI 事件相距不到 20 ms 的密集區，並以影片逐組確認：

- 同一個鼓點在短時間內重複：列為疑似 MIDI 重複事件或快速連擊，不能直接當成錯誤。
- 不同鼓點在短時間內出現：列為可能的同時擊打，不能誤判為 MIDI 問題。
- 影片時間用 `MIDI 時間 + 14,500 ms` 作為目前候選偏移，CSV 時間用 `MIDI 時間 + 16,000 ms`。

## 已確認統計

來源：`Data/Derived/Song_Collection_COM/S20260805_P01_song01/midi_events.csv`

- MIDI 事件：1,820 筆
- 相鄰事件間隔小於 20 ms：429 組相鄰關係
- 同一 MIDI note 的短間隔關係：113 組
- 不同 MIDI note 的短間隔關係：316 組
- 多事件密集區：287 區
- 含同一 note 重複的密集區：126 區
- 僅不同 note 的密集區：161 區

## 使用方式

1. 開啟 `Apps/Song_Collection_COM/HoloGrip_MIDI_20MS影片稽核_0808_ZH_TW.html`。
2. 頁面會先載入 0805 MIDI；選擇 `midi_events.csv` 可換成其他 MIDI 事件 CSV。
3. 選擇原始 100 Hz CSV 和 `IMG_6060.MOV`，才能顯示手套活動線與影片。
4. 以「上一個／下一個密集資料區」導覽；同一鼓點重複區要優先和流音確認。
5. 對每一區選擇判定、寫入備註，再匯出稽核 CSV，作為會議紀錄，不直接當成最終訓練標籤。

## 尚未證明的事項

## 驗證結果

- HTML 內嵌 MIDI gzip Base64 可成功解碼為 1,820 筆事件。
- JavaScript node --check 通過。
- 以頁面實際分組邏輯重算：429 對、同音 113 對、不同音 316 對、密集區 287 區、同音區 126 區、不同音區 161 區。
- git diff --check 通過。

## Residuals

- Browser 工具的 URL 安全政策拒絕開啟本機 file:// 頁面，因此這次沒有自動截圖或瀏覽器點擊證據。
- 影片和原始 CSV 需要使用者在頁面中選檔；這是瀏覽器本機檔案權限限制，不是資料解析失敗。

- MIDI 事件檔本身無法精確證明左手或右手；`hand_candidate` 只作為候選資訊。
- 是否真的在 20 ms 內連打，要以影片中鼓棒與鼓面接觸畫面確認。
- `+16,000 ms` 與 `+14,500 ms` 是目前對齊候選值，正式標記前仍需用可辨識擊打確認。

## 0808 最新分類進度

主頁異常導覽已改成只使用異常區索引，不會跳到雙手乾淨區；同一 note 的 20 ms 判斷也改為完整檢查任意配對。新增可重跑統計腳本：

`C:\Users\Typelin_Station\Desktop\HoloGrip\Apps\Song_Collection_COM\midi_quality_audit_0808_ZH_TW.py`

原始 100 Hz CSV 實際重算結果：Hi-Hat note 46 有 153 對 20 ms 內重複，涉及 291 筆唯一事件；±90 ms 異常 705、有效分母 1,115、高信心合計 325；±80 ms 異常 665、有效分母 1,155、高信心合計 342。先前 100 筆是統計腳本漏加 +16,000 ms 偏移造成的錯誤低估，已修正。正式報告在：

`C:\Users\Typelin_Station\Desktop\HoloGrip\Docs\Current\HoloGrip_有效MIDI與異常分類報告_0808_ZH_TW.md`

目前仍需影片／WAV 確認 Hi-Hat 快速重複是否為真連擊或電子鼓重複觸發，以及雙手候選的兩筆 MIDI 左右手分配。
