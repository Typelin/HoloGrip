# HoloGrip 三頁網頁工具工作紀錄

- goal: 將 MIDI/CSV 對齊、同時 MIDI 影片手別標記與資料集進度總覽整理為三個清楚分工的本機 HTML 工具。
- done_definition: 三個 HTML 各自對應對齊、同時 MIDI 人工手別標記與資料集進度判讀；三頁可互相導覽，資料交接路徑明確，並通過 JavaScript 與 0805 實際資料檢查。
- status: shipped
- completed_steps: 確認三頁現有功能與 0805 triage 統計；第三頁未將待人工資料誤算為可訓練；三頁已加入一致導覽；人工審核 BAT 改為只開啟標記頁，重建流程分離為獨立 BAT。
- artifacts: `C:\Users\Typelin_Station\Desktop\HoloGrip\Apps\Song_Collection_COM\HoloGrip_MIDI_CSV對齊檢查_0807_ZH_TW.html`、`C:\Users\Typelin_Station\Desktop\HoloGrip\Apps\Song_Collection_COM\HoloGrip_同時MIDI影片手別標記_0807_ZH_TW.html`、`C:\Users\Typelin_Station\Desktop\HoloGrip\Apps\Song_Collection_COM\HoloGrip_資料集最終進度總覽_0807_ZH_TW.html`
- next_step: 已完成；使用者依第 1 -> 第 2 -> 第 3 頁順序操作，人工標記完成後將第 2 頁下載 CSV 載入第 3 頁。
- evidence: 三個 HTML 的 JavaScript 語法檢查通過；0805 實際輸出重算為 247 個高信心單點、257 組同時 MIDI、586 個同時節點、194 組雙點同時 MIDI、441 個基準窗口、635 個基準標籤。
- residuals: 本機 browser 對 file:// 頁面有安全政策限制，無法自動截圖；未安裝額外 DOM 測試依賴。
- updated_at: 2026-08-07
