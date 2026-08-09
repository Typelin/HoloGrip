# KEEPALIVE｜對齊基底與預覽解析度修正

## Result

已將 MIDI→CSV 對齊改為使用 `sensor_time_ms` 正規化後的 10 ms 感測器時間格，取消會把未來約 40 ms 峰值提前套用到目前事件的局部最大值平滑。10 ms 基底每格保留左右手各自的最大 `abs(accel_magnitude_g - 1.0)`；20、50、100 ms 僅將 10 ms 基底做算術平均供畫面預覽，不參與對齊。

## Evidence

- 原始 CSV：74,592 列，左右手各 37,296 列。
- MIDI：1,820 筆 mapped NoteOn。
- `sensor_time_ms` 起點：399,053；感測器時間長度：373,139 ms。
- 新實跑候選偏移：`+16,030 ms`。
- 第一筆 MIDI：0 ms、note 46 Hi-Hat；對齊 CSV 16,030 ms，左手 0.083093g、右手 6.092579g。
- 新報告：`Data/Derived/Song_Collection_COM/現場MIDI_CSV對齊_0809_ZH_TW_20260809_sensor_clock/alignment_report.json`
- 新前端：`Data/Derived/Song_Collection_COM/現場MIDI_CSV對齊_0809_ZH_TW_20260809_sensor_clock/HoloGrip_MIDI_CSV對齊現場檢查_0809_ZH_TW.html`
- 主模板已同步：`Apps/Song_Collection_COM/HoloGrip_MIDI_CSV對齊檢查_0807_ZH_TW.html`
- Python self-test、Python compile、HTML 8 個 JavaScript 區塊語法檢查與 `git diff --check` 通過。

## Residuals

- `+16,030 ms` 是全部 MIDI 事件的自動候選偏移，不代表每一筆事件的峰值都會剛好落在同一個 10 ms 點；前端懸停資訊現在會顯示左右峰值位置與距離。
- `abs(accel_magnitude_g - 1.0)` 是由原始加速度推導的活動度，不是原始 `ax/ay/az` 本身；正式訓練仍須保留並使用原始 CSV 欄位。
- 尚未用瀏覽器自動化開啟 `file://` 頁面做截圖；本次驗證採實際資料生成、Python 測試與 Node JavaScript 語法檢查。
