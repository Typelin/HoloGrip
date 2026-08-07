# HoloGrip MIDI/CSV 手別標記進度與整理計畫

更新日期：2026-08-07

> 資料夾實體遷移已於 2026-08-07 完成。本文件中的目前操作路徑已更新為 `Apps/`、`Firmware/` 與 `Data/` 結構；第 8 節保留整理前盤點，供追溯使用。

## 1. 本次資料與目的

目標是將雙手手套的 100 Hz CSV，配合電子鼓本次實際演奏產生的 MIDI，建立可追溯的左右手鼓點訓練資料。

| 資料 | 目前檔案 | 用途 |
|---|---|---|
| 原始手套資料 | `Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv` | 左右手 IMU，100 Hz |
| 演奏 MIDI | `Data/External/FlowAudio_20260805/Drum Midi_110BPM (0805).mid` | 鼓點時間、7 類鼓點、velocity |
| 演奏影片 | `Data/External/FlowAudio_20260805/IMG_6060.MOV` | 人工確認左右手 |
| 演奏音訊 | `Data/External/FlowAudio_20260805/Drum Audio_110BPM (0805).wav` | 輔助影片初始對時 |

MIDI 有 1,820 個 7 鼓點事件，但沒有原生左手／右手欄位。MIDI 負責「何時、哪個鼓」；影片與手套活動才用於補足「哪隻手」。

## 2. 已確認的時間關係

| 換算 | 目前值 | 說明 |
|---|---:|---|
| MIDI -> CSV | +16,000 ms | 原始手套 CSV 的候選對齊偏移，訓練前仍要保留對齊稽核 |
| MIDI -> 影片 | +14,500 ms | 由影片收音與提供音檔取得的初始偏移；影片用於手別判讀，不用來切訓練窗口 |
| CSV 採樣 | 100 Hz | 一個資料格約 10 ms |
| 影片 | 約 59.94 fps | 一個影格約 16.68 ms，不能視為 1 ms 真值 |

模型使用 MIDI 中心點前後各 80 ms 的 CSV，即總長 160 ms。影片播放也固定為同一段 +/-80 ms，防止人工審核看進前後其他鼓點。

## 3. 標記分流規則

### 3.1 自動高信心單點

一筆 MIDI 可先自動標記為某一手，必須同時符合：

1. 同一個 12 ms onset 群組只有 1 個 MIDI 事件。
2. 前、後 MIDI onset 群組各距離至少 180 ms。
3. 在對齊 CSV 的 +/-80 ms 內，主手活動占比 `max(L, R) / (L + R)` >= 90%。
4. 主手活動峰值 >= 1.0 g，避免兩手都幾乎未動時出現比例假象。

### 3.2 同時 MIDI 群組

同一 onset 群組內有 2 個以上 MIDI 事件時，所有事件必須綁成同一個人工審核項目。不得把同一個「右手候選」複製給組內每個鼓點。

人工在影片中逐一為組內 MIDI 鼓點指定左手、右手或略過。影片若不在該群組的 +/-80 ms 採樣窗口內，群組審核頁會鎖住標記按鈕。

### 3.3 暫不使用的事件

- 單一 MIDI 但左右手活動占比未達 90%：先不自動標記。
- 單一 MIDI 但距離相鄰事件少於 180 ms：窗口容易與前後動作重疊，第一版不納入獨立窗口訓練。

這些原始資料沒有損壞；它們適合未來的連續時間序列或多事件輸出模型，而不是第一版的一點一窗口模型。

## 4. 本次實際分流結果

| 類別 | MIDI 節點 | 全部 MIDI 比例 | 獨立 CSV 窗口 | 處理方式 |
|---|---:|---:|---:|---|
| 高信心單一 MIDI | 247 | 13.6% | 247 | 自動高信心候選 |
| 同時 MIDI 群組 | 586 | 32.2% | 257 群組 | 人工分配左右手 |
| 可標記合計 | 833 | **45.8%** | **504** | 第一輪資料來源 |
| 單點但手別不明確 | 151 | 8.3% | 151 | 暫不使用或後續人工補看 |
| 靠近相鄰 MIDI 的單點 | 836 | 45.9% | 836 | 保留給第二階段序列模型 |
| 合計 | 1,820 | 100.0% | - | - |

同時 MIDI 群組大小：

| 同時 MIDI 數量 | 群組數 |
|---:|---:|
| 2 點 | 194 |
| 3 點 | 54 |
| 4 點 | 9 |

MIDI 並非最多只會同時 2 點。第一版雙手模型只使用「高信心單點 + 已人工分配的 2 點同時群組」：441 個獨立窗口、635 個 MIDI 標籤，約占全部 MIDI 34.9%。3、4 點同時群組留到多事件模型。

## 5. #9/#10 範例

影片約 `0:19.892` 的一次同時打擊，在 MIDI 中是：

| MIDI 事件 | MIDI 時間 | 鼓點 | note |
|---:|---:|---|---:|
| #9 | 5.392 s | 小鼓 | 38 |
| #10 | 5.398 s | Hi-Hat | 46 |

兩點相差僅 5.7 ms，已放在同時群組 #9。該群組 CSV 峰值為左 5.176 g、右 5.924 g，主手只占 53.4%，不能自動指派右手；必須用影片決定小鼓與 Hi-Hat 各自由哪隻手打出。

## 6. 鼓點數量與訓練判斷

完成同時群組人工標記後，最多可取得下列節點數：

| 鼓點 | 可用 MIDI 節點 |
|---|---:|
| Hi-Hat | 485 |
| 小鼓 | 174 |
| Ride | 107 |
| 落地 Tom | 34 |
| Crash | 17 |
| 高音 Tom | 9 |
| 中音 Tom | 7 |

這足以建立第一版訓練與驗證流程，但不足以做可靠的「左右手 x 7 鼓點」完整模型。尤其高音 Tom、中音 Tom、Crash 必須在下一輪補錄。建議每一手、每一鼓點增加至少 100 個乾淨單打。

第一版資料單位應為：

```text
輸入：同一個 160 ms 窗口中的左手手套 + 右手手套訊號
輸出：左手鼓點類別、右手鼓點類別
```

## 7. 已建立的工具與輸出

### 自動分流與同時 MIDI 人工審核

啟動人工審核：`Apps/Song_Collection_COM/run_hand_label_triage_0807.bat`

它會開啟：`Apps/Song_Collection_COM/HoloGrip_同時MIDI影片手別標記_0807_ZH_TW.html`，不會重建既有分流輸出。

只有原始 CSV、MIDI、對齊偏移或篩選規則改變時，才執行：`Apps/Song_Collection_COM/rebuild_hand_label_triage_0807.bat`。它會先執行 `Apps/Song_Collection_COM/hand_label_triage_0807.py`，再開啟人工審核頁。

群組審核頁要載入：

1. `Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/manual_simultaneous_groups.json`
2. `Data/External/FlowAudio_20260805/IMG_6060.MOV`

產出目錄：`Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/`

| 檔案 | 意義 |
|---|---|
| `auto_accepted_single_events.csv` | 247 筆自動高信心單點候選 |
| `manual_simultaneous_groups.json` | 257 組人工影片審核輸入 |
| `manual_simultaneous_groups.csv` | 群組總覽 |
| `excluded_events.csv` | 本輪不使用的事件與原因 |
| `hand_label_triage_summary.json` | 可供程式讀取的統計 |

第 2 頁完成手別判讀後，按「下載同時 MIDI 手別 CSV」。該下載檔是第 3 頁「第 2 頁下載的人工手別 CSV」欄位要載入的檔案；原本的 `manual_simultaneous_groups.csv` 只是群組總覽，不能取代人工手別 CSV。

## 8. 整理前資料夾盤點（歷史紀錄）

目前根目錄有幾個正確但混雜的區塊：

| 目前位置 | 正確角色 | 整理判斷 |
|---|---|---|
| `CSV_Data/` | 原始 CSV | 保留為不可修改的 raw data 根目錄 |
| `Derived_Data/` | 對齊、分流、標記等推導結果 | 保留為唯一 derived data 根目錄 |
| `流音給/` | 外部交付的 MIDI、WAV、MOV | 應改為明確的 external/source 名稱，但先不搬動 |
| `Song_Collection_COM/` | COM 收集與 MIDI 標記工具 | 應保留為目前活躍程式區 |
| `UDP_version_release/` | UDP 收集版本 | 保留但需和 COM 程式明確分開 |
| `Gloves_Firmware_INO*` | 韌體 | 需區分舊版與 COM 版 |
| `Docs/` | 規格、計畫、電池紀錄 | 需要區分 current、logs、archive |
| `old/`、`CSV_Data_Legacy/`、`outputs/`、根目錄暫存檔 | 舊資料或暫存輸出 | 需要盤點後封存 |

## 9. 已完成的整理方式

### 已完成：穩定資料路徑與分類

1. 原始資料統一在 `Data/Raw/`，推導標記統一在 `Data/Derived/`，流音交付素材統一在 `Data/External/FlowAudio_20260805/`。
2. COM 與 UDP 程式統一在 `Apps/`，兩套韌體統一在 `Firmware/`。
3. 此文件與 `README.md` 為目前 MIDI/CSV 流程入口。
4. 後續新增資料遵守：原始檔只進 `Data/Raw/` 或 `Data/External/`；產生物只進 `Data/Derived/`。
5. 所有提供人閱讀的繁體中文報告與說明文件，檔名統一以 `_ZH_TW` 結尾。原始資料、程式、BAT、韌體與第三方交付檔維持原檔名，避免破壞工具路徑與可追溯性。

### 遷移後實際結構

目標結構：

```text
HoloGrip/
  Apps/
    Song_Collection_COM/
    UDP_Collection/
  Firmware/
    COM/
    Legacy/
  Data/
    Raw/
      Song/
      UDP/
    Derived/
      Song/
    External/
      FlowAudio_0805/
  Docs/
    Current/
    Logs/
    Archive/
  Presentations/
  Archive/
```

遷移前已建立 Git 基線、更新 BAT/Python 路徑並完成靜態驗證。搬遷後仍需在下次連接實體手套時，執行一次 COM 實機收集與 MIDI triage 冒煙測試。

## 10. 下一步

1. 執行同時 MIDI 群組人工審核，先完成所有 2 點群組。
2. 下載群組左右手 CSV，建立第一版 441 個窗口的基礎訓練集。
3. 以 blocked time split 訓練第一版雙手模型；不隨機拆分 CSV 列。
4. 補錄高音 Tom、中音 Tom、Crash 的左右手乾淨單打資料。
5. 核准第二階段目錄遷移表後，再進行實體搬檔。
