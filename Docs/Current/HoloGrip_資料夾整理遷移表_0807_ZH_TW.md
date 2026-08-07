# HoloGrip 資料夾整理遷移表

更新日期：2026-08-07

> 狀態：已完成。此文件保留搬遷設計與原始對照；實際結果請看 `HoloGrip_資料夾整理完成紀錄_0807_ZH_TW.md`。

## 結論

現在應該整理，但必須分階段進行。HoloGrip 根目錄目前混放了活躍收集程式、韌體、原始資料、推導資料、外部素材、測試、簡報與舊版本；直接拖曳資料夾會使 Python、BAT 和文件中的既有路徑失效。

本次盤點的根目錄總量約 2.56 GB，其中 `Presentations/` 約 1.00 GB、`流音給/` 約 916 MB、`vids/` 約 602 MB。真正需要保護的活躍資料流程是：COM 收集 -> `CSV_Data/` -> `Derived_Data/` -> MIDI/影片人工標記。

## 目前不得直接搬動的活躍路徑

| 目前路徑 | 為何不能直接搬動 |
|---|---|
| `CSV_Data/` | COM 與 UDP 收集程式均直接寫入此目錄。 |
| `Derived_Data/` | MIDI 對齊、手別分流與人工審核輸出位於此處。 |
| `流音給/` | `hand_label_triage_0807.py` 與 MIDI 標記 BAT 直接讀取 MIDI、影片、音訊。 |
| `Song_Collection_COM/` | 目前唯一正在使用的 COM 收集、MIDI 對齊與手別標記工具。 |
| `UDP_version_release/` | 保留為獨立 UDP 路線；程式會以目前位置推導專案根目錄。 |

## 目標結構

```text
HoloGrip/
  Apps/
    Song_Collection_COM/
    UDP_Collection/
  Firmware/
    COM/
    UDP_Legacy/
  Data/
    Raw/
      Song_Collection_COM/
      UDP_Collections/
      Legacy/
    Derived/
      Song_Collection_COM/
    External/
      FlowAudio_20260805/
  Docs/
    Current/
    Logs/
    Archive/
  Presentations/
    Current/
    Archive/
  Media/
    Demo/
  Tests/
    Latency/
  Archive/
    Legacy_Code/
  outputs/
```

## 遷移對照表

| 現在位置 | 目標位置 | 分類 | 搬動前必要動作 |
|---|---|---|---|
| `Song_Collection_COM/` | `Apps/Song_Collection_COM/` | 活躍程式 | 將所有專案根目錄推導與 BAT 路徑改為新位置，並跑 Python 語法檢查。 |
| `UDP_version_release/` | `Apps/UDP_Collection/` | 保留程式 | 改正專案根目錄推導與 README 指令，避免 UDP/COM 混用。 |
| `Gloves_Firmware_INO_COM/` | `Firmware/COM/` | 活躍韌體 | 更新 README 與燒錄說明。 |
| `Gloves_Firmware_INO/` | `Firmware/UDP_Legacy/` | 舊韌體 | 確認不再由任何活躍 BAT/文件作為 COM 燒錄入口。 |
| `CSV_Data/` | `Data/Raw/` | 原始資料 | 修改 COM/UDP 寫入位置；確認新測試 CSV 可建立。 |
| `Derived_Data/` | `Data/Derived/` | 推導資料 | 修改 MIDI pipeline、triage 與人工審核說明路徑。 |
| `流音給/` | `Data/External/FlowAudio_20260805/` | 外部來源 | 修改 MIDI/影片/音訊預設路徑，並重新跑一次 triage。 |
| `CSV_Data_Legacy/` | `Data/Raw/Legacy/` | 歷史原始資料 | 只搬動，不轉檔、不改內容。 |
| `Latency_Test/` | `Tests/Latency/` | 測試工具 | 更新測試報告內相對路徑後搬動。 |
| `old/` | `Archive/Legacy_Code/` | 舊程式 | 只封存，不納入目前執行入口。 |
| `vids/` | `Media/Demo/` | 舊示範影片 | 先確認沒有被簡報以絕對路徑連結。 |
| `簡報1/` | `Presentations/Archive/簡報1/` | 舊簡報素材 | 與既有簡報資料夾合併管理。 |
| `Presentations/` | 保持名稱，內部分為 `Current/` 與 `Archive/` | 簡報 | 先確定目前要對外使用的 PPT，再分流舊版。 |
| 根目錄論文 DOCX | `Docs/Current/` | 現行文件 | 更新 README 連結（如需）。 |
| `battery_test_timer.py` 與電池紀錄 | `Tools/`、`Docs/Logs/` | 工具與紀錄 | 調整程式目前使用的相對 `Docs/` 輸出路徑。 |
| `temp_docx_text.txt`、`問題點.txt` | `Docs/Archive/` | 暫存／舊紀錄 | 確認後封存。 |
| `__pycache__/` | 不遷移 | Python 快取 | 最後一步可安全清除；不影響程式邏輯。 |

## 執行順序

1. 建立此遷移表與搬動前檔案清單，不改實體路徑。
2. 將活躍程式的資料目錄集中為設定值，更新 README、BAT 與 Markdown 文件。
3. 以原路徑執行一次 COM 收集啟動、MIDI triage 與 Python 語法檢查。
4. 建立目標目錄，依序搬動活躍程式、韌體、資料與外部素材；每一批搬動後立即執行對應驗證。
5. 最後才封存舊 CSV、舊程式、簡報素材與暫存檔；不刪除原始研究資料。

## 驗收條件

- COM 收集程式仍可啟動，且寫出的 CSV 位於新 `Data/Raw/Song_Collection_COM/`。
- MIDI pipeline 與 hand-label triage 能找到 MIDI、CSV、影片與輸出位置。
- 人工審核 HTML 能載入 JSON 與影片。
- UDP 路線仍保留獨立入口，不會寫入 COM 的資料集。
- 每一批搬動前後檔案數與總大小一致；不使用刪除操作。
- README 的所有使用者操作路徑與實際檔案一致。

## 本次決策範圍

本次已完成路徑改造、實體搬遷與檔案數量核對。搬遷前 Git 基線為 `b297e3c`；原始 CSV、MIDI、影片、音訊沒有刪除或轉檔。
