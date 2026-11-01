# 現場當晚流程（第 9 次 → 第十次包）

目的：現場當天晚上，把今天的資料標記、算出盲測成績、重新訓練並做出下一個現場包，隔天直接用。
所有指令在這個資料夾執行，先設定 `PYTHONIOENCODING=utf-8`。原始 Raw／MIDI 只讀不改；前測一律不進訓練。

```bash
cd /c/Users/Typelin_Station/Desktop/HoloGrip/Research/NextRound_1105
export PYTHONIOENCODING=utf-8
```

## 0. 收資料（現場結束後）
- Raw：`第九次的包_暫定/Data/Raw/`（蒐集）與 `Data/模型實測/S20261105*/`（前測，含現場 `predictions.csv`）。
- MIDI：放進一個資料夾（例如 `Desktop/今天`），前測 MIDI 檔名填在下一步。

## 1. MIDI ↔ Raw 配對（約 1～2 分鐘）
```bash
python pair_midi_raw.py R9
```
輸出 `out/pairing_R9.md`：每個 MIDI 最像的 Raw 與吻合率。「要人工看」＝吻合 < 0.85 或第二像差距 < 0.15，多半是中途重錄（作廢那份不要用）。

## 2. 填 `rounds.json` 的 `R9`（人工確認）
```json
"R9": {"enabled": true, "date": "20261105", "package": "第九次的包_暫定",
       "midi_dir": "C:/Users/Typelin_Station/Desktop/今天", "pretest_midi": "測試1.mid",
       "single": {"ride L": ["Ride", "<Raw 檔名，不含 .csv>"], ...},
       "songs":  {"測試2": ["song", "<Raw 檔名>"], "測試3": ["four_drum", "<Raw 檔名>"]},
       "void": ["<作廢的 Raw>"]}
```
- 單手補強的 MIDI 檔名要以 ` L`／` R` 結尾（手別從檔名來）；`pairing_R9.json` 的 `proposal` 可直接參考。
- 前測不要放進 `single`／`songs`。前測有幾份錄音，`pretest_midi` 就給幾個檔名（依錄音檔名排序對應）。

## 3. 一鍵跑完
```bash
python run_night.py R9 --name 第十次的包_暫定 --ordinal 第十次 --date 2026-11-06
```
| 步驟 | 內容 | 輸出 |
|---|---|---|
| 1 label | 自動標記（10/27 規則） | `第九次的包_暫定/Data/自動標記_20261105/` |
| 2 live | **今天現場實際判斷的盲測成績**（讀現場 `predictions.csv`） | `out/score_live_R9.md` |
| 3 replay | 同一份前測用第九次包重播（應與 2 幾乎相同） | `out/score_replay_R9.md` |
| 4 build | 第 4～9 次重新訓練（第九次包同一套做法） | `out/model_第十次.joblib` |
| 5 package | 建 `第十次的包_暫定`：裝模型、改指紋、凍結、測試、GUI 測試 | `out/package_第十次.json` |
| 6 full | 第十次包重播全部前測（開發分數） | `out/score_full_第十次.md` |

總表：`out/night_R9.md`。某一步失敗會停下，記錄在 `out/log_night_R9_<步驟>.log`；修好後用 `--skip 1,2,3` 跳過已完成的步驟。

加 `--lodo` 會再做六天留一天（每天各訓練一次，約多 25 分鐘），輸出 `out/score_lodo_第十次.md`；時間不夠可以 11/06 之後再跑。
如果 `第十次的包_暫定` 已經存在（例如重跑），`make_package.py` 會拒絕覆蓋；確認裡面 `Data/` 沒有現場資料後，手動加 `--replace` 重跑第 5 步。

## 4. 隔天
- 用 `第十次的包_暫定` 照平常流程：`00_CHECK.bat` → `1_測前次.bat`。
- 11/06 晚上同樣流程：`rounds.json` 填 `R10`（`package` 改 `第十次的包_暫定`），再跑 `run_night.py R10 --name 第十一次的包_暫定 ...`（如需要）。

## 大約時間（這台電腦）
配對 1～2 分鐘；標記約 3 分鐘；重訓 7.4 分鐘；建包＋測試約 4 分鐘；重播全部前測約 5 分鐘。不含留一天約 20～25 分鐘。留一天每天約 3 分鐘訓練，共六天，再加約 5 分鐘評分。

## 已驗證（2026-11-01 演練，用 10/27 當作「新的一天」）
| 檢查 | 結果 |
|---|---|
| `pair_midi_raw.py`（10/27 的 16 個 MIDI、22 份 Raw） | 每個 MIDI 都配到正確的 Raw，吻合 0.977～1.0；作廢的重錄都排在第二 |
| `autolabel_day.py`（設定檔 `dryrun_1027.json`，輸出到 `out/dryrun_labels_1027`） | 12 份標記與 10/27 正式標記逐筆相同（0 差異），5,741 下 |
| `rounds_data.load_round` | 訓練列、104 維特徵、前測答案都與第 8 次快取完全相同 |
| `score.py --live`（10/27 現場 `predictions.csv`） | 246/301＝81.7%（去重複觸發後；現場當時記 245/303） |
| `score.py`（第九次包重播） | 1729/1795＝96.3%，與之前的驗證相同 |
| `build_model.py --days none`（用第 4～8 次重建第九次模型） | 19,358 下；七份前測每一下的判斷與第九次包相同（1729/1795、未達 5/92 格）。模型位元組不同（GPU 訓練與說明文字），判斷結果相同 |
| `make_package.py`（演練包 `第十次的包_演練`） | 凍結、參考檔、16/16 測試、GUI 測試、自檢全部通過 |
| 留一天（`build_model.py --hold D` ×5＋`score.py --lodo`，現場引擎） | 1582/1795＝88.1%，每天都與研究台相同（R4 273、R5 156、R6 127、R7 769、R8 257） |

注意：`第十次的包_演練` 只用於演練，它的模型等於第九次包，不要拿去現場。正式的 `第十次的包_暫定` 由 11/05 當晚的 `run_night.py` 產生。
