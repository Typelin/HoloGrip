# HoloGrip 交接手冊（給第一次接手的人或 AI）

> 最後更新：2026-10-29（第九次包加入歸零方向保護）。目的：只讀這一份，就能知道專案在做什麼、檔案在哪、怎麼重現所有數字、哪些坑不要再踩。
> 細部時間軸見 `Research/AutoLabel_1027/研究歷程_20261027_ZH_TW.md`；較早的進度見根目錄 `最新進度追蹤_ZH_TW.md`。

---

## 1. 專案一頁說明

**做什麼**：鼓手雙手戴感測手套（WitMotion IMU，100 Hz），只靠手套資料即時判斷每一下打的是哪顆鼓、哪隻手。電子鼓的 MIDI 只當答案（標記與計分），**現場推論時絕不讀 MIDI**。

**7 顆鼓 × 2 手 = 14 格**：小鼓、高音 Tom、中音 Tom、落地 Tom、Hi-Hat、Crash、Ride（左 L／右 R）。

**硬性規則（使用者定的，不可違反）**
1. 前測（每次現場第一份「每鼓左 20／右 20」的錄音）**永遠不進訓練**，只當考卷。
2. 照時間軸即時判斷：每一下只能用這一下＋打擊後最多 80 ms，以及**之前**的打擊；不得用整段錄音、未來打擊或 MIDI。
3. 不用地磁／絕對方位當特徵（感測器左右角度只看「相對歸零」）。
4. 訓練與大量運算一律用 GPU。
5. 回報時把「全部訓練（開發分數）」和「留一天（誠實跨天）」分開，不把開發分數講成盲測。
6. 不刷韌體；原始 Raw／MIDI 永不修改。
7. GitHub repo `Typelin/HoloGrip` 是**公開**的：不上傳錄音、MIDI、影片、模型檔、各包的 `Data/`、`QA/`、`Model/`、研究的 `out/`。
8. 不要增加現場步驟（現場流程固定：歸零 → 左右各 10 下小鼓 → 照常打）；能由系統容錯的就由系統做。
9. 準確率優先；不用「模擬」或假設情境下結論，要用真實的 4～8 次資料考。
10. commit 格式 `feat|fix|docs|chore: 描述`，結尾 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`；commit 前人工看 `git diff --cached` 沒有憑證；**只在使用者要求時 commit／push**。

**現況（2026-10-29）**
- 第九次包（`第九次的包_暫定/`）：五場前測（第 4～8 次）**全部訓練 96.3%**、**留一天 88.1%**（答案已去重複觸發）。
- 歸零方向保護（推算 v3）：**已裝進第九次包**（模型指紋 29f8f2d4…）；現場程式逐筆重播全部 40 份錄音 21102／21466＝98.3%，前測 1729／1795 不變（見 §8）。
- 第八次包已還原為 10/27 現場使用的版本，**不要再改它**。

---

## 2. 環境

| 項目 | 值 |
|---|---|
| 位置 | `C:\Users\Typelin_Station\Desktop\HoloGrip`（git repo；分支 `wip/0919-hologrip-vnext`） |
| Python | 3.12.10（`%LocalAppData%\Programs\Python\Python312\python.exe`；包的 `.bat` 也用它，找不到時用 `py -3.12`） |
| 套件 | numpy 2.4.6、scipy 1.18.1、scikit-learn 1.9.0、xgboost 3.4.1、torch 2.11.0+cu128、joblib 1.5.3、mido 1.3.3、pyserial 3.5（包內 `Program/requirements.txt`） |
| GPU | RTX 5090 Laptop 24 GB（xgboost `device='cuda'`、torch MLP） |
| 推送 | SSH 沒金鑰，用：`git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push https://github.com/Typelin/HoloGrip.git wip/0919-hologrip-vnext:wip/0919-hologrip-vnext` |
| 硬體 | 兩隻手套，COM4＝右、COM5＝左（**手別以封包內 `HAND_ID` 為準**，不是 COM 順序）；USB 序列 460800 baud、100 Hz |

**跑 Python 一律加 `PYTHONIOENCODING=utf-8`**（路徑和輸出都有中文）。研究腳本從自己的資料夾執行（`cd Research/AutoLabel_1027 && python xxx.py`）。

---

## 3. 檔案地圖

### 3.1 各次現場包（每次一個資料夾）
| 資料夾 | 狀態 |
|---|---|
| `第四次的包`、`第五次的包_暫定2026-10-13`、`第六次的包_暫定2026-10-16`、`第七次的包_暫定2026-10-20` | 歷史；內含當次錄音 `Data/` |
| `第八次的包_暫定2026-10-27` | **10/27 現場用的版本，已凍結**；`Data/` 有 10/27 全部錄音與標記 |
| `第九次的包_暫定` | **目前最新**，日期未定；`Data/` 空的 |

一個包的結構：
```
00_CHECK.bat           → Program/seventh_entry.py check   （軟體自檢）
1_測前次.bat           → seventh_entry.py live            （即時判斷＋同步存 Raw；前測、實測用）
2_蒐集這次.bat         → seventh_entry.py collect         （只收 Raw 100 Hz）
3_檢查這次MIDI.bat     → seventh_entry.py midi
4_對齊MIDI與CSV.bat    → seventh_entry.py align
5_開啟資料夾.bat
Model/venue_free_committee_1023.joblib   模型（名稱的 1023 是版本代號，沿用）
Model/detector_venue_free_1p5g.json      打擊偵測參數
Program/                                 全部程式（見 §5）
package_manifest.json                    檔案指紋（啟動時逐一比對，不符就拒絕執行）
Data/Raw  Data/模型實測  Data/全程監聽  Data/對齊與異常報告  Data/MIDI
QA/                                      測試輸出（不上傳）
```

### 3.2 10/27（第 8 次）資料
| 東西 | 位置 |
|---|---|
| 前測錄音＋現場預測 | `第八次的包_暫定2026-10-27/Data/模型實測/S20261027_P01_sevendrum_take01_.../`（raw CSV、`predictions.csv`、`session_manifest.json`） |
| 歌曲、四鼓、單手 Raw | `第八次的包_暫定2026-10-27/Data/Raw/`（檔名開頭是舊場次名 `S20261020_...`，不影響內容） |
| MIDI | `C:\Users\Typelin_Station\Desktop\今天\`（`測試1.mid`＝前測、`測試2改.mid`＝歌曲、`測試3改.mid`＝四鼓、`crash L.mid`…單手） |
| 影片 | 同上資料夾 `20261027_115924.mp4`（歌曲，對 song01）、`20261027_123510.mp4`（四鼓，對 song02_take02） |
| 自動標記 | `第八次的包_暫定2026-10-27/Data/自動標記_20261027/*_自動標記_20261027.json` |

MIDI ↔ Raw 對應（已核對，吻合 93～99%）：

| MIDI | Raw（Data/Raw 檔名中段） |
|---|---|
| 測試2改 | `P01_song01_take01_..._115924` |
| 測試3改 | `P01_song02_take02_..._123510` |
| ride L／R | `P01_L_Ride_take01`／`P01_R_Ride_take01` |
| fioor ton L／R（落地） | `P01_L_GND_take02`／`P01_R_GND_take01` |
| ton2 L／R（中音） | `P01_L_MID_take01`／`P01_R_MID_take02` |
| ton1 L／R（高音） | `P01_L_HI_take01`／`P01_R_HI_take01` |
| crash L／R | `L_Crash_take02`／`R_Crash_take01` |

作廢（中途重錄，不用）：`song02_take01`、`L_GND_take01`、`L_Crash_take01`、`R_MID_take01`（三份）。

### 3.3 研究資料夾（`Research/`）
| 資料夾 | 內容 |
|---|---|
| `AlignmentAudit_20261020` | 建立第 4～7 次研究快取（`實驗快取.joblib`）、保守 IMU 標記 |
| `AutoLabel_v2_20261021` | 手別自動標記 v2（`hand_autolabel_v2.py`；`out/loro.joblib` 是訓練資料） |
| `BestModel_20261021` | 第八次包模型研究：`common.py`（訓練）、`hit_clf.py`（打擊判斷）、`lodo_*.py`、`verify_1023.py`（第八次包重播） |
| `AutoLabel_1027` | **10/27 之後的全部研究**（標記、五天留一天、第九次包、歸零推算），見 §6 |

### 3.4 共用快取（都不在 git）
| 檔案 | 內容 | 由誰產生 |
|---|---|---|
| `第七次的包_暫定2026-10-20/Data/實驗_姿態模型自動標記_20261020/實驗快取.joblib` | 第 4～7 次每份錄音的 events／raw_signals／items（2.1 g 偵測＋104 維特徵）、前測 truth、`detector` 參數 | `AlignmentAudit_20261020/auto_label_experiment.py prepare` |
| `Research/BestModel_20261021/out/dataset2.joblib` | 第 4～7 次訓練列 13,617 下（x 104 維、hand、drum、session、group、g） | `BestModel_20261021/build_features2.py` |
| `Research/BestModel_20261021/out/hit_cand_1p5.joblib` | 每份錄音 1.5 g 候選峰（g、det 18 維、x、ctx） | `hit_cand.py 1p5` |
| `Research/BestModel_20261021/out/pretest_truth_hands.joblib` | 第 4～7 次前測答案（手別由手套峰值） | BestModel 研究 |
| `Research/AutoLabel_1027/out/r8_cache.joblib` | 第 8 次的候選峰＋5,741 訓練列 | `lodo5.py`／`final_1027.setup()` 自動建立（刪掉會重建） |
| `Research/AutoLabel_1027/out/cache/{full,R4..R8}.joblib` | 各考法的委員（原始 1 種子＋對齊 1 種子）＋前測已接受打擊 | `cache_models_1027.py` |
| `.../out/cache/yawfree_*.joblib`、`layouts_*.joblib` | 不看左右角度的分類器、每天鼓位 | `yawfree_1027.py`、`layouts_1027.py` |

---

## 4. 資料格式

### 4.1 Raw CSV（每列一個感測樣本）
`session_id, sample_index, hand(L/R), song_time_ms, received_time_ms, sensor_time_ms, packet_id, ax_g, ay_g, az_g, accel_magnitude_g, yaw_deg, pitch_deg, roll_deg, cal_yaw_deg, cal_pitch_deg`
- `yaw_deg` 是感測器自己的方向（陀螺積分，**無地磁**，每次連線起點任意）；`cal_yaw_deg`＝相對歸零。
- 同目錄 `*.manifest.json`：`calibration`（歸零當下的 yaw／pitch）、`zero_quality`（歸零 0.5 秒內的晃動 `yaw_std` 等）。
- 程式讀入後：`absolute_yaw`＝yaw_deg、`yaw`＝相對歸零；**歸零 yaw＝absolute_yaw − yaw**。

### 4.2 MIDI 音符 → 鼓（`midi_label_pipeline.py` ZONE_BY_NOTE）
37/38/40→小鼓、48→高音、45→中音、43→落地、46→Hi-Hat、49→Crash、51/59→Ride；36（大鼓）、44（踏板）是腳，排除。

### 4.3 104 維特徵（`venue_free_runtime_1020.feature`）
- `x[0:83]`：軌跡摘要（擊打前 240 ms～後 80 ms，以手自己的方向為準，與歸零方向無關）。
- `x[83:104]`：7 個時間點（相對峰值 −120、−80、−40、−20、0、+20、+40 ms）的「相對歸零旋轉」rotvec ÷ π。公式 `Hd·Rt·A⁻¹·Hd⁻¹`（`zero_relative_rotation`）。
- 由旋轉區塊取擊打瞬間的左右角／上下角：`LF.yp(X)`（`lodo_full.yp`：第 5 個時間點 ×π → euler ZYX）。
- **歸零左右偏 δ ⇔ 每個 `x[83:104]` 旋轉左乘 Rz(−δ)**；`x[0:83]`、上下角、翻滾不變（§8 的基礎）。

### 4.4 打擊判斷特徵（126 維）
`[peak_g, 偵測器形狀 18, x 104, ctx 3]`，ctx＝另一手 ±40 ms 最大 |a|、log1p(距上一候選 ms，上限 3000)、g／上一候選 g。研究端 `hit_clf.feats(c)`，現場端 `runtime.hit_probability`，兩者順序相同。

### 4.5 自動標記 JSON（`Data/自動標記_20261027/*.json`）
`labels[id] = {mark: L|R|exclude|review, drum, source, rule, comment}`；`events[]` 含 `midi_ms`、`csv_ms`（已對齊）、`vel`、`training_peak_ms`（擁有的偵測峰，null＝沒有進訓練）。

### 4.6 模型檔（`Model/venue_free_committee_1023.joblib`，第九次包）
主要欄位：`portable`（6 個委員 all_0..all_5，每個含 L／R 的 XGBoost JSON＋numpy MLP）、`member_weights`（all_0..2 原始各 0.133、all_3..5 對齊各 0.2）、`classes`（**順序是 `['Crash','Hi-Hat','Ride','中音 Tom','小鼓','落地 Tom','高音 Tom']`，跟研究程式的 DRUMS 順序不同**）、`drum_rotation_means`（對齊後各手各鼓的旋轉均值）、`adaptation`（confidence 0.8、min_confident_hits 20、passes 2）、`position_prior`（σ 12°、權重 0.5）、`hit_gate`（XGBoost、門檻 0.4、126 維）、`detector_config`（1.5 g）、`training_events` 19358。

---

## 5. 現場判斷流程（第九次包 `Program/venue_free_runtime_1023.py` 的 `StreamEngine`）

每隻手各自處理，每個候選峰在峰後 80 ms 釋出：
1. **候選**：`causal_hit_detector_1016`，1.5 g 門檻。
2. **打擊判斷**：`hit_probability ≥ 0.4` 才算真打擊；否則丟掉、不進任何歷史。
3. **委員判鼓**：6 個委員加權平均（`probabilities`）。
4. **逐鼓位置校正**（2 輪）：用這手**之前**、分數 ≥ 0.8 的打擊，算「各鼓的訓練均值 − 本段觀測均值」的加權平均，平移旋轉區塊；累積 20 下有把握的才開始。
5. **自動錨點**：每手前 5 下先暫存；第 5 下時，若 5 下姿勢都在中位數 8° 內且模型都判小鼓第一 → 當小鼓錨點；否則若都判同一顆鼓、平均分數 ≥ 0.7 → 用那顆鼓當錨點；都不是就照常累積（`ANCHOR_RULE`）。錨定後立刻開始校正。
6. **位置先驗**：最終機率 × exp(−d²／2σ²)^0.5，d＝這下（校正後）左右／上下角與各鼓訓練位置的距離，**左右角用圓周差**。
7. **姿勢連續**：分數 < 0.9、同手上一下在 600 ms 內、角度差 ≤ 16°、上一顆鼓在這一下前兩名 → 沿用上一顆鼓（`CONTINUITY`）。

**指紋凍結機制**：`seventh_core.load_components()` 會比對 `package_manifest.json` 的模型／程式指紋，以及 `seventh_core.MODEL_SHA256`。改了模型或程式之後必須：
1. 更新 `seventh_core.py` 的 `MODEL_SHA256`（若換模型）與 `freeze_seventh_package.py` 第 16 行的 assert 指紋；
2. 執行 `python Program/freeze_seventh_package.py` 重寫 manifest；
3. 再跑測試。

---

## 6. 研究程式（`Research/AutoLabel_1027/`）怎麼用

| 檔案 | 用途 | 跑法／輸出 |
|---|---|---|
| `autolabel_1027.py` | 10/27 自動標記（單手照規則；歌曲／四鼓：保守規則＋v2 手別＋排除規則） | `python autolabel_1027.py` → `第八次.../Data/自動標記_20261027/` |
| `truth_clean_1027.py` | 去掉電子鼓重複觸發（同鼓 150 ms 內、手套只有一下） | 函式 `dedupe(truth, cand)` |
| `lodo5.py` | 第 8 次接進第 4～7 次流程：R8 候選、訓練列、前測答案（`r8_cache.joblib`） | 被其他腳本 import |
| `final_1027.py` | `setup()`：載入全部訓練列與答案；五天留一天＋全部訓練評分 | 多數腳本的入口 |
| `final_1027_rc.py` | 對齊（`recenter`）＋訓練 | |
| `mix_1027.py`、`auto_anchor_1027.py`、`song_anchor_1027.py` | 混合／雙模型、自動錨點、歌曲錨點的留一天比較 | `out/*.json` |
| `cache_models_1027.py` | 為 full／R4..R8 訓練並快取委員（每組原始＋對齊各 1 種子） | `out/cache/*.joblib`（約數十分鐘） |
| `exp_1027.py` | **快速實驗台**：`evaluate(tag, opts)`、`summary()`；可模擬歸零偏差（`zero_err`）與各種推算選項 | 秒～分鐘級 |
| `trace_1027.py` | 逐下追蹤某份前測某隻手的判斷過程 | |
| `yawfree_1027.py` | 不看左右角度的分類器（歸零推算用） | `out/cache/yawfree_*.joblib` |
| `verify_9th.py` | 第九次包**現場程式逐筆重播**五場前測（去重複答案），寫 `out/replay_9th/`（包測試的參考檔） | 期望 1729／1795 |
| `eval_all_9th.py` | 第九次包模型批次考第 4～8 次**全部 40 份錄音** | 期望全部 20636／21466 |
| `zero_tolerance_9th.py` | 真實重播＋只移歸零的容錯測試 | |
| `layouts_1027.py`、`layout_geom_1027.py`、`diag_pretest_left.py` | 鼓位診斷 | |

`exp_1027.py` 的最佳設定（＝第九次包）：
```python
o = dict(E.DEFAULT, w_rc=0.6, anchor_top=1, conf=0.8, pos_w=0.5, pos_sigma=12.0,
         pos_all=True, anchor_any=True, anchor_minp=0.7)
```
歸零推算 v3 額外選項：
```python
AZ = dict(auto_zero=True, az_consistent=True, az_mode='yawfree', yf_conf=0.9, yf_agree=0.5,
          yf_round=15.0, yf_drums=1, az_verify=True, yf_thr=40.0, az_margin=0.0, yf_max=60.0, az_persist=10)
```

---

## 7. 重現關鍵數字（照做應得到相同結果）

```bash
cd /c/Users/Typelin_Station/Desktop/HoloGrip/第九次的包_暫定
PYTHONIOENCODING=utf-8 python Program/test_seventh_package.py      # 16 tests OK
PYTHONIOENCODING=utf-8 python Program/test_seventh_native_gui.py   # "passed": true

cd ../Research/AutoLabel_1027
PYTHONIOENCODING=utf-8 python verify_9th.py      # R4 278/290 R5 158/160 R6 135/140 R7 860/904 R8 298/301，合計 1729/1795，heading +90 identical: True
PYTHONIOENCODING=utf-8 python eval_all_9th.py    # out/eval_all_9th.json；全部 20636/21466（96.1%）
```
留一天（研究台，1 種子，去重複答案）：
```python
import exp_1027 as E, truth_clean_1027 as TC, final_1027 as FB
FB.setup(); _l = E.load
E.load = lambda t: (lambda M: (M.__setitem__('truth', {k: TC.dedupe(v, FB.H.CAND[k])[0] for k, v in M['truth'].items()}), M)[1])(_l(t))
res = {}
for D in E.DAYS: res.update(E.evaluate(D, o, days=[D]))
# 合計 1582/1795 = 88.1%（R4 273/290、R5 156/160、R6 127/140、R7 769/904、R8 257/301）
```

| 指標 | 數字 |
|---|---|
| 10/27 現場（第八次包）前測 | 245／303＝80.9%（左 69.1%、右 94.7%） |
| 第九次包，全部訓練，五場前測 | 1729／1795＝96.3% |
| 第九次包，留一天，五場前測 | 1582／1795＝88.1% |
| 第九次包，全部 40 份錄音 | 20636／21466＝96.1%（前測 96.3、訓練錄音 96.1；左手 94.5、右手 98.0） |
| ＋歸零推算 v3（研究台） | 21102／21466＝98.3%（只有 2 份改變；前測、留一天不變） |

---

## 8. 目前判斷邏輯的由來（為什麼是這些參數）

| 元件 | 解決的問題 | 證據 |
|---|---|---|
| 第 8 次資料加入訓練 | 今天的鼓位（左手 Tom 低、間距壓縮） | 全部訓練今天前測左手中音 4→19、Ride 0→20 |
| 對齊（每段平移到共同鼓位） | 每段「小鼓相對歸零」差 ±30°（左 +2～+67°），同顆鼓落在不同位置 | 只加資料不對齊，今天前測仍 82.4%、左手 Ride 0/20 |
| 原始＋對齊雙模型 0.4／0.6 | 對齊模型前測強、歌曲弱；原始相反 | 留一天：原始 79.2／77.9、對齊 82.7／74.0、雙模型 83.9／75.6（前測／歌曲） |
| 自動錨點（小鼓第一或任意同一顆鼓） | 前 5 下強制當小鼓，遇到開頭不是小鼓（R4 從 Hi-Hat 開始）會錨錯 | R4 左手 Hi-Hat 10→17+／21 |
| 位置先驗 σ12° 權重 0.5（圓周） | 位置明明在 Ride，模型依揮擊動作判 Tom | R5 左中音、R7 左 Ride 達標；留一天 84.3→87.4 |
| 校正信心 0.8 | 用沒把握的打擊估偏移會被帶偏 | 94.6 vs 94.3 |
| 答案去重複觸發 | 電子鼓同一下送兩個同鼓音符 | R4 右 Crash 20/24 → 20/20 |

### 歸零方向保護（推算 v3，已裝進第九次包）
- 問題：10/27 左手落地、中音單手補強歸零時左手方向差約 50°（感測器打小鼓讀數各段都約 150°，差在歸零那一刻），第九次包只對 17/245、23/270。
- 做法：不看左右角度的分類器（`yawfree_1027.py`，約 65% 準）找有把握的打擊 → 「訓練左右角 − 實際左右角」投票 → 過半在 ±15° 內一致、取 15° 倍數、|估計| 介於 40～60°、連續 10 下不變、且修正後完整模型對已打過的打擊較有把握 → 整段歷史一起轉回來，再走原本的校正。
- 結果：40 份錄音只有那 2 份改變（→ 239/245、267/270）；前測、留一天完全不變。
- 現場實作：`venue_free_runtime_1023.py` 的 `_zero_estimate`（投票、連續 10 下、驗證）、`_replay`（推算值改變時用批次重建這隻手的校正統計、錨點、姿勢連續狀態）、`heading_free`、`rotate_zero`；模型檔 `zero_estimator`（每手一個 XGBoost＋參數）；每下輸出多一欄 `zero_correction_deg`。
- 驗證：`stream_takes_9th.py`、`stream_all_9th.py`（現場程式逐筆重播）與研究台每一下一致；10/27 左手落地 239／245、中音 267／270，R6 歌曲 2 不觸發。
- 延遲（這台電腦）：歸零正常時每下約 +3.5 ms（歌曲中位數 22.8 → 26.3 ms）；保護啟動的那段中位數約 38 ms、重建那一下約 0.1 秒。驗證時只在推算值不為 0 時才多算委員，平常不多算。

---

## 9. 踩過的坑（症狀 → 原因 → 避免方式）

1. **批次考試前測只有 40/290**：現場模型 `bundle['classes']` 順序與研究 `K.DRUMS` 不同，機率欄位錯位。→ 用 `[bundle['classes'].index(d) for d in K.DRUMS]` 重排（`eval_all_9th.py` 已處理）。
2. **啟動報「檔案指紋不符」／「不是本次凍結的場地無關模型」**：換模型或改程式沒有更新 `MODEL_SHA256`、freeze 的 assert、manifest。→ 依 §5 三步驟。
3. **git checkout 還原後指紋仍不符**：`core.autocrlf=true` 會把 LF 改成 CRLF。→ 用 `git show HEAD:路徑 > 路徑` 寫回原始位元組。
4. **還原第八次包後，第八次包測試失敗**：第八次包測試的參考重播檔放在 `Research/BestModel_20261021/out/replay_1023`，曾被新模型的重播覆蓋。→ 重跑 `BestModel_20261021/verify_1023.py` 重建；第九次包用自己的 `out/replay_9th`。
5. **把新模型裝進已結束的第八次包**：使用者要求舊包保持現場原狀。→ 新東西一律建新包；舊包只動指紋相關時要先問。
6. **歸零容錯模擬結果全部相同**：同時改了 absolute_yaw 與歸零＝整個場地轉（本來就不變）。→ 只移歸零：`s['yaw'] -= δ`（absolute 不動）；研究台用 `rotate_zero`（左乘 Rz(−δ)）。
7. **「錯過」其實是 MIDI 重複觸發**：Crash／Ride 偶爾同一下送兩個音（相隔 11～130 ms），放寬計分窗沒用。→ `truth_clean_1027.dedupe`（以手套峰值判斷，左右手各一下的快速連打保留）。
8. **串音假音被標成打擊**：打 Tom 時線碰到小鼓產生力度 3～10 的假小鼓；`ton2 L.mid` 每下中音都有假 Hi-Hat。→ 力度 < 20 且 30 ms 內有強 3 倍的另一鼓 → 排除；單手補強非目標鼓全部排除。
9. **單手補強開頭小鼓不一定剛好左 10 右 10**（落地 R 左手打 17 下）。→ 錨點的手別看手套峰值，不數下數。
10. **R6 右手 Ride 在 ±180° 邊界**：位置距離用一般差值會差 358°。→ 左右角一律用圓周差 `(d+180)%360−180`。
11. **歸零推算的陷阱**：用「模型最有把握的方向」會選到假方向（0° 時只剩 66%）；估計中途改變但歷史沒重算會互相矛盾；R6 右手會穩定地估成 90°（加上限 60°）；自然偏差 ±25～30° 會被誤修（門檻 40°）。
12. **縮放＋平移校正、左右上下分開的先驗、每天各自的鼓位樣板、沒把握不回答**：試過都沒有整體幫助（數字在研究歷程）。
13. **姿勢連續會把上一顆鼓的錯誤帶進下一段**（前測換鼓的第一下）；關掉又會讓別處變差。→ 目前保留（分數 < 0.9、前兩名條件）。
14. **給使用者檔案連結**：這個工作階段的工作目錄是 `D:\Antigravity_proj`，HoloGrip 在桌面，相對路徑連結會打不開。→ 用絕對路徑或直接傳檔。
15. **研究參數是看這五場前測挑的**：全部訓練分數偏樂觀；下次新的一天看留一天（約 85～88%）。
16. **研究腳本讀命令列參數要在 import 之前**：`lodo_full` 被 import 時會把 `sys.argv` 改成 `['x','1p5']`，之後再讀參數會拿到 `1p5`。
17. **背景工作 10 分鐘上限**：40 份錄音逐筆重播超過 10 分鐘會被中斷（輸出停在某份錄音、沒有總計）→ 分批跑（`stream_all_9th.py "R4,R5"`）。
18. **大型背景工作**：上百萬次比對的暴力搜尋（MIDI 對齊每 10 ms 試一次）會跑很久 → 改用時間差直方圖找候選位移，再精修。

---

## 10. 未解問題與下一步

**仍未到 17／20 的格子（全部訓練，七份前測逐份看，共 90 格裡 5 格）**：R6 左手中音 5／10（左手中音與落地姿勢只差 2°）；R7 #01 右手中音 14／20；R7 #12（移鼓後）左手小鼓 24／30、右手落地 12／20；R7 #15（移鼓後）右手 Ride 14／20。
- 原因：移鼓後新一段鼓的前幾下落在兩顆鼓的中間（如 #12 右手落地在 −105～−112°，介於學到的中音 −86° 與落地 −143°）；姿勢連續規則把錯誤延續（#12 右手落地 #137、#138 模型已判落地 0.85 仍被改成中音）。
- 試過無效：姿勢連續改成軟記憶（Markov 濾波，前測 93.7～94.8%、未達格 7～10）、連續規則加分數比例條件（92.1～94.0%、未達格 9～11）。目前的硬規則整體仍最好。
- 注意：七份前測要逐份算每格；把 R7 三份合併會掩蓋個別格子。

**下一步候選（依優先序）**
1. 委員從 3＋3 增加到 5＋5（減少整段翻轉），確認延遲 < 80 ms。
2. 歌曲跨天準確率（留一天約 75～78%）是最大提升空間。
3. 補資料：雙手 Hi-Hat 單手補強（10/27 沒錄）、左手 Ride、刻意移鼓位的錄音。

**每次現場後的標準流程**
1. 用 MIDI 對前測算正確率（不訓練）。
2. MIDI ↔ Raw 配對（打擊時間直方圖找位移）、找出作廢重錄。
3. `autolabel` 類腳本建標記（單手照規則；歌曲／四鼓自動＋排除）。
4. 抽查影片幾個案例（同時兩鼓、連打、可疑的同手同時）。
5. 加入訓練列 → 五天（或 N 天）留一天＋全部訓練評分。
6. 新包：換模型、更新指紋、freeze、`verify_*` 逐筆重播、包測試、GUI 測試。
7. 寫研究歷程；使用者同意後 commit／push（不含資料與模型）。
