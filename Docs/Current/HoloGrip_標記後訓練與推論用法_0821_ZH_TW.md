# HoloGrip 標記後：怎麼訓練、怎麼用、思路有沒有偏

更新日期：2026-08-21  
狀態：Song 2 真值已完成；MLP 基準已訓完並落碟。缺的是「推論接線」，不是再蒐一次孤立點擊。

---

## 先給結論

1. **思路沒有走錯。** 從「每顆鼓打滿 X 下 → KNN → 立刻 DEMO」改成「實際演奏 → MIDI 對齊 → 人工真值 → 事件窗口分類」是正確方向。
2. **現在已經有可載入的模型。** 不是空白等訓練。最佳模型是 **MLP (64, 32)**，5-fold 準確率 93.48%。
3. **93% 不是現場準確率。** 那是同一首歌、打亂後的交叉驗證。能證明「這批真值上分得開左右手」，不能證明「換一首歌、換一天、現場 MIDI 觸發也一樣」。
4. **下一步不該再訓一次同一套。** 下一步是把 `joblib` 接到「MIDI 觸發 → 切 ±80 ms → 抽同一組特徵 → predict」。先離線重播 Song 2，通過再上手套。

---

## 舊 KNN 與現在 MLP：不是同一個題目

```text
舊版（空氣鼓 DEMO）
  左手自己一個 KNN，右手自己一個 KNN
  輸入：打擊瞬間 3 維  yaw / pitch / swing_depth
  輸出：這隻手打到哪一顆鼓（Zone 0~6）
  資料：每顆鼓孤立點滿 N 下 → 同場立刻 DEMO

現行（電子鼓 + 雙手手套）
  一個分類器同時看左右手窗口
  輸入：MIDI 擊中前後 ±80 ms 的 IMU 統計（22 維）
  輸出：這一筆 MIDI 是左手還是右手
  資料：實際打整首 Song 2 + 影片真值
```

舊版效果不好，原因對得上，不是「沒換成深度學習」：

| 舊做法 | 為什麼現場會垮 |
|---|---|
| 孤立點擊當訓練 | 真打會重疊、回彈、預備動作，分佈不一樣 |
| 每手各訓各的 | 從不比較「這一刻哪隻手能量比較大」 |
| 3 維姿態猜鼓位 | 電子鼓 MIDI 已經知道鼓位；手套該補的是手別 |
| 同場 train 完立刻 DEMO | 等於拿剛才剛看過的動作考自己 |
| KNN | 對抖動敏感，推論還要翻整份樣本 |

MIDI **沒有左右手**。手套的工作是補這一個缺口。鼓位交給 MIDI，手別交給 IMU。這才是能落地的產品切法。

---

## 訓練這一步：其實已經做完

腳本：`Apps/Song_Collection_COM/train_hand_classifier_0821_ZH_TW.py`

```text
真值 CSV（311 筆，單手 307）
        +
Raw 100Hz 左右手 CSV
        │
        ▼  每筆 MIDI：csv_center_ms ± 80 ms
22 維特徵（左 8 + 右 8 + 不對稱 4 + velocity + drum_zone）
        │
        ▼  5-Fold Stratified CV
比較 RF / SVM / MLP / GB / LR
        │
        ▼  用全資料再 fit 一次最佳模型
hologrip_song2_best_hand_classifier.joblib
```

若要重跑（通常不必）：

```powershell
python Apps/Song_Collection_COM/train_hand_classifier_0821_ZH_TW.py
```

或帶舊 KNN 對照的互動版：

```powershell
python Apps/Song_Collection_COM/run_complete_model_benchmark_interactive.py
```

產出：

- 模型：`Data\Derived\...\hologrip_song2_best_hand_classifier.joblib`
- 報告：`model_benchmark_report_0821.json`

**不要**為了「感覺要訓練」再打一輪每顆鼓 200 下。那會退回舊題目。

---

## 訓練出來的模型怎麼用

模型不會自己聽手套。它只吃 **22 個數字**，吐出 `0=左 / 1=右`。

```text
現場（或離線重播）
  ① 觸發：電子鼓 MIDI onset（現階段唯一可靠觸發）
  ② 對齊：同一套時間軸上取左右手 IMU [t-80ms, t+80ms]
  ③ 特徵：必須呼叫與訓練相同的 extract_features
  ④ 推論：clf.predict([feat]) → L 或 R
  ⑤ 雙手組：MIDI 差 ≤ 12 ms 的兩筆，各自預測、不要合成一筆亂標
```

偽代碼：

```python
import joblib
clf = joblib.load(r".../hologrip_song2_best_hand_classifier.joblib")

# feat = 與 train_hand_classifier_0821 完全同一條特徵函式
pred = clf.predict([feat])[0]          # 0=L, 1=R
proba = clf.predict_proba([feat])[0]   # 把握度
```

三條硬規則：

1. **特徵函式只能有一份。** 訓練與推論複製兩套數字，現場一定漂。
2. **現階段觸發用 MIDI，不要用舊版單手 peak。** 舊 peak 是為了猜鼓位；新手別模型假設「已經知道這一擊發生」。
3. **`velocity` 與 `drum_zone_id` 現場也要來自 MIDI。** 產品若永遠接電子鼓，這不是作弊。若哪天要「沒有 MIDI、純手套」，這兩維必須拿掉重訓。

特徵重要性前四名全是雙手能量差（`mean_mag_diff`、`energy_diff`…），物理上就是「哪隻手比較用力」。這表示模型主要靠 IMU，不是靠鼓種偷分。

---

## 現在缺的那一截（所以才覺得「還不能跑」）

```text
[已有] 真值 311
[已有] 訓練腳本 + joblib
[已有] 3D / 影片重播（看標記，不是看模型）
[沒有] 離線：對 Song 2 每筆 MIDI 跑 predict，對答案
[沒有] 線上：MIDI + 即時 IMU → 閃左手/右手
```

所以不是「做不出能跑的模型」，是 **模型在碟上，推論迴路還沒焊**。

建議閘門（一次只過一關）：

1. **離線重播 Song 2**：用同一首的 MIDI + CSV 跑模型，對 `final_hand`。這關不過，現場不必上。
2. **新打一小段（同一人、同一套手套）**：測的是「換一次演奏還能不能用」，不是再做 5-fold。
3. **現場 DEMO**：電子鼓 MIDI 當觸發，畫面只顯示手別。

CNN / LSTM 現在不要上。307 筆單手樣本撐不住時序網路；那是 10 首以上的事。

---

## 「算不算打」與「打哪／哪隻手」是兩層

主人直覺對：**先判定這一下算不算打，再分類。** 現行 MLP **沒有**做第一層。它假設觸發已經發生。

```text
第 0 層  觸發（onset）     這一下算不算打？
第 1 層  鼓位（zone）      打到哪一顆？
第 2 層  手別（hand）      哪隻手？
```

| 層 | 舊空氣鼓 DEMO | 現行 Song 2（電子鼓） | 現況 |
|---|---|---|---|
| 0 算不算打 | 有。`HitDetector`：mag>1.7、局部峰值、防彈跳、濾抬手 | **電子鼓 MIDI onset** | 手套偵測器還在採集程式裡，**沒接到 MLP** |
| 1 打哪顆 | 舊 KNN（yaw/pitch/揮幅）效果差 | **MIDI note → 鼓位** | 不缺，只要接電子鼓 |
| 2 哪隻手 | 不需要（每隻手套自己報） | **MLP 22 維** | 已訓、未接推論 |

所以不是「完全沒有打擊判斷」，是 **新模型把第 0／1 層外包給 MIDI**，只訓第 2 層。

兩條產品線不要混：

1. **接電子鼓（現在該走的）**：MIDI = 算打 + 打哪顆；手套 MLP = 哪隻手。3D 頁現在就是這條的重播。
2. **不接鼓、純手套**：必須把舊 `HitDetector` 接回來當第 0 層，再決定第 1 層還要不要猜鼓位。這是另一個專案切片，不是現在 93% 那個模型能單獨完成的。

打擊判斷程式還在：

- `Apps/Song_Collection_COM/song_collection_server.py` → `HitDetector`
- 條件：峰值 mag > 1.7g、局部最大、debounce 150/250 ms、排除抬手與水平揮

它現在只負責**採集時寫 hit_events**，沒有餵給 MLP。

---

## 必須承認的限制（論文／老師會問）

- 5-fold 有打亂，同一樂句相鄰鼓點可能漏進訓練與測試 → **數字偏樂觀**。
- 只有一首歌、一個人。
- 4 筆雙手齊擊沒進分類器。
- joblib 是 sklearn MLP，CPU 推論沒問題，不是嵌入式韌體模型。

這些都不否定方向。它們只說明：先做離線重播，再談現場準確率。