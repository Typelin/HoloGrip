# HoloGrip 兩階段推論：現行能力 93.5%

日期：2026-08-21  
資料：Song 2，`S20260812_P01_song02_T1127_cleaned`  
評估集：單手真值 307 筆（全曲 311 含 4 筆雙手齊擊，產品評估排除雙手）  
推論輸入：只有 ESP32 + JY901S 100 Hz IMU。MIDI／影片只當老師，不進現場模型。

可播 Demo（即時正確率，播完定格全曲數字）：

[HoloGrip_Song2_產品Demo_即時正確率_0821_ZH_TW.html](file:///C:/Users/Typelin_Station/Desktop/HoloGrip/Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/HoloGrip_Song2_產品Demo_即時正確率_0821_ZH_TW.html)

---

## 0. 現行能力

Song 2 離線、無 MIDI 推論，產品端到端（手+鼓都對）**287／307 = 93.5%**。

這不是重訓鼓位模型得到的，也不是把「有沒有打」犧牲掉去換「打哪一顆」。現場兩階段架構：先判斷有沒有打，再把該手峰值窗口丟進 7 類鼓位模型。手別來自韌體 `HAND_ID`。

請不要把這個 93.5% 和手別 MLP 的 5-Fold 93.48% 混在同一張表。後者是「已知有事件，判左手／右手」的論文基準；產品要解的是「有打且打哪一顆」。

---

## 1. 系統切成兩段

```text
手套 100 Hz IMU
    │
    ▼
┌─────────────────────────┐
│ 第一段  HitDetector     │  這一瞬間算不算打？
│ 局部峰值 + 2.3g 門檻    │  輸出：trigger_ms, peak_ms, peak_g
│ 重打 100 ms／輕打 150 ms│
│ ≥4g 旁路抬手／水平濾波  │
└───────────┬─────────────┘
            │ 只在「有打」時往下走
            ▼
┌─────────────────────────┐
│ 第二段  7 類鼓位 MLP    │  這一打是哪一顆？
│ 該手 peak ±80 ms 窗口   │  小鼓／三顆 Tom／HH／Crash／Ride
│ 17 維 IMU 特徵          │
└─────────────────────────┘
手別 = 封包 HAND_ID
```

評估：真值 `csv_center_ms`、同手、±90 ms 配對。配對成功才看鼓名。

| 指標 | 定義 | 本曲 |
|---|---|---|
| 打擊召回 | 配對成功 / 307 | **301／307 = 98.0%** |
| 打擊精確率 | 配對成功 / 系統輸出 337 | **301／337 = 89.3%** |
| 條件鼓位正確率 | 鼓名對 / 配對成功 | **287／301 = 95.3%** |
| **產品端到端** | 手+鼓都對 / 307 | **287／307 = 93.5%** |
| 鼓位錯 | 打到了但鼓名不同 | 14 |
| 多偵測 | 系統有、真值沒配到 | 36 |
| 漏打 | 真值有、系統沒配到 | 6 |

精確率的分母含曲前熱身與 GT 未標到的實打（例如 00:23–00:25 的 7–14g）。那些不是現場誤觸的主體。

---

## 2. 現行偵測器參數

採集 UI 預設不動（歷史 `hit_events` 定義）。產品線（離線重放、現場 `LiveRecognizer`）走 `PRODUCT_DETECTOR_KWARGS`：

| 參數 | 值 | 作用 |
|---|---|---|
| `mag_min` | 2.3g | 擋低 g 幽靈 |
| `debounce_heavy_s` | 100 ms | 放行 Crash／重打連擊 |
| `debounce_light_s` | 150 ms | 放行 Hi-Hat／Ride 密打 |
| `motion_bypass_mag` | 4.0g | 大力側向擊不當成抬手／水平而拒絕 |

鼓位模型檔 `hologrip_song2_imu_zone_classifier.joblib` 維持訓練時權重。條件鼓位能到 95.3%，是因為送進模型的 ±80 ms 窗口對準真打峰值：幽靈較少搶 ±90 ms 真值槽，連打取到的是這一擊而不是上一擊的尾。

程式：

- `Apps/Song_Collection_COM/product_hit_and_zone.py` → `PRODUCT_DETECTOR_KWARGS`
- `Apps/Song_Collection_COM/song_collection_server.py` → `HitDetector`
- 評估 JSON：`Song2_product_hit_zone_eval_0821.json`

---

## 3. 剩餘錯誤（本曲上限）

漏打 6 筆，全在 00:39–00:51，峰值 2.48–3.81g（低於 4g 旁路）：左手 Hi-Hat 1、Crash 5。

鼓位錯 14 筆，幾乎是空間鄰居：落地 Tom ↔ 中音 Tom、中音 Tom ↔ 高音 Tom、Crash ↔ 小鼓。這是下一階段分類器的靶，不是再調門檻。

多偵測 36 筆裡含曲前 4、GT 沒標的實打約 11、對側共振與其他。補標 00:23–00:25 會讓精確率看起來更高，那是標記覆蓋，不是偵測器回退。

---

## 4. 兩個 93% 不要混

| | 產品 93.5% | 手別論文 93.48% |
|---|---|---|
| 任務 | 有打且打哪一顆（7 類） | 已知事件，判 L／R |
| 輸入 | 單手擊中窗口 | 雙手事件窗口 |
| 現場 | **用這套** | 不用（手別已在封包） |

---

## 5. 現場

`Apps/Song_Collection_COM/run_live_hit_and_zone_0821_ZH_TW.bat`

重產 Demo：`Apps/Song_Collection_COM/run_raw_csv_hit_zone_demo_0821_ZH_TW.bat`