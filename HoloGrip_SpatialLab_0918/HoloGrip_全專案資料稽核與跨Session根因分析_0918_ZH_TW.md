# HoloGrip 全專案資料稽核與跨 Session 根因分析（2026-09-18）

## 1. 分析範圍

### 第一次正式真實演奏收集：2026-08-05
- Raw CSV: `Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv`
- Raw rows: 74,592（L/R 各 37,296）
- MIDI events: 1,820
- MIDI 品質不作完整真值使用。
- 針對跨 Session 診斷，只使用 `auto_accepted_single_events.csv` 的 247 個高信心單擊事件。

### 第二次正式收集：2026-08-12 Song2
- Raw CSV: `Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv`
- Raw rows: 16,039（L=8,019, R=8,020）
- 清理後 MIDI: 311
- 可用單手訓練事件: 307
- 有人工/影片整理後 Ground Truth，為目前最可靠資料。

### 2026-09-18 現場模擬
- Live session: `HoloGrip_SpatialLab_0918/live_sessions/20260918_141423`
- Repeatability: `HoloGrip_SpatialLab_0918/repeatability_sessions/20260918_143857`

---

## 2. 硬體與資料傳輸層

### PASS
- 08/05：L/R 約 99.96 Hz，packet gap=0，backwards=0。
- 08/12：L 約 99.96 Hz，R 約 99.98 Hz，packet gap=0，backwards=0。
- 09/18：雙 XIAO ESP32-C6 實測可同時約 100 Hz。
- USB 燒錄問題已定位為 Arduino `serial-monitor.exe` 殘留鎖 COM，而不是感測資料品質問題。

### 判定
目前沒有證據支持「USB / 100 Hz 掉包」是鼓位分類崩壞主因。

---

## 3. 歸零系統的真實狀態

Raw CSV 證明每個 session 的 `yaw_offset` / `pitch_offset` 在整場完全固定（std=0），所以軟體 offset 沒有自己漂。

但兩次按下歸零時的實際姿態並不完全一致：

- 08/05 L: yaw0≈178.28°, pitch0≈26.49°
- 08/12 L: yaw0≈-175.78°（circular 約差 5.94°）, pitch0≈9.33°（差 17.16°）
- 08/05 R: yaw0≈18.45°, pitch0≈29.71°
- 08/12 R: yaw0≈3.57°（差 14.88°）, pitch0≈29.55°（差 0.16°）

錄製第一秒仍接近歸零姿勢，可推得 Roll 零位：
- 08/05 L≈+24.58°, R≈-16.55°
- 08/12 L≈+22.41°, R≈-42.68°

右手 Roll 的 session 零位差約 26.1°，但舊 pipeline 完全沒有 Roll 校正。

---

## 4. 舊 Feature Pipeline 的明確問題

舊 17D：
`max_mag, mean_mag, std_mag, energy, max_ax, max_ay, max_az, jerk_max, mean_yaw, mean_pitch, mean_roll, std_yaw, std_pitch, center_yaw, center_pitch, swing_depth, yaw_range`

### 問題 A：Roll 是絕對值，未歸零
跨 session 手套綁法直接進入模型。

### 問題 B：ax/ay/az 是 sensor-local 絕對軸
手套旋轉後，同一物理動作會換成不同三軸分布。

### 問題 C：Yaw 是圓周角，卻直接線性 mean/std/range
`+179° -> -179°` 物理只差 2°，舊 `yaw_range` 可被算成約 358°。
09/18 live 已實際出現此類 wrap 問題。

### 問題 D：Euler 逐軸相減不是完整 3D 相對旋轉
尤其 Pitch 接近 ±90° 時會出現 gimbal-lock / Yaw-Roll 耦合。
09/18 repeatability 固定姿勢 Pitch 約 70–84°，Euler 看起來 Yaw/Roll 爆 50–80°；完整旋轉矩陣判定實際姿態差遠小於該數字。

---

## 5. 左右手混模是第二個明確設計缺陷

08/12 同一鼓、不同手的中心姿態差非常大。

例：
- Crash center yaw: L≈-6.7°, R≈+38.5°（差約45°）
- Crash center pitch: L≈48.8°, R≈20.8°（差約28°）
- Ride center yaw: L≈-35.9°, R≈-95.7°（差約60°）

舊模型沒有 hand feature，等於讓 307 筆資料同時學 7 鼓 × 2 手約 14 個姿態子群。

實驗：
- 舊混模 Group CV: 84.4%
- + hand_flag: 87.3%
- 左右手分模: 88.0%

跨 08/12 -> 08/05：
- 舊混模: 49.4%
- + hand_flag: 56.7%
- 左右手分模: 60.3%

---

## 6. Wrist orientation 不等於 XYZ 鼓位

08/12 右手：
- Hi-Hat: center yaw≈+40.0°, pitch≈17.9°
- Crash: center yaw≈+38.5°, pitch≈20.8°

兩顆真實空間位置不同的鼓，在腕部 Yaw/Pitch 幾乎重疊。

因此：
- IMU 姿態可作分類 feature；
- 但不能把 Yaw/Pitch 直接當作「手在空間中的位置」。
- 舊 3D/角度 UI 只能解讀成姿態參考，不是 Cartesian drum location。

---

## 7. 同一 Session 內也存在 posture non-stationarity

08/12 前段 vs 後段，同鼓同手中心可移動：
- L Mid Tom center_pitch 約 -31.8°
- L Floor Tom 約 -19.9°
- L High Tom 約 -19.5°
- R Hi-Hat 約 -10.7°
- 多類 Roll 約變 10–47°

這不一定是 IMU 漂移，更可能包含演奏段落/手臂高度/連打方式改變。
但它證明「每顆鼓 = 固定 Yaw/Pitch 點」不成立。

---

## 8. 09/18 Repeatability 的正確解讀

A -> B Euler 差：
- L: yaw -49.2°, pitch -11.0°, roll -20.9°
- R: yaw -80.1°, pitch -3.2°, roll -76.3°

但重力向量方向差：
- L 約 10.8°
- R 約 9.4°

以 ZYX 完整姿態矩陣估計：
- L 約 31.1°
- R 約 10.4°

結論：
- 不是 IMU 自己「漂了 80°」；
- Euler 表示在高 Pitch 區放大了視覺上的漂移；
- 同時使用者回到基準姿勢仍存在約 10–30° 真實腕部姿態重現誤差。

---

## 9. 原 0826 驗證的限制

原驗證採 `StratifiedGroupKFold`，group=floor(time_ms/4000)，比 random split 正確。
但仍只有：
- 同一演奏者
- 同一 session
- 同一穿戴
- 同一 Song2

因此是 within-session temporal generalization，不是 cross-session / re-wear validation。

原 strict end-to-end：
- F1 ≈ 0.787
- hit detection recall ≈ 0.961
- Crash recall ≈ 0.535

Crash 在原 session 裡本來就是弱類別。

---

## 10. 直接跨 Session 壓力測試

用 08/12 frozen MLP 直接預測 08/05 的 247 個高信心單擊：
- Accuracy: 49.4%
- 17D NN OOD 只有約 1.2%

表示舊 OOD 指標也不足以保護跨 session 分類。

主要錯法：
- 08/05 Hi-Hat 大量被判 Crash
- Floor Tom 大量被判 Ride

---

## 11. 關鍵突破：完整 local-zero 3D relative rotation

實驗 pipeline：

1. 每隻手、每個 session 用錄製開始第一秒建立完整 `R0`。
2. 原始 Yaw/Pitch/Roll -> 旋轉矩陣 `R`。
3. 姿態使用 `R_rel = R0^T R`，不做 Euler 軸向相減。
4. 加速度轉至 calibration-local frame：
   `a_local = R0^T R a_sensor`
5. 使用 rotation 6D representation、window mean/std、rotation angle range、magnitude/jerk、local-frame acceleration。
6. 左右手分開訓練。

### 結果
08/12 4 秒 Group CV：
- Accuracy: 96.1%
- Balanced Accuracy: 95.2%
- Macro-F1: 95.1%

08/12 train -> 08/05 high-confidence cross-session：
- Accuracy: 92.7%
- Balanced Accuracy: 80.8%
- Macro-F1: 75.7%

主要有足夠樣本的 08/05 類別：
- Snare: 33/38 = 86.8%
- Floor Tom: 23/28 = 82.1%
- Hi-Hat: 142/142 = 100%
- Ride: 24/26 = 92.3%

合計 222/234 = 94.9%。

少樣本類別：
- High Tom: 2 events
- Mid Tom: 3 events
- Crash: 8 events，correct 3/8
因此不能用 08/05 對這三類下穩定結論。

### 判定
這是目前最強證據：
**舊資料本身仍有高價值；主要問題不是「IMU 完全不可用」，而是錯誤的坐標表示、缺少完整姿態歸零、以及左右手條件化不足。**

---

## 12. 建議的新正式架構

### Calibration
- 每隻手各自 hold 1–2 秒固定標準姿勢。
- 用多 sample 平均建立 `R0`，不要只取單 packet。
- 最好直接讀 JY901S quaternion；若暫時只有 Euler，先轉 rotation matrix 後再做 relative rotation。
- 保存 calibration metadata：device serial、hand、R0、時間。

### Features
- 不再使用 raw `mean_roll`。
- 不再直接使用 sensor-local `max_ax/max_ay/max_az`。
- 不再對 wrapped yaw 直接 mean/std/range。
- 改用：
  - relative rotation 6D / quaternion
  - local-frame acceleration
  - magnitude / jerk
  - rotation-window dispersion/range
  - detector 的 vertical motion features 可做額外候選。

### Model
- 優先左右手獨立模型。
- 若要單模型，至少 hand 必須是 explicit conditioning。
- 不再把 pooled Yaw/Pitch posture map 當正式 decision boundary。

### Validation
最低要求：
1. session A train -> session B test
2. session B train -> session A test（若標記足夠）
3. Leave-one-session-out
4. 之後再做 leave-one-person-out

正式 accuracy 不再只報同 session 4 秒 block CV。

### Residual hard classes
Crash/Hi-Hat 特別需要額外受控資料。
建議每次 session 開始：
- 每鼓 × 每可用手 5–10 次 calibration hit
- 至少強化 Crash / Hi-Hat / Floor Tom / Ride
- 可做 prototype adaptation / small fine-tune

---

## 13. 最終根因排序

1. **完整姿態沒有相對化（Roll 未校正）** — 最大已驗證問題。
2. **左右手混在同一模型但未提供 hand** — 明確已驗證。
3. **Euler subtraction / yaw wrap 的數學表示錯誤** — live 已觸發。
4. **sensor-local acceleration axes 受穿戴旋轉影響**。
5. **wrist orientation 本身不是 Cartesian hand position**，同鼓/不同鼓存在姿態重疊。
6. **只有單一 session 真正完整標記，驗證過於 within-session**。
7. Crash 原本就是弱類別，且 08/05 可用 Crash 太少。
8. USB / 100 Hz 傳輸目前不是主因。

## 14. 專案現況判斷

HoloGrip 不是「資料全部作廢」。
08/05 與 08/12 Raw CSV 證明硬體採樣鏈路穩定，且 full-relative-rotation 實驗能把跨 session 主要類別拉回高準確率。

真正需要重做的是：
- calibration contract
- feature coordinate system
- hand conditioning
- validation protocol

舊 frozen model 不適合作為下一次正式跨 session 現場驗證的最終模型。
