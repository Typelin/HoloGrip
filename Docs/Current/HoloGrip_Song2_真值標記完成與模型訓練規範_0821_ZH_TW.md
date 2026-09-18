# HoloGrip Song 2 真值標記完成與模型訓練規範

更新日期：2026-08-21  
狀態：標記完成（100% 覆蓋），模型基準已訓練完畢

---

## 一、標記完成現況與數據統計

本曲（Song 2，T1127 Cleaned）共 311 筆標準 MIDI 鼓點事件，已全數完成真值標註：

| 類別 | 筆數 | 佔比 | 說明 |
|---|---:|---:|---|
| **左手 (L)** | 141 | 45.3% | 小鼓 25、Hi-Hat 24、Crash 20、Ride 20、Tom 52 |
| **右手 (R)** | 166 | 53.4% | Hi-Hat 56、Crash 23、Ride 21、小鼓 18、Tom 48 |
| **雙手齊擊 (DUAL)** | 4 | 1.3% | #50 小鼓、#51 高音Tom、#52 中音Tom、#53 落地Tom |
| **排除 (EXCLUDE)** | 0 | 0.0% | 無幽靈 MIDI |
| **合計** | **311** | **100.0%** | **全曲覆蓋率 100%** |

---

## 二、特徵工程架構（22 維動態特徵）

以 MIDI 擊中中心為基準，擷取手套 Raw 100Hz IMU 之 `±80 ms` 窗口（16 個採樣點），計算以下 22 維特徵向量：

1. **左手特徵（8 維）**：`L_max_mag`, `L_mean_mag`, `L_std_mag`, `L_energy`, `L_max_ax`, `L_max_ay`, `L_max_az`, `L_jerk_max`
2. **右手特徵（8 維）**：`R_max_mag`, `R_mean_mag`, `R_std_mag`, `R_energy`, `R_max_ax`, `R_max_ay`, `R_max_az`, `R_jerk_max`
3. **雙手非對稱交互特徵（4 維）**：
   - `energy_diff` ($E_R - E_L$)
   - `energy_ratio_asym` ($(E_R - E_L) / (E_R + E_L)$)
   - `max_mag_diff` ($\max(Mag_R) - \max(Mag_L)$)
   - `mean_mag_diff` ($\text{mean}(Mag_R) - \text{mean}(Mag_L)$)
4. **音樂上下文特徵（2 維）**：`velocity` (力度), `drum_zone_id` (鼓件種類 ID)

---

## 三、模型 5-Fold 交叉驗證基準評測

使用 307 筆單手樣本進行 5-Fold Stratified Cross-Validation：

| 模型名稱 | 準確率 (Accuracy) | F1-Score | 精確率 (Precision) | 召回率 (Recall) |
|---|---:|---:|---:|---:|
| **MLP 神經網路 (64, 32)** | **93.48% (±1.46%)** | **94.00%** | **94.30%** | **93.98%** |
| **SVM (線性核 Linear)** | **91.54% (±2.11%)** | **92.01%** | **93.97%** | **90.41%** |
| **SVM (徑向基核 RBF)** | **90.88% (±1.65%)** | **91.55%** | **91.90%** | **91.60%** |
| **邏輯斯蒂迴歸 (Logistic)** | **90.56% (±1.87%)** | **91.18%** | **92.37%** | **90.41%** |
| **梯度提升 (Gradient Boosting)** | **90.23% (±1.46%)** | **90.91%** | **91.90%** | **90.41%** |
| **隨機森林 (Random Forest)** | **87.63% (±3.00%)** | **88.47%** | **90.02%** | **87.40%** |

### 特徵重要性排行（Top 5）
1. `mean_mag_diff` (14.46%) - 雙手平均加速度強度差
2. `energy_diff` (13.32%) - 雙手活動能量差
3. `energy_ratio_asym` (13.24%) - 雙手能量非對稱比
4. `max_mag_diff` (7.90%) - 雙手峰值加速度差
5. `L_std_mag` (6.62%) - 左手加速度標準差

---

## 四、檔案輸出與調用清單

1. 訓練腳本：[`Apps/Song_Collection_COM/train_hand_classifier_0821_ZH_TW.py`](file:///C:/Users/Typelin_Station/Desktop/HoloGrip/Apps/Song_Collection_COM/train_hand_classifier_0821_ZH_TW.py)
2. 最佳模型權重：[`Data/Derived/.../hologrip_song2_best_hand_classifier.joblib`](file:///C:/Users/Typelin_Station/Desktop/HoloGrip/Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/hologrip_song2_best_hand_classifier.joblib)
3. 評測報告 JSON：[`Data/Derived/.../model_benchmark_report_0821.json`](file:///C:/Users/Typelin_Station/Desktop/HoloGrip/Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/model_benchmark_report_0821.json)
