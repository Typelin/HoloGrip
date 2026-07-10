# HoloGrip 系統學術研究數據規格與分析指南 (Research Data Specification)

本文件旨在為 HoloGrip 系統的科研與論文寫作提供**嚴謹的實驗設計 (Experimental Design)、物理量定義與統計分析指南**。
本指南針對 Pre-Type-C 版本與 Type-C 分離供電版本之對比實驗進行設計，詳細說明各數據變數在學術論文中的統計價值與分析方法。

---

## 1. 核心研究假說 (Research Hypotheses)

在撰寫論文時，改版對比實驗通常旨在驗證以下三個核心科學假說：

* **假說一 (H1) — 傳輸穩定度與抖動 (Jitter) 的改善**：
  * *陳述*：Type-C 分離供電版本由於隔離了 Wi-Fi 射頻發送時的高頻瞬態電流波動，能顯著降低網路傳輸延遲的抖動（Jitter）與封包丟失率。
  * *驗證指標*：`timing_error_ms` 的變異數（Variance）與 `loss_rate`。
* **假說二 (H2) — 動作分類精確度與信噪比 (SNR) 的提升**：
  * *陳述*：分離供電降低了電源纹波對 IMU 內部運算與 ADC 採樣的電磁干擾（降低感測器噪聲），進而使機器學習模型（KNN）的分類準確率顯著提升。
  * *驗證指標*：`predicted_zone` 與 `Zone_ID` 的混淆矩陣、`confidence` 置信度分佈。
* **假說三 (H3) — 人機交互定時精確度 (Temporal Precision) 的優化**：
  * *陳述*：極致且穩定的延遲回饋，能幫助使用者在節奏任務中建立更精準的內部時鐘（Internal Clock），顯著降低敲擊時間誤差。
  * *驗證指標*：`timing_error_ms` 與 `inter_hit_interval_ms` 的變異係數 (CV)。

---

## 2. 實驗變數設計 (Variables Classification)

在學術論文中，應將收集到的數據欄位清晰界定為以下三類變數：

### 2.1 自變量 (Independent Variable) — 實驗控制條件
* **`hardware_version` (硬體版本)**：核心分類自變量，包含 `Pre-Type-C` (對照組) 與 `Type-C` (實驗組)。

### 2.2 因變量 (Dependent Variables) — 觀測的物理效應與指標
* **動作特徵 (Kinematics)**：`Feature_Yaw`, `Feature_Pitch`, `Feature_SwingDepth`。
* **分類效能 (Classifier Performance)**：`predicted_zone` (預測區域), `confidence` (置信度)。
* **時間與定時 (Temporal Performance)**：`actual_hit_time_ms` (手套實質打擊時間), `timing_error_ms` (拍點誤差), `inter_hit_interval_ms` (相鄰打擊間隔)。

### 2.3 協變量 / 控制變數 (Covariates & Control Variables) — 排除干擾源
* **`participant_id` (受試者 ID)**：控制受試者個體生理結構與揮擊習慣的差異（在多因素方差分析中作為隨機效應處理）。
* **`session_id` / `trial_id`**：控制實驗順序效應（如受試者隨時間產生的疲勞效應或熟練效應）。
* **`bpm`**：控制音樂速度對人體敲擊精確度的物理影響。

---

## 3. 欄位物理意義與學術價值深度解密

以下欄位是從客戶提出的變數中，經篩選具有**真實科研與統計價值**的物理量：

| 數據欄位名稱 | 物理與生理意義 | 論文寫作與統計分析價值 |
| :--- | :--- | :--- |
| **`actual_hit_time_ms`** | **手套端物理打擊時間** | 基於 ESP32 的 `millis()` 開機時間戳。這是唯一**不受網絡延遲與電腦排程干擾**的黃金物理時間基準，用於計算絕對動作時間。 |
| **`inter_hit_interval_ms`**<br>(IHI) | **相鄰打擊間隔時間** | 計算公式：$IHI_i = t_i - t_{i-1}$。<br>在人體工學與運動控制學（Motor Control）中，IHI 的**變異係數 (Coefficient of Variation, CV)** 是衡量人體運動定時穩定度（Motor Timing Variability）的黃金指標。 |
| **`timing_error_ms`** | **拍點定時誤差** | 計算公式：$E = t_{actual} - t_{target}$。<br>1. **平均值（恆常誤差 CE）**：代表受試者習慣性搶拍（負值）或慢拍（正值）。<br>2. **標準差（變異誤差 VE）**：直接代表玩家的**時間控制精度 (Temporal Precision)**。 |
| **`confidence`** | **機器學習置信度** | 由 KNN 鄰近點距離加權計算得出。可用於評估感測器噪聲降低後，AI 模型對邊界動作判定的「把握度」，可用於繪製 **ROC 曲線**。 |
| **`Feature_SwingDepth`** | **相對揮幅深度** | 擊中前 400ms 內 Pitch 的 $\text{max} - \text{min}$ 差值。代表敲擊動作的物理振幅（強度），可用於研究「供電穩定度是否影響高強度動作下的信號解算」。 |

---

## 4. 統計分析 (Statistical Analysis) 與圖表建議

在收集完改版前後的兩套加強版 CSV 數據後，建議論文團隊在 Python (Pandas/SciPy) 或 R 中進行以下統計檢定與可視化：

### 4.1 傳輸延遲與丟包分析 (ANOVA / t-檢定)
* **檢定方法**：使用 **獨立樣本 t-檢定 (Independent t-test)** 對比 `Pre-Type-C` 與 `Type-C` 的延遲波動。
* **圖表建議**：繪製 **延遲分佈機率密度曲線 (PDF - Probability Density Function)**。若改版成功，Type-C 組的曲線將更窄、更集中，代表 Jitter 極低。

### 4.2 動作定時精確度分析 (箱線圖 Boxplot)
* **檢定方法**：對比兩組的 `timing_error_ms`。
* **圖表建議**：繪製 **Timing Error 箱線圖**。如果改版後的物理延遲更低且更穩定，受試者的 Timing Error 箱線圖將會顯著收窄，平均值更接近 0ms，證明「閉環回饋優化提升了人體定時效能」。

### 4.3 AI 模型識別率分析 (混淆矩陣 Confusion Matrix)
* **檢定方法**：對比 `Zone_ID` (實質鼓組) 與 `predicted_zone` (預測鼓組)，計算 Precision, Recall 與 F1-Score。
* **圖表建議**：繪製兩組的 **混淆矩陣熱力圖 (Heatmap)**。用以直觀證明「電源雜訊降低後，AI 對相鄰鼓組（如小鼓與中鼓）的邊界模糊判定顯著減少」。

---

*本數據規格設計已完整整合至 HoloGrip Studio 接收端伺服器 (server.py) 中。當前版本實驗完成後，請直接匯出 CSV，數據將完全相容於本規範。*
