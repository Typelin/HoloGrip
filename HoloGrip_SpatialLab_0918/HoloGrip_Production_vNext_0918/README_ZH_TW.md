# HoloGrip Production vNext 0918

這是下一次學校實測用的 production candidate。**不要再用舊 frozen 17D 模型當主模型。**

## 下一次到學校怎麼跑

1. 插上左右兩隻手套。
2. 雙擊 `00_啟動Production_vNext_學校實戰.bat`。
3. 按 **1. 2秒姿態歸零**，雙手保持標準起始姿勢。
4. 按 **2. 開始一擊校準**。
   - 畫面會依序要求 hand × 鼓。
   - 每個提示只打一擊。
   - 如果某個 hand × 鼓真的不會用到，可以按「跳過目前 hand×鼓」。
5. 校準完成後，程式會自動把這些 few-shot hits 各加入一次 base training set，重新 fit 左右手模型。
6. 按 **3. 正式辨識**，再開始真正演奏 / blind validation。

## Production 固定規則

- 2 秒完整姿態 R0 校正。
- full-relative rotation + local-frame acceleration，31D feature。
- 左右手獨立 MLP。
- per-hand 120 ms refractory，避免一擊多算。
- I2C 壞 frame 永遠不進 HitDetector / classifier。
- 壞 frame 仍會寫入 Raw log，方便事後追查。
- few-shot calibration hit 與正式 prediction 分開保存。

## 已驗證結果

### Base model
- 8/12 4 秒 Group CV：accuracy 約 96.1%。
- 8/12 train → 8/5 high-confidence cross-session：accuracy 約 92.7%。

### Production detector
- 120 ms refractory：
  - hit precision 約 93.3%
  - hit recall 約 95.4%
  - hit F1 約 94.4%
- production end-to-end replay：
  - matched hit 後鼓類 accuracy 約 95.9%
  - exact joint F1 約 90.5%

### Few-shot 現場適應
以 8/5 模擬新 session，每個有足夠資料的 hand×class 隨機取 1 個 calibration hit：
- 30/30 次 overall accuracy 都比 base 提升。
- 平均 accuracy 約 96.1%。
- 最差約 94.2%，最好約 97.9%。
- macro-F1 30/30 次都提升。

注意：8/5 並沒有完整涵蓋所有 14 個 hand×drum 組合，所以這不是「14 組全部都已被歷史資料證明」；它支持的是：少量 session-specific calibration 對重新穿戴 / session shift 很有幫助。

## 右手 I2C

右手曾在高速壓力段出現大量壞 frame，但後續 40 秒純硬體壓力測試：
- 約 100 Hz
- 最高約 15 g
- 0 invalid frame
- 0 packet gap
- 0 back/dup

因此判定是**間歇性資料鏈異常**，不是每次必現。Production 仍永久保留 frame validation。
