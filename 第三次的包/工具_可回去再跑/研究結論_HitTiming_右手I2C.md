# HoloGrip 0918｜Hit Timing 與右手 I2C 鑑識研究

## 1. 右手硬體 / I2C 結論

### 當前 next-person live session
Session: `next_person_realdrum_sessions/20260918_202322`

- 左手 hits: 25
- 右手 hits: 0
- 右手 invalid frames: 516
  - all_zero: 379
  - stale_high_plateau: 81
  - i2c_same_word: 56
- 最大單一故障 episode:
  - 20:26:35.116 → 20:26:41.093
  - 505 invalid frames
  - 5.977 s
- 故障時右手有效資料 run 中位數只有 3 packets（約 30 ms）
- HitDetector 需要至少 45 packets（約 450 ms）buffer，因此該段右手根本不可能產生合法 hit。

### 目前硬體狀態
故障後立即做 15 s 雙手 raw health test：
- COM4 R: 1499 packets / 15 s = 99.93 Hz, invalid=0, packet gap=0
- COM5 L: 1498 packets / 15 s = 99.87 Hz, invalid=0, packet gap=0

結論：右手不是永久故障，而是 intermittent I2C/JY901S 資料鏈異常。

### 歷史真鼓 Raw
- 2026-08-12: L 8019 / R 8020 frames，兩手此類 invalid = 0
- 2026-08-05: L 37296 / R 37296 frames，兩手此類 invalid = 0

=> 訓練資料沒有被現在這種右手 toxic frame 污染；這是目前硬體狀態的新問題。

### Firmware bug
Gloves_Right_R.ino 每次 loop 先把 ax/ay/az/yaw/pitch/roll 初始化為 0。
如果 I2C endTransmission/requestFrom/available 失敗，仍會 Serial.printf 一包資料。
因此 I2C 失敗會被 firmware 包裝成「正常 D packet 的 0 值」送給 host。

需要修：
- I2C read 必須回傳 success/fail
- fail 時不可輸出正常 D packet
- 建議 retry 1 次；連續 fail 時重新 Wire.begin / setClock
- packet 加 validity/status bit 或 error counter
- host 仍保留 FrameValidator

## 2. Hit timing / detector 結論

### Detector 現況
- 100 Hz
- target = buffer[-4]，即當前封包約 30 ms 前
- PEAK_LAG_MS = 30 ms
- feature window = peak ±80 ms
- detector heavy debounce = 100 ms
- light debounce = 150 ms
- production 額外 fixed refractory = 120 ms

### 8/12 307 GT 重放
Raw detector:
- detections: 337
- matched by peak ±90 ms: 296
- miss: 11
- extra: 41
- peak timing error median: -3.27 ms
- timing error p10/p90: 約 -23.4 / +10.7 ms

=> 正常真鼓情境的 peak 時機本身很準；10~30 ms jitter 不是主因。

### Feature timing sensitivity
以 8/12 GT 為中心，把 feature window 整體平移：

- -40 ms: 97.39%
- -30 ms: 99.02%
- -20 ms: 99.67%
- -10 / 0 / +10 ms: 100%
- +20 ms: 99.35%
- +40 ms: 97.39%
- -60 ms: 92.51%
- +60 ms: 95.11%
- -80 ms: 70.68%
- +80 ms: 82.08%
- ±100 ms 後約剩 50%

=> ±20~40 ms 影響有限；真正跨到 80~120 ms 才會大幅換鼓。

## 3. 「同一位置卻跳多顆鼓」的核心證據

Current live 左手 25 hits：
- Hi-Hat 17
- 高音 Tom 5
- Floor Tom 3

大量 hit 間隔：
- 125 ms
- 140 ms
- 141 ms

典型 pair：
- 3~4 g 低峰
- 125~140 ms 後 5~7 g 高峰
- 同時 rel yaw/pitch 從一側姿態翻到另一側姿態

這符合「同一次揮擊循環，pre-impact / impact 被 detector 當成兩擊」。

### 真鼓 isolated GT 驗證
挑 106 個同手前後 >=300 ms 沒其他真音符的 isolated GT：
只有 1 個出現雙峰，且：
- 第一峰: GT -145.3 ms, 2.47 g
- 第二峰: GT -35.3 ms, 8.46 g
- gap: 110 ms
- 第二峰更接近真正 MIDI impact

=> detector 的確可能先抓到一個弱 pre-impact local peak，再抓到真正較強 impact peak。

## 4. Peak replacement 實驗

規則：
若早期 candidate <= 3.0~3.5 g，且 160 ms 內出現 >=1.4x 的更強 peak，則 suppress 早峰、保留後峰。

8/12 GT：
原始 detector：
- Hit F1 = 0.9193
- exact joint F1 ≈ 0.8944

Peak replacement：
- Hit F1 ≈ 0.9395~0.9397
- exact joint F1 ≈ 0.9143~0.9172
- FP 41 → 約 26~27
- recall 仍約 96%

=> 比單純把 fixed refractory 拉長更好。
160~180 ms fixed refractory 會明顯掉 recall，不建議。

## 5. 利用 Raw 時間鄰域做 temporal augmentation

舊訓練：每個 307 GT 只抽一個 center window。

新研究：每個 GT 用 -20 / 0 / +20 ms 三個 window，label 不變，grouped CV 仍按時間區塊隔離。

8/12 grouped CV：
- center-only: 96.09%
- aug20: 94.79%（略降）

8/12 → 8/5 真鼓 cross-date：
- center-only: 92.71%
- aug20: 95.14%

Timing robustness on 8/5:
- shift -40 ms: 75.71% → 82.59%
- shift -20 ms: 90.28% → 93.12%
- shift 0 ms: 92.71% → 95.14%

新 timing-robust 模型已保存：
`vnext_timing_robust_aug20/hologrip_timing_robust_L.joblib`
`vnext_timing_robust_aug20/hologrip_timing_robust_R.joblib`

8/5 247 真鼓事件：
- overall: 95.14%
- L: 88.89%
- R: 96.21%
- LEFT <-> RIGHT catastrophic errors: 0 / 204

## 6. 根因排序

1. **当前右手 intermittent I2C/JY901S data corruption**：硬件 / firmware 层，严重且确定。
2. **HitDetector 的弱 pre-impact peak + 强 impact peak 双抓问题**：确定存在，可通过 peak replacement 改善。
3. **训练只用单一 center window，缺少 timing jitter robustness**：已通过 ±20 ms temporal augmentation 改善跨日真鼓 92.7% → 95.1%。
4. **一般 10~30 ms peak timing jitter**：不是主要问题，因为真实 GT 上 timing median ~-3 ms，±40 ms 模型仍约 97%。
5. **今天自由空挥的空间动作差异**：会影响模型，但不能拿来解释当前所有异常，因为当前同一位置乱跳里已发现明确双峰 detector 问题。

## 7. 下一版应实施

- Firmware hardening：I2C fail 不得输出正常 packet；retry/reinit；输出 validity/error counter。
- Host: FrameValidator 保留。
- Detector: 替换 fixed 120ms gate 为 pending candidate / peak replacement。
- Classifier: 使用 timing-robust aug20 per-hand MLP。
- Live pipeline: candidate peak → 160ms 内 peak resolution → ±80ms 31D feature → timing-robust MLP。
- 再用 8/12 strict replay + 8/5 cross-date + 本地同位置重复击打做三重验证。
