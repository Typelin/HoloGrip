# HoloGrip 第三次驗證 SOP（新人 + 新歌）

文件日期：2026-09-14

## 這次驗證要回答的唯一核心問題

在 **同一套鼓、同一個歸零位置、封版 2026-08-26 模型完全不重訓** 的條件下，新的流音學生打新的歌曲時：

1. 手套與 COM 串流是否健康；
2. HitDetector 是否有抓到真實擊打；
3. 已抓到的擊打，17 維 IMU → MLP(64,32) 的七鼓分類是否正確。

三件事必須分開報告。不要只看一個總正確率，也不要因為 GUI 一筆判錯就立刻改模型。

---

## 當天一定要帶／準備

- [ ] 兩隻原本 HoloGrip 手套，保持原本的 L / R `HAND_ID` 韌體
- [ ] 兩條確定穩定的 USB Type-C 線，最好多帶 1–2 條備用
- [ ] 驗證用筆電 + 充電器；Windows 關閉睡眠／省電斷 USB
- [ ] 同一套電子鼓、同一套鼓位配置、同一個站位／歸零原點
- [ ] 電子鼓端 **原始 MIDI 錄製**；不要只錄音或只看螢幕
- [ ] 新歌 BPM（若 MIDI 自己有 tempo meta event 仍建議另外記錄）
- [ ] 建議同步錄一段手機影片，畫面能看到演奏者與鼓組；它是事故稽核，不是模型輸入
- [ ] 使用 participant ID，例如 `P02`；不需要把學生姓名寫進資料檔

## 封版資產

第三次正式 zero-shot 驗證固定使用：

`Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/hologrip_song2_七鼓點模型_真正驗證版_0826.joblib`

預期 SHA-256：

`c2f9ce16750887011219a6f46fee5095c2e4f50c8df3f94548b3c5d806b2f660`

新的驗證 GUI 啟動時會檢查這個 hash；不相符就直接拒絕作為正式第三次驗證。

---

## 要開哪一支程式

專案根目錄直接執行：

`HoloGrip第三次驗證.bat`

或：

`Apps/Song_Collection_COM/run_round3_validation_0914_ZH_TW.bat`

**不要同時開**舊的歌曲採集 GUI、Arduino Serial Monitor、PuTTY 或其他會占用同一個 COM 的程式。

這個第三次驗證 GUI 已經同時負責：

- 現場七鼓模型推論；
- **每一筆 100 Hz Raw CSV 保存**；
- 每一次 HitDetector + MLP prediction 保存；
- 歸零／預檢／慢打／正式歌曲 marker 保存。

因此不需要第二支程式搶 COM。

---

## 正式流程

### A. 連線

1. 插上兩隻手套。
2. 開啟 `HoloGrip第三次驗證.bat`。
3. 掃描 COM，選兩個不同 COM，按「連接」。
4. 確認畫面同時出現左手、右手，兩邊約 100 Hz。
5. 如果兩個 COM 都送出同一個 `HAND_ID`，停止，不做正式驗證。

### B. 先開始留證據

1. Session 建議填：`R3_YYYYMMDD_P02`。
2. 按「開始場次」。
3. 從這一刻起，GUI 開始同步保存 Raw 100 Hz。

### C. 固定原點歸零

1. 請學生站在與第二次收集相同的位置。
2. 雙手回到原本規定的歸零姿勢。
3. 按原 GUI 的「歸零」。
4. 每次重插手套／重新連 COM 後，都必須重新歸零；正式歌曲途中不要隨意重歸零。

### D. 5 秒靜止 Preflight

按「5秒靜止預檢」，5 秒內不要打鼓、不要大幅移動。

GUI 會檢查：

- 約 80–120 Hz 封包率；
- 全零封包；
- packet ID gap；
- 靜止時 Yaw/Pitch 漂移；
- 靜止加速度是否接近 1 g。

`PASS` 才進下一階段最理想。`WARN` 不是神經網路失敗：先重插線、確認感測器、重新回原點歸零，再做一次 preflight。

### E. 七鼓慢打診斷（不訓練）

先按「七鼓慢打開始」。每換一顆鼓前，先在 GUI 按對應的鼓名按鈕，再只打那一顆 **5–10 下**：

1. 小鼓
2. 高音 Tom
3. 中音 Tom
4. 落地 Tom
5. Hi-Hat
6. Crash
7. Ride

這一段的用途是確認「新人站在同一鼓組空間時，模型有沒有整體跑掉」。

**禁止**用這一段重新 fit 模型；否則後面的新人測試就不再是 zero-shot 驗證。

慢打的判讀：

- 少數相鄰鼓互相混淆：記錄，仍可進正式歌曲。
- 所有鼓都集中偏成同一顆／完全不合理：先排查歸零與手套座標，不要直接開始正式歌曲。
- 人有打但 GUI 常常完全沒事件：HitDetector 問題，不是七鼓 MLP 問題。

### F. 正式新人 + 新歌

1. 電子鼓端開始保存原始 MIDI。
2. HoloGrip GUI 按「正式歌曲開始」。
3. 演奏完整歌曲；不要途中改門檻、改模型、重訓。
4. 結束後按「歌曲結束」。
5. 停止並保存電子鼓 MIDI 原檔；不要編輯覆蓋原檔。
6. HoloGrip GUI 按「停止場次」。

---

## 每一個 HoloGrip 場次會留下什麼

位置：

`Data/Validation/Round3/<session>_<timestamp>/`

核心檔案：

- `manifest.json`：模型 hash 與場次契約
- `raw_100hz.csv`：兩手每一筆原始 100 Hz + 原始角度 + 歸零後 Yaw/Pitch
- `predictions.csv`：HitDetector 產生的事件與 MLP 七鼓輸出
- `markers.csv`：CALIBRATE / PREFLIGHT / SLOW_DRUM / SONG_START / SONG_END
- `preflight.json`：5 秒硬體／漂移預檢
- `session_summary.json` / `.md`：場次保存狀態

Raw CSV 是第三次兜底最重要的檔案。現場 GUI 就算看起來很差，只要 Raw + MIDI 都保存成功，回來仍然能準確查出是哪一層出錯。

---

## 回來怎麼算正式結果

只檢查 HoloGrip 場次本身（還沒放 MIDI）：

```bat
python Apps\Song_Collection_COM\analyze_round3_validation_0914_ZH_TW.py "Data\Validation\Round3\你的場次資料夾"
```

有電子鼓 MIDI 後：

```bat
python Apps\Song_Collection_COM\analyze_round3_validation_0914_ZH_TW.py "Data\Validation\Round3\你的場次資料夾" --midi "你的電子鼓.mid" --bpm 110
```

若 MIDI 內有 tempo meta event，可先不填 `--bpm`；若歌曲有變速或 MIDI tempo 不可靠，必須另外核對。

分析報告會分開給：

1. Raw/COM 健康；
2. 七鼓慢打「已偵測事件」的一致率；
3. 正式歌曲 HitDetector Recall / Precision；
4. 已抓到擊打之後的七鼓分類正確率；
5. 嚴格「時間 + 鼓位」Precision / Recall / F1。

---

## 如果當場看到「打小鼓卻顯示完全不同鼓」

不要立刻重訓。依序做：

1. 看兩手是否仍約 100 Hz，是否有 zero/gap。
2. 回固定原點看歸零後 Yaw/Pitch 是否接近 0；若不是，停止正式歌曲並重新歸零。
3. 再打 5 下同一顆小鼓，看錯誤是否穩定重現。
4. 如果 HitDetector 每一下都有事件，但鼓位穩定錯：記為 zone domain shift。
5. 如果很多實際擊打根本沒 event：記為 HitDetector 問題。
6. **不要當場拿新人資料覆蓋封版模型。** 正式 zero-shot 結果要完整保存。

正式結果不漂亮並不等於整個 HoloGrip 不可行；它會告訴下一階段到底需要：HitDetector 調整、座標校正，還是多演奏者／個人化模型。

---

## Plan B：正式 zero-shot 完成後才能做

如果新人結果差，可以另外開一個「救援／個人化校正」實驗，例如每顆鼓少量 5–10 下做 calibration，再比較 before/after。

但報告必須分成：

- **Zero-shot frozen model**：第三次正式成績；
- **Personalized/calibrated model**：後續救援成績。

兩者不能混在一起。這樣第三次即使不好看，也會是一份可信、可診斷、可繼續研究的結果，而不是一次不可解釋的 Demo 失敗。
