# HoloGrip｜第三次的包

這一包只做四件事：

1. **先測目前凍結模型**
2. **再收第三次新 Raw**
3. **檢查第三次 MIDI / Raw 是否異常**
4. **做 MIDI ↔ Raw 對齊，回來再決定怎麼訓練**

不要在現場臨時重訓模型。

---

## 現場只按這幾個

### 00_先檢查雙手與模型_ZH_TW.bat

出發前一次、到場插好兩手後再一次。

它會檢查：

- 兩顆 XIAO ESP32-C6
- 左右 hand ID
- 雙手約 100 Hz
- packet gap
- all_zero / i2c_same_word / stale_high_plateau
- timing-robust 左右模型能否正常載入
- 模型契約是否為 31D → MLP(64,32) → 7 drums

**00 沒 PASS，不要直接開始正式測試。**

---

### 01_測目前凍結模型_保存Raw_ZH_TW.bat

這是「考上一版模型」。

目前主模型：

- 8/12 真鼓 GT
- 每個 GT 使用 -20 / 0 / +20 ms temporal augmentation
- 左右手各一顆 31D → 64 → 32 → 7 MLP
- 8/12 → 8/5 真鼓跨日：**95.14%**
- 8/5 側邊事件 LEFT ↔ RIGHT 大錯：**0 / 204**
- Hit resolver：弱前峰最多等待 160 ms；如果後面出現更強 impact，壓掉前峰
- 壞 frame 不進模型

操作：

1. 戴好兩隻手套。
2. 雙手放固定正前方 neutral 姿勢。
3. 按 **穩定 R0 歸零**。程式會先等每手 300 個連續乾淨 frame，並檢查 3 秒窗口前後的完整姿態漂移 ≤2°、重力方向漂移 ≤1.5°；沒有穩定就不會接受 R0。
4. 等兩手顯示 `R0 STABLE ✓` 並進入 Preview；此時**還不是正式測試 0 秒**。
5. 準備好讓流音端同步錄 MIDI 後，按 **開始正式測試**。這一刻就是 `song_time_ms = 0`。
6. 真鼓上直接打，不做每鼓校準、不重訓。
7. 若要做逐鼓測試，可用畫面上的標記選：
   - Hi-Hat
   - Crash
   - 小鼓
   - 高音 Tom
   - 中音 Tom
   - Ride
   - 落地 Tom
8. 建議每顆先打 10 下，再打一段 30～60 秒自然 groove。
9. 打完按 **結束正式測試**；程式會把正式段寫成 `raw_100hz_formal.csv`。
10. 同一次演奏請同步保存原始 MIDI。
11. 關閉程式後資料會留在：

`Data\上次模型測試\<timestamp>\`

其中：
- `raw_100hz_formal.csv` = 正式測試 0 ms 起算，可直接做 MIDI 對齊
- `resolved_hits.csv` = 模型每一擊答案
- `formal_test_manifest.json` = 正式段起訖、時長、sample 數

同一次演奏的 MIDI 放：

`Data\上次模型測試_MIDI\`

**這段資料只是在驗證目前凍結模型，不要跟第三次新訓練 Raw 混。**

---

### 02_收集第三次Raw_100Hz_ZH_TW.bat

01 完全關閉、COM 釋放後再開。

這一段才是第三次新資料。

建議：

- 開始前 neutral 靜止 2～3 秒
- 真實七鼓
- 正常打擊，不要只空揮
- 每顆鼓最好至少 20 個乾淨 hit
- 再加一段自然 groove / song
- 同步錄 MIDI
- 有影片就一起錄
- 結束後再留 2～3 秒才停止

Raw 自動放：

`Data\第三次Raw\`

第三次 MIDI 放：

`Data\第三次MIDI\`

---

### 03_檢查第三次MIDI異常_ZH_TW.bat

雙擊後會自動挑：

- `Data\第三次MIDI` 最新 MIDI
- `Data\第三次Raw` 最新 CSV

會檢查：

- unknown MIDI note
- excluded MIDI note
- 同 tick 多音
- 同 tick 同 note 重複
- <100 ms 過密 onset
- 極弱 velocity
- 七鼓 MIDI 數量
- Raw 左右手 Hz
- packet gap
- all_zero / i2c_same_word / stale_high_plateau

報告寫到：

`Data\對齊與異常報告\`

**它只報告，不會修改原 MIDI / Raw。**

如果看到 REVIEW，不要直接把整批當乾淨 training labels。

---

### 04_對齊第三次MIDI與Raw_ZH_TW.bat

03 看完後再做。

預設資料夾已改成：

- Raw：`Data\第三次Raw`
- MIDI：`Data\第三次MIDI`
- Output：`Data\對齊與異常報告`

自動 offset 只能當候選，仍要人工看開頭 / 中段 / 結尾與異常區。

---

### 05_打開第三次資料_ZH_TW.bat

直接開 Data。

---

# 第三次最短流程

```
00 先檢查
↓
01 測目前凍結模型
   └─ 同一次演奏錄一份 MIDI
↓
完全關閉 01
↓
02 收第三次 Raw
   └─ 同步錄第三次 MIDI + 最好有影片
↓
03 檢查 MIDI + Raw 異常
↓
04 MIDI ↔ Raw 對齊
↓
確認檔案完整再離場
```

---

# 為什麼第三次資料仍然有價值

目前家裡 Field v2 的 154 個 resolved hit，拿去跟 8/12 真鼓訓練雲比較：

- 左手 training NN P99 約 3.00；家裡中位數約 10.96
- 右手 training NN P99 約 3.16；家裡中位數約 6.84
- **154 / 154 全部是 OOD**

而且家裡「左上 Crash 模擬」旋轉窗口明顯比真鼓大：

- 8/12 左手 rot_range 中位數約 0.274 rad
- 家裡左上 Crash 模擬約 1.036 rad

所以家裡空揮亂跳不能直接等同「真鼓方案已死」。

第三次真鼓資料的價值不是單純把同一個點再複製很多次，而是補：

- 新穿戴
- 新 session
- 可能的新受試者
- 真正鼓面 impact / rebound
- 新 MIDI 時機
- 新揮擊速度與力度

如果第三次真鼓仍然大量 OOD，下一版就應該做 **multi-session / cross-person training**，而不是繼續只拿 8/12 單 session 擴增。

---

# 已知風險：右手 I2C

目前右手曾出現間歇性：

- i2c_same_word
- all_zero
- stale_high_plateau

歷史 8/5、8/12 真鼓 Raw 沒有這種污染，但現在硬體會間歇發作。

第三次包有：

`Firmware_第三次候選\`

修正內容：

- I2C 讀失敗 **不再輸出假的正常 D packet**
- 讀取重試
- 連續失敗時重新初始化 I2C
- 錯誤只輸出 E line

兩手候選韌體已經通過 XIAO ESP32-C6 編譯。

原版保留：

`Firmware_原版回退\`

**目前第三次候選韌體只是準備好，尚未自動燒進手套。**

---

# 目前不要做的事

- 不要用家裡純空揮結果當正式 accuracy
- 不要把第三次 MIDI 自動清掉重複事件後覆蓋原檔
- 不要現場把新資料餵進模型再回頭算「盲測」
- 不要把 01 的驗證 Raw 和 02 的第三次訓練 Raw 混在一起
- 右手如果 00 或畫面 health 報 BAD，先修資料鏈再打正式資料

---

## 可選底層診斷：上電姿態重現性

位置：

`工具_可回去再跑\06_上電姿態重現性測試.bat`

用途：驗證同一個真實 A/B 姿勢在不同斷電、上電姿勢後，IMU Raw orientation 與 A↔B 相對旋轉是否可重現。

流程：A0 → 不斷電移到 B → B_before → B 姿勢斷電重插 → B_after → 不斷電回 A → A_after_Bboot → A 姿勢再斷電重插 → A_reboot。

這項測試可以直接檢查目前 `R0.T @ R` 的跨開機假設是否成立；不是模型 accuracy 測試。
