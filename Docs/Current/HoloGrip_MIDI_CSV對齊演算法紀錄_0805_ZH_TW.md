# 0805 MIDI ↔ 手套 CSV 對齊演算法紀錄

## 目前決策

本次 0805 資料先採用 `+16,000 ms` 作為 MIDI → CSV 的工作對齊基準。

這個數值不是由音樂 MIDI 樂譜推算，也不是假設值。現有流程以手套活動度與 MIDI 事件時間做整體偏移搜尋；在 0805 實際資料中，`+16,000 ms` 得到最高候選分數。現場目前確認「歸零後，第一個 MIDI 鼓點對應到 CSV 約 +16 秒處的第一個主要活動峰值」，因此本次先以此值產生候選標記。

仍要區分兩件事：

- `+16,000 ms`：本次工作基準，可以用來繼續產生候選標記。
- 正式真值：仍需確認第一、中段、最後段，以及同時鼓點與低信心事件後才能升級。

## 來源資料

- MIDI：`Data/External/FlowAudio_20260805/Drum Midi_110BPM (0805).mid`
- Raw CSV：`Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv`
- BPM：`110`
- MIDI：Format 0、PPQN 96、1,820 個映射事件
- CSV：74,592 列，左右手各 37,296 列，時間長度 373,125 ms

### 實際時間範圍

- MIDI 第一個 mapped 鼓點：`0 ms`
- MIDI 最後一個 mapped 鼓點：`353,551.136 ms`（5:53.551）
- MIDI track end：`353,619.318 ms`（5:53.619）
- CSV 結束：`373,125 ms`（6:13.125）
- 套用 `+16,000 ms` 後，最後一個 MIDI 鼓點：`369,551.136 ms`（6:09.551）
- 最後一個 MIDI 鼓點後的 CSV 尾端：約 `3,573.864 ms`

因此這組 MIDI 不是 3 分鐘；它本身約 5 分 54 秒，與 6 分 13 秒 CSV 接近。CSV 前 `0 ～ 16 秒` 是 MIDI 對齊前的前置區，CSV 最後約 3.574 秒是 MIDI 事件結束後的尾端。

## 活動度定義

Raw CSV 已有 `accel_magnitude_g`。若需要從三軸重新計算：

```text
accel_magnitude_g(t) = sqrt(ax_g(t)^2 + ay_g(t)^2 + az_g(t)^2)
activity(t) = abs(accel_magnitude_g(t) - 1.0)
```

`1.0 g` 是靜止時重力的基準。活動度越大，表示加速度總量偏離靜止重力越多；它是動作變化訊號，不是敲擊力度。MIDI 的 `velocity` 是另一個欄位，不能和活動度混為一談。

目前 Python pipeline 的活動度處理是：

1. 以 10 ms bin 保留左右手活動度。
2. 使用約 ±40 ms 的 local-max smoothing，保留短促峰值。
3. 合併左右手時取 `max(R, L)` 作為整體對齊分數。

HTML 載入新 CSV 時，為了畫面效能，以每 100 ms 區間的最大活動度繪圖；公式本身不變。

## `+16 秒`候選的計算

對每一筆 MIDI 事件 `i`：

```text
aligned_csv_time_i = midi_time_i + offset_ms
```

對每一個候選偏移 `offset_ms`，計算：

```text
score(offset) = mean(
    max(activity_R(aligned_csv_time_i),
        activity_L(aligned_csv_time_i))
)
```

目前搜尋以 10 ms 為間隔。0805 的結果：

- `+15,990 ms`：4.437879
- `+16,000 ms`：4.462075，最高候選
- `+16,010 ms`：4.403325

附近分數仍接近，所以這個方法能找出合理起點，但不能單獨證明絕對同步。WAV 或已知同步擊仍是正式確認方式。

## 標記方式

不需要手動標記全部 1,820 筆事件。建議流程是：

1. 先確認一個整體 offset，例如本次工作基準 `+16,000 ms`。
2. 對所有 MIDI 事件套用同一個 offset。
3. 依 MIDI note 直接轉成 Zone。
4. 以 CSV 對應時間窗取出左右手活動度與原始感測資料。
5. 只人工複核低信心、同時事件、峰值不明顯或時間疑似錯位的事件。

固定 Zone 對照：

| MIDI note | Zone | 鼓點 |
| ---: | ---: | --- |
| 37、38 | 0 | 小鼓 |
| 48 | 1 | 高音 Tom |
| 45 | 2 | 中音 Tom |
| 43 | 3 | 落地 Tom |
| 46 | 4 | Hi-Hat |
| 49 | 5 | Crash |
| 51 | 6 | Ride |

MIDI 沒有原生左右手欄位。左右手只能由 CSV 活動度產生 `hand_candidate`，因此手別信心低或左右同時有峰值的事件不能直接當成正式手別真值。

## 視窗標準

### 建議採用：固定基礎視窗

目前 pipeline 的候選視窗為：

```text
window_start = aligned_csv_time - 200 ms
window_end   = aligned_csv_time + 300 ms
```

100 Hz 下約為每隻手 51 個時間點。固定視窗的理由是：

- 每筆訓練輸入尺寸一致。
- 模型不會因為某一筆峰值較大就看到不同長度資料。
- 方便比較不同鼓點、不同歌曲與不同受試者。

### 動態判斷只做 QC，不改變基礎視窗

下一步可在較寬的檢查範圍內計算 local baseline 與 peak，例如：

```text
threshold = max(fixed_floor, baseline + 3 * MAD)
```

若出現下列情況，標記 `review_required`：

- 視窗內沒有超過 threshold 的明顯峰值。
- 視窗內出現多個相近峰值。
- 相鄰 MIDI 事件使視窗重疊。
- 左右手分數太接近，無法可靠判斷手別。

動態判斷的用途是找出需要人工看的事件，不是讓每筆資料自動使用不同長度的訓練視窗。

## 目前沒有 MIDI 對應的部分

- CSV `0 ～ 16 秒`：前置錄製資料，尚未落在 MIDI 事件上。
- CSV `6:09.551 ～ 6:13.125`：最後一個 mapped 鼓點後的尾端資料。
- MIDI 鼓點之間的 CSV：沒有 MIDI hit 的連續動作或背景資料，不代表遺失。
- 0805 報告中的 unknown note 與 excluded note 都是 `0`；1,820 筆事件都能落到目前 7 個 Zone。

## 正式訓練前檢查門檻

- 確認第一筆、歌曲中段、最後一筆的時間是否都合理。
- 檢查同時發生的 MIDI note 是否要保留為多標籤事件。
- 複核 `hand_candidate` 低信心事件。
- 確認重疊視窗的資料處理規則。
- 只有通過複核的事件，才能把 `pending_manual_alignment_review` 升級成正式訓練標記。
