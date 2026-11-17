# HoloGrip → MIDI 輸出（Ableton Live）

`holo_midi_out.py`：把手套即時判斷的每一下，轉成 MIDI 音符送給音樂軟體（Ableton Live）。
判斷引擎直接用模型包（預設 `../../11-17的包`，可用環境變數 `HOLOGRIP_PACKAGE` 指定）裡凍結的程式與模型，不修改；只在「判斷出一下且被接受」時多送一個音符。

## 功能
- **輸出埠**：
  - macOS：建立虛擬埠「HoloGrip」，Ableton 的 MIDI 輸入選它即可。
  - Windows：選既有的輸出埠。
    - 「Microsoft GS Wavetable Synth」：Windows 內建，可以直接聽到鼓聲，適合測試。
    - 要接 Ableton，需要另外裝 loopMIDI，建立一個虛擬埠後，在這裡選它。
- **音符對照**：每手每鼓一個 MIDI 音符（0～127），左右手可以設不同，例如左手 Crash 49、右手 Crash 57。每格旁邊有「試打」按鈕；可以單顆鼓靜音；按「儲存設定」存到 `midi_map.json`。
  - 預設音符與現場電子鼓 MIDI 相同：小鼓 38、Hi-Hat 42、Crash 49、高音 48、中音 45、Ride 51、落地 43。
  - 預設通道 10，音長 100 ms。
- **力道**：固定值；或跟手套打擊峰值（g）線性對應到 1～127，預設 1.5 g → 1、8 g → 127。
- **手套演出**：連線 → 歸零（手掌貼鼓面）→ 開始演出 → 停止。流程與前測相同，Raw 與判斷會存到 `模型包/Data/演出/模型實測/`。開始後每手先打 5 下小鼓（自動錨點）。
- **重播測試**：選一份錄好的 Raw CSV，照真實速度餵給引擎，送出同樣的 MIDI。不用手套就能測試音色和延遲。
- **即時畫面**：每一下的手、鼓、音符、力道、判斷時間；最近 200 下判斷時間的中位、95%、最大值；本程式的 CPU 和記憶體。

## 安裝
```
pip install -r requirements_midi.txt
python holo_midi_out.py
```
模型包本身需要的套件：`numpy scipy scikit-learn joblib threadpoolctl xgboost pyserial`，見模型包的 `Program/requirements.txt`。

## 效能（11/16、11/17 在本機實測）
- 電腦：Intel Core Ultra 9 275HX，64 GB；Windows 11；Python 3.12。
- 測試資料：用 11/17 09:52 的前測（165 秒、300 下）重播。

| 項目 | 結果 |
|---|---|
| 載入模型 | 約 10 秒 |
| 記憶體 | 約 830～860 MB（大部分是委員會模型） |
| CPU | 一個核心的 11～14%（照真實速度重播時） |
| 每下判斷時間（照真實速度、含 MIDI 送出） | 中位 44 ms、95% 89 ms、最大 281 ms |
| 送出 MIDI | 157/157 下都送出；左手 Crash 49、右手 Crash 57 分開正確 |

- 判斷時間大部分花在委員會模型：10 個成員的 XGBoost 加 MLP，每下約 4～5 次 XGBoost 預測，每次約 4 ms；另外歸零方向估計每下約 28 ms。
- 偶爾出現的 200～460 ms，是歸零方向修正觸發時，把前面的打擊重算一次。
- 從打擊到出聲 ≈ 偵測固定等待 80 ms + 判斷時間（中位 44 ms）+ 無線傳輸 + 音樂軟體的音訊緩衝。
- 推論只用 CPU、單執行緒（`nthread=1`），不需要 GPU。

## macOS 相容性
- 引擎程式只用 Python 標準庫、numpy、scipy、xgboost；模型以 XGBoost JSON 和 numpy 陣列存放，跨平台可讀。程式裡沒有 Windows 專用的部分；`os.startfile` 只用在原 GUI 的「開資料夾」按鈕，這裡不會用到。
- 需要的套件在 Apple Silicon（arm64）都有現成的安裝檔。xgboost 在 macOS 可能需要先 `brew install libomp`，到 Mac 上第一次執行時確認。
- tkinter：python.org 版的 Python 內建；用 Homebrew 的 Python 要另外裝 `python-tk`。
- 手套連線：
  - 無線（UDP-L／UDP-R）：與 Windows 相同。
  - USB：埠名是 `/dev/cu.usbserial-…` 這類，不是 COM4／COM5，要在下拉選單裡選。
- 速度：M1 以後的 Apple Silicon 單核效能與本機差不多，預估判斷時間相近；Intel 舊款 Mac 可能慢 1.5～2 倍。**實際數字要在 Mac 上用「重播測試」量一次**，畫面上會直接顯示中位與 95%。

## Ableton Live 設定
1. 偏好設定 → Link, Tempo & MIDI：
   - MIDI 輸入選「HoloGrip」（Mac）或 loopMIDI 的埠（Windows），Track 打開。
2. 新增一個 MIDI 軌，放 Drum Rack（或任何鼓組），監聽設為 In 或 Auto，軌道開啟錄音預備（Arm）。
3. 在本程式按「試打」，確認每顆鼓的音色對應；要換音色就改音符編號。
4. 音訊緩衝（Buffer Size）設小一點，例如 128 samples，可以少大約 3～10 ms 延遲。
