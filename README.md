# HoloGrip: Wireless Air Drum System (無線空氣鼓系統)

HoloGrip 是一款基於九軸慣性量測單元 (IMU) 與 ESP32 控制板的穿戴式無線空氣鼓實時打擊與動作分類系統。本系統採用 10ms (100Hz) 極低延遲 UDP 傳輸，並整合了動態重力自適應基準、自適應防彈跳冷卻時間、防抬手與水平晃動過濾等先進嵌入式信號處理演算法，搭配電腦端 KNN 分類器實現高精度的虛擬擊鼓體驗。

---

## 📁 專案結構 (Project Directory Structure)

本專案倉庫主要包含以下模組與說明文檔：

* **📁 `UDP_version_release/`** (原 `UDP正式版`)
  * 包含實時打擊接收與動作分類的電腦端 GUI 軟體 `server.py`。
* **📁 `Song_Collection_COM/`**
  * 流音歌曲資料收集的有線 COM 專用入口；CSV 與 UDP 資料分開保存。
* **📁 `Gloves/`**
  * 包含手套端最新 ESP32 UDP 發送韌體 `Gloves.ino`。
* **📁 `Docs/`**
  * 包含數據集結構與科研分析規格說明文檔。
* **📁 `Latency_Test/`**
  * 包含延遲測試、功耗分析與傳感器信號評估的基準代碼與實測報告。

---

## 🚀 快速開始 (Quick Start)

### 1. 手套端韌體燒錄 (ESP32)
1. 使用 Arduino IDE 打開 [Gloves/Gloves.ino](Gloves/Gloves.ino)。
2. 配置 Wi-Fi 名稱與密碼 (SSID / Password)。
3. 在 `targetIP` 中配置運行電腦端伺服器的 IP 地址（如行動熱點網關 `192.168.137.1`）。
4. 設定 `HAND_ID` (右手燒錄 `'R'`，左手燒錄 `'L'`)。
5. 連接手套開發板並上傳燒錄。

### 2. 電腦端伺服器運行
1. 確保電腦已連入同一個熱點 Wi-Fi。
2. 安裝 Python 相關依賴庫：
   ```bash
   pip install customtkinter scikit-learn
   ```
3. 運行伺服器：
   ```bash
   python UDP_version_release/server.py
   ```
4. 在介面中進行校準、訓練與擊鼓 Demo。

---

## 📊 數據收集與學術分析 (Data Specification)

為了學術研究與性能對比，系統支持**雙模式數據分流儲存**。詳細的欄位定義與物理意義請參閱 [Docs/HoloGrip_數據集欄位規格說明書_ZH_TW.md](Docs/HoloGrip_數據集欄位規格說明書_ZH_TW.md)：
* **訓練資料集 (Train Data)**：由 14 個核心資料欄位組成，供機器學習模型離線學習。
* **打擊報告集 (Report Data)**：由 16 個核心資料欄位組成，於 Demo 模式下獨立從頭累計，供真實環境下的時延、丟包率與 AI 辨識準確率 (Accuracy) 分析。

---

*本專案供 HoloGrip 研發團隊、林老師團隊及學術交接使用。*

## 歌曲資料收集端（流音系使用）

歌曲資料收集請使用獨立的收集程式，不需要啟動 KNN 訓練介面。兩種入口輸出相同的時間軸欄位，差別只在傳輸方式：

### UDP 無線版

第一次使用先安裝介面套件：

```bash
python -m pip install -r UDP_version_release/requirements-song-collection.txt
```

```bash
python UDP_version_release/song_collection_udp.py
```

適用於手套透過 Wi-Fi 將 `D,R,...`／`D,L,...` 封包送到電腦 `8888` port 的情況。
Windows 也可以直接雙擊 `UDP_version_release/run_song_collection_udp.bat`。

### COM 有線版

先安裝一次歌曲收集端套件：

```bash
python -m pip install -r Song_Collection_COM/requirements.txt
```

請使用獨立入口：

```bash
python Song_Collection_COM/song_collection_com.py
```

COM 版需要使用 `Gloves_Firmware_INO_COM/Gloves_Firmware_INO_COM.ino`。左右手各燒錄一次，分別將 `HAND_ID` 設成 `R` 與 `L`，並在介面選擇兩個 COM 埠。
Windows 也可以直接雙擊 `Song_Collection_COM/run_song_collection_com.bat`。

### 介面中的兩種資料模式

* `100 Hz 原始串流`：每個感測封包都寫入 CSV，不經過彈跳、峰值或動作過濾，適合事後重新標註與重新設計演算法。
* `彈跳後打擊事件`：只寫入通過現有局部峰值、抬手／水平動作過濾與 150–250 ms 動態防彈跳的有效打擊。

按下「開始歌曲收集」時建立 `song_time_ms = 0` 起點；請在按下後立即播放與 MIDI 對應的歌曲。COM CSV 會寫入 `CSV_Data/Song_Collection_COM/`；UDP CSV 則寫入 `CSV_Data/UDP_Collections/`，並由背景 writer 執行緒寫檔。
