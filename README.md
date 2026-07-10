# HoloGrip: Wireless Air Drum System (無線空氣鼓系統)

HoloGrip 是一款基於九軸慣性量測單元 (IMU) 與 ESP32 控制板的穿戴式無線空氣鼓實時打擊與動作分類系統。本系統採用 10ms (100Hz) 極低延遲 UDP 傳輸，並整合了動態重力自適應基準、自適應防彈跳冷卻時間、防抬手與水平晃動過濾等先進嵌入式信號處理演算法，搭配電腦端 KNN 分類器實現高精度的虛擬擊鼓體驗。

---

## 📁 專案結構 (Project Directory Structure)

本專案倉庫主要包含以下模組與說明文檔：

* **📁 `UDP_version_release/`** (原 `UDP正式版`)
  * 包含實時打擊接收與動作分類的電腦端 GUI 軟體 `server.py`。
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
