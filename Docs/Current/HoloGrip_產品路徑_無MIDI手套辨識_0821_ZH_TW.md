# 做法：左手／右手打哪一顆（無 MIDI 推論）

現場：ESP 只傳 100 Hz。Python 輸出 `(手, 鼓)`。  
MIDI／影片只當老師，不進模型。

```text
手套 100Hz → HitDetector（有沒有打）→ 該手 ±80ms IMU → 7 類鼓位
手別 = 封包 HAND_ID
偵測：2.3g、重打100ms／輕打150ms、≥4g 放行動作濾波
```

## Song 2 現行數字

| 指標 | 結果 |
|---|---|
| **手+鼓都對** | **287／307（93.5%）** |
| 打擊召回 | 301／307（98.0%） |
| 打擊精確率 | 301／337（89.3%） |
| 偵測後鼓位 | 287／301（95.3%） |
| 鼓位錯／多偵測／漏打 | 14／36／6 |

即時 Demo（播完定格上表）：

[HoloGrip_Song2_產品Demo_即時正確率_0821_ZH_TW.html](file:///C:/Users/Typelin_Station/Desktop/HoloGrip/Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/HoloGrip_Song2_產品Demo_即時正確率_0821_ZH_TW.html)

報告：`Docs/Current/HoloGrip_兩階段推論_偵測優化帶動鼓位_0821_ZH_TW.md`

現場手套 GUI（插 USB → 打）：

1. 桌面 `HoloGrip現場打擊.bat`，或 `Apps/Song_Collection_COM/run_live_hit_and_zone_0821_ZH_TW.bat`
2. 掃描兩個 COM → 連接 → 左／右都約 100 Hz → 面向鼓組按歸零 → 打
3. 畫面輸出「哪隻手 + 打哪一顆」。手別 = 手套 HAND_ID，不要重燒韌體，不要開採集 UI

腳本：`Apps/Song_Collection_COM/run_live_hit_and_zone_0821_ZH_TW.py`