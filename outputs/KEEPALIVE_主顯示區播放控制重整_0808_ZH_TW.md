# KEEPALIVE｜主顯示區播放控制重整

## 已完成

- 設定區把 MIDI、CSV、影片、WAV、報告檔案與時間設定集中到頁面前方。
- 影片改為整行上方顯示。
- MIDI＋CSV 局部活動圖改為影片下方整行顯示。
- 播放、上一筆／下一筆 MIDI、倍速、雙手乾淨區、異常區控制移到圖表下方。
- 自動播放起點只強制對齊一次；播放期間不再逐幀寫入影片 `currentTime`。
- 手動事件切換與手動時間滑桿仍會強制跳到影片／WAV 對應位置。
- 移除設定區第二個播放速度選擇，統一使用下方播放控制的倍速。

## 14.52 秒說明

`MIDI → 影片 +14,520 ms` 是影片音訊、外部 WAV 與 MIDI onset 交叉比對後的目前候選；`MIDI → WAV +0 ms` 是另一條獨立證據。第一個 Crash 的 MIDI／WAV／影片時間約為 `4318.182 / 4320 / 18840 ms`。

## 本次補強

- MIDI → CSV 偏移與採樣窗口移到資料設定區；工作流程導覽改放在設定之後。
- 播放控制分成完整時間軸、目前窗口、區段導覽三組，全部位於 MIDI＋CSV 圖表下方。
- 完整播放自然結束與手動切換事件時，會停止媒體並退出自動跟隨模式。
- `+14,520 ms` 不是只看影像、音譜或 WAV onset 單獨推導的不可變真值，仍需用清楚的 Crash／Snare 畫面確認手別與鼓面。

## 驗證待辦

- 已完成 5 個 script block 語法檢查。
- 待使用者本機重新開啟 HTML，確認實際畫面和自動播放時媒體自然前進。

## 0808 交付狀態

- 已完成四方資料分析：影片 +14,520 ms、WAV +0 ms、CSV +16,000 ms；60 秒局部搜尋未發現需要拉伸的漂移。
- 已修正連續時間邏輯：總覽、局部折線、採樣窗口、密集區層都使用連續 S.time。
- 已加入「播放目前選取 MIDI 片段」：以選取事件為中心播放 ±採樣窗口，影片／WAV／時間軸同步掃過後停止。
- 已將 WAV 音訊區移到事件導覽之後，播放複核控制列設為可見的 sticky 區域。
- 待做：重新開啟 HTML 做使用者端驗收；file:// 仍可能阻擋預設影片／WAV，需在資料設定選檔。

交付路徑：

- C:\Users\Typelin_Station\Desktop\HoloGrip\Apps\Song_Collection_COM\HoloGrip_MIDI_CSV對齊檢查_0807_ZH_TW.html
- C:\Users\Typelin_Station\Desktop\HoloGrip\Tools\Audio\analyze_full_midi_wav_video_csv_alignment_0808_ZH_TW.py
- C:\Users\Typelin_Station\Desktop\HoloGrip\Data\Derived\Song_Collection_COM\S20260805_P01_song01\video_alignment_0808\full_alignment_report_0808_ZH_TW.json
